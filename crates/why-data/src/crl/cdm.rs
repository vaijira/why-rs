//! Causal decision models (Def. 8.1.3), environments, and causal RL tasks
//! (Def. 8.1.4, Algorithm 19).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use rssn::symbolic::core::Expr;

use super::{
    CrlError, Dataset, NucReport, Plan, Policy, PolicySpace, Reward, SbcReport, State, Uniform,
    conditions, diagram, optimize,
};
use crate::fairness::SplitMix64;
use crate::graph::dseparation::Admg;
use crate::scm::{Model, Value, eval, free_vars};

/// Something an agent can watch (`see`) or act in (`do`), one episode at a
/// time.
pub trait Environment {
    /// The variables of every sample, in order.
    fn variables(&self) -> Vec<String>;

    /// The values an action may take, if the environment knows the action.
    fn domain(&self, action: &str) -> Option<Vec<String>>;

    /// One sample `V ~ P(V)`: the behaviour policy picks the actions.
    ///
    /// # Errors
    ///
    /// Whatever the environment cannot evaluate.
    fn observe(&mut self, uniform: &mut dyn FnMut() -> f64) -> Result<Vec<String>, CrlError>;

    /// One sample `V ~ P_π(V)`: the agent picks each action of `space` from
    /// `policy`, after observing its input states.
    ///
    /// # Errors
    ///
    /// Whatever the environment cannot evaluate, or a policy that is not a
    /// distribution over the action's domain.
    fn intervene(
        &mut self,
        space: &PolicySpace,
        policy: &dyn Policy,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Result<Vec<String>, CrlError>;
}

/// One step of simulating the model.
enum Step<'a> {
    /// Evaluate a structural equation.
    Equation(&'a str, &'a Expr),
    /// Draw stage `i`'s action from the policy.
    Action(usize),
}

/// A causal decision model `⟨M*, Π, R⟩` (Def. 8.1.3).
///
/// It knows the environment `M*` completely, so it can compute everything
/// exactly by enumerating the exogenous variables:
///
/// * [`Cdm::evaluate`] gives `E_π[R(Y); M*]`;
/// * [`Cdm::optimal`] solves the planning problem `argmax_π E_π[R(Y); M*]`
///   (Eq. 8.10);
/// * [`Cdm::observational_data`] and [`Cdm::interventional_data`] give the
///   exact weighted tables of `P(V)` and `P_π(V)`, on which [`super::ipw`]
///   and [`super::dp`] return population quantities.
///
/// It is also an [`Environment`], so online learners can sample from it.
///
/// ```
/// use why_data::crl::{Cdm, PolicySpace, Reward, Deterministic, State};
/// use why_data::scm::{expr, Dist, Model};
///
/// // A bandit with an unobserved confounder (Ex. 8.18, in tenths):
/// // X ← U < 8, Y ← U < 4 − X.
/// let u = Dist::Categorical(vec![0.1; 10]);
/// let model = Model::new(
///     [
///         ("X".to_string(), expr::lt(expr::var("U"), expr::int(8))),
///         ("Y".to_string(), expr::lt(expr::var("U"), expr::sub(expr::int(4), expr::var("X")))),
///     ],
///     [("U".to_string(), u)],
/// )
/// .unwrap();
/// let cdm = Cdm::new(model, PolicySpace::bandit("X"), Reward::total(["Y"])).unwrap();
///
/// let arm0 = Deterministic(|_: &str, _: &State<'_>| "0".to_string());
/// assert!((cdm.evaluate(&arm0).unwrap() - 0.4).abs() < 1e-12);
/// assert!(!cdm.nuc().unwrap().holds());
/// ```
#[derive(Debug, Clone)]
pub struct Cdm {
    model: Model,
    space: PolicySpace,
    reward: Reward,
    /// The domain of every endogenous variable: its observational support,
    /// widened to both booleans for a boolean variable, unless overridden.
    domains: HashMap<String, Vec<Value>>,
}

impl Cdm {
    /// Builds a CDM. It checks `space` against the model's [`diagram`]
    /// (Def. 8.1.1) and that every reward signal is endogenous.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidSpace`] or [`CrlError::Unknown`] for a bad space or
    /// signal. [`CrlError::Scm`] if the model cannot be evaluated.
    pub fn new(model: Model, space: PolicySpace, reward: Reward) -> Result<Self, CrlError> {
        space.validate(&diagram(&model))?;
        let world = model.world(model.observational()).world();
        for y in reward.signals() {
            if !world.variables().contains(y) {
                return Err(CrlError::Unknown(y.clone()));
            }
        }

        let mut support: HashMap<String, Vec<Value>> = HashMap::new();
        for (mut env, _) in model.exogenous_assignments() {
            for (k, e) in world.equations() {
                let v = eval(e, &env)?;
                env.insert(k.clone(), v);
                let s = support.entry(k.clone()).or_default();
                if !s.contains(&v) {
                    s.push(v);
                }
            }
        }
        for s in support.values_mut() {
            if s.iter().all(|v| matches!(v, Value::Bool(_))) {
                *s = vec![Value::Bool(false), Value::Bool(true)];
            }
            s.sort();
        }
        Ok(Self {
            model,
            space,
            reward,
            domains: support,
        })
    }

    /// Overrides the domain of an action, e.g. to allow a value the
    /// behaviour policy never takes.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] if `action` is not endogenous, and
    /// [`CrlError::InvalidInput`] for an empty domain.
    pub fn with_domain(
        mut self,
        action: &str,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<Self, CrlError> {
        let entry = self
            .domains
            .get_mut(action)
            .ok_or_else(|| CrlError::Unknown(action.to_string()))?;
        let values: Vec<Value> = values.into_iter().collect();
        if values.is_empty() {
            return Err(CrlError::InvalidInput(format!("empty domain for `{action}`")));
        }
        *entry = values;
        Ok(self)
    }

    /// The environment `M*`.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The policy space `Π`.
    #[must_use]
    pub fn space(&self) -> &PolicySpace {
        &self.space
    }

    /// The reward function `R`.
    #[must_use]
    pub fn reward(&self) -> &Reward {
        &self.reward
    }

    /// The causal diagram of `M*` (see [`diagram`]).
    #[must_use]
    pub fn diagram(&self) -> Admg {
        diagram(&self.model)
    }

    /// Whether NUC holds for `Π` in `M*` (Def. 8.2.1).
    ///
    /// # Errors
    ///
    /// As [`super::nuc`].
    pub fn nuc(&self) -> Result<NucReport, CrlError> {
        conditions::nuc(&self.model, &self.space)
    }

    /// Whether `Π` is back-door admissible w.r.t. `Y` in the diagram of `M*`
    /// (Def. 8.2.3).
    ///
    /// # Errors
    ///
    /// As [`super::sequential_backdoor`].
    pub fn sequential_backdoor(&self) -> Result<SbcReport, CrlError> {
        let signals: Vec<&str> = self.reward.signals().iter().map(String::as_str).collect();
        conditions::sequential_backdoor(&self.diagram(), &self.space, &signals)
    }

    /// `π_UNIF` over the domains of `Π`'s actions.
    #[must_use]
    pub fn uniform(&self) -> Uniform {
        Uniform::new(self.space.actions().into_iter().map(|a| {
            (
                a.to_string(),
                self.domains[a].iter().map(ToString::to_string).collect::<Vec<_>>(),
            )
        }))
    }

    /// `E_π[R(Y); M*]`, exactly.
    ///
    /// # Errors
    ///
    /// [`CrlError::PolicyNotNormalized`] if a rule is not a distribution over
    /// its action's domain. [`CrlError::Scm`] if an equation cannot be
    /// evaluated.
    pub fn evaluate(&self, policy: &dyn Policy) -> Result<f64, CrlError> {
        let steps = self.steps(Some(&self.space))?;
        let signals = self.reward.signals();
        let mut total = 0.0;
        self.enumerate(&steps, Some((&self.space, policy)), &mut |env, p| {
            let y: Vec<f64> = signals.iter().map(|s| env[s].as_int() as f64).collect();
            total += p * self.reward.apply(&y);
        })?;
        Ok(total)
    }

    /// The exact table of `P(V)`: one weighted row per assignment of the
    /// endogenous variables with positive probability.
    ///
    /// # Errors
    ///
    /// [`CrlError::Scm`] if an equation cannot be evaluated.
    pub fn observational_data(&self) -> Result<Dataset, CrlError> {
        let steps = self.steps(None)?;
        self.table(&steps, None)
    }

    /// The exact table of `P_π(V)` under `do(π)`.
    ///
    /// # Errors
    ///
    /// As [`Cdm::evaluate`].
    pub fn interventional_data(&self, policy: &dyn Policy) -> Result<Dataset, CrlError> {
        let steps = self.steps(Some(&self.space))?;
        self.table(&steps, Some((&self.space, policy)))
    }

    /// Solves the planning problem `π* = argmax_{π ∈ Π} E_π[R(Y); M*]`
    /// (Eq. 8.10) over deterministic policies.
    ///
    /// It runs [`optimize`] on the exact table of `P_{π_UNIF}(V)`. NUC holds
    /// in that submodel (Lemma 8.2.3), and every action has positive
    /// probability after every reachable history, so the result is exact.
    ///
    /// # Errors
    ///
    /// As [`optimize`].
    pub fn optimal(&self) -> Result<Plan, CrlError> {
        let data = self.interventional_data(&self.uniform())?;
        optimize(&data, &self.space, &self.reward)
    }

    /// The order in which to evaluate the model: topological in the
    /// observational world, or in `G_π` (actions read their `S_i`) under a
    /// policy over `space`.
    fn steps(&self, space: Option<&PolicySpace>) -> Result<Vec<Step<'_>>, CrlError> {
        let world = self.model.world(self.model.observational()).world();
        let Some(space) = space else {
            return Ok(world
                .equations()
                .iter()
                .map(|(k, e)| Step::Equation(k, e))
                .collect());
        };
        for a in space.actions() {
            if !self.domains.contains_key(a) {
                return Err(CrlError::Unknown(a.to_string()));
            }
        }

        let vars = world.variables();
        let endogenous: HashSet<&str> = vars.iter().map(String::as_str).collect();
        let stage_of: HashMap<&str, usize> = space
            .actions()
            .into_iter()
            .enumerate()
            .map(|(i, a)| (a, i))
            .collect();
        let deps: Vec<Vec<&str>> = vars
            .iter()
            .map(|v| {
                if let Some(&i) = stage_of.get(v.as_str()) {
                    space.stages()[i]
                        .states
                        .iter()
                        .map(String::as_str)
                        .collect()
                } else {
                    let (_, e) = world
                        .equations()
                        .iter()
                        .find(|(k, _)| k == v)
                        .expect("every endogenous variable has an equation");
                    let mut fv = HashSet::new();
                    free_vars(e, &mut fv);
                    vars.iter()
                        .map(String::as_str)
                        .filter(|x| fv.contains(*x) && endogenous.contains(x))
                        .collect()
                }
            })
            .collect();

        // Kahn's algorithm, preferring declaration order among ready nodes.
        let mut done: HashSet<&str> = HashSet::new();
        let mut steps = Vec::with_capacity(vars.len());
        while steps.len() < vars.len() {
            let next = vars
                .iter()
                .enumerate()
                .find(|(i, v)| !done.contains(v.as_str()) && deps[*i].iter().all(|d| done.contains(d)));
            let Some((_, v)) = next else {
                return Err(CrlError::InvalidSpace(
                    "intervening with this policy space creates a cycle".into(),
                ));
            };
            done.insert(v);
            steps.push(match stage_of.get(v.as_str()) {
                Some(&i) => Step::Action(i),
                None => {
                    let (k, e) = world
                        .equations()
                        .iter()
                        .find(|(k, _)| k == v)
                        .expect("every endogenous variable has an equation");
                    Step::Equation(k, e)
                }
            });
        }
        Ok(steps)
    }

    /// `π_i(· | s_i)` over the domain of `X_i`, read from `env`.
    fn rule(
        &self,
        space: &PolicySpace,
        i: usize,
        policy: &dyn Policy,
        env: &HashMap<String, Value>,
    ) -> Result<Vec<(Value, f64)>, CrlError> {
        let stage = &space.stages()[i];
        let rendered: Vec<String> = stage.states.iter().map(|s| env[s].to_string()).collect();
        let pairs: Vec<(&str, &str)> = stage
            .states
            .iter()
            .map(String::as_str)
            .zip(rendered.iter().map(String::as_str))
            .collect();
        let state = State::new(&pairs);
        let probs: Vec<(Value, f64)> = self.domains[&stage.action]
            .iter()
            .map(|v| (*v, policy.probability(&stage.action, &state, &v.to_string())))
            .collect();
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        if probs.iter().any(|(_, p)| *p < 0.0) || (total - 1.0).abs() > 1e-9 {
            return Err(CrlError::PolicyNotNormalized {
                action: stage.action.clone(),
                state: state.to_string(),
                total,
            });
        }
        Ok(probs)
    }

    /// Visits every complete assignment with its probability, branching on
    /// each action according to the policy.
    fn enumerate(
        &self,
        steps: &[Step<'_>],
        policy: Option<(&PolicySpace, &dyn Policy)>,
        visit: &mut dyn FnMut(&HashMap<String, Value>, f64),
    ) -> Result<(), CrlError> {
        for (mut env, p) in self.model.exogenous_assignments() {
            if p > 0.0 {
                self.descend(steps, policy, &mut env, p, visit)?;
            }
        }
        Ok(())
    }

    fn descend(
        &self,
        steps: &[Step<'_>],
        policy: Option<(&PolicySpace, &dyn Policy)>,
        env: &mut HashMap<String, Value>,
        p: f64,
        visit: &mut dyn FnMut(&HashMap<String, Value>, f64),
    ) -> Result<(), CrlError> {
        let Some((step, rest)) = steps.split_first() else {
            visit(env, p);
            return Ok(());
        };
        match step {
            Step::Equation(k, e) => {
                let v = eval(e, env)?;
                env.insert((*k).to_string(), v);
                self.descend(rest, policy, env, p, visit)
            }
            Step::Action(i) => {
                let (space, pi) = policy.expect("actions only appear under a policy");
                let action = &space.stages()[*i].action;
                for (v, q) in self.rule(space, *i, pi, env)? {
                    if q > 0.0 {
                        env.insert(action.clone(), v);
                        self.descend(rest, policy, env, p * q, visit)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn table(
        &self,
        steps: &[Step<'_>],
        policy: Option<(&PolicySpace, &dyn Policy)>,
    ) -> Result<Dataset, CrlError> {
        let vars = self.variables();
        let mut rows: BTreeMap<Vec<Value>, f64> = BTreeMap::new();
        self.enumerate(steps, policy, &mut |env, p| {
            let key: Vec<Value> = vars.iter().map(|v| env[v]).collect();
            *rows.entry(key).or_insert(0.0) += p;
        })?;
        rows.retain(|_, p| *p > 0.0);
        let weights: Vec<f64> = rows.values().copied().collect();
        let records = rows
            .keys()
            .map(|k| k.iter().map(ToString::to_string).collect::<Vec<_>>());
        Ok(Dataset::from_records(vars, records)?.with_weights(weights)?)
    }

    fn sample(
        &self,
        steps: &[Step<'_>],
        policy: Option<(&PolicySpace, &dyn Policy)>,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Result<Vec<String>, CrlError> {
        let mut env: HashMap<String, Value> = HashMap::new();
        for u in self.model.exogenous() {
            let dist = self.model.distribution(u).expect("declared exogenous");
            let options: Vec<(Value, f64)> = dist
                .support()
                .into_iter()
                .map(|v| (v, dist.probability(v)))
                .collect();
            env.insert(u.clone(), draw(&options, uniform()));
        }
        for step in steps {
            match step {
                Step::Equation(k, e) => {
                    let v = eval(e, &env)?;
                    env.insert((*k).to_string(), v);
                }
                Step::Action(i) => {
                    let (space, pi) = policy.expect("actions only appear under a policy");
                    let options = self.rule(space, *i, pi, &env)?;
                    env.insert(space.stages()[*i].action.clone(), draw(&options, uniform()));
                }
            }
        }
        Ok(self
            .variables()
            .iter()
            .map(|v| env[v].to_string())
            .collect())
    }
}

/// Inverse-CDF draw from a finite distribution.
fn draw(options: &[(Value, f64)], u: f64) -> Value {
    let mut target = u * options.iter().map(|(_, p)| p).sum::<f64>();
    for (v, p) in options {
        if *p <= 0.0 {
            continue;
        }
        target -= p;
        if target < 0.0 {
            return *v;
        }
    }
    options
        .iter()
        .rev()
        .find(|(_, p)| *p > 0.0)
        .map_or(options[0].0, |(v, _)| *v)
}

impl Environment for Cdm {
    fn variables(&self) -> Vec<String> {
        self.model
            .world(self.model.observational())
            .world()
            .variables()
            .to_vec()
    }

    fn domain(&self, action: &str) -> Option<Vec<String>> {
        self.domains
            .get(action)
            .map(|d| d.iter().map(ToString::to_string).collect())
    }

    fn observe(&mut self, uniform: &mut dyn FnMut() -> f64) -> Result<Vec<String>, CrlError> {
        let steps = self.steps(None)?;
        self.sample(&steps, None, uniform)
    }

    fn intervene(
        &mut self,
        space: &PolicySpace,
        policy: &dyn Policy,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Result<Vec<String>, CrlError> {
        space.validate(&self.diagram())?;
        let steps = self.steps(Some(space))?;
        self.sample(&steps, Some((space, policy)), uniform)
    }
}

/// How a CRL agent interacts with the environment (`L`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// `L = see`: passive observation, `D ~ P(V)` (off-policy learning).
    See,
    /// `L = do`: intervention, `D ~ P_π(V)` (online learning).
    Do,
}

/// What the agent assumes about `M*` (`A`).
#[derive(Debug, Clone)]
pub enum Assumptions {
    /// Nothing beyond the temporal order of `Π`.
    None,
    /// No Unmeasured Confounder (Def. 8.2.1), taken on faith.
    Nuc,
    /// A causal diagram compatible with `M*`.
    Diagram(Admg),
}

/// Why the agent may evaluate policies from its data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Justification {
    /// NUC is assumed (Thms 8.2.1 and 8.2.2).
    Nuc,
    /// The diagram satisfies the sequential back-door (Thm 8.2.8).
    SequentialBackdoor,
    /// The data come from intervening, so NUC holds in the submodel
    /// (Lemma 8.2.3).
    ExperimentalNuc,
}

impl fmt::Display for Justification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Nuc => "NUC (assumed)",
            Self::SequentialBackdoor => "sequential back-door (Thm 8.2.8)",
            Self::ExperimentalNuc => "experimental NUC (Lemma 8.2.3)",
        })
    }
}

/// A causal reinforcement learning task `⟨L, A, Π, R⟩` (Def. 8.1.4).
#[derive(Debug, Clone)]
pub struct Task {
    /// The learning regime `L`.
    pub regime: Regime,
    /// The structural assumptions `A`.
    pub assumptions: Assumptions,
    /// The policy space `Π`.
    pub space: PolicySpace,
    /// The reward function `R`.
    pub reward: Reward,
}

/// What [`Task::run`] learned.
#[derive(Debug, Clone)]
pub struct Learned {
    /// Why evaluating policies from `data` is valid.
    pub justification: Justification,
    /// Every sample collected.
    pub data: Dataset,
    /// The estimated optimal policy and its estimated value.
    pub plan: Plan,
}

impl Task {
    /// A task from its four components.
    #[must_use]
    pub fn new(regime: Regime, assumptions: Assumptions, space: PolicySpace, reward: Reward) -> Self {
        Self {
            regime,
            assumptions,
            space,
            reward,
        }
    }

    /// Which result licenses off-policy evaluation in this task (Table 8.3,
    /// Fig. 8.12).
    ///
    /// # Errors
    ///
    /// [`CrlError::NotIdentifiable`] for `see` with no assumptions, or with a
    /// diagram where the sequential back-door fails (the message names the
    /// stage and an open path). [`CrlError::InvalidSpace`] if `Π` violates
    /// Def. 8.1.1 in the diagram.
    pub fn justification(&self) -> Result<Justification, CrlError> {
        match (self.regime, &self.assumptions) {
            (Regime::Do, _) => Ok(Justification::ExperimentalNuc),
            (Regime::See, Assumptions::Nuc) => Ok(Justification::Nuc),
            (Regime::See, Assumptions::None) => Err(CrlError::NotIdentifiable(
                "observational data with no structural assumptions identify no policy value; \
                 assume NUC or supply a causal diagram"
                    .into(),
            )),
            (Regime::See, Assumptions::Diagram(g)) => {
                self.space.validate(g)?;
                let signals: Vec<&str> =
                    self.reward.signals().iter().map(String::as_str).collect();
                let report = conditions::sequential_backdoor(g, &self.space, &signals)?;
                match report.stages.iter().find(|s| !s.holds) {
                    None => Ok(Justification::SequentialBackdoor),
                    Some(s) => Err(CrlError::NotIdentifiable(format!(
                        "the sequential back-door fails at {} given {{{}}}{}",
                        s.action,
                        s.conditioning.join(", "),
                        s.open_path
                            .as_ref()
                            .map(|p| format!(": {p} is open"))
                            .unwrap_or_default()
                    ))),
                }
            }
        }
    }

    /// Algorithm 19. It first checks the [`Task::justification`], then
    /// interacts with `env` for `episodes` episodes under the regime. Under
    /// `see` it observes; under `do` it follows `π_UNIF`. Finally it returns
    /// the policy [`optimize`] picks from the data.
    ///
    /// # Errors
    ///
    /// As [`Task::justification`] and [`optimize`]. [`CrlError::Unknown`] if
    /// the environment does not know an action or a column.
    pub fn run(
        &self,
        env: &mut dyn Environment,
        episodes: usize,
        seed: u64,
    ) -> Result<Learned, CrlError> {
        let justification = self.justification()?;
        let mut rng = SplitMix64::new(seed);
        let mut uniform = || rng.unit();
        let pi = uniform_for(env, &self.space)?;
        let mut rows = Vec::with_capacity(episodes);
        for _ in 0..episodes {
            rows.push(match self.regime {
                Regime::See => env.observe(&mut uniform)?,
                Regime::Do => env.intervene(&self.space, &pi, &mut uniform)?,
            });
        }
        let data = Dataset::from_records(env.variables(), rows)?;
        let plan = optimize(&data, &self.space, &self.reward)?;
        Ok(Learned {
            justification,
            data,
            plan,
        })
    }
}

/// `π_UNIF` over the environment's domains for `space`'s actions.
pub(crate) fn uniform_for(env: &dyn Environment, space: &PolicySpace) -> Result<Uniform, CrlError> {
    let mut domains = Vec::with_capacity(space.horizon());
    for a in space.actions() {
        let d = env.domain(a).ok_or_else(|| CrlError::Unknown(a.to_string()))?;
        domains.push((a.to_string(), d));
    }
    Ok(Uniform::new(domains))
}
