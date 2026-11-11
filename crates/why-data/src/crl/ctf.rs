//! Counterfactual decision-making (Sec. 9.4, CRL Task 3).
//!
//! An agent that intercepts its own intended action `X′` before acting can
//! condition on it. In a bandit this optimises the effect of the treatment
//! on the treated `E[Y_{X←x} | X = x′]` (Eq. 9.115) rather than `E_x[Y]`.
//! An optimal counterfactual policy is never worse than an optimal
//! interventional one, and strictly better when `X′` carries information
//! about the confounder (Thms 9.4.1 and 9.4.3).

use std::collections::HashMap;

use super::cdm::{endogenous_domains, evaluate_world, sample_exogenous};
use super::{CrlError, Reward};
use crate::fairness::SplitMix64;
use crate::scm::{Model, Value};

fn value_of(domain: &[Value], s: &str) -> Option<Value> {
    domain.iter().find(|v| v.to_string() == s).copied()
}

fn reward_at(env: &HashMap<String, Value>, reward: &Reward) -> f64 {
    let y: Vec<f64> = reward
        .signals()
        .iter()
        .map(|s| env[s].as_int() as f64)
        .collect();
    reward.apply(&y)
}

fn check_vars(model: &Model, vars: &[&str]) -> Result<(), CrlError> {
    let world = model.world(model.observational()).world();
    for v in vars {
        if !world.variables().iter().any(|w| w == v) {
            return Err(CrlError::Unknown((*v).to_string()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bandits
// ---------------------------------------------------------------------------

/// The effect of the treatment on the treated, `E[R(Y)_{X←x} | X = x′]`,
/// for every intended arm `x′` and realised arm `x` (Def. 9.4.1), computed
/// exactly in an SCM.
#[derive(Debug, Clone, PartialEq)]
pub struct EttTable {
    /// The action variable.
    pub action: String,
    /// The arms, sorted.
    pub arms: Vec<String>,
    /// `P(X = x′)`, the behaviour policy's distribution, per arm.
    pub intended: Vec<f64>,
    /// `values[x′][x] = E[R(Y)_{X←x} | X = x′]`. Rows with `P(x′) = 0` are
    /// zero.
    pub values: Vec<Vec<f64>>,
}

impl EttTable {
    /// `E[R(Y) | X = x]`, which by composition is `E[R(Y)_x | X = x]`
    /// (Eq. 9.104).
    #[must_use]
    pub fn observational(&self) -> Vec<f64> {
        (0..self.arms.len()).map(|x| self.values[x][x]).collect()
    }

    /// `E_x[R(Y)] = Σ_{x′} P(x′) E[R(Y)_x | x′]`.
    #[must_use]
    pub fn interventional(&self) -> Vec<f64> {
        (0..self.arms.len())
            .map(|x| {
                (0..self.arms.len())
                    .map(|xp| self.intended[xp] * self.values[xp][x])
                    .sum()
            })
            .collect()
    }

    /// Eq. 9.123: `E_π[R(Y)] = Σ_{x,x′} E[R(Y)_x | x′] π(x | x′) P(x′)`, for
    /// a counterfactual policy given as `π(x | x′)` over arm indices.
    pub fn evaluate(&self, policy: impl Fn(usize, usize) -> f64) -> f64 {
        let mut total = 0.0;
        for xp in 0..self.arms.len() {
            for x in 0..self.arms.len() {
                total += self.intended[xp] * policy(x, xp) * self.values[xp][x];
            }
        }
        total
    }

    /// The optimal counterfactual policy (Eq. 9.115), as the best arm
    /// index for each intended arm, and its value. Ties go to the earlier
    /// arm.
    #[must_use]
    pub fn optimal(&self) -> (Vec<usize>, f64) {
        let best: Vec<usize> = self.values.iter().map(|row| argmax(row)).collect();
        let value = self.evaluate(|x, xp| f64::from(u8::from(best[xp] == x)));
        (best, value)
    }

    /// The best arm under `do(x)` and its value `max_x E_x[R(Y)]`.
    #[must_use]
    pub fn best_interventional(&self) -> (usize, f64) {
        let v = self.interventional();
        let x = argmax(&v);
        (x, v[x])
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Computes the [`EttTable`] of `action` in `model` by enumerating the
/// exogenous variables: for each unit, the natural choice `X(u) = x′` comes
/// from the observational world and `Y_x(u)` from the world `M_x` sharing
/// `u`.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a variable the model lacks, and
/// [`CrlError::Scm`] if an equation cannot be evaluated.
pub fn ett(model: &Model, action: &str, reward: &Reward) -> Result<EttTable, CrlError> {
    check_vars(model, &[action])?;
    let signals: Vec<&str> = reward.signals().iter().map(String::as_str).collect();
    check_vars(model, &signals)?;
    let domain = endogenous_domains(model)?.remove(action).unwrap_or_default();
    let k = domain.len();
    let mut intended = vec![0.0; k];
    let mut sums = vec![vec![0.0; k]; k];
    for (u, p) in model.exogenous_assignments() {
        if p <= 0.0 {
            continue;
        }
        let natural = evaluate_world(model, &u, &[])?;
        let xp = domain
            .iter()
            .position(|v| *v == natural[action])
            .expect("natural value lies in the domain");
        intended[xp] += p;
        for (x, v) in domain.iter().enumerate() {
            let world = evaluate_world(model, &u, &[(action, *v)])?;
            sums[xp][x] += p * reward_at(&world, reward);
        }
    }
    let values = sums
        .into_iter()
        .zip(&intended)
        .map(|(row, p)| row.into_iter().map(|s| if *p > 0.0 { s / p } else { 0.0 }).collect())
        .collect();
    Ok(EttTable {
        action: action.to_string(),
        arms: domain.iter().map(ToString::to_string).collect(),
        intended,
        values,
    })
}

/// A bandit an agent can play through counterfactual randomization: each
/// episode first reveals the intended arm, then executes the arm chosen.
pub trait CtfEnvironment {
    /// The arms.
    fn arms(&self) -> Vec<String>;

    /// Starts an episode and returns the agent's intended arm `X′`, before
    /// the decision flow executes it.
    ///
    /// # Errors
    ///
    /// Whatever the environment cannot evaluate.
    fn intend(&mut self, uniform: &mut dyn FnMut() -> f64) -> Result<String, CrlError>;

    /// Performs `do(X ← arm)` in the current episode and returns `R(Y)`.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] for an unknown arm or when no episode is
    /// open.
    fn act(&mut self, arm: &str) -> Result<f64, CrlError>;
}

/// An SCM played through counterfactual randomization (Def. 9.4.3): the
/// intended arm is `f_X(u)`, and the reward is `R(Y_x(u))` for the same
/// unit `u`.
#[derive(Debug, Clone)]
pub struct CtfBandit {
    model: Model,
    action: String,
    reward: Reward,
    domain: Vec<Value>,
    unit: Option<HashMap<String, Value>>,
}

impl CtfBandit {
    /// Wraps `model`, with `action` as the arm and `reward` over its
    /// signals.
    ///
    /// # Errors
    ///
    /// As [`ett`].
    pub fn new(model: Model, action: &str, reward: Reward) -> Result<Self, CrlError> {
        check_vars(&model, &[action])?;
        let signals: Vec<&str> = reward.signals().iter().map(String::as_str).collect();
        check_vars(&model, &signals)?;
        let domain = endogenous_domains(&model)?.remove(action).unwrap_or_default();
        Ok(Self {
            model,
            action: action.to_string(),
            reward,
            domain,
            unit: None,
        })
    }
}

impl CtfEnvironment for CtfBandit {
    fn arms(&self) -> Vec<String> {
        self.domain.iter().map(ToString::to_string).collect()
    }

    fn intend(&mut self, uniform: &mut dyn FnMut() -> f64) -> Result<String, CrlError> {
        let u = sample_exogenous(&self.model, uniform);
        let natural = evaluate_world(&self.model, &u, &[])?;
        self.unit = Some(u);
        Ok(natural[&self.action].to_string())
    }

    fn act(&mut self, arm: &str) -> Result<f64, CrlError> {
        let v = value_of(&self.domain, arm)
            .ok_or_else(|| CrlError::InvalidInput(format!("unknown arm `{arm}`")))?;
        let u = self
            .unit
            .take()
            .ok_or_else(|| CrlError::InvalidInput("no episode is open".into()))?;
        let world = evaluate_world(&self.model, &u, &[(&self.action, v)])?;
        Ok(reward_at(&world, &self.reward))
    }
}

/// The trace of a counterfactual bandit learner.
#[derive(Debug, Clone, PartialEq)]
pub struct CtfRun {
    /// The intended arm `X′` of each episode.
    pub intended: Vec<String>,
    /// The arm executed.
    pub realized: Vec<String>,
    /// The reward received.
    pub rewards: Vec<f64>,
}

impl CtfRun {
    /// The cumulative pseudo-regret against the optimal counterfactual
    /// policy, `Σ_t max_x E[Y_x | x′_t] − E[Y_{x_t} | x′_t]`.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] for an arm missing from `table`.
    pub fn pseudo_regret(&self, table: &EttTable) -> Result<Vec<f64>, CrlError> {
        let idx = |a: &str| {
            table
                .arms
                .iter()
                .position(|b| b == a)
                .ok_or_else(|| CrlError::Unknown(a.to_string()))
        };
        let mut total = 0.0;
        self.intended
            .iter()
            .zip(&self.realized)
            .map(|(xp, x)| {
                let row = &table.values[idx(xp)?];
                total += row.iter().copied().fold(f64::NEG_INFINITY, f64::max) - row[idx(x)?];
                Ok(total)
            })
            .collect()
    }
}

/// Algorithm 26, counterfactual randomized controlled trials: for `N`
/// episodes, intercept the intended arm and play a uniformly random arm;
/// afterwards play `argmax_x Ê^{(N)}[Y_{X←x} | X = x′]` (Eq. 9.187).
///
/// A pair `(x′, x)` never explored has no estimate and is skipped. If no
/// arm was explored after `x′`, the first arm is played.
#[derive(Debug, Clone, Copy)]
pub struct CtfRct {
    /// The number of exploration episodes `N`.
    pub trials: usize,
}

impl CtfRct {
    /// Runs for `episodes` episodes in `env`.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] unless `0 < N ≤ T`, and whatever `env`
    /// reports.
    pub fn run(
        &self,
        env: &mut dyn CtfEnvironment,
        episodes: usize,
        seed: u64,
    ) -> Result<CtfRun, CrlError> {
        if self.trials == 0 || self.trials > episodes {
            return Err(CrlError::InvalidInput(format!(
                "need 0 < N ≤ T, got N = {} and T = {episodes}",
                self.trials
            )));
        }
        let arms = env.arms();
        let k = arms.len();
        let mut rng = SplitMix64::new(seed);
        let mut n = vec![vec![0.0; k]; k];
        let mut s = vec![vec![0.0; k]; k];
        let mut run = CtfRun {
            intended: Vec::with_capacity(episodes),
            realized: Vec::with_capacity(episodes),
            rewards: Vec::with_capacity(episodes),
        };
        for t in 1..=episodes {
            let xp_name = env.intend(&mut || rng.unit())?;
            let xp = arm_index(&arms, &xp_name)?;
            let x = if t <= self.trials {
                rng.below(k)
            } else {
                let mut best: Option<(usize, f64)> = None;
                for x in 0..k {
                    if n[xp][x] > 0.0 {
                        let e = s[xp][x] / n[xp][x];
                        if best.is_none_or(|(_, b)| e > b) {
                            best = Some((x, e));
                        }
                    }
                }
                best.map_or(0, |(x, _)| x)
            };
            let r = env.act(&arms[x])?;
            if t <= self.trials {
                n[xp][x] += 1.0;
                s[xp][x] += r;
            }
            run.intended.push(xp_name);
            run.realized.push(arms[x].clone());
            run.rewards.push(r);
        }
        Ok(run)
    }
}

fn arm_index(arms: &[String], a: &str) -> Result<usize, CrlError> {
    arms.iter()
        .position(|b| b == a)
        .ok_or_else(|| CrlError::InvalidInput(format!("intended arm `{a}` is not an arm")))
}

/// Algorithm 27, the counterfactual UCB: one UCB instance per intended arm.
/// In episode `t` it plays the first arm maximising
///
/// ```text
/// Ê[Y_{X←x} | X = x′] + sqrt(log(1/δ) / (2 N(x′, x))),   δ = N_t(x′)^{−4}
/// ```
///
/// where `N_t(x′)` counts the episodes up to `t` whose intended arm was
/// `x′`. An unplayed pair has an infinite index. Its regret is
/// `O(K sqrt(T log T))` against the optimal counterfactual policy
/// (Thm 9.4.4), while any interventional learner can be stuck at linear
/// regret (Cor. 9.4.5).
#[derive(Debug, Clone, Copy, Default)]
pub struct CtfUcb;

impl CtfUcb {
    /// Runs for `episodes` episodes in `env`.
    ///
    /// # Errors
    ///
    /// Whatever `env` reports.
    pub fn run(
        &self,
        env: &mut dyn CtfEnvironment,
        episodes: usize,
        seed: u64,
    ) -> Result<CtfRun, CrlError> {
        let arms = env.arms();
        let k = arms.len();
        let mut rng = SplitMix64::new(seed);
        let mut seen = vec![0usize; k];
        let mut n = vec![vec![0.0; k]; k];
        let mut s = vec![vec![0.0; k]; k];
        let mut run = CtfRun {
            intended: Vec::with_capacity(episodes),
            realized: Vec::with_capacity(episodes),
            rewards: Vec::with_capacity(episodes),
        };
        for _ in 0..episodes {
            let xp_name = env.intend(&mut || rng.unit())?;
            let xp = arm_index(&arms, &xp_name)?;
            seen[xp] += 1;
            let log_inv_delta = 4.0 * (seen[xp] as f64).ln();
            let mut best = 0;
            let mut best_index = f64::NEG_INFINITY;
            for x in 0..k {
                let index = if n[xp][x] == 0.0 {
                    f64::INFINITY
                } else {
                    s[xp][x] / n[xp][x] + (log_inv_delta / (2.0 * n[xp][x])).sqrt()
                };
                if index > best_index {
                    best = x;
                    best_index = index;
                }
            }
            let r = env.act(&arms[best])?;
            n[xp][best] += 1.0;
            s[xp][best] += r;
            run.intended.push(xp_name);
            run.realized.push(arms[best].clone());
            run.rewards.push(r);
        }
        Ok(run)
    }
}

// ---------------------------------------------------------------------------
// MDPs with unobserved confounders
// ---------------------------------------------------------------------------

/// A finite MDP with unobserved confounders (Def. 9.4.4) in its
/// counterfactual form `⟨D(S) × D(X), D(X), T_ctf, R_ctf⟩` (Eq. 9.174).
///
/// The agent sees the state `s` and its intended action `x′`, then picks the
/// realised action `x`. By Lemma 9.4.2 the dynamics are the stage-invariant
/// conditional ETTs
///
/// ```text
/// T_ctf(s, x′, x, s″, x″) = P(S_{i+1, x} = s″ | s, X_i = x′) · P(X = x″ | S = s″)
/// R_ctf(s, x′, x)         = E[Y_{i, x} | s, X_i = x′]
/// ```
///
/// The next intended action factors out because the noise of each stage is
/// i.i.d. (Def. 9.4.4), so it follows the behaviour policy `P(X | S)`.
/// Indices refer to [`CtfMdp::states`] and [`CtfMdp::actions`].
#[derive(Debug, Clone, PartialEq)]
pub struct CtfMdp {
    /// The states.
    pub states: Vec<String>,
    /// The actions.
    pub actions: Vec<String>,
    /// `P(S_1 = s)`.
    pub initial: Vec<f64>,
    /// `intent[s][x′] = P(X = x′ | S = s)`, the behaviour policy.
    pub intent: Vec<Vec<f64>>,
    /// `next[s][x′][x][s″] = P(S_{i+1, x} = s″ | s, X_i = x′)`.
    pub next: Vec<Vec<Vec<Vec<f64>>>>,
    /// `reward[s][x′][x] = E[Y_{i, x} | s, X_i = x′]`.
    pub reward: Vec<Vec<Vec<f64>>>,
}

/// The variables of one stage of an MDP environment, for
/// [`CtfMdp::from_stage_model`].
#[derive(Debug, Clone, Copy)]
pub struct StageVars<'a> {
    /// The current state `S_i`.
    pub state: &'a str,
    /// The action `X_i`, whose natural value is the intended action.
    pub action: &'a str,
    /// The reward signal `Y_i`.
    pub reward: &'a str,
    /// The next state `S_{i+1}`.
    pub next_state: &'a str,
}

/// An optimal action-value function of a [`CtfMdp`].
#[derive(Debug, Clone, PartialEq)]
pub struct CtfQ {
    /// `q[s][x′][x]`.
    pub q: Vec<Vec<Vec<f64>>>,
    /// `policy[s][x′]`, the greedy realised action (ties to the earlier).
    pub policy: Vec<Vec<usize>>,
    /// `Σ_{s,x′} P(s) P(x′ | s) max_x q[s][x′][x]`.
    pub value: f64,
}

/// An optimal action-value function over `(s, x)` only, as an agent that
/// ignores its intended action sees the MDP.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpQ {
    /// `q[s][x]`.
    pub q: Vec<Vec<f64>>,
    /// `policy[s]`.
    pub policy: Vec<usize>,
    /// `Σ_s P(s) max_x q[s][x]`.
    pub value: f64,
}

impl CtfMdp {
    /// Builds the counterfactual MDP of a stationary environment from an SCM
    /// of one stage.
    ///
    /// `model` must contain the state, the action (whose structural equation
    /// is the behaviour policy `f_X`), the reward signal and the next state,
    /// with the state upstream of the action. The state's own distribution
    /// in `model` becomes `P(S_1)`. The conditional ETTs are computed by
    /// enumerating the exogenous variables, sharing each unit across the
    /// natural and intervened worlds.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] for a missing variable,
    /// [`CrlError::InvalidInput`] if the next state takes a value outside the
    /// state's domain or a state has probability zero.
    pub fn from_stage_model(model: &Model, vars: StageVars<'_>) -> Result<Self, CrlError> {
        check_vars(model, &[vars.state, vars.action, vars.reward, vars.next_state])?;
        let domains = endogenous_domains(model)?;
        let s_dom = domains[vars.state].clone();
        let x_dom = domains[vars.action].clone();
        for v in &domains[vars.next_state] {
            if !s_dom.contains(v) {
                return Err(CrlError::InvalidInput(format!(
                    "next state `{}` takes {v}, outside the domain of `{}`",
                    vars.next_state, vars.state
                )));
            }
        }
        let (ns, nx) = (s_dom.len(), x_dom.len());
        let mut p_s = vec![0.0; ns];
        let mut p_sx = vec![vec![0.0; nx]; ns];
        let mut next = vec![vec![vec![vec![0.0; ns]; nx]; nx]; ns];
        let mut reward = vec![vec![vec![0.0; nx]; nx]; ns];
        let pos = |d: &[Value], v: Value| d.iter().position(|w| *w == v).expect("in domain");
        for (u, p) in model.exogenous_assignments() {
            if p <= 0.0 {
                continue;
            }
            let natural = evaluate_world(model, &u, &[])?;
            let s = pos(&s_dom, natural[vars.state]);
            let xp = pos(&x_dom, natural[vars.action]);
            p_s[s] += p;
            p_sx[s][xp] += p;
            for (x, xv) in x_dom.iter().enumerate() {
                let world = evaluate_world(model, &u, &[(vars.action, *xv)])?;
                if world[vars.state] != natural[vars.state] {
                    return Err(CrlError::InvalidInput(format!(
                        "state `{}` depends on action `{}`",
                        vars.state, vars.action
                    )));
                }
                let s2 = pos(&s_dom, world[vars.next_state]);
                next[s][xp][x][s2] += p;
                reward[s][xp][x] += p * world[vars.reward].as_int() as f64;
            }
        }
        if let Some(s) = p_s.iter().position(|p| *p <= 0.0) {
            return Err(CrlError::InvalidInput(format!(
                "state {} has probability zero in the stage model",
                s_dom[s]
            )));
        }
        let intent: Vec<Vec<f64>> = p_sx
            .iter()
            .zip(&p_s)
            .map(|(row, ps)| row.iter().map(|p| p / ps).collect())
            .collect();
        for s in 0..ns {
            for xp in 0..nx {
                let w = p_sx[s][xp];
                for x in 0..nx {
                    if w > 0.0 {
                        reward[s][xp][x] /= w;
                        for v in &mut next[s][xp][x] {
                            *v /= w;
                        }
                    } else {
                        // An intention never formed in state s: no data, so
                        // keep a well-formed but irrelevant row.
                        next[s][xp][x] = vec![1.0 / ns as f64; ns];
                    }
                }
            }
        }
        Ok(Self {
            states: s_dom.iter().map(ToString::to_string).collect(),
            actions: x_dom.iter().map(ToString::to_string).collect(),
            initial: p_s,
            intent,
            next,
            reward,
        })
    }

    /// `T_ctf(s, x′, x, s″, x″)` (Eq. 9.175).
    #[must_use]
    pub fn transition(&self, s: usize, xp: usize, x: usize, s2: usize, x2: usize) -> f64 {
        self.next[s][xp][x][s2] * self.intent[s2][x2]
    }

    fn ns(&self) -> usize {
        self.states.len()
    }

    fn nx(&self) -> usize {
        self.actions.len()
    }

    /// `Σ_{s″,x″} T_ctf(s, x′, x, s″, x″) V(s″, x″)`.
    fn expect(&self, s: usize, xp: usize, x: usize, v: &[Vec<f64>]) -> f64 {
        let mut total = 0.0;
        for ((p, intent), vs) in self.next[s][xp][x].iter().zip(&self.intent).zip(v) {
            if *p > 0.0 {
                let inner: f64 = intent.iter().zip(vs).map(|(q, val)| q * val).sum();
                total += p * inner;
            }
        }
        total
    }

    fn initial_value(&self, v: &[Vec<f64>]) -> f64 {
        (0..self.ns())
            .map(|s| {
                self.initial[s]
                    * (0..self.nx())
                        .map(|xp| self.intent[s][xp] * v[s][xp])
                        .sum::<f64>()
            })
            .sum()
    }

    fn greedy(&self, q: &[Vec<Vec<f64>>]) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let policy: Vec<Vec<usize>> = q
            .iter()
            .map(|row| row.iter().map(|qs| argmax(qs)).collect())
            .collect();
        let v = q
            .iter()
            .zip(&policy)
            .map(|(row, pi)| row.iter().zip(pi).map(|(qs, x)| qs[*x]).collect())
            .collect();
        (policy, v)
    }

    /// Value iteration on the counterfactual Bellman optimality equation
    /// (Eq. 9.182) with discount `γ`, until the largest change is below
    /// `tol`.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] unless `0 < γ < 1` and `tol > 0`.
    pub fn value_iteration(&self, gamma: f64, tol: f64) -> Result<CtfQ, CrlError> {
        check_discount(gamma, tol)?;
        let (ns, nx) = (self.ns(), self.nx());
        let mut v = vec![vec![0.0; nx]; ns];
        loop {
            let q: Vec<Vec<Vec<f64>>> = (0..ns)
                .map(|s| {
                    (0..nx)
                        .map(|xp| {
                            (0..nx)
                                .map(|x| self.reward[s][xp][x] + gamma * self.expect(s, xp, x, &v))
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let (policy, v2) = self.greedy(&q);
            let delta = v
                .iter()
                .flatten()
                .zip(v2.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            v = v2;
            if delta < tol {
                let value = self.initial_value(&v);
                return Ok(CtfQ { q, policy, value });
            }
        }
    }

    /// Backward induction over a finite horizon `H` with the undiscounted
    /// total reward: `Q^{(1)}, …, Q^{(H)}`, in stage order.
    #[must_use]
    pub fn finite_horizon(&self, horizon: usize) -> Vec<CtfQ> {
        let (ns, nx) = (self.ns(), self.nx());
        let mut v = vec![vec![0.0; nx]; ns];
        let mut out = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let q: Vec<Vec<Vec<f64>>> = (0..ns)
                .map(|s| {
                    (0..nx)
                        .map(|xp| {
                            (0..nx)
                                .map(|x| self.reward[s][xp][x] + self.expect(s, xp, x, &v))
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let (policy, v2) = self.greedy(&q);
            v = v2;
            let value = self.initial_value(&v);
            out.push(CtfQ { q, policy, value });
        }
        out.reverse();
        out
    }

    /// The expected total reward over `H` stages of a counterfactual policy
    /// given per stage as `policy[i][s][x′] = x`.
    ///
    /// # Panics
    ///
    /// Panics if `policy` has fewer than `horizon` stages.
    #[must_use]
    pub fn evaluate_finite(&self, policy: &[Vec<Vec<usize>>], horizon: usize) -> f64 {
        let (ns, nx) = (self.ns(), self.nx());
        let mut v = vec![vec![0.0; nx]; ns];
        for i in (0..horizon).rev() {
            v = (0..ns)
                .map(|s| {
                    (0..nx)
                        .map(|xp| {
                            let x = policy[i][s][xp];
                            self.reward[s][xp][x] + self.expect(s, xp, x, &v)
                        })
                        .collect()
                })
                .collect();
        }
        self.initial_value(&v)
    }

    /// The interventional MDP an agent sees when it ignores `X′`:
    /// `T_exp(s, x, s″) = Σ_{x′} P(x′ | s) P(S_{i+1,x} = s″ | s, x′)` and
    /// `R_exp(s, x) = Σ_{x′} P(x′ | s) R_ctf(s, x′, x)`.
    fn experimental_model(&self) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let (ns, nx) = (self.ns(), self.nx());
        let mut t = vec![vec![vec![0.0; ns]; nx]; ns];
        let mut r = vec![vec![0.0; nx]; ns];
        for s in 0..ns {
            for x in 0..nx {
                for xp in 0..nx {
                    let w = self.intent[s][xp];
                    r[s][x] += w * self.reward[s][xp][x];
                    for (acc, p) in t[s][x].iter_mut().zip(&self.next[s][xp][x]) {
                        *acc += w * p;
                    }
                }
            }
        }
        (t, r)
    }

    /// Value iteration over experimental policies `π(x | s)` with discount
    /// `γ`. By Thm 9.4.3 its value never exceeds that of
    /// [`CtfMdp::value_iteration`].
    ///
    /// # Errors
    ///
    /// As [`CtfMdp::value_iteration`].
    pub fn experimental_value_iteration(&self, gamma: f64, tol: f64) -> Result<ExpQ, CrlError> {
        check_discount(gamma, tol)?;
        let (ns, nx) = (self.ns(), self.nx());
        let (t, r) = self.experimental_model();
        let mut v = vec![0.0; ns];
        loop {
            let q: Vec<Vec<f64>> = (0..ns)
                .map(|s| {
                    (0..nx)
                        .map(|x| r[s][x] + gamma * (0..ns).map(|s2| t[s][x][s2] * v[s2]).sum::<f64>())
                        .collect()
                })
                .collect();
            let policy: Vec<usize> = q.iter().map(|row| argmax(row)).collect();
            let v2: Vec<f64> = q.iter().zip(&policy).map(|(row, x)| row[*x]).collect();
            let delta = v.iter().zip(&v2).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            v = v2;
            if delta < tol {
                let value = (0..ns).map(|s| self.initial[s] * v[s]).sum();
                return Ok(ExpQ { q, policy, value });
            }
        }
    }

    /// Backward induction over experimental policies for a finite horizon.
    #[must_use]
    pub fn experimental_finite_horizon(&self, horizon: usize) -> Vec<ExpQ> {
        let (ns, nx) = (self.ns(), self.nx());
        let (t, r) = self.experimental_model();
        let mut v = vec![0.0; ns];
        let mut out = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let q: Vec<Vec<f64>> = (0..ns)
                .map(|s| {
                    (0..nx)
                        .map(|x| r[s][x] + (0..ns).map(|s2| t[s][x][s2] * v[s2]).sum::<f64>())
                        .collect()
                })
                .collect();
            let policy: Vec<usize> = q.iter().map(|row| argmax(row)).collect();
            v = q.iter().zip(&policy).map(|(row, x)| row[*x]).collect();
            let value = (0..ns).map(|s| self.initial[s] * v[s]).sum();
            out.push(ExpQ { q, policy, value });
        }
        out.reverse();
        out
    }
}

fn check_discount(gamma: f64, tol: f64) -> Result<(), CrlError> {
    if !(gamma > 0.0 && gamma < 1.0) || tol <= 0.0 || tol.is_nan() {
        return Err(CrlError::InvalidInput(format!(
            "need 0 < γ < 1 and tol > 0, got γ = {gamma} and tol = {tol}"
        )));
    }
    Ok(())
}

/// An MDP an agent can play through counterfactual randomization, with
/// states and actions as indices.
pub trait CtfMdpEnvironment {
    /// `|D(S)|`.
    fn n_states(&self) -> usize;
    /// `|D(X)|`.
    fn n_actions(&self) -> usize;
    /// Starts an episode: the initial state and the intended action.
    fn reset(&mut self, uniform: &mut dyn FnMut() -> f64) -> (usize, usize);
    /// Performs `do(X_i ← x)`: the reward, the next state and the next
    /// intended action.
    fn step(&mut self, x: usize, uniform: &mut dyn FnMut() -> f64) -> (f64, usize, usize);
}

/// Simulates a [`CtfMdp`]: `(S_{i+1}, X′_{i+1})` is drawn from `T_ctf` and
/// `Y_i` is a Bernoulli draw with mean `R_ctf`. That reproduces the law of
/// the counterfactual submodel exactly for binary rewards, except for any
/// dependence between `Y_i` and `S_{i+1}` given `(s, x′, x)`, which no
/// expected total reward can see.
#[derive(Debug, Clone)]
pub struct CtfMdpSimulator {
    mdp: CtfMdp,
    current: Option<(usize, usize)>,
}

impl CtfMdpSimulator {
    /// Wraps an MDP.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] if a reward is outside `[0, 1]`.
    pub fn new(mdp: CtfMdp) -> Result<Self, CrlError> {
        if mdp.reward.iter().flatten().flatten().any(|r| !(0.0..=1.0).contains(r)) {
            return Err(CrlError::InvalidInput(
                "the simulator draws Bernoulli rewards, so R_ctf must lie in [0, 1]".into(),
            ));
        }
        Ok(Self { mdp, current: None })
    }
}

fn draw_index(p: impl Iterator<Item = f64>, u: f64) -> usize {
    let p: Vec<f64> = p.collect();
    let mut target = u * p.iter().sum::<f64>();
    for (i, q) in p.iter().enumerate() {
        if *q <= 0.0 {
            continue;
        }
        target -= q;
        if target < 0.0 {
            return i;
        }
    }
    p.iter().rposition(|q| *q > 0.0).unwrap_or(0)
}

impl CtfMdpEnvironment for CtfMdpSimulator {
    fn n_states(&self) -> usize {
        self.mdp.ns()
    }

    fn n_actions(&self) -> usize {
        self.mdp.nx()
    }

    fn reset(&mut self, uniform: &mut dyn FnMut() -> f64) -> (usize, usize) {
        let s = draw_index(self.mdp.initial.iter().copied(), uniform());
        let xp = draw_index(self.mdp.intent[s].iter().copied(), uniform());
        self.current = Some((s, xp));
        (s, xp)
    }

    fn step(&mut self, x: usize, uniform: &mut dyn FnMut() -> f64) -> (f64, usize, usize) {
        let (s, xp) = self.current.expect("reset before stepping");
        let y = f64::from(u8::from(uniform() < self.mdp.reward[s][xp][x]));
        let s2 = draw_index(self.mdp.next[s][xp][x].iter().copied(), uniform());
        let x2 = draw_index(self.mdp.intent[s2].iter().copied(), uniform());
        self.current = Some((s2, x2));
        (y, s2, x2)
    }
}

/// The trace of [`CtfUcbvi`].
#[derive(Debug, Clone, PartialEq)]
pub struct UcbviRun {
    /// `Σ_i Y_i` of each episode.
    pub returns: Vec<f64>,
    /// The greedy policy of each episode, `policies[t][i][s][x′] = x`. An
    /// agent ignoring `X′` has identical rows across `x′`.
    pub policies: Vec<Vec<Vec<Vec<usize>>>>,
}

impl UcbviRun {
    /// The cumulative regret `t V* − Σ_{s ≤ t} V^{π^{(s)}}` against the
    /// optimal counterfactual policy, evaluated exactly in `mdp`.
    #[must_use]
    pub fn regret(&self, mdp: &CtfMdp, horizon: usize) -> Vec<f64> {
        let best = mdp.finite_horizon(horizon)[0].value;
        let mut total = 0.0;
        self.policies
            .iter()
            .map(|pi| {
                total += best - mdp.evaluate_finite(pi, horizon);
                total
            })
            .collect()
    }
}

/// Algorithm 28 (Ctf-UCBVI) with Algorithm 29 (UCB-Q-values): UCBVI over
/// the counterfactual MDP, conditioning every estimate on the intercepted
/// intended action.
///
/// Before each episode it estimates `T̂_ctf` and `R̂_ctf` from every step so
/// far (Eq. 9.196) and runs backward induction with the bonus
/// `b = c·H·sqrt(ln(5 |S||X|² T / δ) / N(s, x′, x))` (Eq. 9.197, `c = 7`),
/// keeping each `Q^{(i)}` below `H` and below its previous value. Unvisited
/// cells keep `Q = H`. It then plays `x = argmax Q̂^{(i)}(s, x′, ·)`.
///
/// With `use_intent = false` it is plain UCBVI over `(s, x)`, which by
/// Cor. 9.4.7 can suffer linear regret against the counterfactual optimum.
#[derive(Debug, Clone, Copy)]
pub struct CtfUcbvi {
    /// The error probability `δ`.
    pub delta: f64,
    /// The bonus constant `c` (7 in Eq. 9.197).
    pub bonus: f64,
    /// Whether to condition on the intended action.
    pub use_intent: bool,
}

impl Default for CtfUcbvi {
    fn default() -> Self {
        Self {
            delta: 0.1,
            bonus: 7.0,
            use_intent: true,
        }
    }
}

impl CtfUcbvi {
    /// Runs `episodes` episodes of `horizon` steps in `env`.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] unless `0 < δ < 1`, `c ≥ 0` and the
    /// horizon is positive.
    pub fn run(
        &self,
        env: &mut dyn CtfMdpEnvironment,
        horizon: usize,
        episodes: usize,
        seed: u64,
    ) -> Result<UcbviRun, CrlError> {
        if !(self.delta > 0.0 && self.delta < 1.0) || self.bonus < 0.0 || horizon == 0 {
            return Err(CrlError::InvalidInput(format!(
                "need 0 < δ < 1, c ≥ 0 and H > 0, got δ = {}, c = {}, H = {horizon}",
                self.delta, self.bonus
            )));
        }
        let (ns, nx) = (env.n_states(), env.n_actions());
        // The intent dimension collapses to one cell without intent.
        let ni = if self.use_intent { nx } else { 1 };
        let hf = horizon as f64;
        let log_term = (5.0 * (ns * nx * nx) as f64 * episodes.max(1) as f64 / self.delta).ln();

        let mut n = vec![vec![vec![0.0; nx]; ni]; ns];
        let mut r_sum = vec![vec![vec![0.0; nx]; ni]; ns];
        // Next (s″, x″) counts; x″ collapses without intent.
        let mut t_count = vec![vec![vec![vec![0.0; ns * ni]; nx]; ni]; ns];
        let mut q_prev = vec![vec![vec![vec![hf; nx]; ni]; ns]; horizon];
        let mut rng = SplitMix64::new(seed);
        let mut run = UcbviRun {
            returns: Vec::with_capacity(episodes),
            policies: Vec::with_capacity(episodes),
        };
        for _ in 0..episodes {
            // Algorithm 29.
            let mut v_next = vec![vec![0.0; ni]; ns];
            let mut q_new = q_prev.clone();
            for i in (0..horizon).rev() {
                for s in 0..ns {
                    for c in 0..ni {
                        for x in 0..nx {
                            let cnt = n[s][c][x];
                            if cnt == 0.0 {
                                continue;
                            }
                            let r_hat = r_sum[s][c][x] / cnt;
                            let mut tv = 0.0;
                            for s2 in 0..ns {
                                for c2 in 0..ni {
                                    tv += t_count[s][c][x][s2 * ni + c2] / cnt * v_next[s2][c2];
                                }
                            }
                            let b = self.bonus * hf * (log_term / cnt).sqrt();
                            q_new[i][s][c][x] = q_prev[i][s][c][x].min(hf).min(r_hat + tv + b);
                        }
                    }
                }
                v_next = (0..ns)
                    .map(|s| {
                        (0..ni)
                            .map(|c| q_new[i][s][c].iter().copied().fold(f64::NEG_INFINITY, f64::max))
                            .collect()
                    })
                    .collect();
            }
            q_prev = q_new;
            let policy: Vec<Vec<Vec<usize>>> = (0..horizon)
                .map(|i| {
                    (0..ns)
                        .map(|s| {
                            (0..nx)
                                .map(|xp| argmax(&q_prev[i][s][if self.use_intent { xp } else { 0 }]))
                                .collect()
                        })
                        .collect()
                })
                .collect();

            // Algorithm 28, steps 4–10.
            let mut uniform = || rng.unit();
            let (mut s, mut xp) = env.reset(&mut uniform);
            let mut ret = 0.0;
            for pi in &policy {
                let x = pi[s][xp];
                let (y, s2, x2) = env.step(x, &mut uniform);
                let c = if self.use_intent { xp } else { 0 };
                let c2 = if self.use_intent { x2 } else { 0 };
                n[s][c][x] += 1.0;
                r_sum[s][c][x] += y;
                t_count[s][c][x][s2 * ni + c2] += 1.0;
                ret += y;
                s = s2;
                xp = x2;
            }
            run.returns.push(ret);
            run.policies.push(policy);
        }
        Ok(run)
    }
}
