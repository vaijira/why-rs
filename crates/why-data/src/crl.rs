//! Causal reinforcement learning (chapter 8, "Reinforcement Learning through
//! Causal Lenses").
//!
//! # The objects
//!
//! * [`PolicySpace`] is `Π = {⟨X_1, S_1⟩, …, ⟨X_H, S_H⟩}` (Def. 8.1.1), and
//!   [`PolicySpace::validate`] checks its two non-descendant constraints
//!   against a causal diagram.
//! * [`Reward`] is `R : D(Y) ↦ ℝ` (Def. 8.1.2): total, average, discounted
//!   or custom.
//! * [`Policy`] is a sequence of decision rules `π_i(x_i | s_i)`.
//!   [`Deterministic`], [`Stochastic`], [`Uniform`] and [`TabularPolicy`]
//!   implement it.
//! * [`Cdm`] is a causal decision model `⟨M*, Π, R⟩` (Def. 8.1.3) over an
//!   [`scm::Model`](crate::scm::Model). It evaluates any policy exactly,
//!   plans the optimal one, and acts as the [`Environment`] that online
//!   learners interact with.
//! * [`Task`] is a CRL task `⟨L, A, Π, R⟩` (Def. 8.1.4). [`Task::run`] is
//!   Algorithm 19: interact under the regime `L`, then return a policy
//!   estimate that the assumptions `A` justify.
//!
//! # Off-policy evaluation (regime `see`)
//!
//! [`ipw`] (Thm 8.2.1) and [`dp`] (Thm 8.2.2) evaluate `E_π[R(Y)]` from a
//! [`Dataset`]. Both are the plug-in functional of Eq. 8.66, so on the same
//! data they agree to rounding error, whether or not the assumptions hold.
//! Run on the exact weighted table of `P(V)` ([`Cdm::observational_data`]),
//! they return the exact policy value whenever NUC ([`nuc`], Def. 8.2.1) or
//! the sequential back-door condition ([`sequential_backdoor`], Def. 8.2.3)
//! holds. [`optimize`] searches `Π` for the best deterministic policy. With
//! perfect recall it uses backward induction (Eqs. 8.36–8.39). Otherwise it
//! enumerates every policy.
//!
//! # Online learning (regime `do`)
//!
//! Intervening establishes NUC in the submodel (Lemma 8.2.3, Exp-NUC). So
//! [`Rct`] (Algorithm 20) explores uniformly and then commits to the policy
//! [`optimize`] picks from that data. [`Ucb`] is Algorithm 21 for
//! multi-armed bandits. [`rct_regret_bound`], [`rct_optimal_trials`] and
//! [`ucb_regret_bound`] are Thms 8.2.4 and 8.2.5 and Eq. 8.101.
//!
//! # Departures from the text
//!
//! * **History, not just input states.** The propensities of Thm 8.2.1 and
//!   the value functions of Thm 8.2.2 condition on the whole history
//!   `X̄_{i−1} ∪ S̄_i`, while each decision rule reads only its own `S_i`.
//!   [`PolicySpace::history`] is that union, in temporal order and without
//!   duplicates.
//! * **Positivity is checked, not assumed.** A policy that puts mass on an
//!   action never observed after a history it reaches makes both estimands
//!   undefined. That is reported as [`CrlError::Positivity`]; the term is
//!   never silently dropped.
//! * **Eq. 8.101 uses the natural log.** Minimising Eq. 8.100 gives
//!   `N = 8/Δ² · ln(TΔ²/4)`. The book's "N ≈ 878" for `T = 5000, Δ = 0.1`
//!   comes from `log₁₀`. The natural log gives 2021.
//! * **Eq. 8.119 has a typo.** It should read `Y ← S1 ⊕ X2 ⊕ U3`. Tables 8.7
//!   and 8.8, Eq. 8.124 and the back-door path the text blocks all need
//!   `X2`, not `X1`.
//! * **Ties break towards the earlier action.** That is the first level of
//!   the action column, or the first value of the domain. In the DTR of
//!   Ex. 8.6, `π₂*(0, 0, 1)` is a tie that the book resolves as `X₂ ← 1`.
//!
//! Not implemented: the infinite-horizon MDP planning of Ex. 8.5 and the
//! do-calculus policy identification of Sec. 8.2.3.2. When the sequential
//! back-door fails, [`Task::run`] reports [`CrlError::NotIdentifiable`]
//! instead of falling back to those.

mod cdm;
mod conditions;
mod offpolicy;
mod online;
mod policy;
mod space;
#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use crate::graph::dseparation::{Admg, DSepError};
use crate::scm::{Model, ScmError, free_vars};

pub use crate::fairness::Dataset;
pub use cdm::{Assumptions, Cdm, Environment, Justification, Learned, Regime, Task};
pub use conditions::{NucReport, NucViolation, SbcReport, SbcStage, nuc, sequential_backdoor};
pub use offpolicy::{Evaluation, Plan, QTable, dp, ipw, optimize};
pub use online::{
    OnlineRun, Rct, Ucb, rct_optimal_trials, rct_regret_bound, ucb_regret_bound,
};
pub use policy::{Deterministic, Policy, State, Stochastic, TabularPolicy, TabularRule, Uniform};
pub use space::{PolicySpace, Reward, Stage};

/// Everything that can go wrong in this module.
#[derive(Debug, Clone, PartialEq)]
pub enum CrlError {
    /// The policy space is malformed or violates Def. 8.1.1.
    InvalidSpace(String),
    /// A variable is missing from the graph, model or data.
    Unknown(String),
    /// A graph query failed.
    Graph(DSepError),
    /// The structural causal model rejected an operation.
    Scm(ScmError),
    /// The data could not be built or read.
    Data(String),
    /// A reward signal holds a value that is not a finite number.
    NonNumericReward {
        /// The reward column.
        column: String,
        /// The offending value.
        value: String,
    },
    /// The policy puts mass on an action never observed after a history it
    /// reaches, so neither IPW nor DP is defined.
    Positivity {
        /// 1-based stage index.
        stage: usize,
        /// The history `X̄_{i−1} ∪ S̄_i` as `(variable, value)` pairs.
        history: Vec<(String, String)>,
        /// The unsupported value of `X_i`.
        action: String,
    },
    /// A decision rule's probabilities do not sum to one over the domain of
    /// its action.
    PolicyNotNormalized {
        /// The action variable.
        action: String,
        /// The input state, rendered.
        state: String,
        /// The sum that was found.
        total: f64,
    },
    /// Exhaustive policy search would visit more policies than the cap.
    TooManyPolicies {
        /// How many deterministic policies the space holds.
        count: f64,
        /// The cap that was exceeded.
        cap: usize,
    },
    /// The task's assumptions do not license evaluating policies from its
    /// data.
    NotIdentifiable(String),
    /// An argument is out of range or inconsistent.
    InvalidInput(String),
}

impl fmt::Display for CrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpace(m) => write!(f, "invalid policy space: {m}"),
            Self::Unknown(v) => write!(f, "unknown variable `{v}`"),
            Self::Graph(e) => write!(f, "graph error: {e}"),
            Self::Scm(e) => write!(f, "SCM error: {e}"),
            Self::Data(m) => write!(f, "data error: {m}"),
            Self::NonNumericReward { column, value } => {
                write!(f, "reward signal `{column}` holds non-numeric value `{value}`")
            }
            Self::Positivity {
                stage,
                history,
                action,
            } => {
                let h = render_pairs(history);
                write!(
                    f,
                    "positivity violated at stage {stage}: the policy may choose `{action}` \
                     after history ({h}), which the data never shows"
                )
            }
            Self::PolicyNotNormalized {
                action,
                state,
                total,
            } => write!(
                f,
                "decision rule for `{action}` sums to {total} at state ({state}), not 1"
            ),
            Self::TooManyPolicies { count, cap } => write!(
                f,
                "exhaustive search would visit {count:.3e} policies (cap {cap}); \
                 give the policy space perfect recall to use backward induction"
            ),
            Self::NotIdentifiable(m) => write!(f, "not identifiable: {m}"),
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for CrlError {}

impl From<DSepError> for CrlError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl From<ScmError> for CrlError {
    fn from(e: ScmError) -> Self {
        Self::Scm(e)
    }
}

impl From<crate::fairness::FairnessError> for CrlError {
    fn from(e: crate::fairness::FairnessError) -> Self {
        Self::Data(e.to_string())
    }
}

pub(crate) fn render_pairs(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The causal diagram of an SCM, with its exogenous variables projected out.
///
/// There is a directed edge `V_j → V_i` whenever `V_j` occurs in `f_{V_i}`.
/// There is a bidirected edge `V_i ↔ V_j` whenever their equations share an
/// exogenous variable. `scm`'s exogenous variables are mutually independent,
/// so sharing one is the only way two noises can be dependent.
#[must_use]
pub fn diagram(model: &Model) -> Admg {
    let world = model.world(model.observational()).world();
    let exogenous: HashSet<&str> = model.exogenous().iter().map(String::as_str).collect();
    let mut g = Admg::new();
    for v in world.variables() {
        g.add_node(v);
    }
    let mut children: HashMap<String, BTreeSet<String>> = HashMap::new();
    for v in world.variables() {
        let (_, e) = world
            .equations()
            .iter()
            .find(|(k, _)| k == v)
            .expect("every endogenous variable has an equation");
        let mut parents = HashSet::new();
        free_vars(e, &mut parents);
        let mut parents: Vec<String> = parents.into_iter().collect();
        parents.sort();
        for p in parents {
            if exogenous.contains(p.as_str()) {
                children.entry(p).or_default().insert(v.clone());
            } else {
                g.directed(&p, v);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut noises: Vec<_> = children.into_iter().collect();
    noises.sort();
    for (_, kids) in noises {
        let kids: Vec<&String> = kids.iter().collect();
        for (i, a) in kids.iter().enumerate() {
            for b in &kids[i + 1..] {
                if seen.insert(((*a).clone(), (*b).clone())) {
                    g.bidirected(a, b);
                }
            }
        }
    }
    g
}
