//! Policy spaces (Def. 8.1.1) and reward functions (Def. 8.1.2).

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use super::CrlError;
use crate::graph::dseparation::Admg;

/// One stage `⟨X_i, S_i⟩` of a policy space: an action and the input states
/// its decision rule reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// The action variable `X_i`.
    pub action: String,
    /// The input states `S_i`, in the order a [`super::State`] presents them.
    pub states: Vec<String>,
}

/// A policy space `Π = {⟨X_1, S_1⟩, …, ⟨X_H, S_H⟩}` (Def. 8.1.1).
///
/// Stages are in temporal order: the agent observes `S_1`, picks `X_1`,
/// observes `S_2`, and so on.
///
/// ```
/// use why_data::crl::PolicySpace;
///
/// // The DTR of Ex. 8.2 with perfect recall.
/// let full = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1", "X1", "S2"])]).unwrap();
/// assert_eq!(full.history(1), ["S1", "X1", "S2"]);
/// assert!(full.has_perfect_recall());
///
/// let markov = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S2"])]).unwrap();
/// assert_eq!(markov.history(1), ["S1", "X1", "S2"]);
/// assert!(!markov.has_perfect_recall());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySpace {
    stages: Vec<Stage>,
    /// Every variable of `S̄_H ∪ X̄_H` in temporal order, without duplicates.
    order: Vec<String>,
    /// `history_len[i]` is the length of the prefix of `order` that is the
    /// history `X̄_{i−1} ∪ S̄_i`. `X_i` sits right after it.
    history_len: Vec<usize>,
}

impl PolicySpace {
    /// Builds a policy space from `(action, states)` pairs in temporal order.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidSpace`] if there are no stages, an action repeats,
    /// or `S_i` names `X_i` or a later action. Those are the parts of
    /// Def. 8.1.1 that need no graph. [`PolicySpace::validate`] checks the
    /// rest.
    pub fn new<'a, I, S>(stages: I) -> Result<Self, CrlError>
    where
        I: IntoIterator<Item = (&'a str, S)>,
        S: AsRef<[&'a str]>,
    {
        let stages: Vec<Stage> = stages
            .into_iter()
            .map(|(action, states)| Stage {
                action: action.to_string(),
                states: states.as_ref().iter().map(|s| (*s).to_string()).collect(),
            })
            .collect();
        Self::from_stages(stages)
    }

    /// The multi-armed bandit space `{⟨X, ∅⟩}`.
    #[must_use]
    pub fn bandit(action: &str) -> Self {
        Self::from_stages(vec![Stage {
            action: action.to_string(),
            states: Vec::new(),
        }])
        .expect("a single stage with no states is always valid")
    }

    /// Builds a policy space from [`Stage`]s; see [`PolicySpace::new`].
    ///
    /// # Errors
    ///
    /// As [`PolicySpace::new`].
    pub fn from_stages(stages: Vec<Stage>) -> Result<Self, CrlError> {
        if stages.is_empty() {
            return Err(CrlError::InvalidSpace("no stages".into()));
        }
        let actions: Vec<&str> = stages.iter().map(|s| s.action.as_str()).collect();
        let mut seen = HashSet::new();
        for a in &actions {
            if !seen.insert(*a) {
                return Err(CrlError::InvalidSpace(format!("action `{a}` repeats")));
            }
        }
        for (i, stage) in stages.iter().enumerate() {
            let mut own = HashSet::new();
            for s in &stage.states {
                if !own.insert(s.as_str()) {
                    return Err(CrlError::InvalidSpace(format!(
                        "state `{s}` repeats in S_{}",
                        i + 1
                    )));
                }
                if let Some(j) = actions[i..].iter().position(|a| a == s) {
                    return Err(CrlError::InvalidSpace(format!(
                        "S_{} contains the action X_{} = `{s}`, which is not yet decided",
                        i + 1,
                        i + j + 1
                    )));
                }
            }
        }

        let mut order: Vec<String> = Vec::new();
        let mut history_len = Vec::with_capacity(stages.len());
        for stage in &stages {
            for s in &stage.states {
                if !order.contains(s) {
                    order.push(s.clone());
                }
            }
            history_len.push(order.len());
            order.push(stage.action.clone());
        }
        Ok(Self {
            stages,
            order,
            history_len,
        })
    }

    /// The stages, in temporal order.
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// The horizon `H = |X|`.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.stages.len()
    }

    /// The actions `X_1, …, X_H`.
    #[must_use]
    pub fn actions(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.action.as_str()).collect()
    }

    /// The history `X̄_{i−1} ∪ S̄_i` before the action of stage `i` (0-based),
    /// in temporal order.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.horizon()`.
    #[must_use]
    pub fn history(&self, i: usize) -> Vec<&str> {
        self.order[..self.history_len[i]]
            .iter()
            .map(String::as_str)
            .collect()
    }

    /// Every variable in some `S_i` or `X_i`, in temporal order.
    pub(crate) fn order(&self) -> &[String] {
        &self.order
    }

    pub(crate) fn history_len(&self, i: usize) -> usize {
        self.history_len[i]
    }

    /// Whether every decision rule sees its entire history, i.e.
    /// `S_i = X̄_{i−1} ∪ S̄_i` as sets. Then backward induction is exact.
    #[must_use]
    pub fn has_perfect_recall(&self) -> bool {
        self.stages.iter().enumerate().all(|(i, stage)| {
            let own: HashSet<&str> = stage.states.iter().map(String::as_str).collect();
            let hist: HashSet<&str> = self.history(i).into_iter().collect();
            own == hist
        })
    }

    /// Checks Def. 8.1.1 against a causal diagram. Each `X_i` must be a
    /// non-descendant of `X_{i+1}, …, X_H`. Each `S_i` must be a
    /// non-descendant of `X_i, …, X_H`.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] for a variable the diagram lacks, and
    /// [`CrlError::InvalidSpace`] naming the first violated constraint.
    pub fn validate(&self, g: &Admg) -> Result<(), CrlError> {
        for v in &self.order {
            if g.node(v).is_none() {
                return Err(CrlError::Unknown(v.clone()));
            }
        }
        let actions = self.actions();
        for (i, stage) in self.stages.iter().enumerate() {
            let later: HashSet<&str> = g
                .descendants_of(&actions[i + 1..])?
                .into_iter()
                .collect();
            if later.contains(stage.action.as_str()) {
                return Err(CrlError::InvalidSpace(format!(
                    "X_{} = `{}` is a descendant of a later action",
                    i + 1,
                    stage.action
                )));
            }
            let from_here: HashSet<&str> = g.descendants_of(&actions[i..])?.into_iter().collect();
            if let Some(s) = stage.states.iter().find(|s| from_here.contains(s.as_str())) {
                return Err(CrlError::InvalidSpace(format!(
                    "state `{s}` of S_{} is a descendant of X_{}, …, X_{}",
                    i + 1,
                    i + 1,
                    self.horizon()
                )));
            }
        }
        Ok(())
    }

    /// The manipulated diagram `G_{π_from, …, π_H}` (Sec. 8.2.3.1). For every
    /// stage `j ≥ from` (0-based), it drops every edge into `X_j`, bidirected
    /// ones included, and adds `S → X_j` for each `S ∈ S_j`.
    ///
    /// The result covers every policy in the space at once. A particular
    /// policy can only ignore some of the added edges, which never opens a
    /// path.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] for a variable the diagram lacks.
    pub fn manipulated(&self, g: &Admg, from: usize) -> Result<Admg, CrlError> {
        let later: Vec<&str> = self.actions()[from.min(self.horizon())..].to_vec();
        let mut out = g.cut_edges_into(&later)?;
        for stage in &self.stages[from.min(self.horizon())..] {
            for s in &stage.states {
                if g.node(s).is_none() {
                    return Err(CrlError::Unknown(s.clone()));
                }
                out.directed(s, &stage.action);
            }
        }
        Ok(out)
    }
}

impl fmt::Display for PolicySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (i, s) in self.stages.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "⟨{}, {{{}}}⟩", s.action, s.states.join(", "))?;
        }
        f.write_str("}")
    }
}

/// A user-supplied reward function.
type CustomReward = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// How the reward signals are folded into one number.
#[derive(Clone)]
enum Aggregate {
    Total,
    Average,
    Discounted(f64),
    Custom(CustomReward),
}

/// A reward function `R : D(Y) ↦ ℝ` over reward signals `Y` (Def. 8.1.2).
///
/// ```
/// use why_data::crl::Reward;
///
/// let r = Reward::discounted(["Y1", "Y2", "Y3"], 0.5).unwrap();
/// assert_eq!(r.apply(&[1.0, 1.0, 1.0]), 1.75);
/// assert_eq!(Reward::average(["Y1", "Y2"]).apply(&[1.0, 0.0]), 0.5);
/// ```
#[derive(Clone)]
pub struct Reward {
    signals: Vec<String>,
    aggregate: Aggregate,
}

impl Reward {
    /// `R_total(Y) = Σ Y_i` (Eq. 8.7). With one signal this is `R(Y) = Y`.
    pub fn total<I, S>(signals: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            signals: signals.into_iter().map(Into::into).collect(),
            aggregate: Aggregate::Total,
        }
    }

    /// `R_average(Y) = (1/H) Σ Y_i` (Eq. 8.8).
    pub fn average<I, S>(signals: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            signals: signals.into_iter().map(Into::into).collect(),
            aggregate: Aggregate::Average,
        }
    }

    /// `R_discount(Y) = Σ γ^{i−1} Y_i` (Eq. 8.9).
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] unless `0 < γ < 1`.
    pub fn discounted<I, S>(signals: I, gamma: f64) -> Result<Self, CrlError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if !(gamma > 0.0 && gamma < 1.0) {
            return Err(CrlError::InvalidInput(format!(
                "discount factor {gamma} is not in (0, 1)"
            )));
        }
        Ok(Self {
            signals: signals.into_iter().map(Into::into).collect(),
            aggregate: Aggregate::Discounted(gamma),
        })
    }

    /// Any function of the signals' values, which arrive in the order given
    /// here.
    pub fn custom<I, S, F>(signals: I, f: F) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Self {
            signals: signals.into_iter().map(Into::into).collect(),
            aggregate: Aggregate::Custom(Arc::new(f)),
        }
    }

    /// The reward signals `Y`.
    #[must_use]
    pub fn signals(&self) -> &[String] {
        &self.signals
    }

    /// `R(y)` for values of the signals in [`Reward::signals`] order.
    #[must_use]
    pub fn apply(&self, y: &[f64]) -> f64 {
        match &self.aggregate {
            Aggregate::Total => y.iter().sum(),
            Aggregate::Average => {
                if y.is_empty() {
                    0.0
                } else {
                    y.iter().sum::<f64>() / y.len() as f64
                }
            }
            Aggregate::Discounted(g) => {
                let mut factor = 1.0;
                let mut total = 0.0;
                for v in y {
                    total += factor * v;
                    factor *= g;
                }
                total
            }
            Aggregate::Custom(f) => f(y),
        }
    }

    /// `R` applied to string-valued signals, which must parse as numbers.
    pub(crate) fn apply_str<'a>(
        &self,
        values: impl IntoIterator<Item = &'a str>,
    ) -> Result<f64, CrlError> {
        let mut y = Vec::with_capacity(self.signals.len());
        for (column, v) in self.signals.iter().zip(values) {
            let x: f64 = v.trim().parse().map_err(|_| CrlError::NonNumericReward {
                column: column.clone(),
                value: v.to_string(),
            })?;
            if !x.is_finite() {
                return Err(CrlError::NonNumericReward {
                    column: column.clone(),
                    value: v.to_string(),
                });
            }
            y.push(x);
        }
        Ok(self.apply(&y))
    }
}

impl fmt::Debug for Reward {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.aggregate {
            Aggregate::Total => "total".to_string(),
            Aggregate::Average => "average".to_string(),
            Aggregate::Discounted(g) => format!("discounted(γ = {g})"),
            Aggregate::Custom(_) => "custom".to_string(),
        };
        f.debug_struct("Reward")
            .field("signals", &self.signals)
            .field("aggregate", &kind)
            .finish()
    }
}
