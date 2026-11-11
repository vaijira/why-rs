//! When off-policy evaluation is valid: NUC (Def. 8.2.1) and the sequential
//! back-door condition (Def. 8.2.3).

use std::collections::HashSet;
use std::fmt;

use super::{CrlError, PolicySpace};
use crate::graph::dseparation::Admg;
use crate::scm::{Model, free_vars};

/// One way NUC fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NucViolation {
    /// Condition 1: an endogenous parent of `X_i` is outside its history
    /// `X̄_{i−1} ∪ S̄_i`.
    UnobservedParent {
        /// The action.
        action: String,
        /// The parent missing from the history.
        parent: String,
    },
    /// Condition 2: an exogenous parent of `X_i` also drives another
    /// endogenous variable, so `U_{X_i}` is not independent of `U_j`.
    SharedNoise {
        /// The action.
        action: String,
        /// The shared exogenous variable.
        noise: String,
        /// The other variable it drives.
        with: String,
    },
}

impl fmt::Display for NucViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnobservedParent { action, parent } => {
                write!(f, "parent {parent} of {action} is not in its history")
            }
            Self::SharedNoise {
                action,
                noise,
                with,
            } => write!(f, "{noise} drives both {action} and {with}"),
        }
    }
}

/// The outcome of [`nuc`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NucReport {
    /// Every violation found, in stage order.
    pub violations: Vec<NucViolation>,
}

impl NucReport {
    /// Whether NUC holds.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Checks the No Unmeasured Confounder condition (Def. 8.2.1) for `space` in
/// `model`.
///
/// Every action `X_i` must satisfy two conditions:
///
/// 1. its endogenous parents lie in `X̄_{i−1} ∪ S̄_i`;
/// 2. its exogenous parents drive no other endogenous variable.
///
/// `scm`'s exogenous variables are mutually independent, so condition 2 is
/// exactly `U_{X_i} ⊥ U_j` for every `V_j ≠ X_i`.
///
/// # Errors
///
/// [`CrlError::Unknown`] if an action is not endogenous in `model`.
pub fn nuc(model: &Model, space: &PolicySpace) -> Result<NucReport, CrlError> {
    let world = model.world(model.observational()).world();
    let exogenous: HashSet<&str> = model.exogenous().iter().map(String::as_str).collect();
    let parents_of = |v: &str| -> Option<Vec<String>> {
        let (_, e) = world.equations().iter().find(|(k, _)| k == v)?;
        let mut out = HashSet::new();
        free_vars(e, &mut out);
        let mut out: Vec<String> = out.into_iter().collect();
        out.sort();
        Some(out)
    };

    let mut report = NucReport::default();
    for (i, stage) in space.stages().iter().enumerate() {
        let parents =
            parents_of(&stage.action).ok_or_else(|| CrlError::Unknown(stage.action.clone()))?;
        let history: HashSet<&str> = space.history(i).into_iter().collect();
        for p in &parents {
            if exogenous.contains(p.as_str()) {
                for v in world.variables() {
                    if v == &stage.action {
                        continue;
                    }
                    if parents_of(v).is_some_and(|ps| ps.contains(p)) {
                        report.violations.push(NucViolation::SharedNoise {
                            action: stage.action.clone(),
                            noise: p.clone(),
                            with: v.clone(),
                        });
                    }
                }
            } else if !history.contains(p.as_str()) {
                report.violations.push(NucViolation::UnobservedParent {
                    action: stage.action.clone(),
                    parent: p.clone(),
                });
            }
        }
    }
    Ok(report)
}

/// The sequential back-door test at one stage.
#[derive(Debug, Clone)]
pub struct SbcStage {
    /// The action `X_i`.
    pub action: String,
    /// The conditioning set `X̄_{i−1} ∪ S̄_i`.
    pub conditioning: Vec<String>,
    /// Whether `(X_i ⊥ Y | X̄_{i−1}, S̄_i)` holds in
    /// `G_{X̲_i, π_{i+1}, …, π_H}`.
    pub holds: bool,
    /// When it fails, one open back-door path, rendered.
    pub open_path: Option<String>,
}

/// The outcome of [`sequential_backdoor`].
#[derive(Debug, Clone)]
pub struct SbcReport {
    /// One test per stage, in stage order.
    pub stages: Vec<SbcStage>,
}

impl SbcReport {
    /// Whether the policy space is back-door admissible.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.stages.iter().all(|s| s.holds)
    }
}

/// Checks the sequential back-door condition (Def. 8.2.3): for every `i`,
///
/// ```text
/// (X_i ⊥ Y | X̄_{i−1}, S̄_i)  in  G_{X̲_i, π_{i+1}, …, π_H}
/// ```
///
/// where the graph is [`PolicySpace::manipulated`] from stage `i + 1`, with
/// the edges out of `X_i` removed. Reward signals that are already in the
/// conditioning set are trivially independent and are skipped.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a variable the diagram lacks, and
/// [`CrlError::InvalidInput`] if a reward signal is an action.
pub fn sequential_backdoor(
    g: &Admg,
    space: &PolicySpace,
    signals: &[&str],
) -> Result<SbcReport, CrlError> {
    let actions = space.actions();
    for y in signals {
        if g.node(y).is_none() {
            return Err(CrlError::Unknown((*y).to_string()));
        }
        if actions.contains(y) {
            return Err(CrlError::InvalidInput(format!(
                "reward signal `{y}` is also an action"
            )));
        }
    }
    let mut stages = Vec::with_capacity(space.horizon());
    for (i, stage) in space.stages().iter().enumerate() {
        let z = space.history(i);
        let ys: Vec<&str> = signals.iter().copied().filter(|y| !z.contains(y)).collect();
        let manipulated = space
            .manipulated(g, i + 1)?
            .cut_edges_out_of(&[stage.action.as_str()])?;
        let holds = ys.is_empty() || manipulated.is_d_separated(&[&stage.action], &ys, &z)?;
        let open_path = if holds {
            None
        } else {
            let mut found = None;
            for y in &ys {
                if let Some(p) = manipulated
                    .paths(&stage.action, y, &z)?
                    .into_iter()
                    .find(|p| p.is_open())
                {
                    found = Some(p.to_string());
                    break;
                }
            }
            found
        };
        stages.push(SbcStage {
            action: stage.action.clone(),
            conditioning: z.iter().map(|s| (*s).to_string()).collect(),
            holds,
            open_path,
        });
    }
    Ok(SbcReport { stages })
}
