//! Quantities across an abstraction and Q-τ consistency (Def. 15.2.3).

use std::collections::{BTreeMap, BTreeSet};

use super::{Abstraction, AbstractionError};
use crate::scm::{Model, Value, WorldId};

/// The most low-level interventions [`layer_consistency`] enumerates.
const MAX_INTERVENTIONS: usize = 4096;

/// One term of a counterfactual quantity: the clusters `event` take the
/// given high-level levels in the world `do(intervention)`.
///
/// For a low-level quantity `Q` (Eq. 15.19) the intervention is low-level,
/// `x_{L,i}`, and must cover whole clusters. For a high-level quantity
/// `τ(Q)` (Eq. 15.20) it is high-level, `x_{H,i}`, by cluster name. The
/// event is high-level in both, since `Q` sums over every low-level value
/// that `τ` maps to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    /// `x_i`, as `(variable, value)` pairs.
    pub intervention: Vec<(String, String)>,
    /// `y_{H,i}`, as `(cluster, level)` pairs.
    pub event: Vec<(String, String)>,
}

impl Term {
    /// A term from string pairs.
    #[must_use]
    pub fn new(intervention: &[(&str, &str)], event: &[(&str, &str)]) -> Self {
        let own = |v: &[(&str, &str)]| {
            v.iter()
                .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
                .collect()
        };
        Self {
            intervention: own(intervention),
            event: own(event),
        }
    }
}

/// A high-level model: anything that gives the joint distribution of every
/// high-level variable, in cluster order, across worlds that share one draw
/// of the exogenous variables.
pub trait HighLevel {
    /// The joint of all high-level variables in each world
    /// `do(worlds[k])`, with the worlds' tuples concatenated in order. The
    /// interventions are high-level, by cluster name.
    ///
    /// # Errors
    ///
    /// Whatever the model reports for an unknown name or level.
    fn joint(
        &mut self,
        tau: &Abstraction,
        worlds: &[Vec<(String, String)>],
    ) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError>;
}

/// A structural model over the high-level variables, named as the clusters,
/// with integer (or boolean) values whose decimal form is the level.
impl HighLevel for Model {
    fn joint(
        &mut self,
        tau: &Abstraction,
        worlds: &[Vec<(String, String)>],
    ) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError> {
        let blocks: Vec<Vec<String>> = (0..tau.len())
            .map(|i| vec![tau.name_at(i).to_string()])
            .collect();
        joint_over(self, worlds, &blocks, |_, values| values.to_vec())
    }
}

/// The names `names` have in the world `do(x)` of `m`.
pub(crate) fn names_in(
    m: &mut Model,
    x: &[(String, String)],
    names: &[String],
) -> Result<Vec<String>, AbstractionError> {
    if x.is_empty() {
        return Ok(names.to_vec());
    }
    let values = x
        .iter()
        .map(|(k, v)| Ok((k.as_str(), parse(v)?)))
        .collect::<Result<Vec<_>, AbstractionError>>()?;
    let w: WorldId = m.intervene(&values)?;
    let base = m.world(m.observational()).world().variables().to_vec();
    let renamed = m.world(w).world().variables();
    names
        .iter()
        .map(|n| {
            base.iter()
                .position(|b| b == n)
                .map(|i| renamed[i].clone())
                .ok_or_else(|| AbstractionError::UnknownVariable(n.clone()))
        })
        .collect()
}

pub(crate) fn parse(v: &str) -> Result<Value, AbstractionError> {
    v.parse::<i64>()
        .map(Value::Int)
        .map_err(|_| AbstractionError::Data(format!("{v} is not an integer level")))
}

/// The joint over `blocks` of variables in each world, each block reduced
/// to one string by `reduce(block index, values)`.
fn joint_over(
    m: &mut Model,
    worlds: &[Vec<(String, String)>],
    blocks: &[Vec<String>],
    reduce: impl Fn(usize, &[String]) -> Vec<String>,
) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError> {
    let flat: Vec<String> = blocks.iter().flatten().cloned().collect();
    let mut columns: Vec<String> = Vec::new();
    for x in worlds {
        columns.extend(names_in(m, x, &flat)?);
    }
    let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
    let table = m.world(m.observational()).probability_table(Some(&refs))?;
    let mut out: BTreeMap<Vec<String>, f64> = BTreeMap::new();
    for (row, p) in &table.rows {
        if *p <= 0.0 {
            continue;
        }
        let text: Vec<String> = row.iter().map(|v| v.as_int().to_string()).collect();
        let mut key = Vec::new();
        let mut at = 0;
        for _ in worlds {
            for (b, block) in blocks.iter().enumerate() {
                key.extend(reduce(b, &text[at..at + block.len()]));
                at += block.len();
            }
        }
        *out.entry(key).or_default() += p;
    }
    Ok(out)
}

/// The joint over every high-level variable, in cluster order, of the
/// low-level model in each world `do(worlds[k])` (low-level interventions),
/// each cluster mapped through `τ`.
pub(crate) fn low_joint(
    m: &mut Model,
    tau: &Abstraction,
    worlds: &[Vec<(String, String)>],
) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError> {
    let blocks: Vec<Vec<String>> = (0..tau.len()).map(|i| tau.members_at(i).to_vec()).collect();
    joint_over(m, worlds, &blocks, |b, values| {
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        vec![tau.map_at(b, &refs)]
    })
}

/// Sums the joint over the entries that satisfy every term's event.
fn sum_events(
    tau: &Abstraction,
    joint: &BTreeMap<Vec<String>, f64>,
    terms: &[Term],
) -> Result<f64, AbstractionError> {
    let n = tau.len();
    let wanted: Vec<Vec<(usize, &str)>> = terms
        .iter()
        .enumerate()
        .map(|(k, t)| {
            t.event
                .iter()
                .map(|(c, l)| Ok((k * n + tau.index(c)?, l.as_str())))
                .collect::<Result<Vec<_>, AbstractionError>>()
        })
        .collect::<Result<_, _>>()?;
    Ok(joint
        .iter()
        .filter(|(key, _)| wanted.iter().flatten().all(|(i, l)| key[*i] == *l))
        .map(|(_, p)| p)
        .sum::<f64>()
        + 0.0)
}

/// Checks that a low-level intervention covers whole clusters only.
fn check_low(tau: &Abstraction, x: &[(String, String)]) -> Result<(), AbstractionError> {
    for (v, _) in x {
        let c = tau
            .cluster_of(v)
            .ok_or_else(|| AbstractionError::Partial(v.clone()))?;
        for m in tau.members(c)? {
            if !x.iter().any(|(w, _)| w == m) {
                return Err(AbstractionError::Partial(c.to_string()));
            }
        }
    }
    Ok(())
}

/// A low-level quantity `Q` (Eq. 15.19) under `m`: the probability that
/// every term's clusters map through `τ` to the term's levels in its world,
/// `Σ_{y_L : τ(y_L) = y_H} P(Y_{L,*} = y_L)`.
///
/// # Errors
///
/// [`AbstractionError::Partial`] for an intervention that covers part of a
/// cluster or a projected-away variable, and the model's errors.
pub fn low_level(
    m: &mut Model,
    tau: &Abstraction,
    terms: &[Term],
) -> Result<f64, AbstractionError> {
    for t in terms {
        check_low(tau, &t.intervention)?;
    }
    let worlds: Vec<Vec<(String, String)>> = terms.iter().map(|t| t.intervention.clone()).collect();
    let joint = low_joint(m, tau, &worlds)?;
    sum_events(tau, &joint, terms)
}

/// `τ(Q)` under a high-level model, the terms' interventions high-level.
fn high_level(
    m: &mut impl HighLevel,
    tau: &Abstraction,
    terms: &[Term],
) -> Result<f64, AbstractionError> {
    let worlds: Vec<Vec<(String, String)>> = terms.iter().map(|t| t.intervention.clone()).collect();
    let joint = m.joint(tau, &worlds)?;
    sum_events(tau, &joint, terms)
}

/// `Q` under the low-level `low` and `τ(Q)` under the high-level `high`
/// (Def. 15.2.3): `high` is Q-τ consistent with `low` when they agree. The
/// terms' interventions are low-level, and `τ` maps them up.
///
/// # Errors
///
/// Those of [`low_level`] and of the high-level model.
pub fn q_consistency(
    low: &mut Model,
    high: &mut impl HighLevel,
    tau: &Abstraction,
    terms: &[Term],
) -> Result<(f64, f64), AbstractionError> {
    let q = low_level(low, tau, terms)?;
    let lifted: Vec<Term> = terms
        .iter()
        .map(|t| {
            let pairs: Vec<(&str, &str)> = t
                .intervention
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            Ok(Term {
                intervention: tau.tau(&pairs)?,
                event: t.event.clone(),
            })
        })
        .collect::<Result<_, AbstractionError>>()?;
    Ok((q, high_level(high, tau, &lifted)?))
}

/// The largest disagreement [`layer_consistency`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    /// The low-level interventions of the worlds.
    pub worlds: Vec<Vec<(String, String)>>,
    /// The high-level levels, every cluster in every world, in order.
    pub event: Vec<String>,
    /// `Q` under the low-level model.
    pub low: f64,
    /// `τ(Q)` under the high-level model.
    pub high: f64,
}

/// The outcome of [`layer_consistency`].
#[derive(Clone, Debug, PartialEq)]
pub struct LayerCheck {
    /// Whether every query agreed within the tolerance.
    pub holds: bool,
    /// How many joint distributions were compared.
    pub compared: usize,
    /// The largest disagreement, if any query disagreed.
    pub worst: Option<Mismatch>,
}

/// Every value each variable takes with positive probability.
pub(crate) fn supports(m: &Model, vars: &[String]) -> Result<Vec<Vec<String>>, AbstractionError> {
    let obs = m.world(m.observational());
    vars.iter()
        .map(|v| {
            if !obs.world().variables().iter().any(|n| n == v) {
                return Err(AbstractionError::UnknownVariable(v.clone()));
            }
            let t = m
                .world(m.observational())
                .probability_table(Some(&[v.as_str()]))?;
            Ok(t.rows
                .iter()
                .filter(|(_, p)| *p > 0.0)
                .map(|(r, _)| r[0].as_int().to_string())
                .collect())
        })
        .collect()
}

/// Every assignment of `vars` over their `domains`.
pub(crate) fn assignments(
    vars: &[String],
    domains: &[Vec<String>],
    cap: usize,
) -> Result<Vec<Vec<(String, String)>>, AbstractionError> {
    let mut out: Vec<Vec<(String, String)>> = vec![Vec::new()];
    for (v, d) in vars.iter().zip(domains) {
        if out.len().saturating_mul(d.len()) > cap {
            return Err(AbstractionError::TooLarge(cap));
        }
        out = out
            .into_iter()
            .flat_map(|a| {
                d.iter().map(move |l| {
                    let mut a = a.clone();
                    a.push((v.clone(), l.clone()));
                    a
                })
            })
            .collect();
    }
    Ok(out)
}

/// Whether `high` is `L_layer`-τ consistent with `low` (Def. 15.2.3), by
/// comparing the joint over every high-level variable:
///
/// * layer 1: in the natural world;
/// * layer 2: in every world `do(x_L)`, for each union of clusters and each
///   assignment of its members over the values they take;
/// * layer 3: in every pair of those worlds.
///
/// Every quantity of the layer's form over one world (two for layer 3) is a
/// sum of entries of these joints, so they agree exactly when the joints
/// do.
///
/// # Errors
///
/// [`AbstractionError::TooLarge`] past 4096 interventions, and the models'
/// errors.
pub fn layer_consistency(
    low: &mut Model,
    high: &mut impl HighLevel,
    tau: &Abstraction,
    layer: u8,
    tolerance: f64,
) -> Result<LayerCheck, AbstractionError> {
    let mut interventions: Vec<Vec<(String, String)>> = vec![Vec::new()];
    if layer >= 2 {
        let n = tau.len();
        for mask in 1u64..(1u64 << n) {
            let vars: Vec<String> = (0..n)
                .filter(|i| mask >> i & 1 == 1)
                .flat_map(|i| tau.members_at(i).to_vec())
                .collect();
            let domains = supports(low, &vars)?;
            interventions.extend(assignments(&vars, &domains, MAX_INTERVENTIONS)?);
            if interventions.len() > MAX_INTERVENTIONS {
                return Err(AbstractionError::TooLarge(MAX_INTERVENTIONS));
            }
        }
    }
    let mut sets: Vec<Vec<Vec<(String, String)>>> =
        interventions.iter().map(|x| vec![x.clone()]).collect();
    if layer >= 3 {
        for i in 0..interventions.len() {
            for j in i + 1..interventions.len() {
                sets.push(vec![interventions[i].clone(), interventions[j].clone()]);
            }
        }
    }
    let mut worst: Option<Mismatch> = None;
    for worlds in &sets {
        let l = low_joint(low, tau, worlds)?;
        let lifted: Vec<Vec<(String, String)>> = worlds
            .iter()
            .map(|x| {
                let pairs: Vec<(&str, &str)> =
                    x.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                tau.tau(&pairs)
            })
            .collect::<Result<_, _>>()?;
        let h = high.joint(tau, &lifted)?;
        let keys: BTreeSet<&Vec<String>> = l.keys().chain(h.keys()).collect();
        for k in keys {
            let (a, b) = (
                l.get(k).copied().unwrap_or(0.0),
                h.get(k).copied().unwrap_or(0.0),
            );
            if worst
                .as_ref()
                .is_none_or(|w| (a - b).abs() > (w.low - w.high).abs())
            {
                worst = Some(Mismatch {
                    worlds: worlds.clone(),
                    event: k.clone(),
                    low: a,
                    high: b,
                });
            }
        }
    }
    let holds = worst
        .as_ref()
        .is_none_or(|w| (w.low - w.high).abs() <= tolerance);
    Ok(LayerCheck {
        holds,
        compared: sets.len(),
        worst: worst.filter(|w| (w.low - w.high).abs() > tolerance),
    })
}
