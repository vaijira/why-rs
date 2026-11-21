//! Data-checkable relaxations of the AIC: Def. 15.2.5 and the partitions of
//! Eqs. 15.67 and 15.68 behind the causal coarsening theorem (Thm 15.2.5).

use std::collections::BTreeMap;

use super::consistency::{names_in, supports};
use super::{Abstraction, AbstractionError, AicViolation};
use petgraph::Direction;
use petgraph::visit::EdgeRef;

use crate::fairness::Dataset;
use crate::graph::dseparation::{Admg, EdgeKind};
use crate::scm::Model;

/// The weighted joint over `vars`, normalized.
fn table(d: &Dataset, vars: &[&str]) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError> {
    for v in vars {
        if d.levels(v).is_none() {
            return Err(AbstractionError::UnknownVariable((*v).to_string()));
        }
    }
    let mut t: BTreeMap<Vec<String>, f64> = BTreeMap::new();
    for r in 0..d.len() {
        let key = vars
            .iter()
            .map(|v| d.value(v, r).expect("checked").to_string())
            .collect();
        *t.entry(key).or_default() += d.weights().map_or(1.0, |w| w[r]);
    }
    let total: f64 = t.values().sum();
    if total <= 0.0 {
        return Err(AbstractionError::Data("no mass".into()));
    }
    t.values_mut().for_each(|w| *w /= total);
    Ok(t)
}

/// Checks the conditional abstract invariance condition (Def. 15.2.5) on
/// the observational data `p`, in the local form the text gives once the
/// C-DAG is known: for every cluster `C_i` and every two values of its
/// C-DAG parents' members with the same image under `τ`, the distribution of
/// `τ_{C_i}(c_i)` is the same, within `tolerance`. `g` is the low-level
/// diagram, from which the C-DAG is built.
///
/// Def. 15.2.5 itself compares all of `τ(V_L)` given any union of clusters.
/// Conditioning on a cluster moves the distribution of its ancestors, so
/// that form rejects even Ex. 15.7's invariant clustering of cholesterol.
/// There, `P(X | HDL = 0, LDL = 0) ≠ P(X | HDL = 1, LDL = 1)`.
///
/// Only values the data show are compared. Returns the first violation; its
/// `outputs` are the cluster's level and the two probabilities.
///
/// # Errors
///
/// [`AbstractionError::UnknownVariable`] when the data lack a clustered
/// variable, and the errors of [`Abstraction::cdag`].
pub fn conditional_invariance(
    p: &Dataset,
    g: &Admg,
    tau: &Abstraction,
    tolerance: f64,
) -> Result<Option<AicViolation>, AbstractionError> {
    let gc = tau.cdag(g)?;
    let graph = gc.graph();
    for i in 0..tau.len() {
        let name = tau.name_at(i);
        let node = gc.node(name).expect("a C-DAG node per cluster");
        let mut parents: Vec<usize> = graph
            .edges_directed(node, Direction::Incoming)
            .filter(|e| *e.weight() == EdgeKind::Directed)
            .map(|e| tau.index(&graph[e.source()]).expect("a cluster"))
            .collect();
        parents.sort_unstable();
        parents.dedup();
        if parents.is_empty() {
            continue;
        }
        let inputs: Vec<String> = parents
            .iter()
            .flat_map(|&j| tau.members_at(j).to_vec())
            .collect();
        let members = tau.members_at(i).to_vec();
        let mut vars: Vec<&str> = inputs.iter().map(String::as_str).collect();
        vars.extend(members.iter().map(String::as_str));
        let joint = table(p, &vars)?;
        let k = inputs.len();
        // pa -> distribution of τ_{C_i}(c_i).
        let mut given: BTreeMap<Vec<String>, BTreeMap<String, f64>> = BTreeMap::new();
        for (row, w) in &joint {
            let c: Vec<&str> = row[k..].iter().map(String::as_str).collect();
            *given
                .entry(row[..k].to_vec())
                .or_default()
                .entry(tau.map_at(i, &c))
                .or_default() += w;
        }
        let mut groups: BTreeMap<Vec<(String, String)>, Vec<&Vec<String>>> = BTreeMap::new();
        for x in given.keys() {
            let pairs: Vec<(&str, &str)> = inputs
                .iter()
                .zip(x)
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            groups.entry(tau.tau(&pairs)?).or_default().push(x);
        }
        for xs in groups.values() {
            let base = &given[xs[0]];
            let base_total: f64 = base.values().sum();
            for x in &xs[1..] {
                let other = &given[*x];
                let other_total: f64 = other.values().sum();
                for level in base.keys().chain(other.keys()) {
                    let a = base.get(level).copied().unwrap_or(0.0) / base_total;
                    let b = other.get(level).copied().unwrap_or(0.0) / other_total;
                    if (a - b).abs() > tolerance {
                        let pairs = |x: &Vec<String>| {
                            inputs.iter().cloned().zip(x.iter().cloned()).collect()
                        };
                        return Ok(Some(AicViolation {
                            cluster: name.to_string(),
                            first: pairs(xs[0]),
                            second: pairs(x),
                            outputs: (format!("P({name} = {level}) = {a:.6}"), format!("{b:.6}")),
                        }));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Groups `values` whose distributions (one probability per outcome)
/// agree within `tolerance`, keeping the order in which groups first
/// appear.
fn group(values: Vec<(String, Vec<f64>)>, tolerance: f64) -> Vec<Vec<String>> {
    let mut out: Vec<(Vec<f64>, Vec<String>)> = Vec::new();
    for (v, dist) in values {
        match out
            .iter_mut()
            .find(|(d, _)| d.iter().zip(&dist).all(|(a, b)| (a - b).abs() <= tolerance))
        {
            Some((_, members)) => members.push(v),
            None => out.push((dist, vec![v])),
        }
    }
    out.into_iter().map(|(_, m)| m).collect()
}

/// The observational partition of `cause`'s levels (Eq. 15.67): two levels
/// share a block when `P(effect | cause)` agrees within `tolerance`. The
/// blocks are the coarsest clustering with that property.
///
/// # Errors
///
/// [`AbstractionError::UnknownVariable`] when the data lack a column.
pub fn observational_partition(
    p: &Dataset,
    cause: &str,
    effect: &str,
    tolerance: f64,
) -> Result<Vec<Vec<String>>, AbstractionError> {
    let joint = table(p, &[cause, effect])?;
    let effects: Vec<String> = p.levels(effect).expect("checked by table").to_vec();
    let mut causes: Vec<String> = Vec::new();
    for k in joint.keys() {
        if !causes.contains(&k[0]) {
            causes.push(k[0].clone());
        }
    }
    let rows = causes
        .into_iter()
        .map(|c| {
            let total: f64 = joint
                .iter()
                .filter(|(k, _)| k[0] == c)
                .map(|(_, w)| w)
                .sum();
            let dist = effects
                .iter()
                .map(|e| {
                    joint
                        .get(&vec![c.clone(), e.clone()])
                        .copied()
                        .unwrap_or(0.0)
                        / total
                })
                .collect();
            (c, dist)
        })
        .collect();
    Ok(group(rows, tolerance))
}

/// The interventional partition of `cause`'s levels (Eq. 15.68): two levels
/// share a block when `P(effect_{cause = level})` agrees within
/// `tolerance`. The levels are those `cause` takes with positive
/// probability.
///
/// By the causal coarsening theorem (Thm 15.2.5) this partition is, for
/// almost every model, coarser than [`observational_partition`].
///
/// # Errors
///
/// [`AbstractionError::UnknownVariable`] for a name the model lacks, and
/// the model's errors.
pub fn interventional_partition(
    m: &mut Model,
    cause: &str,
    effect: &str,
    tolerance: f64,
) -> Result<Vec<Vec<String>>, AbstractionError> {
    let levels = supports(m, &[cause.to_string(), effect.to_string()])?;
    let (causes, effects) = (levels[0].clone(), levels[1].clone());
    let rows = causes
        .into_iter()
        .map(|c| {
            let name = names_in(m, &[(cause.to_string(), c.clone())], &[effect.to_string()])?;
            let t = m
                .world(m.observational())
                .probability_table(Some(&[name[0].as_str()]))?;
            let dist = effects
                .iter()
                .map(|e| {
                    t.rows
                        .iter()
                        .filter(|(r, _)| r[0].as_int().to_string() == *e)
                        .map(|(_, p)| p)
                        .sum::<f64>()
                })
                .collect();
            Ok((c, dist))
        })
        .collect::<Result<Vec<_>, AbstractionError>>()?;
    Ok(group(rows, tolerance))
}
