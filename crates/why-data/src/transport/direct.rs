//! Direct transportability (Sec. 11.2.1): Thm 11.2.1, Algorithm 32 and
//! structural covariate shift.

use std::collections::BTreeSet;

use super::{SelectionDiagram, TransportError, as_refs};
use crate::graph::dseparation::EdgeKind;

/// Thm 11.2.1: `P*(y | x) = P^source(y | x)` for every pair of models
/// compatible with the diagram if and only if `S_source ⊥ Y | X`.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source and
/// [`TransportError::Graph`] for unknown or overlapping variables.
pub fn is_directly_transportable(
    d: &SelectionDiagram,
    source: &str,
    y: &str,
    x: &[&str],
) -> Result<bool, TransportError> {
    d.is_s_admissible(source, &[y], x)
}

/// Algorithm 32 (ListDTRDist): every `X ⊆ candidates` with
/// `S_source ⊥ Y | X`, that is, every directly transportable
/// `P*(y | x) = P^source(y | x)`.
///
/// The recursion branches on including or excluding one candidate at a time
/// and prunes a branch as soon as Lemma 11.2.3 shows it holds no separator:
/// some `X` with `I ⊆ X ⊆ R` separates if and only if
/// `An(Y ∪ I) ∩ R` does. Each set is therefore found with polynomial delay.
/// Sets come out in the order the leaves are reached, the included branch
/// first.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source or variable, and
/// [`TransportError::InvalidInput`] if `y` is a candidate.
pub fn list_dtr_dist(
    d: &SelectionDiagram,
    source: &str,
    y: &str,
    candidates: &[&str],
) -> Result<Vec<Vec<String>>, TransportError> {
    d.check(&[y])?;
    d.check(candidates)?;
    if candidates.contains(&y) {
        return Err(TransportError::InvalidInput(format!(
            "the outcome `{y}` cannot condition itself"
        )));
    }
    let delta = d.delta(source)?;
    let g = d.augmented();
    let s = d.s_node(source);
    let r: BTreeSet<String> = candidates.iter().map(|c| (*c).to_string()).collect();
    let mut out = Vec::new();
    // With no discrepancy every set separates; the S-node is isolated.
    let trivially = delta.is_empty();
    let exists_sep = |i: &BTreeSet<String>, r: &BTreeSet<String>| -> Result<bool, TransportError> {
        if trivially {
            return Ok(true);
        }
        let mut seeds: Vec<&str> = vec![y];
        seeds.extend(i.iter().map(String::as_str));
        let an: BTreeSet<String> = g
            .ancestors_of(&seeds)?
            .into_iter()
            .map(str::to_string)
            .collect();
        let x0: Vec<String> = an.intersection(r).cloned().collect();
        Ok(g.is_d_separated(&[&s], &[y], &as_refs(&x0))?)
    };
    recurse(&exists_sep, &BTreeSet::new(), &r, &mut out)?;
    Ok(out)
}

fn recurse(
    exists_sep: &impl Fn(&BTreeSet<String>, &BTreeSet<String>) -> Result<bool, TransportError>,
    i: &BTreeSet<String>,
    r: &BTreeSet<String>,
    out: &mut Vec<Vec<String>>,
) -> Result<(), TransportError> {
    if !exists_sep(i, r)? {
        return Ok(());
    }
    if i == r {
        out.push(i.iter().cloned().collect());
        return Ok(());
    }
    let w = r
        .difference(i)
        .next()
        .expect("I ⊊ R leaves a variable")
        .clone();
    let mut with = i.clone();
    with.insert(w.clone());
    recurse(exists_sep, &with, r, out)?;
    let mut without = r.clone();
    without.remove(&w);
    recurse(exists_sep, i, &without, out)
}

/// Def. 11.2.3, structural covariate shift: the features are exactly `Y`'s
/// parents, no source's discrepancy set contains `Y`, and `Y` shares no
/// unobserved confounder with a feature.
///
/// Under SCS, `E_{P*}[Y | x] = E_{P^k}[Y | x]` for every source (Ex. 11.8).
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown variable.
pub fn structural_covariate_shift(
    d: &SelectionDiagram,
    features: &[&str],
    y: &str,
) -> Result<bool, TransportError> {
    d.check(&[y])?;
    d.check(features)?;
    let parents = d.parents(y);
    let wanted: BTreeSet<String> = features.iter().map(|s| (*s).to_string()).collect();
    if parents != wanted {
        return Ok(false);
    }
    if d.sources.iter().any(|(_, delta)| delta.contains(y)) {
        return Ok(false);
    }
    let g = d.graph().graph();
    let confounded = g.edge_indices().any(|e| {
        if g[e] != EdgeKind::Bidirected {
            return false;
        }
        let (a, b) = g.edge_endpoints(e).expect("valid edge");
        let (a, b) = (g[a].as_str(), g[b].as_str());
        (a == y && wanted.contains(b)) || (b == y && wanted.contains(a))
    });
    Ok(!confounded)
}
