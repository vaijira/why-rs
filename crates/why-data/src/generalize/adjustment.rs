//! Generalizing experiments by covariate adjustment (Sec. 12.1, 12.1.1).

use std::collections::BTreeSet;

use crate::graph::dseparation::{Admg, DSepError};
use crate::transport::{Domain, Estimand, Regime, SelectionDiagram, TransportError, subsets};

fn owned(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn refs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

/// Checks that `X` and `Y` are non-empty, that `X`, `Y` and `Z` are pairwise
/// disjoint, and that every name is a variable of `g`.
fn validate(g: &Admg, x: &[&str], y: &[&str], z: &[&str]) -> Result<(), DSepError> {
    if x.is_empty() {
        return Err(DSepError::Empty("X"));
    }
    if y.is_empty() {
        return Err(DSepError::Empty("Y"));
    }
    let mut seen = BTreeSet::new();
    for v in x.iter().chain(y).chain(z) {
        if g.node(v).is_none() {
            return Err(DSepError::Unknown((*v).to_string()));
        }
        if !seen.insert(*v) {
            return Err(DSepError::Overlap((*v).to_string()));
        }
    }
    Ok(())
}

/// `(a ⊥ b | c)` in `g`, true when either side is empty.
fn separated(g: &Admg, a: &[&str], b: &[&str], c: &[&str]) -> Result<bool, DSepError> {
    if a.is_empty() || b.is_empty() {
        return Ok(true);
    }
    g.is_d_separated(a, b, c)
}

/// Condition (12.15): `(Z_i ⊥ Y | W, X)` in `G_X̄`, which lets `Z_i` enter
/// the conditioning set of `P(y | do(x), w)` by rule 1.
fn cond_12_15(g: &Admg, x: &[&str], y: &[&str], zi: &str, w: &[&str]) -> Result<bool, DSepError> {
    let gx = g.cut_edges_into(x)?;
    let mut given = w.to_vec();
    given.extend_from_slice(x);
    separated(&gx, &[zi], y, &given)
}

/// Condition (12.16): `(Z_i ⊥ X | W)` in `G_{X̄(W)}`, with `X(W)` the
/// treatments that are not ancestors of `W`. This is rule 3:
/// `P(z_i | do(x), w) = P(z_i | w)`.
fn cond_12_16(g: &Admg, x: &[&str], zi: &str, w: &[&str]) -> Result<bool, DSepError> {
    let an_w: BTreeSet<&str> = if w.is_empty() {
        BTreeSet::new()
    } else {
        g.ancestors_of(w)?.into_iter().collect()
    };
    let x_of_w: Vec<&str> = x.iter().copied().filter(|v| !an_w.contains(v)).collect();
    separated(&g.cut_edges_into(&x_of_w)?, &[zi], x, w)
}

/// `IsEAdmissible` (Algorithm 40), returning the order it found: an order
/// `Z_1 < Z_2 < …` of `Z` witnessing the e-adjustment criterion
/// (Def. 12.1.2), or `None` when `Z` does not satisfy it.
///
/// The non-descendants of `X` come first, in the order given. Each
/// descendant `Z_i` satisfies (12.15) `(Z_i ⊥ Y | Z_{<i}, X)` in `G_X̄`, or
/// (12.16) `(Z_i ⊥ X | Z_{<i})` in `G_{X̄(Z_{<i})}`. By Lemma 12.1.8 any
/// descendant that passes can be peeled off the end, so the greedy search
/// is complete (Thm 12.1.9): at most `n² − n` separation tests.
///
/// # Errors
///
/// [`DSepError`] for unknown names, an empty `X` or `Y`, or sets that are not
/// pairwise disjoint.
pub fn e_adjustment_order(
    g: &Admg,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<Option<Vec<String>>, DSepError> {
    validate(g, x, y, z)?;
    let de: BTreeSet<&str> = g.descendants_of(x)?.into_iter().collect();
    let mut remaining: Vec<&str> = z.to_vec();
    let mut tail: Vec<String> = Vec::new();
    'peel: loop {
        let candidates: Vec<&str> = remaining.iter().copied().filter(|v| de.contains(v)).collect();
        if candidates.is_empty() {
            let mut order = owned(&remaining);
            order.extend(tail.into_iter().rev());
            return Ok(Some(order));
        }
        for c in candidates {
            let rest: Vec<&str> = remaining.iter().copied().filter(|v| *v != c).collect();
            if cond_12_15(g, x, y, c, &rest)? || cond_12_16(g, x, c, &rest)? {
                tail.push(c.to_string());
                remaining = rest;
                continue 'peel;
            }
        }
        return Ok(None);
    }
}

/// Whether `Z` satisfies the e-adjustment criterion (Def. 12.1.2) relative to
/// `(X, Y)`, which by Thm 12.1.2 is exactly when
/// `P(y | do(x)) = Σ_z P(y | do(x), z) P(z)`.
///
/// ```
/// use why_data::generalize::is_e_admissible;
/// use why_data::graph::dseparation::Admg;
///
/// // X → Z → Y, X → Y: the mediator Z is d-connected to Y given X, and X
/// // causes it, so averaging z-specific effects over P(z) is wrong.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// g.directed("X", "Y");
/// assert!(!is_e_admissible(&g, &["X"], &["Y"], &["Z"]).unwrap());
/// // Without the edge Z → Y it is harmless.
/// let mut h = Admg::new();
/// h.directed("X", "Z");
/// h.directed("X", "Y");
/// assert!(is_e_admissible(&h, &["X"], &["Y"], &["Z"]).unwrap());
/// ```
///
/// # Errors
///
/// As [`e_adjustment_order`].
pub fn is_e_admissible(g: &Admg, x: &[&str], y: &[&str], z: &[&str]) -> Result<bool, DSepError> {
    Ok(e_adjustment_order(g, x, y, z)?.is_some())
}

/// Condition (ii) of the tr-adjustment criterion: `(Y ⊥ S | Z, X)` in
/// `D^Δ_X̄`, with only `source`'s S-node drawn.
fn s_separated(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<bool, TransportError> {
    let (aug, s) = d.augmented_for(source)?;
    let Some(s) = s else { return Ok(true) };
    let mut given = z.to_vec();
    given.extend_from_slice(x);
    Ok(separated(&aug.cut_edges_into(x)?, &[&s], y, &given)?)
}

/// The tr-adjustment criterion for multiple treatments (Def. 12.1.3): `Z`
/// satisfies e-adjustment in the diagram, and `(Y ⊥ S | Z, X)` in `D^Δ_X̄`.
/// By Thm 12.1.3 this is exactly when [`tr_adjustment`]'s formula
/// `P*(y | do(x)) = Σ_z P^source(y | do(x), z) P*(z)` holds.
///
/// The criterion compares one source with the target; with several sources
/// in the diagram, the others play no part.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source, and
/// [`TransportError::Graph`] for malformed sets.
pub fn is_tr_admissible(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<bool, TransportError> {
    if !is_e_admissible(d.graph(), x, y, z)? {
        return Ok(false);
    }
    s_separated(d, source, x, y, z)
}

/// The tr-adjustment criterion for a single treatment (Def. 12.1.1).
///
/// With `Z_nd` the non-descendants of `X` in `Z` and `Z_p` the descendants
/// that are d-connected to `Y` given `Z_nd, X` in `G_X̄` (Eq. 12.5):
/// (i) `(Z_p ⊥ X | Z \ Z_p)`, and (ii) `(Y ⊥ S | Z, X)` in `D^Δ_X̄`. By
/// Lemma 12.1.4 it agrees with [`is_tr_admissible`] whenever `X` is a
/// singleton.
///
/// Condition (i) is read in `G_{X̄(Z \ Z_p)}`, as (12.16) is, since the
/// text does not name a graph. A test checks the lemma on random diagrams.
///
/// # Errors
///
/// [`TransportError::InvalidInput`] if `x` is not a single variable, and as
/// [`is_tr_admissible`] otherwise.
pub fn is_tr_admissible_singleton(
    d: &SelectionDiagram,
    source: &str,
    x: &str,
    y: &[&str],
    z: &[&str],
) -> Result<bool, TransportError> {
    let g = d.graph();
    validate(g, &[x], y, z)?;
    let de: BTreeSet<&str> = g.descendants_of(&[x])?.into_iter().collect();
    let (zd, znd): (Vec<&str>, Vec<&str>) = z.iter().copied().partition(|v| de.contains(v));
    let gx = g.cut_edges_into(&[x])?;
    let mut given = znd.clone();
    given.push(x);
    let mut zp = Vec::new();
    for v in zd {
        if !separated(&gx, &[v], y, &given)? {
            zp.push(v);
        }
    }
    let rest: Vec<&str> = z.iter().copied().filter(|v| !zp.contains(v)).collect();
    let an_rest: BTreeSet<&str> = if rest.is_empty() {
        BTreeSet::new()
    } else {
        g.ancestors_of(&rest)?.into_iter().collect()
    };
    let cut: Vec<&str> = if an_rest.contains(x) { vec![] } else { vec![x] };
    if !separated(&g.cut_edges_into(&cut)?, &zp, &[x], &rest)? {
        return Ok(false);
    }
    s_separated(d, source, &[x], y, z)
}

/// The transport formula of Thm 12.1.3 (Eq. 12.18):
///
/// ```text
/// P*(y | do(x)) = Σ_z P^source(y | z; do(x)) P*(z)
/// ```
///
/// the z-specific effects measured in the source, averaged over the target's
/// covariate distribution.
///
/// ```
/// use why_data::generalize::tr_adjustment;
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::SelectionDiagram;
///
/// // Example 12.1 (Fig. 12.1(a)): age Z differs between LA and NYC.
/// let mut g = Admg::new();
/// g.directed("Z", "X");
/// g.directed("Z", "Y");
/// g.directed("X", "Y");
/// let d = SelectionDiagram::new(g).with_source("LA", &["Z"]).unwrap();
/// let e = tr_adjustment(&d, "LA", &["X"], &["Y"], &["Z"]).unwrap();
/// assert_eq!(e.to_string(), "sum_{z} P^LA(y | z; do(x)) P^*(z)");
/// ```
///
/// # Errors
///
/// [`TransportError::NotTransportable`] when `Z` fails the tr-adjustment
/// criterion, and as [`is_tr_admissible`] otherwise.
pub fn tr_adjustment(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<Estimand, TransportError> {
    if !is_tr_admissible(d, source, x, y, z)? {
        return Err(TransportError::NotTransportable(format!(
            "{{{}}} is not tr-admissible for the effect of {{{}}} on {{{}}}",
            z.join(", "),
            x.join(", "),
            y.join(", ")
        )));
    }
    Ok(adjustment_estimand(
        &Domain::source(source),
        &Domain::Target,
        x,
        y,
        z,
    ))
}

/// `Σ_z P^experiment(y | z; do(x)) P^weights(z)`. The experiment's `x` sits
/// among the conditioning variables, which is how an atomic regime is read.
pub(crate) fn adjustment_estimand(
    experiment: &Domain,
    weights: &Domain,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Estimand {
    let mut given = owned(z);
    given.extend(owned(x));
    let effect = Estimand::P {
        domain: experiment.clone().under(Regime::new().atomic(x)),
        vars: owned(y),
        given,
    };
    if z.is_empty() {
        return effect;
    }
    let covariates = Estimand::P {
        domain: weights.clone(),
        vars: owned(z),
        given: Vec::new(),
    };
    Estimand::sum(owned(z), Estimand::product(vec![effect, covariates]))
}

/// Thm 12.1.6: a subset of a tr-admissible `Z` that has no descendant of `X`
/// and is still tr-admissible — the largest one, the first found when
/// several tie. `None` when `Z` itself is not tr-admissible.
///
/// A set without descendants of `X` trivially satisfies e-adjustment, so only
/// `(Y ⊥ S | Z', X)` is searched for.
///
/// # Errors
///
/// [`TransportError::TooLarge`] when `Z` has more than 20 non-descendants of
/// `X`, and as [`is_tr_admissible`] otherwise.
pub fn pre_treatment_subset(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<Option<Vec<String>>, TransportError> {
    if !is_tr_admissible(d, source, x, y, z)? {
        return Ok(None);
    }
    let de: BTreeSet<&str> = d.graph().descendants_of(x)?.into_iter().collect();
    let nd: Vec<String> = z.iter().filter(|v| !de.contains(*v)).map(|s| (*s).to_string()).collect();
    let mut all = subsets(&nd)?;
    all.reverse();
    for cand in all {
        if s_separated(d, source, x, y, &refs(&cand))? {
            return Ok(Some(cand));
        }
    }
    unreachable!("Thm 12.1.6 guarantees a pre-treatment tr-admissible subset")
}

/// A smallest tr-admissible set, searched among the non-descendants of `X`
/// outside `Y`, which by Thm 12.1.6 loses no generality. `None` when there is
/// none, so that no covariate adjustment transports the effect.
///
/// # Errors
///
/// [`TransportError::TooLarge`] beyond 20 candidates, and as
/// [`is_tr_admissible`] otherwise.
pub fn find_tr_adjustment(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &[&str],
) -> Result<Option<Vec<String>>, TransportError> {
    let g = d.graph();
    validate(g, x, y, &[])?;
    let de: BTreeSet<&str> = g.descendants_of(x)?.into_iter().collect();
    let cands: Vec<String> = g
        .variables()
        .into_iter()
        .filter(|v| !de.contains(v) && !y.contains(v))
        .map(str::to_string)
        .collect();
    for cand in subsets(&cands)? {
        if s_separated(d, source, x, y, &refs(&cand))? {
            return Ok(Some(cand));
        }
    }
    Ok(None)
}
