//! Algorithm 33 (Score-TR): transporting the score function of a feature set
//! from several source domains.

use std::collections::BTreeSet;

use super::{Domain, Estimand, SelectionDiagram, TransportError, as_refs, subsets};
use crate::graph::identification::identify;

/// What [`score_tr`] found for a transportable feature set.
#[derive(Clone, Debug)]
pub struct ScoreTr {
    /// `P*(y | z)` in terms of the source distributions.
    pub estimand: Estimand,
    /// `Z̃`: the features rule 2 of the do-calculus turned into interventions
    /// (lines 2–5).
    pub intervened: Vec<String>,
    /// `W`: the ancestors of `Y ∪ (Z \ Z̃)` once `Z̃` is removed (line 6).
    pub ancestral: Vec<String>,
    /// Each c-component of `W` and the source its c-factor is read from.
    pub components: Vec<(Vec<String>, Domain)>,
}

/// Algorithm 33 (Score-TR): whether the score function `E_{P*}[Y | z]` of the
/// feature set `z` is transportable from the sources of `d`, and the
/// estimand `P*(y | z)` when it is. Sound and complete (Thm 11.2.4).
///
/// 1. Repeatedly move a feature `Z0` with `Y ⊥ Z0 | Z \ {Z0}` in
///    `G[V \ Z̃]` with the edges out of `Z0` cut into `Z̃`, so that
///    `P*(y | z) = P*(y | z \ z̃, do(z̃))`.
/// 2. Let `W` be the ancestors of `Y ∪ (Z \ Z̃)` in `G[V \ Z̃]`, so the
///    numerator is a marginal of `Q*[W]`.
/// 3. `Q*[W]` factorises over the c-components `C_t` of `G[W]`. Each is read
///    from a source `i` with `C_t ∩ Δ^{i,*} = ∅`, where `Q*[C_t] = Q^i[C_t]`;
///    if there is none, the score is not transportable.
///
/// The result is `sum_{W \ (Y ∪ Z')} ∏ Q / sum_{W \ Z'} ∏ Q`, `Z' = Z \ Z̃`.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::{SelectionDiagram, score_tr};
///
/// // Covariate shift: S → X → Y, so E[Y | x] transports as is.
/// let mut g = Admg::new();
/// g.directed("X", "Y");
/// let d = SelectionDiagram::new(g).with_source("1", &["X"]).unwrap();
/// let s = score_tr(&d, &["X"], "Y").unwrap();
/// assert_eq!(s.estimand.to_string(), "P^1(y | x)");
/// ```
///
/// # Errors
///
/// [`TransportError::NotTransportable`] naming the c-component no source can
/// supply, [`TransportError::Unknown`] for an unknown variable, and
/// [`TransportError::InvalidInput`] if `y` is among the features.
pub fn score_tr(d: &SelectionDiagram, z: &[&str], y: &str) -> Result<ScoreTr, TransportError> {
    d.check(&[y])?;
    d.check(z)?;
    if z.contains(&y) {
        return Err(TransportError::InvalidInput(format!(
            "the outcome `{y}` cannot be a feature"
        )));
    }
    let g = d.graph();
    let all: Vec<String> = g.variables().into_iter().map(str::to_string).collect();    let mut zs: BTreeSet<String> = z.iter().map(|s| (*s).to_string()).collect();
    let mut tilde: BTreeSet<String> = BTreeSet::new();

    // Lines 2–5: exhaust rule 2.
    let remaining = |tilde: &BTreeSet<String>| -> Vec<String> {
        all.iter().filter(|v| !tilde.contains(*v)).cloned().collect()
    };
    loop {
        let sub = g.induced_subgraph(&as_refs(&remaining(&tilde)))?;
        let mut moved = None;
        for z0 in &zs {
            let rest: Vec<&str> = zs.iter().filter(|v| *v != z0).map(String::as_str).collect();
            let cut = sub.cut_edges_out_of(&[z0])?;
            if cut.is_d_separated(&[y], &[z0], &rest)? {
                moved = Some(z0.clone());
                break;
            }
        }
        match moved {
            Some(z0) => {
                zs.remove(&z0);
                tilde.insert(z0);
            }
            None => break,
        }
    }

    // Line 6: W = An(Z' ∪ {Y}) in G[V \ Z̃].
    let sub = g.induced_subgraph(&as_refs(&remaining(&tilde)))?;
    let mut seeds: Vec<&str> = zs.iter().map(String::as_str).collect();
    seeds.push(y);
    let w: Vec<String> = sub
        .ancestors_of(&seeds)?
        .into_iter()
        .map(str::to_string)
        .collect();

    // Line 7: c-components of G[W].
    let gw = sub.induced_subgraph(&as_refs(&w))?;
    let mut components = gw.c_components();
    for c in &mut components {
        c.sort();
    }
    components.sort();

    // Lines 8–15: read each c-factor from an invariant source.
    //
    // `Q[C] = P(c | do(s))` for any `S ⊇ Pa(C)` disjoint from `C`, and the
    // parents of `W` lie in `W ∪ Z̃`. Identifying it on the latent projection
    // onto `W ∪ Z̃` keeps every other variable out of the estimand, and the
    // projection preserves identifiability.
    let mut outer: BTreeSet<String> = w.iter().cloned().collect();
    outer.extend(tilde.iter().cloned());
    let kept: Vec<String> = outer.iter().cloned().collect();
    let proj = crate::crl::latent_projection(g, &as_refs(&kept))
        .map_err(|e| TransportError::InvalidInput(e.to_string()))?;
    let mut factors = Vec::new();
    let mut assignment = Vec::new();
    for c in &components {
        let rest: Vec<String> = kept.iter().filter(|v| !c.contains(v)).cloned().collect();
        let mut found = None;
        for (name, delta) in &d.sources {
            if c.iter().any(|v| delta.contains(v)) {
                continue;
            }
            // Q^i[C] = P^i(c | do(v \ c)); identifiable unless C hides a hedge.
            if let Ok(mut f) = identify(&proj, &as_refs(&rest), &as_refs(c)) {
                f.avoid_capture(&outer);
                found = Some((name.clone(), f));
                break;
            }
        }
        let Some((name, f)) = found else {
            return Err(TransportError::NotTransportable(format!(
                "no source shares the mechanisms of the c-component {{{}}}",
                c.join(", ")
            )));
        };
        let domain = Domain::Source(name);
        factors.push(Estimand::from_formula(&f, &domain));
        assignment.push((c.clone(), domain));
    }

    // Line 16.
    let prod = Estimand::product(factors);
    let sum_num: Vec<String> = w
        .iter()
        .filter(|v| *v != y && !zs.contains(*v))
        .cloned()
        .collect();
    let sum_den: Vec<String> = w.iter().filter(|v| !zs.contains(*v)).cloned().collect();
    let mut estimand = Estimand::ratio(
        Estimand::sum(sum_num, prod.clone()),
        Estimand::sum(sum_den, prod),
    );
    let rank = all
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    estimand.sort_given(&rank);
    Ok(ScoreTr {
        estimand,
        intervened: tilde.into_iter().collect(),
        ancestral: w,
        components: assignment,
    })
}

/// The transportable feature sets among `features` and the maximal ones.
#[derive(Clone, Debug)]
pub struct FeatureSets {
    /// Every transportable subset, smallest first.
    pub transportable: Vec<Vec<String>>,
    /// The maximal transportable subsets (Def. 11.2.5): no transportable
    /// proper superset exists. By Prop. 11.2.5, each is at least as accurate
    /// in the target as any transportable subset of it.
    pub maximal: Vec<Vec<String>>,
}

/// Runs [`score_tr`] on every subset of `features` and keeps the
/// transportable and maximal ones.
///
/// There can be exponentially many maximal sets (Ex. 11.12), so this is
/// exhaustive and capped at 20 features.
///
/// # Errors
///
/// [`TransportError::TooLarge`] beyond the cap, and whatever [`score_tr`]
/// reports other than non-transportability.
pub fn transportable_feature_sets(
    d: &SelectionDiagram,
    features: &[&str],
    y: &str,
) -> Result<FeatureSets, TransportError> {
    let mut transportable = Vec::new();
    for s in subsets(features)? {
        match score_tr(d, &s, y) {
            Ok(_) => {
                let mut v: Vec<String> = s.iter().map(|x| (*x).to_string()).collect();
                v.sort();
                transportable.push(v);
            }
            Err(TransportError::NotTransportable(_)) => {}
            Err(e) => return Err(e),
        }
    }
    let maximal = transportable
        .iter()
        .filter(|a| {
            !transportable
                .iter()
                .any(|b| b.len() > a.len() && a.iter().all(|v| b.contains(v)))
        })
        .cloned()
        .collect();
    Ok(FeatureSets {
        transportable,
        maximal,
    })
}
