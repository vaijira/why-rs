//! Mixed policy learning without context (Sec. 9.3.1, CRL Task 2): where to
//! intervene.
//!
//! An agent that may intervene on any subset of `X⋆` faces `2^|X⋆|`
//! intervention sets. Most can be discarded from the causal diagram alone:
//!
//! * a set that is not a **minimal intervention set** (Def. 9.3.2) has the
//!   same rewards as one of its subsets ([`is_minimal_intervention_set`],
//!   Prop. 9.3.1);
//! * a MIS that is not **possibly optimal** (Def. 9.3.3) is dominated by
//!   another MIS in every compatible SCM ([`pomis`], Thm 9.3.2).
//!
//! Thm 9.3.2 as printed projects `G` onto `X′ ∪ {Y}`. That cannot be right:
//! the empty set would then always be a POMIS, while Eq. 9.82 shows that in
//! Fig. 9.8(a) `µ_∅ ≤ µ*_{X1}` in every SCM. The projection is onto
//! `X⋆ ∪ {Y}`, as the text before the theorem says and as in Lee &
//! Bareinboim (2018). [`pomis`] follows that.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::CrlError;
use crate::graph::dseparation::{Admg, EdgeKind};

/// The most candidate variables [`pomis`] and
/// [`minimal_intervention_sets`] enumerate subsets of.
const SUBSET_CAP: usize = 20;

fn check(g: &Admg, names: &[&str]) -> Result<(), CrlError> {
    for n in names {
        if g.node(n).is_none() {
            return Err(CrlError::Unknown((*n).to_string()));
        }
    }
    Ok(())
}

/// The latent projection of `g` onto `keep` (Verma & Pearl).
///
/// Each bidirected edge is first made an explicit latent parent of its two
/// endpoints. Then, among the kept variables, `A → B` whenever `g` has a
/// directed path from `A` to `B` whose interior is latent, and `A ↔ B`
/// whenever some latent variable reaches both through directed paths with a
/// latent interior. The projection preserves d-separation and the
/// identifiability of every effect among the kept variables.
///
/// # Errors
///
/// [`CrlError::Unknown`] if a kept variable is not in `g`.
pub fn latent_projection(g: &Admg, keep: &[&str]) -> Result<Admg, CrlError> {
    check(g, keep)?;
    let keep: HashSet<&str> = keep.iter().copied().collect();
    let names: Vec<&str> = g.variables();
    let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut confounded: Vec<(&str, &str)> = Vec::new();
    for e in g.graph().edge_indices() {
        let (a, b) = g.graph().edge_endpoints(e).expect("valid edge");
        let (a, b) = (g.graph()[a].as_str(), g.graph()[b].as_str());
        match g.graph()[e] {
            EdgeKind::Directed => children.entry(a).or_default().push(b),
            EdgeKind::Bidirected => confounded.push((a, b)),
        }
    }
    // Kept variables reachable from `start`'s children through latent nodes.
    let reach = |start: &[&str]| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = start.to_vec();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if keep.contains(n) {
                out.insert(n.to_string());
            } else {
                stack.extend(children.get(n).into_iter().flatten().copied());
            }
        }
        out
    };

    let mut out = Admg::new();
    for n in &names {
        if keep.contains(n) {
            out.add_node(n);
        }
    }
    let mut directed = BTreeSet::new();
    for a in names.iter().filter(|n| keep.contains(*n)) {
        let kids: Vec<&str> = children.get(a).cloned().unwrap_or_default();
        for b in reach(&kids) {
            if b != *a {
                directed.insert((a.to_string(), b));
            }
        }
    }
    let mut bidirected = BTreeSet::new();
    let mut add_pairs = |set: &BTreeSet<String>| {
        let v: Vec<&String> = set.iter().collect();
        for (i, a) in v.iter().enumerate() {
            for b in &v[i + 1..] {
                bidirected.insert(((*a).clone(), (*b).clone()));
            }
        }
    };
    for l in names.iter().filter(|n| !keep.contains(*n)) {
        let kids: Vec<&str> = children.get(l).cloned().unwrap_or_default();
        add_pairs(&reach(&kids));
    }
    for (a, b) in &confounded {
        add_pairs(&reach(&[a, b]));
    }
    for (a, b) in directed {
        out.directed(&a, &b);
    }
    for (a, b) in bidirected {
        out.bidirected(&a, &b);
    }
    Ok(out)
}

/// Prop. 9.3.1: `X′` is a minimal intervention set for `Y` iff
/// `X′ ⊆ An(Y)` in `G_{X̄′}`, i.e. every `X ∈ X′` reaches `Y` through a
/// directed path that avoids the rest of `X′`.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a variable `g` lacks.
pub fn is_minimal_intervention_set(g: &Admg, set: &[&str], y: &str) -> Result<bool, CrlError> {
    check(g, set)?;
    check(g, &[y])?;
    if set.contains(&y) {
        return Err(CrlError::InvalidInput(format!("`{y}` is both reward and action")));
    }
    let cut = g.cut_edges_into(set)?;
    let an: HashSet<&str> = cut.ancestors_of(&[y])?.into_iter().collect();
    Ok(set.iter().all(|x| an.contains(x)))
}

/// `MUCT(G, Y)` (Def. 9.3.4): the smallest set containing `Y`, inside
/// `H = G[An(Y)]`, that is closed under descendants and under bidirected
/// edges. Built by growing `{Y}` with c-components and descendants until
/// it stops changing. Sorted.
///
/// # Errors
///
/// [`CrlError::Unknown`] if `y` is not in `g`.
pub fn muct(g: &Admg, y: &str) -> Result<Vec<String>, CrlError> {
    check(g, &[y])?;
    let an = g.ancestors_of(&[y])?;
    let h = g.induced_subgraph(&an)?;
    let components = h.c_components();
    let mut t: BTreeSet<String> = BTreeSet::from([y.to_string()]);
    loop {
        let mut grown = t.clone();
        for c in &components {
            if c.iter().any(|v| t.contains(v)) {
                grown.extend(c.iter().cloned());
            }
        }
        let seeds: Vec<&str> = grown.iter().map(String::as_str).collect();
        grown.extend(h.descendants_of(&seeds)?.into_iter().map(str::to_string));
        if grown == t {
            return Ok(t.into_iter().collect());
        }
        t = grown;
    }
}

/// `IB(G, Y) = Pa(T) \ T` with `T = MUCT(G, Y)`: the interventional border.
/// Intervening on it is at least as good as intervening on nothing in every
/// compatible SCM. Sorted.
///
/// # Errors
///
/// [`CrlError::Unknown`] if `y` is not in `g`.
pub fn interventional_border(g: &Admg, y: &str) -> Result<Vec<String>, CrlError> {
    let t: HashSet<String> = muct(g, y)?.into_iter().collect();
    let mut out = BTreeSet::new();
    for e in g.graph().edge_indices() {
        if g.graph()[e] != EdgeKind::Directed {
            continue;
        }
        let (a, b) = g.graph().edge_endpoints(e).expect("valid edge");
        let (a, b) = (&g.graph()[a], &g.graph()[b]);
        if t.contains(b) && !t.contains(a) {
            out.insert(a.clone());
        }
    }
    Ok(out.into_iter().collect())
}

/// Every subset of `candidates`, smallest first, then lexicographically by
/// position.
fn subsets<'a>(candidates: &[&'a str]) -> Result<Vec<Vec<&'a str>>, CrlError> {
    if candidates.len() > SUBSET_CAP {
        return Err(CrlError::TooManyPolicies {
            count: 2f64.powi(candidates.len() as i32),
            cap: 1 << SUBSET_CAP,
        });
    }
    let n = candidates.len();
    let mut masks: Vec<u32> = (0..(1u32 << n)).collect();
    masks.sort_by_key(|m| (m.count_ones(), (0..n).map(|i| (m >> i) & 1 == 0).collect::<Vec<_>>()));
    Ok(masks
        .into_iter()
        .map(|m| (0..n).filter(|i| m >> i & 1 == 1).map(|i| candidates[i]).collect())
        .collect())
}

/// Every minimal intervention set among the subsets of `candidates`
/// (`X⋆`), smallest first.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a variable `g` lacks, and
/// [`CrlError::TooManyPolicies`] for more than 20 candidates.
pub fn minimal_intervention_sets(
    g: &Admg,
    candidates: &[&str],
    y: &str,
) -> Result<Vec<Vec<String>>, CrlError> {
    let mut out = Vec::new();
    for s in subsets(candidates)? {
        if is_minimal_intervention_set(g, &s, y)? {
            out.push(s.iter().map(|v| (*v).to_string()).collect());
        }
    }
    Ok(out)
}

/// Every possibly-optimal minimal intervention set (Def. 9.3.3) among the
/// subsets of `candidates` (`X⋆`), smallest first.
///
/// By Thm 9.3.2, with `H` the latent projection of `G` onto `X⋆ ∪ {Y}`, a
/// MIS `X′` is a POMIS iff `IB(H_{X̄′}, Y) = X′`. Every other intervention
/// set is beaten or matched by a POMIS in every SCM compatible with `G`, so
/// a bandit agent need only play POMIS arms.
///
/// ```
/// use why_data::crl::pomis;
/// use why_data::graph::dseparation::Admg;
///
/// // Fig. 9.8(a): X1 → X2 → Y with X2 ↔ Y.
/// let mut g = Admg::new();
/// g.directed("X1", "X2");
/// g.directed("X2", "Y");
/// g.bidirected("X2", "Y");
/// let sets = pomis(&g, &["X1", "X2"], "Y").unwrap();
/// assert_eq!(sets, [vec!["X1".to_string()], vec!["X2".to_string()]]);
/// ```
///
/// # Errors
///
/// As [`minimal_intervention_sets`].
pub fn pomis(g: &Admg, candidates: &[&str], y: &str) -> Result<Vec<Vec<String>>, CrlError> {
    check(g, candidates)?;
    check(g, &[y])?;
    let mut keep: Vec<&str> = candidates.to_vec();
    keep.push(y);
    let h = latent_projection(g, &keep)?;
    let mut out = Vec::new();
    for s in subsets(candidates)? {
        if !is_minimal_intervention_set(&h, &s, y)? {
            continue;
        }
        let ib = interventional_border(&h.cut_edges_into(&s)?, y)?;
        let mut sorted: Vec<String> = s.iter().map(|v| (*v).to_string()).collect();
        sorted.sort();
        if ib == sorted {
            out.push(s.iter().map(|v| (*v).to_string()).collect());
        }
    }
    Ok(out)
}
