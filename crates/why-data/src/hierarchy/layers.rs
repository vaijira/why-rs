//! The layers L1, L2, L2.25, L2.5 and L3 (Sec. 20.1), the counterfactual
//! reachability and ancestor sets, and single-world intervention graphs
//! (Sec. 20.3.2).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{HierarchyError, Term};
use crate::graph::dseparation::{Admg, EdgeKind};

/// A causal diagram by index: parents, children and a topological order.
#[derive(Clone, Debug)]
pub(super) struct Dag {
    pub names: Vec<String>,
    idx: HashMap<String, usize>,
    pub pa: Vec<Vec<usize>>,
    pub ch: Vec<Vec<usize>>,
    pub order: Vec<usize>,
}

impl Dag {
    pub fn new(g: &Admg) -> Result<Self, HierarchyError> {
        let names: Vec<String> = g.topological_order()?;
        let idx: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let n = names.len();
        let (mut pa, mut ch) = (vec![Vec::new(); n], vec![Vec::new(); n]);
        let graph = g.graph();
        for e in graph.edge_indices() {
            if graph[e] == EdgeKind::Directed {
                let (a, b) = graph.edge_endpoints(e).expect("edge exists");
                let (a, b) = (idx[&graph[a]], idx[&graph[b]]);
                pa[b].push(a);
                ch[a].push(b);
            }
        }
        Ok(Self {
            names,
            idx,
            pa,
            ch,
            order: (0..n).collect(),
        })
    }

    pub fn idx(&self, v: &str) -> Result<usize, HierarchyError> {
        self.idx
            .get(v)
            .copied()
            .ok_or_else(|| HierarchyError::UnknownVariable(v.to_string()))
    }

    /// The ancestors of `v`, itself included, in `G` with the edges into
    /// `cut` removed.
    pub fn an(&self, v: usize, cut: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut out = BTreeSet::from([v]);
        let mut stack = vec![v];
        while let Some(x) = stack.pop() {
            if cut.contains(&x) {
                continue;
            }
            for &p in &self.pa[x] {
                if out.insert(p) {
                    stack.push(p);
                }
            }
        }
        out
    }

    /// The descendants of `v`, itself included.
    pub fn de(&self, v: usize) -> BTreeSet<usize> {
        let mut out = BTreeSet::from([v]);
        let mut stack = vec![v];
        while let Some(x) = stack.pop() {
            for &c in &self.ch[x] {
                if out.insert(c) {
                    stack.push(c);
                }
            }
        }
        out
    }

    /// Checks that every name in the term exists.
    pub fn check(&self, t: &Term) -> Result<(), HierarchyError> {
        self.idx(&t.var)?;
        for k in t.sub.keys() {
            self.idx(k)?;
        }
        Ok(())
    }

    /// `‖W_t‖`: the subscript restricted to the ancestors of `W` in
    /// `G_T̄`, without `W` itself (a term `W_{t}` with `W ∈ T` is read as
    /// `W_{t∖w}`).
    pub fn restrict(&self, var: &str, sub: &BTreeMap<String, u8>) -> Term {
        let w = self.idx[var];
        let cut: BTreeSet<usize> = sub
            .keys()
            .map(|k| self.idx[k])
            .filter(|&k| k != w)
            .collect();
        let an = self.an(w, &cut);
        Term {
            var: var.to_string(),
            sub: sub
                .iter()
                .filter(|(k, _)| {
                    let i = self.idx[k.as_str()];
                    i != w && an.contains(&i)
                })
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }
}

/// Whether two subscripts agree on the variables they share.
pub(super) fn consistent(a: &BTreeMap<String, u8>, b: &BTreeMap<String, u8>) -> bool {
    a.iter().all(|(k, v)| b.get(k).is_none_or(|w| w == v))
}

/// The union of consistent subscripts, or `None` when two disagree.
fn union<'a>(
    subs: impl IntoIterator<Item = &'a BTreeMap<String, u8>>,
) -> Option<BTreeMap<String, u8>> {
    let mut out = BTreeMap::new();
    for s in subs {
        for (k, v) in s {
            if *out.entry(k.clone()).or_insert(*v) != *v {
                return None;
            }
        }
    }
    Some(out)
}

/// Whether a set of terms holds two potential responses of one variable
/// under different regimes.
fn repeats(set: &BTreeSet<Term>) -> bool {
    let mut seen: BTreeMap<&str, &BTreeMap<String, u8>> = BTreeMap::new();
    set.iter()
        .any(|t| seen.insert(&t.var, &t.sub).is_some_and(|s| *s != t.sub))
}

/// A layer of the augmented causal hierarchy PCH* (Thm 20.1.6), from the
/// least to the most expressive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    /// Observational: every subscript is empty.
    L1,
    /// Interventional: one regime for every variable.
    L2,
    /// Realizable when counterfactual randomization passes one value of a
    /// variable to all its children (Def. 20.1.11).
    L225,
    /// Realizable when counterfactual randomization can pass a different
    /// value to each child (Def. 20.1.12).
    L25,
    /// Counterfactual.
    L3,
}

fn restricted(d: &Dag, query: &[Term]) -> Result<Vec<Term>, HierarchyError> {
    query
        .iter()
        .map(|t| {
            d.check(t)?;
            Ok(d.restrict(&t.var, &t.sub))
        })
        .collect()
}

/// `An(W*)` (Def. 20.2.5): for each term `Y_x`, every `W_z` with `W` an
/// ancestor of `Y` in `G_X̄` outside `X` and `z = x ∩ An(W)` in `G_X̄`.
///
/// # Errors
///
/// [`HierarchyError::UnknownVariable`] for an unknown name, and
/// [`HierarchyError::Graph`] for a cyclic graph.
pub fn ancestor_set(g: &Admg, query: &[Term]) -> Result<BTreeSet<Term>, HierarchyError> {
    let d = Dag::new(g)?;
    let mut out = BTreeSet::new();
    for t in restricted(&d, query)? {
        let cut: BTreeSet<usize> = t.sub.keys().map(|k| d.idx[k]).collect();
        for w in d.an(d.idx[&t.var], &cut) {
            if !cut.contains(&w) {
                out.insert(d.restrict(&d.names[w], &t.sub));
            }
        }
    }
    Ok(out)
}

/// `CRS(W*)` (Def. 20.1.16): for each term `Y_x`, `‖W_x‖` for every
/// `W ∈ An(Y) ∪ De(X)` outside `X`, and `‖W_{x∖w}‖` for those in `X`. Then
/// the entries of each variable are merged into one when the union of the
/// subscripts of the terms that produced them changes none of those terms
/// (see the module's book notes).
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::hierarchy::{Term, reachability_set};
///
/// // Ex. 20.12: in the fork Z <- X -> Y, CRS(Z_x, Y_x') holds Z_x and Z_x'.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("X", "Y");
/// let crs = reachability_set(&g, &[Term::new("Z", &[("X", 1)]), Term::new("Y", &[("X", 0)])]).unwrap();
/// let shown: Vec<String> = crs.iter().map(ToString::to_string).collect();
/// assert_eq!(shown, ["X", "Y_{X=0}", "Y_{X=1}", "Z_{X=0}", "Z_{X=1}"]);
/// ```
///
/// # Errors
///
/// As [`ancestor_set`].
pub fn reachability_set(g: &Admg, query: &[Term]) -> Result<BTreeSet<Term>, HierarchyError> {
    let d = Dag::new(g)?;
    let terms = restricted(&d, query)?;
    // Each term's entries, by variable.
    let mut per: Vec<BTreeMap<usize, Term>> = Vec::new();
    for t in &terms {
        let mut reach = d.an(d.idx[&t.var], &BTreeSet::new());
        for k in t.sub.keys() {
            reach.extend(d.de(d.idx[k]));
        }
        per.push(
            reach
                .into_iter()
                .map(|w| (w, d.restrict(&d.names[w], &t.sub)))
                .collect(),
        );
    }
    let mut out = BTreeSet::new();
    for r in 0..d.names.len() {
        let holders: Vec<usize> = (0..terms.len())
            .filter(|&i| per[i].contains_key(&r))
            .collect();
        if holders.is_empty() {
            continue;
        }
        let merged = union(holders.iter().map(|&i| &terms[i].sub)).filter(|x| {
            holders
                .iter()
                .all(|&i| d.restrict(&terms[i].var, x) == terms[i])
        });
        match merged {
            Some(x) => {
                out.insert(d.restrict(&d.names[r], &x));
            }
            None => out.extend(holders.iter().map(|&i| per[i][&r].clone())),
        }
    }
    Ok(out)
}

/// The lowest layer of PCH* that holds `P(W*)` for every SCM compatible
/// with `g`:
///
/// * L1 when every (restricted) subscript is empty;
/// * L2 when the subscripts unite into one regime `x` with `‖W_x‖ = W_t`
///   for every term, and no term is of an intervened variable;
/// * L2.25 when `CRS(W*)` has no variable under two regimes and no two
///   entries with inconsistent subscripts (Lemma 20.1.4 (1));
/// * L2.5 when `An(W*)` has no variable under two regimes
///   (Lemma 20.1.4 (2));
/// * L3 otherwise.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::hierarchy::{Layer, Term, layer};
///
/// // Ex. 20.17 / Fig. 20.10: X -> Z -> Y, X -> Y. The NDE's unnested term
/// // P(Y_{x'z}, Z_x) is in L2.5 but not L2.25; the ETT's P(Y_x, X) is in L2.25.
/// let mut g = Admg::new();
/// for (a, b) in [("X", "Z"), ("Z", "Y"), ("X", "Y")] {
///     g.directed(a, b);
/// }
/// let nde = [Term::new("Y", &[("X", 0), ("Z", 1)]), Term::new("Z", &[("X", 1)])];
/// assert_eq!(layer(&g, &nde).unwrap(), Layer::L25);
/// let ett = [Term::new("Y", &[("X", 1)]), Term::natural("X")];
/// assert_eq!(layer(&g, &ett).unwrap(), Layer::L225);
/// // The probability of sufficiency needs Y twice: L3.
/// let ps = [Term::new("Y", &[("X", 1)]), Term::natural("X"), Term::natural("Y")];
/// assert_eq!(layer(&g, &ps).unwrap(), Layer::L3);
/// ```
///
/// # Errors
///
/// As [`ancestor_set`].
pub fn layer(g: &Admg, query: &[Term]) -> Result<Layer, HierarchyError> {
    let d = Dag::new(g)?;
    let terms = restricted(&d, query)?;
    if terms.iter().all(|t| t.sub.is_empty()) {
        return Ok(Layer::L1);
    }
    if let Some(x) = union(terms.iter().map(|t| &t.sub))
        && terms
            .iter()
            .all(|t| !x.contains_key(&t.var) && d.restrict(&t.var, &x) == *t)
    {
        return Ok(Layer::L2);
    }
    let crs = reachability_set(g, query)?;
    let clash = crs.iter().any(|a| {
        crs.iter()
            .any(|b| a.var != b.var && !consistent(&a.sub, &b.sub))
    });
    if !repeats(&crs) && !clash {
        return Ok(Layer::L225);
    }
    if !repeats(&ancestor_set(g, query)?) {
        return Ok(Layer::L25);
    }
    Ok(Layer::L3)
}

/// A single-world intervention graph `G(x)` (Algorithm 71): every
/// intervened `X` split into a random half, which keeps the edges into
/// `X`, and a fixed half `x`, which takes the edges out of it. Each random
/// node `V` is labelled `V(a)`, with `a` the fixed nodes above it.
#[derive(Clone, Debug)]
pub struct Swig {
    labels: BTreeMap<String, Term>,
    fixed: BTreeMap<String, u8>,
    random: Admg,
}

/// Builds the SWIG of `g` for `do(x)` (Algorithm 71).
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::hierarchy::{Term, swig};
///
/// // Fig. 20.28(d): X -> Z -> Y under do(x, z).
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// let s = swig(&g, &[("X", 1), ("Z", 0)]).unwrap();
/// assert_eq!(s.label("Z").unwrap(), &Term::new("Z", &[("X", 1)]));
/// assert_eq!(s.label("Y").unwrap(), &Term::new("Y", &[("Z", 0)]));
/// ```
///
/// # Errors
///
/// [`HierarchyError::UnknownVariable`] for an unknown name, and
/// [`HierarchyError::Graph`] for a cyclic graph.
pub fn swig(g: &Admg, x: &[(&str, u8)]) -> Result<Swig, HierarchyError> {
    let d = Dag::new(g)?;
    let fixed: BTreeMap<String, u8> = x.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    for k in fixed.keys() {
        d.idx(k)?;
    }
    let labels = d
        .names
        .iter()
        .map(|v| (v.clone(), d.restrict(v, &fixed)))
        .collect();
    let xs: Vec<&str> = fixed.keys().map(String::as_str).collect();
    let random = g.cut_edges_out_of(&xs)?;
    Ok(Swig {
        labels,
        fixed,
        random,
    })
}

impl Swig {
    /// The label `V(a)` of the random node of `v`.
    #[must_use]
    pub fn label(&self, v: &str) -> Option<&Term> {
        self.labels.get(v)
    }

    /// The fixed nodes.
    #[must_use]
    pub fn fixed(&self) -> &BTreeMap<String, u8> {
        &self.fixed
    }

    /// Whether the random nodes `a` and `b` are d-separated given `c` in
    /// the SWIG. The fixed nodes are constants with only outgoing edges, so
    /// they block every path through them.
    ///
    /// # Errors
    ///
    /// [`HierarchyError::UnknownVariable`] for an unknown name.
    pub fn separated(&self, a: &[&str], b: &[&str], c: &[&str]) -> Result<bool, HierarchyError> {
        Ok(self.random.is_d_separated(a, b, c)?)
    }

    /// Prop. 20.3.4: `P(W*)` is in L2.25 when the subscripts are consistent
    /// and every `‖W_t‖` is a node of the SWIG for the union of the
    /// subscripts. `self` must be that SWIG.
    #[must_use]
    pub fn holds(&self, query: &[Term], g: &Admg) -> bool {
        let Ok(d) = Dag::new(g) else { return false };
        query.iter().all(|t| {
            d.check(t).is_ok()
                && consistent(&t.sub, &self.fixed)
                && t.sub.keys().all(|k| self.fixed.contains_key(k))
                && self.labels.get(&t.var) == Some(&d.restrict(&t.var, &t.sub))
        })
    }

    /// The counterfactual adjustment criterion (Def. 20.3.9) for
    /// `P(y_x) = Σ_l P(y | x, l) P(l)`: `X ⊥ Y(x) | L` in the SWIG for
    /// `do(x)`, with every `L` a factual node (no fixed node above it).
    /// `self` must be the SWIG for `do(x)` with `x` the single treatment.
    ///
    /// # Errors
    ///
    /// [`HierarchyError::UnknownVariable`] for an unknown name, and
    /// [`HierarchyError::InvalidInput`] when the SWIG is not for one
    /// treatment.
    pub fn adjusts(&self, y: &str, l: &[&str]) -> Result<bool, HierarchyError> {
        let [x] = self.fixed.keys().map(String::as_str).collect::<Vec<_>>()[..] else {
            return Err(HierarchyError::InvalidInput(
                "the adjustment criterion needs a SWIG for a single treatment".into(),
            ));
        };
        for v in l {
            let lab = self
                .labels
                .get(*v)
                .ok_or_else(|| HierarchyError::UnknownVariable((*v).to_string()))?;
            if !lab.sub.is_empty() || *v == x {
                return Ok(false);
            }
        }
        self.separated(&[x], &[y], l)
    }
}
