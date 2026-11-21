//! The scopes of a G-constrained NCM (Defs. 13.1.10 and 13.1.11).

use std::collections::{BTreeMap, BTreeSet};

use petgraph::visit::EdgeRef;

use super::NcmError;
use crate::graph::dseparation::{Admg, EdgeKind};

/// The complete confounded components (C²-components, Def. 13.1.10) of `g`:
/// the maximal sets of variables every pair of which is joined by a
/// bidirected edge.
///
/// They are the maximal cliques of the bidirected part of `g`, so a variable
/// with no bidirected edge is a component of its own, and the components
/// cover every variable but may overlap. Each list is sorted, and so is the
/// list of lists.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::ncm::c2_components;
///
/// // A <-> B <-> C, with no edge between A and C: two cliques, not one
/// // c-component's worth.
/// let mut g = Admg::new();
/// g.bidirected("A", "B");
/// g.bidirected("B", "C");
/// g.add_node("D");
/// assert_eq!(
///     c2_components(&g),
///     vec![vec!["A", "B"], vec!["B", "C"], vec!["D"]]
/// );
/// ```
#[must_use]
pub fn c2_components(g: &Admg) -> Vec<Vec<String>> {
    let names: Vec<String> = g.variables().iter().map(|s| (*s).to_string()).collect();
    let mut adj: BTreeMap<&str, BTreeSet<&str>> = names
        .iter()
        .map(|n| (n.as_str(), BTreeSet::new()))
        .collect();
    for e in g.graph().edge_references() {
        if *e.weight() == EdgeKind::Bidirected {
            let (a, b) = (
                g.graph()[e.source()].as_str(),
                g.graph()[e.target()].as_str(),
            );
            if a != b {
                adj.get_mut(a).expect("node").insert(b);
                adj.get_mut(b).expect("node").insert(a);
            }
        }
    }
    let mut out = Vec::new();
    let all: BTreeSet<&str> = adj.keys().copied().collect();
    bron_kerbosch(&adj, BTreeSet::new(), all, BTreeSet::new(), &mut out);
    let mut out: Vec<Vec<String>> = out
        .into_iter()
        .map(|c| c.into_iter().map(str::to_string).collect())
        .collect();
    out.sort();
    out
}

/// Bron–Kerbosch with pivoting: every maximal clique extending `r`.
fn bron_kerbosch<'a>(
    adj: &BTreeMap<&'a str, BTreeSet<&'a str>>,
    r: BTreeSet<&'a str>,
    mut p: BTreeSet<&'a str>,
    mut x: BTreeSet<&'a str>,
    out: &mut Vec<BTreeSet<&'a str>>,
) {
    if p.is_empty() && x.is_empty() {
        out.push(r);
        return;
    }
    let pivot = p
        .union(&x)
        .max_by_key(|u| adj[*u].intersection(&p).count())
        .copied()
        .expect("p or x is non-empty");
    let candidates: Vec<&str> = p.difference(&adj[pivot]).copied().collect();
    for v in candidates {
        let mut r2 = r.clone();
        r2.insert(v);
        let p2 = p.intersection(&adj[v]).copied().collect();
        let x2 = x.intersection(&adj[v]).copied().collect();
        bron_kerbosch(adj, r2, p2, x2, out);
        p.remove(v);
        x.insert(v);
    }
}

/// The scopes a G-constrained NCM is built from (Def. 13.1.11): one shared
/// noise `Û_C` per C²-component `C` of `G`, and for each variable its
/// parents in `G`.
///
/// Training is what fixes the networks' parameters; this fixes only which
/// inputs each network sees, which is all that `G`'s constraints restrict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcmStructure {
    /// The variables in a topological order of `G`'s directed edges.
    pub order: Vec<String>,
    /// Each variable's parents in `G`, in `order`.
    pub parents: BTreeMap<String, Vec<String>>,
    /// The C²-components; noise `i` is shared by the variables of
    /// `confounders[i]`.
    pub confounders: Vec<Vec<String>>,
}

impl NcmStructure {
    /// Def. 13.1.11 applied to `g`.
    ///
    /// ```
    /// use why_data::graph::dseparation::Admg;
    /// use why_data::ncm::NcmStructure;
    ///
    /// // Ex. 13.10: D -> S -> B with D <-> B.
    /// let mut g = Admg::new();
    /// g.directed("D", "S");
    /// g.directed("S", "B");
    /// g.bidirected("D", "B");
    /// let s = NcmStructure::from_graph(&g).unwrap();
    /// assert_eq!(s.order, ["D", "S", "B"]);
    /// assert_eq!(s.confounders, vec![vec!["B", "D"], vec!["S"]]);
    /// assert_eq!(s.noises_of("B"), [0]);
    /// assert_eq!(s.parents["B"], ["S"]);
    /// ```
    ///
    /// # Errors
    ///
    /// [`NcmError::Graph`] if the directed edges have a cycle.
    pub fn from_graph(g: &Admg) -> Result<Self, NcmError> {
        let order = g.topological_order()?;
        let rank: BTreeMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i))
            .collect();
        let mut parents: BTreeMap<String, Vec<String>> =
            order.iter().map(|v| (v.clone(), Vec::new())).collect();
        for e in g.graph().edge_references() {
            if *e.weight() == EdgeKind::Directed {
                let (a, b) = (&g.graph()[e.source()], &g.graph()[e.target()]);
                let list = parents.get_mut(b).expect("node");
                if !list.contains(a) {
                    list.push(a.clone());
                }
            }
        }
        for list in parents.values_mut() {
            list.sort_by_key(|v| rank[v.as_str()]);
        }
        Ok(Self {
            order,
            parents,
            confounders: c2_components(g),
        })
    }

    /// The indices of the noises `var`'s mechanism reads: the C²-components
    /// that contain it.
    #[must_use]
    pub fn noises_of(&self, var: &str) -> Vec<usize> {
        self.confounders
            .iter()
            .enumerate()
            .filter(|(_, c)| c.iter().any(|v| v == var))
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether no noise is shared, so every interventional distribution is
    /// identifiable (Corol. 13.1.10).
    #[must_use]
    pub fn is_markovian(&self) -> bool {
        self.confounders.iter().all(|c| c.len() == 1)
    }

    /// The ancestors of `targets` once the edges into `cut` are removed,
    /// `targets` included, in `order`.
    #[must_use]
    pub fn ancestors(&self, targets: &BTreeSet<String>, cut: &BTreeSet<String>) -> Vec<String> {
        let mut keep: BTreeSet<String> = targets.clone();
        let mut stack: Vec<String> = targets.iter().cloned().collect();
        while let Some(v) = stack.pop() {
            if cut.contains(&v) {
                continue;
            }
            for p in &self.parents[&v] {
                if keep.insert(p.clone()) {
                    stack.push(p.clone());
                }
            }
        }
        self.order
            .iter()
            .filter(|v| keep.contains(*v))
            .cloned()
            .collect()
    }
}
