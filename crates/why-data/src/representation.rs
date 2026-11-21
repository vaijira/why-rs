//! Causal representation learning: chapter 18, "Causal Representation
//! Learning".
//!
//! Latent generative factors `V` (with a causal diagram, possibly
//! confounded) are mixed by an unknown diffeomorphism into a
//! high-dimensional `X` (Def. 18.2.1). A learned proxy model gives a
//! representation `V̂`, and the question is which `V̂_j` can only depend on
//! which `V_i` (Def. 18.2.3). The answer depends on the latent selection
//! diagram (Def. 18.2.2) and on which distributions are available:
//! observational or interventional, from which domains, and with which
//! targets. Comparing two distributions cancels the factors their
//! difference leaves invariant (Prop. 18.3.2). What remains, the ΔQ set
//! (Def. 18.3.2), constrains the Jacobian of `V̂` with respect to `V`.
//!
//! | Item | Book |
//! | --- | --- |
//! | [`delta_v`] | the changed variable set ΔV (Def. 18.3.1) |
//! | [`delta_q`] | the ΔQ set (Def. 18.3.2) |
//! | [`crid`], [`Cdm`] | Algorithm 65 (CRID) and the causal disentanglement map (Def. 18.5.1) |
//!
//! # Regimes
//!
//! A regime is a domain (0-based) and a list of intervention targets. A
//! target is `"V1"` (a soft intervention with a mechanism of its own),
//! `"V1:2"` (mechanism 2, shared with any other `"V1:2"`, even in another
//! domain), or either with a `!` after the name, `"V2!"` or `"V2!:1"`, for a
//! marginal intervention `σ(∅)` that cuts the variable's parents and
//! confounders. An empty list is the domain's observational distribution.
//! Unlabelled mechanisms always differ, as the book assumes when no index is
//! given.
//!
//! # How CRID is reconstructed
//!
//! Alg. 65 calls three sub-procedures the book does not print ("Alg. ??").
//! They are rebuilt here from the propositions they cite.
//!
//! * **Canceled variables (Prop. 18.3.3).** Fix a baseline whose marginal
//!   targets are exactly `T`, and compare every distribution whose marginal
//!   targets contain `T`. Each comparison `j` gives, for a canceled variable
//!   `c ∉ ΔQ_j`, a linear equation `Σ_{q ∈ ΔQ_j} A_{jq} ∂v̂_q/∂v_c = 0` with
//!   generic coefficients (Ex. 18.21). The unknowns are the entries the
//!   causal disentanglement map (CDM) still allows. A set of comparisons
//!   whose matching saturates its unknowns forces them to zero, which
//!   removes those CDM edges. This is the "reduce the required number of
//!   distributions" step of Ex. 18.38, where one comparison suffices in the
//!   second epoch because `V3 → V̂1` is already gone.
//! * **Within ΔQ sets (Prop. 18.3.4).** For a set of comparisons whose
//!   ΔQ sets cover `V_tar`, condition (3′) is a matching of "each variable
//!   twice, each edge of `E` once" onto distinct comparisons.
//! * **Canceled from ΔQ (Prop. 18.3.6)**, as stated, for single targets.
//!
//! `Pa^{T+}` (Prop. 18.3.2) depends on the topological order. ΔQ must hold
//! for every order, and the union over orders of the extended parents of a
//! union of c-components is its parent set, so `ΔQ = C(ΔV) ∪ Pa(C(ΔV))` in
//! `G_T`.
//!
//! # Book notes
//!
//! * Thm 18.5.1 says "no edges from `V_tar` to `V̂_en`" means `V_tar` is ID
//!   w.r.t. `V_en`. Def. 18.5.1 reads the CDM the other way: no edge
//!   `V_i → V̂_j` means `V_j` is ID w.r.t. `V_i`, and Ex. 18.38 agrees.
//!   [`Cdm::identifiable`] follows the definition.
//! * Ex. 18.24 says the targets `{V1} × 6, {V2, V3} × 2` fail condition
//!   (3′). That holds for the baseline `σ(1)` it uses. CRID tries every
//!   baseline (Alg. 65, line 7), and with a `{V2, V3}` distribution as the
//!   baseline all eight ΔQ sets are `V`, which satisfies (3′). So CRID
//!   finds `V3` ID w.r.t. `V1` there too.
//! * Ex. 18.18 lists `p(v4 | v2)` as invariant. With Fig. 18.12(c)'s
//!   `V3 → V4`, the factor is `p(v4 | v3)` (as Ex. 18.19 writes).
//!
//! Not implemented: indirect generative factors (Sec. 18.4) and the learning
//! experiments of Sec. 18.6. CRID's output does not depend on them: it
//! is a graphical procedure.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::discovery::subsets;
use crate::graph::dseparation::{Admg, DSepError, EdgeKind};

/// What can go wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum RepresentationError {
    /// A graph operation failed or named an unknown variable.
    Graph(String),
    /// A regime or S-node names a variable the graph lacks.
    UnknownVariable(String),
    /// More regimes than the search over subsets can handle (16).
    TooLarge(usize),
}

impl From<DSepError> for RepresentationError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e.to_string())
    }
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => f.write_str(e),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::TooLarge(n) => write!(f, "{n} regimes are more than CRID enumerates"),
        }
    }
}

impl std::error::Error for RepresentationError {}

/// One intervention target: the mechanism (`None` = a fresh, unshared
/// one) and whether it is marginal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Target {
    mechanism: Option<String>,
    marginal: bool,
}

/// A parsed regime: its domain and targets by variable index.
#[derive(Clone, Debug)]
struct Regime {
    domain: usize,
    targets: BTreeMap<usize, Target>,
}

impl Regime {
    fn marginal(&self) -> BTreeSet<usize> {
        self.targets
            .iter()
            .filter(|(_, t)| t.marginal)
            .map(|(v, _)| *v)
            .collect()
    }
}

/// A latent selection diagram with its regimes, ready for comparisons.
struct Setup {
    names: Vec<String>,
    g: Admg,
    /// S-node children per pair of domains.
    s: BTreeMap<(usize, usize), BTreeSet<usize>>,
    regimes: Vec<Regime>,
}

impl Setup {
    fn new(
        g: &Admg,
        s: &[(usize, usize, &[&str])],
        regimes: &[(usize, &[&str])],
    ) -> Result<Self, RepresentationError> {
        let mut names: Vec<String> = g.variables().iter().map(|v| (*v).to_string()).collect();
        names.sort();
        let idx = |v: &str| {
            names
                .iter()
                .position(|n| n == v)
                .ok_or_else(|| RepresentationError::UnknownVariable(v.to_string()))
        };
        let mut sm: BTreeMap<(usize, usize), BTreeSet<usize>> = BTreeMap::new();
        for (a, b, ch) in s {
            let e = sm.entry((*a.min(b), *a.max(b))).or_default();
            for c in *ch {
                e.insert(idx(c)?);
            }
        }
        let regimes = regimes
            .iter()
            .map(|(d, ts)| {
                let targets = ts
                    .iter()
                    .map(|t| {
                        let (head, mech) = match t.split_once(':') {
                            Some((h, m)) => (h, Some(m.to_string())),
                            None => (*t, None),
                        };
                        let (var, marginal) = match head.strip_suffix('!') {
                            Some(v) => (v, true),
                            None => (head, false),
                        };
                        Ok((
                            idx(var)?,
                            Target {
                                mechanism: mech,
                                marginal,
                            },
                        ))
                    })
                    .collect::<Result<_, RepresentationError>>()?;
                Ok(Regime {
                    domain: *d,
                    targets,
                })
            })
            .collect::<Result<_, RepresentationError>>()?;
        Ok(Self {
            names,
            g: g.clone(),
            s: sm,
            regimes,
        })
    }

    /// `G_T`: `G` with the arrowheads into `T` removed.
    fn cut(&self, t: &BTreeSet<usize>) -> Result<Admg, RepresentationError> {
        let ts: Vec<&str> = t.iter().map(|&i| self.names[i].as_str()).collect();
        Ok(self.g.cut_edges_into(&ts)?)
    }

    /// ΔV between regimes `a` and `b` (Def. 18.3.1).
    fn delta_v(&self, a: &Regime, b: &Regime) -> BTreeSet<usize> {
        let same = |x: Option<&Target>, y: Option<&Target>| match (x, y) {
            (None, None) => true,
            (Some(x), Some(y)) => x.mechanism.is_some() && x == y,
            _ => false,
        };
        let mut out = BTreeSet::new();
        for v in 0..self.names.len() {
            let (x, y) = (a.targets.get(&v), b.targets.get(&v));
            // (1) intervened differently.
            if !same(x, y) {
                out.insert(v);
                continue;
            }
            // (2) the domains differ there, unless one shared mechanism
            // overrides both.
            if a.domain != b.domain && x.is_none() {
                let pair = (a.domain.min(b.domain), a.domain.max(b.domain));
                if self.s.get(&pair).is_some_and(|c| c.contains(&v)) {
                    out.insert(v);
                }
            }
        }
        out
    }

    /// ΔQ = C(ΔV) ∪ Pa(C(ΔV)) in `G_T` (Def. 18.3.2).
    fn delta_q(&self, a: &Regime, b: &Regime, gt: &Admg) -> BTreeSet<usize> {
        let dv = self.delta_v(a, b);
        let comps = gt.c_components();
        let mut tilde: BTreeSet<usize> = BTreeSet::new();
        for comp in &comps {
            let idx: Vec<usize> = comp
                .iter()
                .map(|v| self.names.iter().position(|n| n == v).expect("a node"))
                .collect();
            if idx.iter().any(|i| dv.contains(i)) {
                tilde.extend(idx);
            }
        }
        let mut out = tilde.clone();
        let graph = gt.graph();
        for e in graph.edge_indices() {
            if graph[e] != EdgeKind::Directed {
                continue;
            }
            let (p, c) = graph.edge_endpoints(e).expect("valid edge");
            let (p, c) = (
                self.names
                    .iter()
                    .position(|n| *n == graph[p])
                    .expect("a node"),
                self.names
                    .iter()
                    .position(|n| *n == graph[c])
                    .expect("a node"),
            );
            if tilde.contains(&c) {
                out.insert(p);
            }
        }
        out
    }
}

/// The changed variable set ΔV between two regimes (Def. 18.3.1): the
/// variables intervened differently, and those an S-node of the two
/// domains points to unless one shared mechanism overrides both. Sorted
/// by name.
///
/// # Errors
///
/// [`RepresentationError::UnknownVariable`] for a name `g` lacks.
pub fn delta_v(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    a: (usize, &[&str]),
    b: (usize, &[&str]),
) -> Result<Vec<String>, RepresentationError> {
    let setup = Setup::new(g, s, &[a, b])?;
    let dv = setup.delta_v(&setup.regimes[0], &setup.regimes[1]);
    Ok(dv.into_iter().map(|i| setup.names[i].clone()).collect())
}

/// The ΔQ set between two regimes (Def. 18.3.2): the c-components of
/// `G_T` that meet ΔV, with their parents, where `T` is the first regime's
/// marginal targets. The variables outside it cancel from the comparison
/// (Prop. 18.3.2).
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::representation::delta_q;
///
/// // Ex. 18.15: V1 <-> V3, a soft intervention on V3 leaves nothing
/// // invariant.
/// let mut g = Admg::new();
/// g.bidirected("V1", "V3");
/// let q = delta_q(&g, &[], (0, &[]), (0, &["V3"])).unwrap();
/// assert_eq!(q, ["V1", "V3"]);
/// ```
///
/// # Errors
///
/// [`RepresentationError::UnknownVariable`] for a name `g` lacks.
pub fn delta_q(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    a: (usize, &[&str]),
    b: (usize, &[&str]),
) -> Result<Vec<String>, RepresentationError> {
    let setup = Setup::new(g, s, &[a, b])?;
    let gt = setup.cut(&setup.regimes[0].marginal())?;
    let dq = setup.delta_q(&setup.regimes[0], &setup.regimes[1], &gt);
    Ok(dq.into_iter().map(|i| setup.names[i].clone()).collect())
}

/// A causal disentanglement map (Def. 18.5.1): a bipartite graph from the
/// latent factors `V` to their learned representations `V̂`. An edge
/// `V_i → V̂_j` means `V̂_j` may depend on `V_i`; its absence means `V_j` is
/// identifiable with respect to `V_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cdm {
    names: Vec<String>,
    /// `edge[i][j]`: `V_i → V̂_j`.
    edge: Vec<Vec<bool>>,
}

impl Cdm {
    fn index(&self, v: &str) -> Result<usize, RepresentationError> {
        self.names
            .iter()
            .position(|n| n == v)
            .ok_or_else(|| RepresentationError::UnknownVariable(v.to_string()))
    }

    /// Whether `V̂_rep` may depend on `V_on`.
    ///
    /// # Errors
    ///
    /// [`RepresentationError::UnknownVariable`] for an unknown name.
    pub fn depends(&self, rep: &str, on: &str) -> Result<bool, RepresentationError> {
        Ok(self.edge[self.index(on)?][self.index(rep)?])
    }

    /// Whether `tar` is ID w.r.t. `en` (Def. 18.2.3): no `V̂_t`, `t ∈ tar`,
    /// depends on any `V_e`, `e ∈ en`.
    ///
    /// # Errors
    ///
    /// [`RepresentationError::UnknownVariable`] for an unknown name.
    pub fn identifiable(&self, tar: &[&str], en: &[&str]) -> Result<bool, RepresentationError> {
        for t in tar {
            for e in en {
                if self.depends(t, e)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// The representations each `V̂` may depend on, as `V̂1 <- {V1, V2}`.
    #[must_use]
    pub fn edges(&self) -> Vec<String> {
        (0..self.names.len())
            .map(|j| {
                let from: Vec<&str> = (0..self.names.len())
                    .filter(|&i| self.edge[i][j])
                    .map(|i| self.names[i].as_str())
                    .collect();
                format!("{}^ <- {{{}}}", self.names[j], from.join(", "))
            })
            .collect()
    }
}

impl fmt::Display for Cdm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.edges().join("; "))
    }
}

/// Whether every slot can be given its own comparison, `slots[k]` listing
/// the comparisons slot `k` may take (bipartite matching).
fn saturates(slots: &[Vec<usize>], n: usize) -> bool {
    fn augment(
        k: usize,
        slots: &[Vec<usize>],
        seen: &mut [bool],
        owner: &mut [Option<usize>],
    ) -> bool {
        for &c in &slots[k] {
            if !seen[c] {
                seen[c] = true;
                if owner[c].is_none_or(|o| augment(o, slots, seen, owner)) {
                    owner[c] = Some(k);
                    return true;
                }
            }
        }
        false
    }
    let mut owner = vec![None; n];
    (0..slots.len()).all(|k| augment(k, slots, &mut vec![false; n], &mut owner))
}

/// Algorithm 65 (CRID): the causal disentanglement map the latent
/// selection diagram and the regimes license.
///
/// `g` is the latent causal graph over `V` (bidirected edges are latent
/// confounders), `s` the S-node children per pair of domains, and `regimes`
/// the available distributions (see the module docs for the syntax). It
/// starts from the complete map and removes edges with Props. 18.3.3,
/// 18.3.4 and 18.3.6 until nothing changes.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::representation::crid;
///
/// // Ex. 18.38: V1 -> V2 -> V3 with V1 <-> V2, the domains differ at V3;
/// // observations in both domains, a soft intervention on V3 in domain 2
/// // and a marginal one on V2 in domain 1.
/// let mut g = Admg::new();
/// g.directed("V1", "V2");
/// g.directed("V2", "V3");
/// g.bidirected("V1", "V2");
/// let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["V3"])];
/// let cdm = crid(&g, &s, &[(0, &[]), (1, &[]), (1, &["V3"]), (0, &["V2!"])]).unwrap();
/// assert_eq!(cdm.edges(), ["V1^ <- {V1}", "V2^ <- {V2}", "V3^ <- {V2, V3}"]);
/// assert!(cdm.identifiable(&["V2", "V3"], &["V1"]).unwrap());
/// ```
///
/// # Errors
///
/// [`RepresentationError::UnknownVariable`] for a name `g` lacks, and
/// [`RepresentationError::TooLarge`] past 16 regimes.
pub fn crid(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    regimes: &[(usize, &[&str])],
) -> Result<Cdm, RepresentationError> {
    if regimes.len() > 16 {
        return Err(RepresentationError::TooLarge(regimes.len()));
    }
    let setup = Setup::new(g, s, regimes)?;
    let n = setup.names.len();
    let mut cdm = Cdm {
        names: setup.names.clone(),
        edge: vec![vec![true; n]; n],
    };
    let ts: BTreeSet<BTreeSet<usize>> = setup.regimes.iter().map(Regime::marginal).collect();
    loop {
        let before = cdm.edge.clone();
        for t in &ts {
            let gt = setup.cut(t)?;
            let within: Vec<&Regime> = setup
                .regimes
                .iter()
                .filter(|r| t.is_subset(&r.marginal()))
                .collect();
            for (bi, base) in within.iter().enumerate() {
                if base.marginal() != *t {
                    continue;
                }
                let qs: Vec<BTreeSet<usize>> = within
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != bi)
                    .map(|(_, r)| setup.delta_q(base, r, &gt))
                    .collect();
                canceled(&mut cdm, &qs);
                within_delta_q(&mut cdm, &qs, &gt, &setup.names)?;
            }
        }
        for t in &ts {
            if t.is_empty() {
                continue;
            }
            let gt = setup.cut(t)?;
            from_canceled(&mut cdm, &gt, &setup.names)?;
        }
        if cdm.edge == before {
            return Ok(cdm);
        }
    }
}

/// Prop. 18.3.3 with the CDM's zeros: for each canceled variable `c`, a set
/// of comparisons (none containing `c`) whose ΔQ sets give a matching that
/// saturates the still-allowed `c → V̂_q` removes those edges.
// `c` is a variable index, used beyond indexing `cdm.edge`.
#[allow(clippy::needless_range_loop)]
fn canceled(cdm: &mut Cdm, qs: &[BTreeSet<usize>]) {
    let n = cdm.names.len();
    for c in 0..n {
        let usable: Vec<usize> = (0..qs.len()).filter(|&j| !qs[j].contains(&c)).collect();
        for subset in subsets(&usable) {
            if subset.is_empty() {
                continue;
            }
            let union: BTreeSet<usize> =
                subset.iter().flat_map(|&j| qs[j].iter().copied()).collect();
            let unknowns: Vec<usize> = union.iter().copied().filter(|&q| cdm.edge[c][q]).collect();
            if unknowns.is_empty() {
                continue;
            }
            let slots: Vec<Vec<usize>> = unknowns
                .iter()
                .map(|q| {
                    subset
                        .iter()
                        .copied()
                        .filter(|&j| qs[j].contains(q))
                        .collect()
                })
                .collect();
            if saturates(&slots, qs.len()) {
                for q in unknowns {
                    cdm.edge[c][q] = false;
                }
            }
        }
    }
}

/// Prop. 18.3.4: for comparisons covering `V_tar` (the union of their ΔQ
/// sets), condition (3′) is a matching of each variable twice and each edge
/// of `E` once onto distinct comparisons. Then every pair conditionally
/// independent in `G_T(V_tar)` is disentangled both ways.
fn within_delta_q(
    cdm: &mut Cdm,
    qs: &[BTreeSet<usize>],
    gt: &Admg,
    names: &[String],
) -> Result<(), RepresentationError> {
    let all: Vec<usize> = (0..qs.len()).collect();
    for subset in subsets(&all) {
        if subset.len() < 2 {
            continue;
        }
        let tar: BTreeSet<usize> = subset.iter().flat_map(|&j| qs[j].iter().copied()).collect();
        if tar.len() < 2 || 2 * tar.len() > subset.len() {
            continue;
        }
        let tar_names: Vec<&str> = tar.iter().map(|&i| names[i].as_str()).collect();
        let sub = gt.induced_subgraph(&tar_names)?;
        let rest = |a: usize, b: usize| -> Vec<&str> {
            tar.iter()
                .filter(|&&k| k != a && k != b)
                .map(|&k| names[k].as_str())
                .collect()
        };
        let tv: Vec<usize> = tar.iter().copied().collect();
        let mut edges = Vec::new();
        let mut independent = Vec::new();
        for (x, &a) in tv.iter().enumerate() {
            for &b in &tv[x + 1..] {
                let sep = sub.is_d_separated(&[&names[a]], &[&names[b]], &rest(a, b))?;
                if sep {
                    independent.push((a, b));
                } else if subset
                    .iter()
                    .any(|&j| qs[j].contains(&a) && qs[j].contains(&b))
                {
                    edges.push((a, b));
                }
            }
        }
        if independent.is_empty()
            || independent
                .iter()
                .all(|&(a, b)| !cdm.edge[a][b] && !cdm.edge[b][a])
        {
            continue;
        }
        let mut slots: Vec<Vec<usize>> = Vec::new();
        for &v in &tv {
            let can: Vec<usize> = subset
                .iter()
                .copied()
                .filter(|&j| qs[j].contains(&v))
                .collect();
            slots.push(can.clone());
            slots.push(can);
        }
        for &(a, b) in &edges {
            slots.push(
                subset
                    .iter()
                    .copied()
                    .filter(|&j| qs[j].contains(&a) && qs[j].contains(&b))
                    .collect(),
            );
        }
        if saturates(&slots, qs.len()) {
            for (a, b) in independent {
                cdm.edge[a][b] = false;
                cdm.edge[b][a] = false;
            }
        }
    }
    Ok(())
}

/// Prop. 18.3.6: if no other representation depends on `V_t`, and `V_t` is
/// separated from the rest in `G_T`, then `V̂_t` depends on no other
/// variable.
fn from_canceled(cdm: &mut Cdm, gt: &Admg, names: &[String]) -> Result<(), RepresentationError> {
    let n = names.len();
    for t in 0..n {
        let others: Vec<usize> = (0..n).filter(|&q| q != t).collect();
        if others.is_empty() || others.iter().any(|&q| cdm.edge[t][q]) {
            continue;
        }
        let rest: Vec<&str> = others.iter().map(|&q| names[q].as_str()).collect();
        if gt.is_d_separated(&[&names[t]], &rest, &[])? {
            for q in others {
                cdm.edge[q][t] = false;
            }
        }
    }
    Ok(())
}
