//! Learning with interventional data: augmented diagrams (Defs. 16.2.5,
//! 16.2.11), I-FCI (Algorithms 56-58), Ψ-FCI (Algorithm 59) and Ψ-PC
//! (Algorithm 60), and the equivalence characterizations (Thms 16.2.6,
//! 16.2.8).

use std::collections::{BTreeMap, BTreeSet};

use super::fci::{Engine, subsets};
use super::{DiscoveryError, Invariance, Pag, mag, markov_equivalent};
use crate::graph::dseparation::Admg;

/// One F-node of an augmented diagram: its name (`F1`, `F2`, …), the
/// symmetric difference `K` of the pair of targets it stands for (its
/// children), and that pair, as indices into the targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FNode {
    /// `F1`, `F2`, ….
    pub name: String,
    /// `K = I Δ J`, the variables it points to.
    pub targets: Vec<String>,
    /// The pair `(i, j)`, `i < j`, of targets or distributions.
    pub pair: (usize, usize),
}

/// A target's entries as variable → mechanism (`""` when unnamed).
fn parse(target: &[&str], g: &Admg) -> Result<BTreeMap<String, String>, DiscoveryError> {
    target
        .iter()
        .map(|t| {
            let (v, m) = t.split_once(':').unwrap_or((t, ""));
            if g.node(v).is_none() {
                return Err(DiscoveryError::UnknownVariable(v.to_string()));
            }
            Ok((v.to_string(), m.to_string()))
        })
        .collect()
}

/// The symmetric difference (Def. 16.2.2): variables in exactly one target,
/// or in both under different mechanisms.
fn symmetric_difference(a: &BTreeMap<String, String>, b: &BTreeMap<String, String>) -> Vec<String> {
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    keys.into_iter()
        .filter(|k| a.get(*k) != b.get(*k))
        .cloned()
        .collect()
}

fn check_reserved(names: &[&str], count: usize) -> Result<(), DiscoveryError> {
    for k in 1..=count {
        let f = format!("F{k}");
        if names.contains(&f.as_str()) {
            return Err(DiscoveryError::Reserved(f));
        }
    }
    Ok(())
}

/// The F-nodes for `targets`, over pairs `(i, j)` with `i < j` in order:
/// one per distinct non-empty symmetric difference when `unique`
/// (Def. 16.2.5, Alg. 57), else one per pair (Def. 16.2.11, Alg. 59).
fn fnodes(targets: &[BTreeMap<String, String>], unique: bool) -> Vec<FNode> {
    let mut out: Vec<FNode> = Vec::new();
    for i in 0..targets.len() {
        for j in i + 1..targets.len() {
            let k = symmetric_difference(&targets[i], &targets[j]);
            if unique && (k.is_empty() || out.iter().any(|f| f.targets == k)) {
                continue;
            }
            out.push(FNode {
                name: format!("F{}", out.len() + 1),
                targets: k,
                pair: (i, j),
            });
        }
    }
    out
}

/// The augmented diagram of `g` w.r.t. `targets` (one target per
/// distribution, entries `"X"` or `"X:2"`): `g` plus F-nodes pointing to
/// their symmetric differences. With `unique`, one F-node per distinct
/// non-empty symmetric difference (Def. 16.2.5, for known targets).
/// Without it, one per pair of targets (Def. 16.2.11, for unknown ones).
///
/// ```
/// use why_data::discovery::augment;
/// use why_data::graph::dseparation::Admg;
///
/// // Ex. 16.19: X -> Z -> Y with targets ⟨∅, X, Z, XZ⟩ makes six F-nodes,
/// // two for each of {X}, {Z} and {X, Z}.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// let (_, f) = augment(&g, &[&[], &["X"], &["Z"], &["X", "Z"]], false).unwrap();
/// let ks: Vec<String> = f.iter().map(|f| f.targets.join("")).collect();
/// assert_eq!(ks, ["X", "Z", "XZ", "XZ", "Z", "X"]);
/// let (_, f) = augment(&g, &[&[], &["X"], &["Z"], &["X", "Z"]], true).unwrap();
/// assert_eq!(f.len(), 3);
/// ```
///
/// # Errors
///
/// [`DiscoveryError::UnknownVariable`] for a target naming a variable `g`
/// lacks, and [`DiscoveryError::Reserved`] when a variable is named like an
/// F-node.
pub fn augment(
    g: &Admg,
    targets: &[&[&str]],
    unique: bool,
) -> Result<(Admg, Vec<FNode>), DiscoveryError> {
    let parsed = targets
        .iter()
        .map(|t| parse(t, g))
        .collect::<Result<Vec<_>, _>>()?;
    let f = fnodes(&parsed, unique);
    check_reserved(&g.variables(), f.len())?;
    let mut out = g.clone();
    for node in &f {
        out.add_node(&node.name);
        for k in &node.targets {
            out.directed(&node.name, k);
        }
    }
    Ok((out, f))
}

/// The MAG over `V ∪ F` of an augmented diagram.
fn augmented_mag(a: &Admg) -> Result<Pag, DiscoveryError> {
    mag(a, &a.variables())
}

/// Whether `g1` and `g2` are I-Markov equivalent under the known targets
/// (Thm 16.2.6): their augmented MAGs are Markov equivalent.
///
/// # Errors
///
/// Those of [`augment`], and [`DiscoveryError::Mismatch`] for graphs over
/// different variables.
pub fn i_markov_equivalent(
    g1: &Admg,
    g2: &Admg,
    targets: &[&[&str]],
) -> Result<bool, DiscoveryError> {
    let (a1, _) = augment(g1, targets, true)?;
    let (a2, _) = augment(g2, targets, true)?;
    markov_equivalent(&augmented_mag(&a1)?, &augmented_mag(&a2)?)
}

/// Whether `⟨g1, t1⟩` and `⟨g2, t2⟩` are Ψ-Markov equivalent (Thm 16.2.8):
/// the MAGs of their augmented diagrams (Def. 16.2.11, F-nodes indexed by
/// the same pairs of distributions) are Markov equivalent.
///
/// ```
/// use why_data::discovery::{i_markov_equivalent, psi_markov_equivalent};
/// use why_data::graph::dseparation::Admg;
///
/// // Ex. 16.15: X -> Y with X <-> Y under ⟨∅, {X}⟩ looks like Y -> X under
/// // ⟨∅, {X, Y}⟩, although with known targets ⟨∅, {X}⟩ they differ.
/// let mut g1 = Admg::new();
/// g1.directed("X", "Y");
/// g1.bidirected("X", "Y");
/// let mut g2 = Admg::new();
/// g2.directed("Y", "X");
/// assert!(psi_markov_equivalent(&g1, &[&[], &["X"]], &g2, &[&[], &["X", "Y"]]).unwrap());
/// assert!(!i_markov_equivalent(&g1, &g2, &[&[], &["X"]]).unwrap());
/// ```
///
/// # Errors
///
/// [`DiscoveryError::InvalidInput`] when the tuples of targets differ in
/// length, and those of [`i_markov_equivalent`].
pub fn psi_markov_equivalent(
    g1: &Admg,
    t1: &[&[&str]],
    g2: &Admg,
    t2: &[&[&str]],
) -> Result<bool, DiscoveryError> {
    if t1.len() != t2.len() {
        return Err(DiscoveryError::InvalidInput(
            "the target tuples differ in length".into(),
        ));
    }
    let (a1, _) = augment(g1, t1, false)?;
    let (a2, _) = augment(g2, t2, false)?;
    markov_equivalent(&augmented_mag(&a1)?, &augmented_mag(&a2)?)
}

/// Phase I of the interventional learners over `names ∪ F`: variables are
/// separated by conditional independence in distribution 0 (any one would
/// do under soft interventions), and an F-node from `Y` by the invariance
/// `P_i(y | w) = P_j(y | w)` of its pair (Algs. 58 and InvToSep).
fn interventional_skeleton(
    names: &[&str],
    f: &[FNode],
    r12: bool,
    known: bool,
    test: &mut impl Invariance,
) -> Result<Engine, DiscoveryError> {
    check_reserved(names, f.len())?;
    let n = names.len();
    let mut all: Vec<&str> = names.to_vec();
    all.extend(f.iter().map(|x| x.name.as_str()));
    let f_idx: Vec<usize> = (n..n + f.len()).collect();
    let marks = f
        .iter()
        .enumerate()
        .map(|(k, node)| {
            let ks = known.then(|| {
                node.targets
                    .iter()
                    .filter_map(|t| names.iter().position(|v| v == t))
                    .collect()
            });
            (n + k, ks)
        })
        .collect();
    Engine::skeleton(&all, marks, r12, |i, j| match (i >= n, j >= n) {
        (true, true) => Ok(Some(Vec::new())),
        (false, false) => {
            let others: Vec<usize> = (0..n).filter(|&k| k != i && k != j).collect();
            for w in subsets(&others) {
                let ws: Vec<&str> = w.iter().map(|&k| names[k]).collect();
                if test.independent(0, names[i], names[j], &ws)? {
                    let mut sep = w;
                    sep.extend(f_idx.iter().copied());
                    return Ok(Some(sep));
                }
            }
            Ok(None)
        }
        _ => {
            let (fk, y) = if i >= n { (i, j) } else { (j, i) };
            let (a, b) = f[fk - n].pair;
            let others: Vec<usize> = (0..n).filter(|&k| k != y).collect();
            for w in subsets(&others) {
                let ws: Vec<&str> = w.iter().map(|&k| names[k]).collect();
                if test.invariant(a, b, names[y], &ws)? {
                    let mut sep = w;
                    sep.extend(f_idx.iter().copied().filter(|&x| x != fk));
                    return Ok(Some(sep));
                }
            }
            Ok(None)
        }
    })
}

/// I-FCI (Algorithms 56-58): the I-PAG from distributions whose targets are
/// known. `targets[k]` generated distribution `k` of `test`.
///
/// F-nodes stand for the distinct symmetric differences (Alg. 57) and are
/// named `F1`, `F2`, … in the order their pairs first appear. After Phases
/// I and II, the FCI rules run together with R11 (edges out of F-nodes) and
/// R12 (an F-node with `K = {X}` adjacent to `Y ∉ K` makes `X → Y`).
///
/// # Errors
///
/// [`DiscoveryError::InvalidInput`] when the number of targets differs from
/// the number of distributions, [`DiscoveryError::Reserved`] for a variable
/// named like an F-node, and whatever the tests report.
pub fn i_fci(
    names: &[&str],
    targets: &[&[&str]],
    test: &mut impl Invariance,
) -> Result<Pag, DiscoveryError> {
    if targets.len() != test.distributions() {
        return Err(DiscoveryError::InvalidInput(format!(
            "{} targets for {} distributions",
            targets.len(),
            test.distributions()
        )));
    }
    let mut g = Admg::new();
    for v in names {
        g.add_node(v);
    }
    let parsed = targets
        .iter()
        .map(|t| parse(t, &g))
        .collect::<Result<Vec<_>, _>>()?;
    let f = fnodes(&parsed, true);
    let mut engine = interventional_skeleton(names, &f, true, true, test)?;
    engine.colliders();
    engine.fci_rules();
    Ok(engine.pag)
}

/// The F-nodes of the unknown-target learners: one per pair of
/// distributions, with no known symmetric difference.
fn pair_nodes(m: usize) -> Vec<FNode> {
    let mut out = Vec::new();
    for i in 0..m {
        for j in i + 1..m {
            out.push(FNode {
                name: format!("F{}", out.len() + 1),
                targets: Vec::new(),
                pair: (i, j),
            });
        }
    }
    out
}

/// Ψ-FCI (Algorithm 59): the Ψ-PAG from distributions whose targets are
/// unknown. One F-node per pair of distributions, `F1`, `F2`, … for
/// `(0, 1), (0, 2), …, (1, 2), …`. After Phases I and II, R+ (edges out of
/// F-nodes) and the FCI rules run until none applies.
///
/// # Errors
///
/// [`DiscoveryError::Reserved`] for a variable named like an F-node, and
/// whatever the tests report.
pub fn psi_fci(names: &[&str], test: &mut impl Invariance) -> Result<Pag, DiscoveryError> {
    let f = pair_nodes(test.distributions());
    let mut engine = interventional_skeleton(names, &f, false, false, test)?;
    engine.colliders();
    engine.fci_rules();
    Ok(engine.pag)
}

/// Ψ-PC (Algorithm 60): the Ψ-PDAG of a Markovian system from
/// distributions whose targets are unknown. It shares Ψ-FCI's skeleton
/// and colliders, then orients edges out of F-nodes and applies Meek's
/// rules R1-R3 (Fig. 16.17). Undirected edges are `o-o`.
///
/// # Errors
///
/// Those of [`psi_fci`].
pub fn psi_pc(names: &[&str], test: &mut impl Invariance) -> Result<Pag, DiscoveryError> {
    let f = pair_nodes(test.distributions());
    let mut engine = interventional_skeleton(names, &f, false, false, test)?;
    engine.colliders();
    engine.pc_rules();
    Ok(engine.pag)
}
