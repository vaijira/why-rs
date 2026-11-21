//! Multi-domain structural learning (chapter 17): augmented selection
//! diagrams (Def. 17.2.4), MD-Markov equivalence (Thm 17.2.5), MD-PC
//! (Algorithm 61) and MD-FCI (Algorithm 64), with the node creation of
//! Algorithm 62 and the tests of Algorithm 63.

use std::collections::{BTreeMap, BTreeSet};

use super::fci::{Engine, UNKNOWN_SEPSET, subsets};
use super::interventional::FNode;
use super::{DiscoveryError, Invariance, Pag, mag, markov_equivalent};
use crate::graph::dseparation::Admg;

/// One distribution of a multi-domain study: its domain (0-based), its
/// intervention targets (entries `"X"` or `"X:2"`), and whether the
/// targets are known. An empty known target is the domain's observational
/// distribution.
pub type Regime<'a> = (usize, &'a [&'a str], bool);

/// An S-node of an augmented selection diagram: `S12` for domains 1 and 2
/// (1-based in the name), with its children.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SNode {
    /// `S12`, `S13`, ….
    pub name: String,
    /// The pair of domains `(i, j)`, 0-based, `i < j`.
    pub domains: (usize, usize),
    /// The variables it points to.
    pub targets: Vec<String>,
}

/// A regime with its targets parsed (variable → mechanism).
struct Parsed {
    domain: usize,
    targets: BTreeMap<String, String>,
    known: bool,
}

fn parse(setup: &[Regime<'_>], names: &BTreeSet<String>) -> Result<Vec<Parsed>, DiscoveryError> {
    setup
        .iter()
        .map(|(d, t, known)| {
            let targets = t
                .iter()
                .map(|e| {
                    let (v, m) = e.split_once(':').unwrap_or((e, ""));
                    if !names.contains(v) {
                        return Err(DiscoveryError::UnknownVariable(v.to_string()));
                    }
                    Ok((v.to_string(), m.to_string()))
                })
                .collect::<Result<_, _>>()?;
            Ok(Parsed {
                domain: *d,
                targets,
                known: *known,
            })
        })
        .collect()
}

/// The number of domains, checking that each has a distribution.
fn domains(setup: &[Parsed]) -> Result<usize, DiscoveryError> {
    let n = setup.iter().map(|p| p.domain + 1).max().unwrap_or(0);
    for d in 0..n {
        if !setup.iter().any(|p| p.domain == d) {
            return Err(DiscoveryError::InvalidInput(format!(
                "domain {} has no distribution",
                d + 1
            )));
        }
    }
    Ok(n)
}

/// The symmetric difference across a pair of regimes (Def. 17.2.1). Within
/// a domain it compares mechanisms. Across domains, interventions never
/// share a mechanism (Assumption 17.2.3), so it is the union.
fn delta(a: &Parsed, b: &Parsed) -> Vec<String> {
    if a.domain == b.domain {
        let keys: BTreeSet<&String> = a.targets.keys().chain(b.targets.keys()).collect();
        keys.into_iter()
            .filter(|k| a.targets.get(*k) != b.targets.get(*k))
            .cloned()
            .collect()
    } else {
        let keys: BTreeSet<&String> = a.targets.keys().chain(b.targets.keys()).collect();
        keys.into_iter().cloned().collect()
    }
}

fn observational(p: &Parsed) -> bool {
    p.known && p.targets.is_empty()
}

/// The F-nodes of Algorithm 62: one per pair of distributions `(m, l)`,
/// `m < l`, except pairs of known observational distributions from
/// different domains (the S-node stands for those).
fn fnodes(setup: &[Parsed]) -> Vec<FNode> {
    let mut out = Vec::new();
    for m in 0..setup.len() {
        for l in m + 1..setup.len() {
            if setup[m].domain != setup[l].domain
                && observational(&setup[m])
                && observational(&setup[l])
            {
                continue;
            }
            out.push(FNode {
                name: format!("F{}", out.len() + 1),
                targets: delta(&setup[m], &setup[l]),
                pair: (m, l),
            });
        }
    }
    out
}

fn s_name(i: usize, j: usize) -> String {
    format!("S{}{}", i + 1, j + 1)
}

/// Builds the augmented selection diagram. `joint` makes each cross-domain
/// F-node also point to its domain pair's S-node children (see
/// [`md_fci`]), unless `known_children` and both targets are known.
fn build(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    setup: &[Regime<'_>],
    joint: bool,
    known_children: bool,
) -> Result<(Admg, Vec<FNode>, Vec<SNode>), DiscoveryError> {
    let names: BTreeSet<String> = g.variables().iter().map(|v| (*v).to_string()).collect();
    let parsed = parse(setup, &names)?;
    let n = domains(&parsed)?;
    let mut f = fnodes(&parsed);
    let mut snodes = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let mut children: BTreeSet<String> = BTreeSet::new();
            for (a, b, ch) in s {
                if (a.min(b), a.max(b)) == (&i, &j) {
                    for c in *ch {
                        if !names.contains(*c) {
                            return Err(DiscoveryError::UnknownVariable((*c).to_string()));
                        }
                        children.insert((*c).to_string());
                    }
                }
            }
            // D_ij: what every cross pair of regimes changes.
            let mut d: Option<BTreeSet<String>> = None;
            for m in parsed.iter().filter(|p| p.domain == i) {
                for l in parsed.iter().filter(|p| p.domain == j) {
                    let k: BTreeSet<String> = delta(m, l).into_iter().collect();
                    d = Some(match d {
                        None => k,
                        Some(prev) => prev.intersection(&k).cloned().collect(),
                    });
                }
            }
            children.extend(d.unwrap_or_default());
            snodes.push(SNode {
                name: s_name(i, j),
                domains: (i, j),
                targets: children.into_iter().collect(),
            });
        }
    }
    if joint {
        for node in &mut f {
            let (a, b) = (&parsed[node.pair.0], &parsed[node.pair.1]);
            if a.domain == b.domain || (known_children && a.known && b.known) {
                continue;
            }
            let pair = (a.domain.min(b.domain), a.domain.max(b.domain));
            let sn = snodes
                .iter()
                .find(|s| s.domains == pair)
                .expect("an S-node per pair");
            let mut k: BTreeSet<String> = node.targets.iter().cloned().collect();
            k.extend(sn.targets.iter().cloned());
            node.targets = k.into_iter().collect();
        }
    }
    let reserved: Vec<String> = f
        .iter()
        .map(|x| x.name.clone())
        .chain(snodes.iter().map(|x| x.name.clone()))
        .collect();
    if let Some(clash) = reserved.iter().find(|r| names.contains(*r)) {
        return Err(DiscoveryError::Reserved(clash.clone()));
    }
    let mut out = g.clone();
    for node in &f {
        out.add_node(&node.name);
        for k in &node.targets {
            out.directed(&node.name, k);
        }
    }
    for node in &snodes {
        out.add_node(&node.name);
        for k in &node.targets {
            out.directed(&node.name, k);
        }
    }
    Ok((out, f, snodes))
}

/// The augmented selection diagram (Def. 17.2.4) of the causal diagram `g`
/// (over the observed variables; bidirected edges are latent confounders).
/// The selection diagram's S-node edges are `s`, as `(i, j, children)` for
/// domains `i`, `j` (0-based). `setup` lists the distributions.
///
/// One F-node per pair of distributions (Alg. 62) points to their symmetric
/// difference: within a domain by mechanism, across domains the union
/// (Assumption 17.2.3). Pairs of known observational distributions from two
/// domains get none. One S-node per pair of domains points to its children
/// in `s` and to `D_ij`, what every cross-domain pair of targets changes.
/// F-nodes are named `F1`, `F2`, … in pair order, and S-nodes `S12`, `S13`,
/// ….
///
/// ```
/// use why_data::discovery::augment_selection;
/// use why_data::graph::dseparation::Admg;
///
/// // Ex. 17.14: X -> Y -> Z with S12 -> X, regimes ⟨{Z}¹, {Y}²⟩.
/// let mut g = Admg::new();
/// g.directed("X", "Y");
/// g.directed("Y", "Z");
/// let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["X"])];
/// let (_, f, sn) = augment_selection(&g, &s, &[(0, &["Z"], true), (1, &["Y"], true)]).unwrap();
/// assert_eq!(f[0].targets, ["Y", "Z"]);
/// assert_eq!(sn[0].targets, ["X", "Y", "Z"]);
/// ```
///
/// # Errors
///
/// [`DiscoveryError::UnknownVariable`] for a name `g` lacks,
/// [`DiscoveryError::InvalidInput`] when a domain has no distribution, and
/// [`DiscoveryError::Reserved`] when a variable is named like an F- or
/// S-node.
pub fn augment_selection(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    setup: &[Regime<'_>],
) -> Result<(Admg, Vec<FNode>, Vec<SNode>), DiscoveryError> {
    build(g, s, setup, false, false)
}

/// The graph whose PAG [`md_fci`] learns (`known_children = false`) or
/// whose PDAG [`md_pc`] learns (`known_children = true`).
#[cfg(test)]
pub(crate) fn learning_target(
    g: &Admg,
    s: &[(usize, usize, &[&str])],
    setup: &[Regime<'_>],
    known_children: bool,
) -> Result<Admg, DiscoveryError> {
    Ok(build(g, s, setup, true, known_children)?.0)
}

/// Whether `⟨G_S, Ψ⟩` and `⟨G′_S, Ψ′⟩` are MD-Markov equivalent
/// (Thm 17.2.5): the MAGs of their augmented selection diagrams are Markov
/// equivalent. The setups must pair up regime by regime.
///
/// # Errors
///
/// [`DiscoveryError::InvalidInput`] when the setups differ in length, and
/// those of [`augment_selection`].
pub fn md_markov_equivalent(
    g1: &Admg,
    s1: &[(usize, usize, &[&str])],
    setup1: &[Regime<'_>],
    g2: &Admg,
    s2: &[(usize, usize, &[&str])],
    setup2: &[Regime<'_>],
) -> Result<bool, DiscoveryError> {
    if setup1.len() != setup2.len() {
        return Err(DiscoveryError::InvalidInput(
            "the setups differ in length".into(),
        ));
    }
    let (a1, _, _) = augment_selection(g1, s1, setup1)?;
    let (a2, _, _) = augment_selection(g2, s2, setup2)?;
    markov_equivalent(&mag(&a1, &a1.variables())?, &mag(&a2, &a2.variables())?)
}

/// Answers the multi-domain learners' questions from a selection diagram,
/// by the graphical MD-Markov characterization (Prop. 17.2.4) on the
/// augmented selection diagram. Conditional independence is d-separation
/// in `g`. `P_m(y | w) = P_l(y | w)` holds when `y` is separated from the
/// pair's F-node and, across domains, the pair's S-node, given `w` and
/// every other F- and S-node (Eq. 17.7 with the separating sets of
/// Alg. 63). Those nodes are constant across the two distributions.
#[derive(Clone, Debug)]
pub struct SelectionOracle {
    g: Admg,
    augmented: Admg,
    fnodes: Vec<FNode>,
    snodes: Vec<SNode>,
    domains: Vec<usize>,
}

impl SelectionOracle {
    /// The oracle of the selection diagram `g` plus `s` under `setup` (see
    /// [`augment_selection`]).
    ///
    /// # Errors
    ///
    /// Those of [`augment_selection`].
    pub fn new(
        g: Admg,
        s: &[(usize, usize, &[&str])],
        setup: &[Regime<'_>],
    ) -> Result<Self, DiscoveryError> {
        let (augmented, fnodes, snodes) = augment_selection(&g, s, setup)?;
        Ok(Self {
            g,
            augmented,
            fnodes,
            snodes,
            domains: setup.iter().map(|r| r.0).collect(),
        })
    }
}

impl Invariance for SelectionOracle {
    fn distributions(&self) -> usize {
        self.domains.len()
    }

    fn independent(
        &mut self,
        _dist: usize,
        x: &str,
        y: &str,
        z: &[&str],
    ) -> Result<bool, DiscoveryError> {
        Ok(self.g.is_d_separated(&[x], &[y], z)?)
    }

    fn invariant(
        &mut self,
        i: usize,
        j: usize,
        y: &str,
        w: &[&str],
    ) -> Result<bool, DiscoveryError> {
        let pair = (i.min(j), i.max(j));
        let (di, dj) = (
            *self
                .domains
                .get(i)
                .ok_or_else(|| DiscoveryError::InvalidInput(format!("no distribution {i}")))?,
            *self
                .domains
                .get(j)
                .ok_or_else(|| DiscoveryError::InvalidInput(format!("no distribution {j}")))?,
        );
        let f = self.fnodes.iter().find(|f| f.pair == pair);
        let mut set: Vec<&str> = f.iter().map(|f| f.name.as_str()).collect();
        if di != dj {
            let sp = (di.min(dj), di.max(dj));
            let s = self
                .snodes
                .iter()
                .find(|s| s.domains == sp)
                .expect("an S-node per pair");
            set.push(s.name.as_str());
        }
        // Every other augmented node is constant across the two
        // distributions, so it is conditioned on (the separating sets of
        // Alg. 63).
        let mut given: Vec<&str> = w.to_vec();
        given.extend(
            self.fnodes
                .iter()
                .map(|o| o.name.as_str())
                .chain(self.snodes.iter().map(|o| o.name.as_str()))
                .filter(|o| !set.contains(o)),
        );
        Ok(self.augmented.is_d_separated(&set, &[y], &given)?)
    }
}

/// Phase I of MD-PC and MD-FCI (Algorithm 63) over `names ∪ F ∪ S`.
fn md_skeleton(
    names: &[&str],
    setup: &[Regime<'_>],
    known_children: bool,
    test: &mut impl Invariance,
) -> Result<Engine, DiscoveryError> {
    let set: BTreeSet<String> = names.iter().map(|s| (*s).to_string()).collect();
    let parsed = parse(setup, &set)?;
    if parsed.len() != test.distributions() {
        return Err(DiscoveryError::InvalidInput(format!(
            "{} regimes for {} distributions",
            parsed.len(),
            test.distributions()
        )));
    }
    let nd = domains(&parsed)?;
    let f = fnodes(&parsed);
    let mut s_pairs = Vec::new();
    for i in 0..nd {
        for j in i + 1..nd {
            s_pairs.push((i, j));
        }
    }
    let n = names.len();
    let mut all: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
    all.extend(f.iter().map(|x| x.name.clone()));
    all.extend(s_pairs.iter().map(|&(i, j)| s_name(i, j)));
    if let Some(clash) = all[n..].iter().find(|r| set.contains(*r)) {
        return Err(DiscoveryError::Reserved(clash.clone()));
    }
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    let f_end = n + f.len();
    // Each augmented node, with its symmetric difference when both targets
    // are known within one domain (Rule 9').
    let mut marks: Vec<(usize, Option<Vec<usize>>)> = f
        .iter()
        .enumerate()
        .map(|(k, node)| {
            let (a, b) = (&parsed[node.pair.0], &parsed[node.pair.1]);
            let h = (a.domain == b.domain && a.known && b.known).then(|| {
                node.targets
                    .iter()
                    .filter_map(|t| names.iter().position(|v| v == t))
                    .collect()
            });
            (n + k, h)
        })
        .collect();
    marks.extend((f_end..all.len()).map(|i| (i, None)));
    let other_f =
        |skip: Option<usize>| -> Vec<usize> { (n..f_end).filter(|&x| Some(x) != skip).collect() };
    Engine::skeleton(&refs, marks, true, |i, j| {
        if i >= n && j >= n {
            return Ok(Some(Vec::new()));
        }
        if i < n && j < n {
            let others: Vec<usize> = (0..n).filter(|&k| k != i && k != j).collect();
            for w in subsets(&others) {
                let ws: Vec<&str> = w.iter().map(|&k| names[k]).collect();
                if test.independent(0, names[i], names[j], &ws)? {
                    let mut sep = w;
                    sep.extend(n..all.len());
                    return Ok(Some(sep));
                }
            }
            return Ok(None);
        }
        let (a, y) = if i >= n { (i, j) } else { (j, i) };
        // The pairs of distributions whose invariance separates `a` from y.
        let pairs: Vec<(usize, usize, Option<usize>)> = if a < f_end {
            let node = &f[a - n];
            vec![(node.pair.0, node.pair.1, Some(a))]
        } else {
            let (di, dj) = s_pairs[a - f_end];
            let mut v = Vec::new();
            for (m, pm) in parsed.iter().enumerate() {
                for (l, pl) in parsed.iter().enumerate() {
                    if m < l
                        && ((pm.domain, pl.domain) == (di, dj)
                            || (pm.domain, pl.domain) == (dj, di))
                    {
                        let fk = f.iter().position(|x| x.pair == (m, l)).map(|k| n + k);
                        v.push((m, l, fk));
                    }
                }
            }
            v
        };
        let others: Vec<usize> = (0..n).filter(|&k| k != y).collect();
        for (m, l, fk) in pairs {
            for w in subsets(&others) {
                let ws: Vec<&str> = w.iter().map(|&k| names[k]).collect();
                if test.invariant(m, l, names[y], &ws)? {
                    let mut sep = w;
                    sep.extend(other_f(fk));
                    return Ok(Some(sep));
                }
            }
        }
        // Markovian, both targets known: an F-node's children are exactly
        // its symmetric difference, though no test separates it (an S-node
        // shares its invariances). No separating set is known.
        if a < f_end && known_children {
            let node = &f[a - n];
            if parsed[node.pair.0].known
                && parsed[node.pair.1].known
                && !node.targets.iter().any(|t| t == names[y])
            {
                return Ok(Some(vec![UNKNOWN_SEPSET]));
            }
        }
        Ok(None)
    })
}

/// MD-FCI (Algorithm 64): the MD-PAG from distributions over several
/// domains, each with known or unknown targets (`setup`, in the order of
/// `test`'s distributions).
///
/// The augmented nodes are those of [`augment_selection`]: `F1`, `F2`, …
/// per pair of distributions and `S12`, … per pair of domains. Phase I
/// separates variables by conditional independence. It separates an
/// augmented node from `Y` by an invariance `P_m(y | w) = P_l(y | w)`: of
/// the F-node's own pair, or of any pair across the S-node's two domains.
/// After Phases IIa and IIb's FCI rules, Rule 8′ orients edges out of F- and
/// S-nodes. Rule 9′ (R12 of I-FCI) acts only on F-nodes of known targets
/// within one domain.
///
/// **What is learned.** By Eq. 17.7 an invariance across two domains
/// separates the pair's F-node and S-node from `Y` together, and nothing in
/// the data separates one without the other. So the learned graph is the
/// PAG of the augmented selection diagram in which each cross-domain F-node
/// also points to the children of its domains' S-node.
///
/// # Errors
///
/// [`DiscoveryError::InvalidInput`] when `setup` and `test` disagree in
/// length or a domain has no distribution, [`DiscoveryError::Reserved`] for
/// a clashing name, and whatever the tests report.
pub fn md_fci(
    names: &[&str],
    setup: &[Regime<'_>],
    test: &mut impl Invariance,
) -> Result<Pag, DiscoveryError> {
    let mut engine = md_skeleton(names, setup, false, test)?;
    engine.colliders();
    engine.fci_rules();
    Ok(engine.pag)
}

/// MD-PC (Algorithm 61): the MD-PDAG of a Markovian selection diagram. It
/// shares MD-FCI's Phase I, except that an F-node whose two targets are
/// known is adjacent to exactly its symmetric difference. Without latents
/// nothing else can reach it. Then colliders, edges out of F- and S-nodes
/// (Rule 5′) and Meek's rules. Undirected edges are `o-o`.
///
/// ```
/// use why_data::discovery::{SelectionOracle, md_pc};
/// use why_data::graph::dseparation::Admg;
///
/// // Ex. 17.23: X -> Z -> Y, the domains differ at X, regimes
/// // ⟨{}¹, {Y}², {Z}²⟩.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["X"])];
/// let setup: [(usize, &[&str], bool); 3] = [(0, &[], true), (1, &["Y"], true), (1, &["Z"], true)];
/// let mut oracle = SelectionOracle::new(g, &s, &setup).unwrap();
/// let pdag = md_pc(&["X", "Y", "Z"], &setup, &mut oracle).unwrap();
/// assert_eq!(
///     pdag.edges(),
///     ["F1 --> Y", "F2 --> Z", "F3 --> Y", "F3 --> Z", "S12 --> X", "X --> Z", "Y <-- Z"]
/// );
/// ```
///
/// # Errors
///
/// Those of [`md_fci`].
pub fn md_pc(
    names: &[&str],
    setup: &[Regime<'_>],
    test: &mut impl Invariance,
) -> Result<Pag, DiscoveryError> {
    let mut engine = md_skeleton(names, setup, true, test)?;
    engine.colliders();
    engine.pc_rules();
    Ok(engine.pag)
}
