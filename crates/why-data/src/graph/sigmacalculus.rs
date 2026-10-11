//! The three rules of σ-calculus (Correa & Bareinboim, 2020).
//!
//! σ-calculus generalises [do-calculus](super::docalculus) from atomic
//! interventions `do(x)` to *policies* `σ_X`: a regime that replaces the
//! mechanism of `X` with one of our choosing. A policy is one of
//!
//! - **idle** — `X` keeps its natural mechanism; nothing changes;
//! - **atomic** — `X` is set to a constant, as `do(x)` does;
//! - **conditional** — `X` is set by a function of some inputs `pa*`;
//! - **stochastic** — `X` is drawn from a distribution over `pa*`.
//!
//! Graphically conditional and stochastic are the same: `X` loses every
//! arrowhead it had and gains `pa* -> X`. That is the σ-graph `G_{σ_X}`
//! ([`sigma_graph`]), and each rule is a d-separation test on it:
//!
//! ```text
//! R1  P(y | z, w; σ_X)      = P(y | w; σ_X)
//!         if (Y _||_ Z | W) in G_{σ_X}
//! R2  P(y | z, w; σ_X, σ_Z) = P(y | z, w; σ_X, σ_Z')
//!         if (Y _||_ Z | W) in G_{σ_X σ_Z \underline{Z}} and G_{σ_X σ_Z' \underline{Z}}
//! R3  P(y | w; σ_X, σ_Z)    = P(y | w; σ_X, σ_Z')
//!         if (Y _||_ Z | W) in G_{σ_X σ_Z \bar{Z(W)}} and G_{σ_X σ_Z' \bar{Z(W)}}
//! ```
//!
//! where `Z(W)` is the `Z`-variables that are not ancestors of `W` in
//! `G_{σ_X}`. Unlike do-calculus, `X` is *not* conditioned on: a policy may
//! still leave `X` random. Rules 2 and 3 compare two policies on `Z`, so both
//! σ-graphs must pass.
//!
//! The book's drawings also carry a regime node `σ_X -> X` per policy. It is
//! omitted here: a root with a single child is never interior to a path
//! between two other variables, so it cannot change any test below.
//!
//! ```
//! use why_data::graph::dseparation::Admg;
//! use why_data::graph::sigmacalculus::{self, Intervention};
//!
//! // Z -> X -> Y, and A confounds X and Y.
//! let mut g = Admg::new();
//! g.directed("Z", "X");
//! g.directed("X", "Y");
//! g.directed("A", "X");
//! g.directed("A", "Y");
//!
//! // Under a policy that sets X from A alone, Z no longer reaches Y.
//! let r = sigmacalculus::rule_1(&g, &["Y"], &["Z"], &[], &[Intervention::conditional("X", &["A"])])
//!     .unwrap();
//! assert!(r.holds);
//! assert_eq!(r.rewrite, "P(y | z; σ_x) = P(y | σ_x)");
//! ```
//!
//! # Where this differs from the book's Python
//!
//! This is a port of `SigmaCalculusInspector`, with three departures. Each
//! was checked by running the Python on the graphs in this module's tests.
//!
//! - Its `G_{σ_X}` only *spares* existing edges from a policy's inputs. An
//!   input that was not already a parent of `X` never gets `pa* -> X`, so the
//!   Python can call `Y` and `Z` separated when the new edge connects them. An
//!   input sharing a bidirected edge with `X` also keeps it. [`sigma_graph`]
//!   builds the graph the way the book's own
//!   `SigmaCalculusUtils.createGraphSigma` (used by its σ-calculus engine)
//!   does: cut everything into `X`, then add `pa* -> X`.
//! - In rule 3 it spares the inputs of `σ_Z` when overlining `Z(W)`. Here the
//!   overline cuts every edge into `Z(W)`, as it does in do-calculus.
//! - `TransformNodes.__init__` takes mutable default lists, which every
//!   instance shares, so the cuts for `σ_Z` leak into the test for `σ_Z'`.
//!   They also leak into every later query in the same process. Each test here
//!   builds its graph from scratch.
//!
//! The first and last can make the Python answer "applicable" when the rule
//! does not license the rewriting. The first two can make it answer "not
//! applicable" when the rule does license it.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::docalculus::{independence, lower, separated};
use super::dseparation::{Admg, DSepError, Path};

/// How a policy sets its target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Policy<'a> {
    /// The target keeps its natural mechanism.
    Idle,
    /// The target is set to a constant.
    Atomic,
    /// The target is set by a function of these inputs.
    Conditional(Vec<&'a str>),
    /// The target is drawn from a distribution given these inputs.
    Stochastic(Vec<&'a str>),
}

impl Policy<'_> {
    /// The inputs `pa*` the target depends on under the policy.
    #[must_use]
    pub fn inputs(&self) -> &[&str] {
        match self {
            Self::Idle | Self::Atomic => &[],
            Self::Conditional(inputs) | Self::Stochastic(inputs) => inputs,
        }
    }
}

/// A policy `σ_X` applied to one variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention<'a> {
    /// The variable whose mechanism is replaced.
    pub target: &'a str,
    /// What replaces it.
    pub policy: Policy<'a>,
}

impl<'a> Intervention<'a> {
    /// Leaves `target` alone.
    #[must_use]
    pub const fn idle(target: &'a str) -> Self {
        Self { target, policy: Policy::Idle }
    }

    /// Sets `target` to a constant — `do(target)`.
    #[must_use]
    pub const fn atomic(target: &'a str) -> Self {
        Self { target, policy: Policy::Atomic }
    }

    /// Sets `target` by a function of `inputs`.
    #[must_use]
    pub fn conditional(target: &'a str, inputs: &[&'a str]) -> Self {
        Self { target, policy: Policy::Conditional(inputs.to_vec()) }
    }

    /// Draws `target` from a distribution given `inputs`.
    #[must_use]
    pub fn stochastic(target: &'a str, inputs: &[&'a str]) -> Self {
        Self { target, policy: Policy::Stochastic(inputs.to_vec()) }
    }
}

/// What a σ-calculus query can get wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SigmaError {
    /// The graph surgery or the d-separation test failed.
    Graph(DSepError),
    /// A policy in `σ_Z` or `σ_Z'` targets a variable outside `Z`.
    StrayTarget(String),
}

impl fmt::Display for SigmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => e.fmt(f),
            Self::StrayTarget(v) => write!(f, "a policy on Z targets {v}, which is not in Z"),
        }
    }
}

impl std::error::Error for SigmaError {}

impl From<DSepError> for SigmaError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

/// Which rule was tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Rule 1 — inserting or deleting an observation.
    InsertDeleteObservation,
    /// Rule 2 — changing the policy on an observed `Z`.
    ChangeObservedPolicy,
    /// Rule 3 — changing the policy on an unobserved `Z`.
    ChangeUnobservedPolicy,
}

impl Rule {
    /// The rule's number, 1 to 3.
    #[must_use]
    pub fn number(self) -> u8 {
        match self {
            Self::InsertDeleteObservation => 1,
            Self::ChangeObservedPolicy => 2,
            Self::ChangeUnobservedPolicy => 3,
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::InsertDeleteObservation => "insertion/deletion of observations",
            Self::ChangeObservedPolicy => "change of policy with observation",
            Self::ChangeUnobservedPolicy => "change of policy without observation",
        };
        write!(f, "Rule {} ({name})", self.number())
    }
}

/// One d-separation test a rule needs.
#[derive(Clone, Debug)]
pub struct Test {
    /// How the graph is written, e.g. `G_{σ_x σ_z \underline{z}}`.
    pub graph_label: String,
    /// The graph the test ran on.
    pub graph: Admg,
    /// Whether the independence holds in it.
    pub holds: bool,
}

/// The outcome of testing a rule.
#[derive(Clone, Debug)]
pub struct Applicability {
    /// Which rule was tested.
    pub rule: Rule,
    /// Whether the rule applies — whether every one of [`Self::tests`] holds.
    pub holds: bool,
    /// The independence that was tested, e.g. `(y _||_ z | w)`.
    pub independence: String,
    /// One test for rule 1; one per policy on `Z`, `σ_Z` then `σ_Z'`, for rules
    /// 2 and 3.
    pub tests: Vec<Test>,
    /// The rewriting the rule licenses, when it holds.
    pub rewrite: String,
    /// When the rule fails, the open paths between `Y` and `Z` that defeat it,
    /// across all failing tests. Enumerated with [`Admg::paths`], so
    /// exponential in the worst case, but only computed on failure.
    pub violating_paths: Vec<Path>,
}

impl fmt::Display for Applicability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.rule)?;
        for t in &self.tests {
            let verdict = if t.holds { "holds" } else { "fails" };
            writeln!(f, "  test:  {} in {}: {verdict}", self.independence, t.graph_label)?;
        }
        for p in &self.violating_paths {
            writeln!(f, "  open:  {p}")?;
        }
        if self.holds {
            writeln!(f, "  holds: YES")?;
            write!(f, "  gives: {}", self.rewrite)
        } else {
            writeln!(f, "  holds: NO")?;
            write!(f, "  the rewriting {} is not licensed", self.rewrite)
        }
    }
}

/// `G_{σ_X}`: `G` with each policy in `sigma` applied, in order.
///
/// An idle policy changes nothing. Every other one deletes all arrowheads into
/// its target — bidirected edges too, since the new mechanism has fresh noise —
/// and a conditional or stochastic one then adds `input -> target` for each of
/// its inputs.
///
/// # Errors
///
/// [`DSepError::Unknown`] for a name the graph does not have, and
/// [`DSepError::Cyclic`] if the added inputs close a directed cycle.
pub fn sigma_graph(g: &Admg, sigma: &[Intervention<'_>]) -> Result<Admg, DSepError> {
    let mut out = g.clone();
    let mut rewired = false;
    for intv in sigma {
        if intv.policy == Policy::Idle {
            continue;
        }
        out = out.cut_edges_into(&[intv.target])?;
        for input in intv.policy.inputs() {
            if out.node(input).is_none() {
                return Err(DSepError::Unknown((*input).to_string()));
            }
            out.directed(input, intv.target);
            rewired = true;
        }
    }
    if rewired {
        out.topological_order()?;
    }
    Ok(out)
}

/// `σ_x, σ_z` for an expression, or `σ_x σ_z` for a graph subscript.
fn regimes(sigma: &[&[Intervention<'_>]], prime: &[bool], sep: &str) -> String {
    sigma
        .iter()
        .zip(prime)
        .flat_map(|(set, &p)| {
            set.iter().map(move |i| {
                let mark = if p { "'" } else { "" };
                format!("σ_{}{mark}", i.target.to_lowercase())
            })
        })
        .collect::<Vec<_>>()
        .join(sep)
}

/// `P(y | z, w; σ_x)`, dropping whichever side of the bar is empty.
fn probability(y: &[&str], observed: &[&[&str]], regimes: &str) -> String {
    let seen: Vec<&str> = observed.iter().flat_map(|o| o.iter().copied()).collect();
    match (seen.is_empty(), regimes.is_empty()) {
        (true, true) => format!("P({})", lower(y)),
        (true, false) => format!("P({} | {regimes})", lower(y)),
        (false, true) => format!("P({} | {})", lower(y), lower(&seen)),
        (false, false) => format!("P({} | {}; {regimes})", lower(y), lower(&seen)),
    }
}

/// `G_{σ_x σ_z \underline{z}}`, or `G` when there is nothing to write.
fn graph_label(parts: &[String]) -> String {
    let parts: Vec<&str> = parts.iter().map(String::as_str).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        "G".to_string()
    } else {
        format!("G_{{{}}}", parts.join(" "))
    }
}

fn decorate(kind: &str, vs: &[&str]) -> String {
    if vs.is_empty() {
        String::new()
    } else {
        format!("\\{kind}{{{}}}", lower(vs))
    }
}

/// Every open path between a `Y` and a `Z` in `g`, skipping any already seen.
fn open_paths(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    w: &[&str],
    seen: &mut HashSet<String>,
    out: &mut Vec<Path>,
) -> Result<(), DSepError> {
    for a in y {
        for b in z {
            for p in g.paths(a, b, w)? {
                if p.is_open() && seen.insert(p.to_string()) {
                    out.push(p);
                }
            }
        }
    }
    Ok(())
}

/// Runs `tests`, then gathers violating paths if any failed.
fn conclude(
    rule: Rule,
    tests: Vec<(String, Admg)>,
    (y, z, w): (&[&str], &[&str], &[&str]),
    rewrite: String,
) -> Result<Applicability, SigmaError> {
    let mut run = Vec::with_capacity(tests.len());
    for (graph_label, graph) in tests {
        let holds = separated(&graph, y, z, w)?;
        run.push(Test { graph_label, graph, holds });
    }
    let holds = run.iter().all(|t| t.holds);
    let mut violating_paths = Vec::new();
    let mut seen = HashSet::new();
    for t in run.iter().filter(|t| !t.holds) {
        open_paths(&t.graph, y, z, w, &mut seen, &mut violating_paths)?;
    }
    Ok(Applicability {
        rule,
        holds,
        independence: independence(y, z, &[], w),
        tests: run,
        rewrite,
        violating_paths,
    })
}

/// Checks that every policy in `σ_Z` and `σ_Z'` targets a member of `Z`.
fn check_targets(z: &[&str], sets: &[&[Intervention<'_>]]) -> Result<(), SigmaError> {
    let z: HashSet<&str> = z.iter().copied().collect();
    for set in sets {
        if let Some(i) = set.iter().find(|i| !z.contains(i.target)) {
            return Err(SigmaError::StrayTarget(i.target.to_string()));
        }
    }
    Ok(())
}

/// Rule 1 — insertion and deletion of observations.
///
/// ```text
/// P(y | z, w; σ_X) = P(y | w; σ_X)   if (Y _||_ Z | W) in G_{σ_X}
/// ```
///
/// With no policies this is plain conditional independence in `G`.
///
/// # Errors
///
/// [`SigmaError::Graph`] for unknown names, sets that are not pairwise
/// disjoint, or policies that make the graph cyclic.
pub fn rule_1(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    w: &[&str],
    sigma_x: &[Intervention<'_>],
) -> Result<Applicability, SigmaError> {
    let sx = regimes(&[sigma_x], &[false], ", ");
    let rewrite = format!("{} = {}", probability(y, &[z, w], &sx), probability(y, &[w], &sx));
    let label = graph_label(&[regimes(&[sigma_x], &[false], " ")]);
    conclude(
        Rule::InsertDeleteObservation,
        vec![(label, sigma_graph(g, sigma_x)?)],
        (y, z, w),
        rewrite,
    )
}

/// Rule 2 — changing the policy on `Z` while `Z` is observed.
///
/// ```text
/// P(y | z, w; σ_X, σ_Z) = P(y | z, w; σ_X, σ_Z')
///     if (Y _||_ Z | W) in G_{σ_X σ_Z \underline{Z}} and G_{σ_X σ_Z' \underline{Z}}
/// ```
///
/// Taking `σ_Z` atomic and `σ_Z'` idle recovers do-calculus rule 2: the action
/// on `Z` becomes an observation.
///
/// # Errors
///
/// As [`rule_1`], plus [`SigmaError::StrayTarget`] if a policy in `sigma_z`
/// or `sigma_z_prime` targets a variable outside `z`.
pub fn rule_2(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    w: &[&str],
    sigma_x: &[Intervention<'_>],
    sigma_z: &[Intervention<'_>],
    sigma_z_prime: &[Intervention<'_>],
) -> Result<Applicability, SigmaError> {
    check_targets(z, &[sigma_z, sigma_z_prime])?;
    let mut tests = Vec::with_capacity(2);
    for (sz, prime) in [(sigma_z, false), (sigma_z_prime, true)] {
        let both: Vec<Intervention<'_>> = sigma_x.iter().chain(sz).cloned().collect();
        let graph = sigma_graph(g, &both)?.cut_edges_out_of(z)?;
        let label = graph_label(&[
            regimes(&[sigma_x, &z_regime(z, sz)], &[false, prime], " "),
            decorate("underline", z),
        ]);
        tests.push((label, graph));
    }
    let rewrite = format!(
        "{} = {}",
        probability(y, &[z, w], &regimes(&[sigma_x, &z_regime(z, sigma_z)], &[false, false], ", ")),
        probability(y, &[z, w], &regimes(&[sigma_x, &z_regime(z, sigma_z_prime)], &[false, true], ", ")),
    );
    conclude(Rule::ChangeObservedPolicy, tests, (y, z, w), rewrite)
}

/// Rule 3 — changing the policy on `Z` while `Z` is not observed.
///
/// ```text
/// P(y | w; σ_X, σ_Z) = P(y | w; σ_X, σ_Z')
///     if (Y _||_ Z | W) in G_{σ_X σ_Z \bar{Z(W)}} and G_{σ_X σ_Z' \bar{Z(W)}}
/// ```
///
/// `Z(W)` is the `Z`-variables that are not ancestors of `W` in `G_{σ_X}`.
/// Taking `σ_Z` atomic and `σ_Z'` idle recovers do-calculus rule 3: the action
/// on `Z` can be dropped.
///
/// # Errors
///
/// As [`rule_2`].
pub fn rule_3(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    w: &[&str],
    sigma_x: &[Intervention<'_>],
    sigma_z: &[Intervention<'_>],
    sigma_z_prime: &[Intervention<'_>],
) -> Result<Applicability, SigmaError> {
    check_targets(z, &[sigma_z, sigma_z_prime])?;
    let g_sigma_x = sigma_graph(g, sigma_x)?;
    let ancestors_of_w: HashSet<&str> = if w.is_empty() {
        HashSet::new()
    } else {
        g_sigma_x.ancestors_of(w)?.into_iter().collect()
    };
    let z_of_w: Vec<&str> = z
        .iter()
        .copied()
        .filter(|v| !ancestors_of_w.contains(v))
        .collect();

    let mut tests = Vec::with_capacity(2);
    for (sz, prime) in [(sigma_z, false), (sigma_z_prime, true)] {
        let both: Vec<Intervention<'_>> = sigma_x.iter().chain(sz).cloned().collect();
        let graph = sigma_graph(g, &both)?.cut_edges_into(&z_of_w)?;
        let label = graph_label(&[
            regimes(&[sigma_x, &z_regime(z, sz)], &[false, prime], " "),
            decorate("bar", &z_of_w),
        ]);
        tests.push((label, graph));
    }
    let rewrite = format!(
        "{} = {}",
        probability(y, &[w], &regimes(&[sigma_x, &z_regime(z, sigma_z)], &[false, false], ", ")),
        probability(y, &[w], &regimes(&[sigma_x, &z_regime(z, sigma_z_prime)], &[false, true], ", ")),
    );
    conclude(Rule::ChangeUnobservedPolicy, tests, (y, z, w), rewrite)
}

/// The policies on `Z` as written: one per member of `Z`, taken from `sz` when
/// it names one and idle otherwise. A regime on `Z` is a regime on all of it,
/// so `σ_z` is printed even for a variable `sz` leaves alone, as the book does.
fn z_regime<'a>(z: &[&'a str], sz: &[Intervention<'a>]) -> Vec<Intervention<'a>> {
    let given: HashMap<&str, &Intervention<'a>> = sz.iter().map(|i| (i.target, i)).collect();
    z.iter()
        .map(|v| given.get(v).map_or_else(|| Intervention::idle(v), |i| (*i).clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::docalculus;

    /// Z -> X -> Y with A -> X and A -> Y.
    fn confounded_chain() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.directed("A", "X");
        g.directed("A", "Y");
        g
    }

    #[test]
    fn sigma_graph_rewires_the_target() {
        let mut g = Admg::new();
        g.directed("B", "X");
        g.bidirected("A", "X");
        g.directed("X", "Y");
        let s = sigma_graph(&g, &[Intervention::conditional("X", &["A"])]).unwrap();
        // B -> X and A <-> X are gone, A -> X is new, X -> Y survives.
        assert!(s.is_d_separated(&["B"], &["X"], &[]).unwrap());
        assert!(!s.is_d_separated(&["A"], &["X"], &[]).unwrap());
        assert_eq!(s.descendants_of(&["A"]).unwrap(), vec!["A", "X", "Y"]);
        assert_eq!(s.variables().len(), g.variables().len());
    }

    #[test]
    fn idle_and_atomic_policies() {
        let g = confounded_chain();
        let idle = sigma_graph(&g, &[Intervention::idle("X")]).unwrap();
        assert_eq!(idle.graph().edge_count(), g.graph().edge_count());
        let atomic = sigma_graph(&g, &[Intervention::atomic("X")]).unwrap();
        assert_eq!(atomic.graph().edge_count(), 2, "only X -> Y and A -> Y remain");
    }

    #[test]
    fn a_policy_closing_a_cycle_is_rejected() {
        let g = confounded_chain();
        let err = sigma_graph(&g, &[Intervention::stochastic("X", &["Y"])]).unwrap_err();
        assert!(matches!(err, DSepError::Cyclic(_)));
        let err = sigma_graph(&g, &[Intervention::stochastic("X", &["nope"])]).unwrap_err();
        assert_eq!(err, DSepError::Unknown("nope".into()));
    }

    #[test]
    fn rule_1_matches_the_book() {
        // Python: applicable True, P(Y | Z; σ_X) = P(Y | σ_X) in G_{σ_X}.
        let r = rule_1(&confounded_chain(), &["Y"], &["Z"], &[], &[Intervention::conditional("X", &["A"])])
            .unwrap();
        assert!(r.holds);
        assert_eq!(r.tests[0].graph_label, "G_{σ_x}");
        assert_eq!(r.independence, "(y _||_ z)");
        assert_eq!(r.rewrite, "P(y | z; σ_x) = P(y | σ_x)");
        assert!(r.violating_paths.is_empty());
    }

    #[test]
    fn rule_1_without_policies_reports_the_open_path() {
        let r = rule_1(&confounded_chain(), &["Y"], &["Z"], &[], &[]).unwrap();
        assert!(!r.holds);
        assert_eq!(r.tests[0].graph_label, "G");
        assert_eq!(r.rewrite, "P(y | z) = P(y)");
        let paths: Vec<String> = r.violating_paths.iter().map(ToString::to_string).collect();
        assert_eq!(paths, ["Y <- X <- Z"]);
    }

    #[test]
    fn rule_1_uses_inputs_that_were_not_parents() {
        // Z -> A, X -> Y, and a policy setting X from A. In G_{σ_X} the new
        // A -> X opens Z -> A -> X -> Y. The book's inspector never adds A -> X
        // and so answers True; its own createGraphSigma would add it.
        let mut g = Admg::new();
        g.directed("Z", "A");
        g.directed("X", "Y");
        let r = rule_1(&g, &["Y"], &["Z"], &[], &[Intervention::conditional("X", &["A"])]).unwrap();
        assert!(!r.holds);
        let paths: Vec<String> = r.violating_paths.iter().map(ToString::to_string).collect();
        assert_eq!(paths, ["Y <- X <- A <- Z"]);
    }

    #[test]
    fn rule_2_with_two_policies() {
        // Z -> Y, Z <- A -> Y. Underlining Z removes Z -> Y; what is left is
        // the back door through A, which σ_Z cuts (Z set from nothing) but σ_Z'
        // keeps (Z set from A). Only the first test passes.
        let mut g = Admg::new();
        g.directed("Z", "Y");
        g.directed("A", "Z");
        g.directed("A", "Y");
        let r = rule_2(
            &g,
            &["Y"],
            &["Z"],
            &[],
            &[],
            &[Intervention::atomic("Z")],
            &[Intervention::conditional("Z", &["A"])],
        )
        .unwrap();
        assert_eq!(r.rule.number(), 2);
        assert!(r.tests[0].holds);
        assert!(!r.tests[1].holds);
        assert!(!r.holds);
        assert_eq!(r.tests[0].graph_label, "G_{σ_z \\underline{z}}");
        assert_eq!(r.tests[1].graph_label, "G_{σ_z' \\underline{z}}");
        assert_eq!(r.rewrite, "P(y | z; σ_z) = P(y | z; σ_z')");
        let paths: Vec<String> = r.violating_paths.iter().map(ToString::to_string).collect();
        assert_eq!(paths, ["Y <- A -> Z"]);

        // Conditioning on A closes that back door in both.
        let r = rule_2(
            &g,
            &["Y"],
            &["Z"],
            &["A"],
            &[],
            &[Intervention::atomic("Z")],
            &[Intervention::conditional("Z", &["A"])],
        )
        .unwrap();
        assert!(r.holds);
    }

    #[test]
    fn rule_3_cuts_the_inputs_of_the_z_policy() {
        // A -> Y, A -> Z and nothing from Z to Y, so how Z is set cannot
        // affect Y. The book's inspector spares A -> Z while overlining Z, sees
        // Z <- A -> Y, and answers False.
        let mut g = Admg::new();
        g.directed("A", "Y");
        g.directed("A", "Z");
        let r = rule_3(
            &g,
            &["Y"],
            &["Z"],
            &[],
            &[],
            &[Intervention::conditional("Z", &["A"])],
            &[Intervention::atomic("Z")],
        )
        .unwrap();
        assert!(r.holds);
        assert_eq!(r.tests[0].graph_label, "G_{σ_z \\bar{z}}");
        assert_eq!(r.rewrite, "P(y | σ_z) = P(y | σ_z')");
    }

    #[test]
    fn rule_3_keeps_ancestors_of_w() {
        // Z -> W, Z <-> Y. Conditioning on W, a descendant of Z, keeps Z's
        // arrowheads, and Z <-> Y defeats the rule. The book's inspector
        // answers True here: the do(z) cut leaks into the idle σ_Z' graph
        // through TransformNodes' shared default lists.
        let mut g = Admg::new();
        g.directed("Z", "W");
        g.bidirected("Z", "Y");
        let sz = [Intervention::atomic("Z")];
        let sz_prime = [Intervention::idle("Z")];
        let without = rule_3(&g, &["Y"], &["Z"], &[], &[], &sz, &sz_prime).unwrap();
        assert!(without.holds);
        let with = rule_3(&g, &["Y"], &["Z"], &["W"], &[], &sz, &sz_prime).unwrap();
        assert!(!with.holds);
        assert_eq!(with.tests[0].graph_label, "G_{σ_z}");
        assert_eq!(with.rewrite, "P(y | w; σ_z) = P(y | w; σ_z')");
    }

    #[test]
    fn atomic_versus_idle_reduces_to_do_calculus() {
        // do-calculus rules 2 and 3 are σ-calculus with σ_Z atomic and σ_Z'
        // idle, σ_X atomic and X added to the conditioning set. Check that
        // the verdicts agree on the docalculus module's graphs.
        let mut g1 = Admg::new();
        g1.directed("Z", "X");
        g1.directed("X", "Y");
        g1.directed("Z", "Y");
        let mut g3 = Admg::new();
        g3.directed("Z", "X");
        g3.directed("X", "Y");
        g3.bidirected("Z", "Y");
        g3.bidirected("X", "Y");
        let sz = [Intervention::atomic("Z")];
        let idle = [Intervention::idle("Z")];
        for g in [g1, g3] {
            for (x, w) in [(&[][..], &["X"][..]), (&["X"][..], &[][..])] {
                let sx: Vec<Intervention<'_>> = x.iter().map(|v| Intervention::atomic(v)).collect();
                let mut xw: Vec<&str> = x.to_vec();
                xw.extend(w);
                // σ-calculus's W is do-calculus's X ∪ W: an atomic policy is a
                // constant, so conditioning on it changes nothing.
                let d2 = docalculus::rule_2(&g, &["Y"], &["Z"], x, w).unwrap().holds;
                let s2 = rule_2(&g, &["Y"], &["Z"], &xw, &sx, &sz, &idle).unwrap().holds;
                assert_eq!(d2, s2, "rule 2, do({x:?}), w = {w:?}");
                let d3 = docalculus::rule_3(&g, &["Y"], &["Z"], x, w).unwrap().holds;
                let s3 = rule_3(&g, &["Y"], &["Z"], &xw, &sx, &sz, &idle).unwrap().holds;
                assert_eq!(d3, s3, "rule 3, do({x:?}), w = {w:?}");
            }
        }
    }

    #[test]
    fn stray_targets_are_reported() {
        let g = confounded_chain();
        let err = rule_2(&g, &["Y"], &["Z"], &[], &[], &[Intervention::atomic("A")], &[]).unwrap_err();
        assert_eq!(err, SigmaError::StrayTarget("A".into()));
    }

    #[test]
    fn unknown_variables_are_reported() {
        let err = rule_1(&confounded_chain(), &["Y"], &["nope"], &[], &[]).unwrap_err();
        assert!(matches!(err, SigmaError::Graph(DSepError::Unknown(_))));
    }
}
