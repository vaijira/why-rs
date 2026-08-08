//! The three rules of do-calculus.
//!
//! Theorem 4.3.1. Each rule licenses rewriting an interventional expression,
//! and each is a d-separation test on a mutilated graph — so all three reduce
//! to [`Admg::is_d_separated`] applied to the right surgery.
//!
//! `G` with a set overlined means every arrowhead into it is deleted
//! ([`Admg::cut_edges_into`]); underlined means every edge leaving it is
//! deleted ([`Admg::cut_edges_out_of`]).
//!
//! ```
//! use why_data::graph::docalculus::{self, Rule};
//! use why_data::graph::dseparation::Admg;
//!
//! // Z -> X -> Y, with X and Y confounded.
//! let mut g = Admg::new();
//! g.directed("Z", "X");
//! g.directed("X", "Y");
//! g.bidirected("X", "Y");
//!
//! // Rule 1: is Z ignorable when predicting Y under do(x)?
//! let applies = docalculus::rule_1(&g, &["Y"], &["Z"], &["X"], &[]).unwrap();
//! assert!(applies.holds);
//! assert_eq!(applies.rule, Rule::InsertDeleteObservation);
//! ```

use std::collections::HashSet;
use std::fmt;

use super::dseparation::{Admg, DSepError};

/// Which rule was tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Rule 1 — inserting or deleting an observation.
    InsertDeleteObservation,
    /// Rule 2 — exchanging an action for an observation.
    ActionObservationExchange,
    /// Rule 3 — inserting or deleting an action.
    InsertDeleteAction,
}

impl Rule {
    /// The rule's number, 1 to 3.
    #[must_use]
    pub fn number(self) -> u8 {
        match self {
            Self::InsertDeleteObservation => 1,
            Self::ActionObservationExchange => 2,
            Self::InsertDeleteAction => 3,
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::InsertDeleteObservation => "insertion/deletion of observations",
            Self::ActionObservationExchange => "action/observation exchange",
            Self::InsertDeleteAction => "insertion/deletion of actions",
        };
        write!(f, "Rule {} ({name})", self.number())
    }
}

/// The outcome of testing a rule.
#[derive(Clone, Debug)]
pub struct Applicability {
    /// Which rule was tested.
    pub rule: Rule,
    /// Whether the rule applies — that is, whether the independence holds in
    /// the mutilated graph.
    pub holds: bool,
    /// How the mutilated graph is written, e.g. `G_{\bar{X}\underline{Z}}`.
    pub graph_label: String,
    /// The independence that was tested, e.g. `(Y ⊥ Z | X, W)`.
    pub independence: String,
    /// The graph the test ran on. The notebook plots this; here it is returned
    /// so callers can inspect or render it.
    pub graph: Admg,
    /// The rewriting the rule licenses, when it holds.
    pub rewrite: String,
}

impl fmt::Display for Applicability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.rule)?;
        writeln!(f, "  test:  {} in {}", self.independence, self.graph_label)?;
        if self.holds {
            writeln!(f, "  holds: YES")?;
            write!(f, "  gives: {}", self.rewrite)
        } else {
            writeln!(f, "  holds: NO")?;
            write!(f, "  the rewriting {} is not licensed", self.rewrite)
        }
    }
}

/// `\bar{x}` / `\underline{x}`, or nothing at all when the set is empty.
fn decorate(kind: &str, vs: &[&str]) -> String {
    if vs.is_empty() {
        String::new()
    } else {
        format!("\\{kind}{{{}}}", lower(vs))
    }
}

/// The subscript of a mutilated graph, `G` itself when nothing was cut.
fn graph_label(parts: &[String]) -> String {
    let joined: String = parts.concat();
    if joined.is_empty() {
        "G".to_string()
    } else {
        format!("G_{{{joined}}}")
    }
}

fn lower(vs: &[&str]) -> String {
    vs.iter()
        .map(|v| v.to_lowercase())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `P(y | do(x), w)`, or `P(y | do(x), z, w)` when `z` is given.
fn probability(y: &[&str], do_x: &[&str], observed: &[&[&str]]) -> String {
    let mut parts = Vec::new();
    if !do_x.is_empty() {
        parts.push(format!("do({})", lower(do_x)));
    }
    for o in observed {
        if !o.is_empty() {
            parts.push(lower(o));
        }
    }
    if parts.is_empty() {
        format!("P({})", lower(y))
    } else {
        format!("P({} | {})", lower(y), parts.join(", "))
    }
}

fn independence(y: &[&str], z: &[&str], x: &[&str], w: &[&str]) -> String {
    let mut given: Vec<&str> = x.to_vec();
    given.extend(w.iter().copied());
    if given.is_empty() {
        format!("({} _||_ {})", lower(y), lower(z))
    } else {
        format!("({} _||_ {} | {})", lower(y), lower(z), lower(&given))
    }
}

/// A d-separation test that treats an empty `Y` or `Z` as vacuously separated,
/// which is what the rules mean when a set is absent.
fn separated(g: &Admg, y: &[&str], z: &[&str], given: &[&str]) -> Result<bool, DSepError> {
    if y.is_empty() || z.is_empty() {
        return Ok(true);
    }
    g.is_d_separated(y, z, given)
}

/// Rule 1 — insertion and deletion of observations.
///
/// ```text
/// P(y | do(x), z, w) = P(y | do(x), w)   if (Y _||_ Z | X, W) in G_xbar
/// ```
///
/// # Errors
///
/// [`DSepError`] for unknown names, or sets that are not pairwise disjoint.
pub fn rule_1(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    x: &[&str],
    w: &[&str],
) -> Result<Applicability, DSepError> {
    let mutilated = g.cut_edges_into(x)?;
    let mut given: Vec<&str> = x.to_vec();
    given.extend(w.iter().copied());
    let holds = separated(&mutilated, y, z, &given)?;
    Ok(Applicability {
        rule: Rule::InsertDeleteObservation,
        holds,
        graph_label: graph_label(&[decorate("bar", x)]),
        independence: independence(y, z, x, w),
        graph: mutilated,
        rewrite: format!(
            "{} = {}",
            probability(y, x, &[z, w]),
            probability(y, x, &[w])
        ),
    })
}

/// Rule 2 — exchanging an action for an observation.
///
/// ```text
/// P(y | do(x), do(z), w) = P(y | do(x), z, w)
///     if (Y _||_ Z | X, W) in G_xbar_zunderline
/// ```
///
/// # Errors
///
/// As [`rule_1`].
pub fn rule_2(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    x: &[&str],
    w: &[&str],
) -> Result<Applicability, DSepError> {
    let mutilated = g.cut_edges_into(x)?.cut_edges_out_of(z)?;
    let mut given: Vec<&str> = x.to_vec();
    given.extend(w.iter().copied());
    let holds = separated(&mutilated, y, z, &given)?;

    let mut both: Vec<&str> = x.to_vec();
    both.extend(z.iter().copied());
    Ok(Applicability {
        rule: Rule::ActionObservationExchange,
        holds,
        graph_label: graph_label(&[decorate("bar", x), decorate("underline", z)]),
        independence: independence(y, z, x, w),
        graph: mutilated,
        rewrite: format!(
            "{} = {}",
            probability(y, &both, &[w]),
            probability(y, x, &[z, w])
        ),
    })
}

/// Rule 3 — insertion and deletion of actions.
///
/// ```text
/// P(y | do(x), do(z), w) = P(y | do(x), w)
///     if (Y _||_ Z | X, W) in G_xbar_z(w)bar
/// ```
///
/// `Z(W)` is the set of `Z`-variables that are *not* ancestors of `W` in
/// `G_xbar`. Only those have their incoming edges cut, which is what makes this
/// rule weaker than cutting all of `Z`.
///
/// # Errors
///
/// As [`rule_1`].
pub fn rule_3(
    g: &Admg,
    y: &[&str],
    z: &[&str],
    x: &[&str],
    w: &[&str],
) -> Result<Applicability, DSepError> {
    let cut_x = g.cut_edges_into(x)?;

    // Z(W): the Z-variables that are not ancestors of W in G_xbar.
    let ancestors_of_w: HashSet<&str> = if w.is_empty() {
        HashSet::new()
    } else {
        cut_x.ancestors_of(w)?.into_iter().collect()
    };
    let z_of_w: Vec<&str> = z
        .iter()
        .copied()
        .filter(|v| !ancestors_of_w.contains(v))
        .collect();

    let mutilated = cut_x.cut_edges_into(&z_of_w)?;
    let mut given: Vec<&str> = x.to_vec();
    given.extend(w.iter().copied());
    let holds = separated(&mutilated, y, z, &given)?;

    let mut both: Vec<&str> = x.to_vec();
    both.extend(z.iter().copied());
    Ok(Applicability {
        rule: Rule::InsertDeleteAction,
        holds,
        graph_label: graph_label(&[decorate("bar", x), decorate("bar", &z_of_w)]),
        independence: independence(y, z, x, w),
        graph: mutilated,
        rewrite: format!(
            "{} = {}",
            probability(y, &both, &[w]),
            probability(y, x, &[w])
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The notebook's G1 for rule 1: Z -> X -> Y, X <-> Y.
    fn g1() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.bidirected("X", "Y");
        g
    }

    /// The notebook's G2 for rule 2: Z -> X, Z -> Y, Y -> X.
    fn g2() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("Z", "Y");
        g.directed("Y", "X");
        g
    }

    /// The notebook's G3 for rule 3: Z -> X -> Y, Z <-> Y, X <-> Y.
    fn g3() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.bidirected("Z", "Y");
        g.bidirected("X", "Y");
        g
    }

    #[test]
    fn cutting_edges_into_removes_bidirected_ones_too() {
        let g = g1();
        let cut = g.cut_edges_into(&["X"]).unwrap();
        // Z -> X and X <-> Y are gone; X -> Y survives.
        assert!(cut.is_d_separated(&["Z"], &["X"], &[]).unwrap());
        assert!(!cut.is_d_separated(&["X"], &["Y"], &[]).unwrap());
        // Every variable is preserved by the surgery.
        assert_eq!(cut.variables().len(), g.variables().len());
    }

    #[test]
    fn cutting_edges_out_of_keeps_bidirected_ones() {
        let g = g1();
        let cut = g.cut_edges_out_of(&["X"]).unwrap();
        // X -> Y is gone but X <-> Y remains, so they are still d-connected.
        assert!(!cut.is_d_separated(&["X"], &["Y"], &[]).unwrap());
        // Z -> X survives.
        assert!(!cut.is_d_separated(&["Z"], &["X"], &[]).unwrap());
    }

    #[test]
    fn rule_1_on_g1() {
        // In G_xbar, Z is disconnected from everything, so observing it tells us
        // nothing about Y once we intervene on X.
        let r = rule_1(&g1(), &["Y"], &["Z"], &["X"], &[]).unwrap();
        assert!(r.holds);
        assert_eq!(r.rule.number(), 1);
        assert_eq!(r.rewrite, "P(y | do(x), z) = P(y | do(x))");
    }

    #[test]
    fn rule_1_fails_without_the_intervention() {
        // With no do(x), Z -> X -> Y leaves Z informative about Y.
        let r = rule_1(&g1(), &["Y"], &["Z"], &[], &[]).unwrap();
        assert!(!r.holds);
    }

    #[test]
    fn rule_2_on_g2_does_not_apply() {
        // The notebook's query: can observing X stand in for do(x), given Z?
        // It cannot — X has no outgoing edges, so underlining it changes
        // nothing, and Y -> X leaves the two adjacent and hence inseparable.
        let r = rule_2(&g2(), &["Y"], &["X"], &[], &["Z"]).unwrap();
        assert!(!r.holds);
        assert_eq!(r.rewrite, "P(y | do(x), z) = P(y | x, z)");
    }

    #[test]
    fn rule_2_applies_in_the_back_door_graph() {
        // Z -> X -> Y with Z -> Y. Underlining X deletes X -> Y, and Z blocks
        // the remaining X <- Z -> Y, so the action becomes an observation.
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.directed("Z", "Y");
        let r = rule_2(&g, &["Y"], &["X"], &[], &["Z"]).unwrap();
        assert!(r.holds);
        assert_eq!(r.rewrite, "P(y | do(x), z) = P(y | x, z)");
    }

    #[test]
    fn rule_3_on_g3_drops_the_action() {
        // In G_{Xbar Zbar}: cutting into X removes Z -> X and X <-> Y, then
        // cutting into Z removes Z <-> Y, leaving Z isolated. So intervening on
        // Z adds nothing once X is intervened on — Z reached Y only through
        // Z -> X -> Y.
        let r = rule_3(&g3(), &["Y"], &["Z"], &["X"], &[]).unwrap();
        assert_eq!(r.rule.number(), 3);
        assert!(r.holds);
        assert_eq!(r.rewrite, "P(y | do(x, z)) = P(y | do(x))");
    }

    #[test]
    fn rule_3_only_cuts_the_non_ancestors_of_w() {
        // Z confounds with Y only through a latent cause, and Z -> W.
        let mut g = Admg::new();
        g.directed("Z", "W");
        g.bidirected("Z", "Y");

        // No W: Z(W) is all of Z, its arrowheads are cut, and Y becomes
        // independent of it.
        let cut = rule_3(&g, &["Y"], &["Z"], &[], &[]).unwrap();
        assert!(cut.holds);

        // Conditioning on W: Z is an ancestor of W, so Z(W) is empty, nothing
        // is cut, and Z <-> Y keeps them dependent.
        let kept = rule_3(&g, &["Y"], &["Z"], &[], &["W"]).unwrap();
        assert_eq!(kept.graph_label, "G", "nothing is cut, so the graph is G itself");
        assert!(!kept.holds);
    }

    #[test]
    fn an_empty_set_is_vacuously_separated() {
        let r = rule_1(&g1(), &["Y"], &[], &["X"], &[]).unwrap();
        assert!(r.holds);
    }

    #[test]
    fn unknown_variables_are_reported() {
        assert!(rule_1(&g1(), &["Y"], &["nope"], &["X"], &[]).is_err());
    }
}
