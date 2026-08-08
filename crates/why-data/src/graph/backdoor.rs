//! The back-door criterion and back-door adjustment.
//!
//! Chapter 4 of the Causal AI book. Given a causal diagram and a treatment /
//! outcome pair, this answers two questions: *is this set of covariates enough
//! to identify the causal effect by adjustment*, and *which sets are*.
//!
//! ```
//! use why_data::graph::backdoor;
//! use why_data::graph::dseparation::Admg;
//!
//! // Z confounds X and Y; adjusting for it identifies the effect.
//! let mut g = Admg::new();
//! g.directed("Z", "X");
//! g.directed("Z", "Y");
//! g.directed("X", "Y");
//!
//! assert!(!backdoor::satisfies(&g, &["X"], &["Y"], &[]).unwrap());
//! assert!(backdoor::satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap());
//! ```

use std::collections::HashSet;

use super::dseparation::{Admg, DSepError};

/// Whether `z` satisfies the back-door criterion relative to `(x, y)`.
///
/// Definition 4.2.2 — `Z` is admissible when
///
/// 1. no variable of `Z` is a descendant of `X`, and
/// 2. `Z` blocks every path between `X` and `Y` that contains an arrow into
///    `X`.
///
/// The second condition is tested as d-separation of `X` and `Y` given `Z` in
/// `G` with `X` underlined — deleting the edges *out of* `X` leaves exactly the
/// paths that arrive at `X` by an arrowhead, so blocking all of those is the
/// same as d-separating in the mutilated graph.
///
/// # Errors
///
/// [`DSepError::Unknown`] for a name the graph does not have, and
/// [`DSepError::Overlap`] if `X`, `Y` and `Z` are not pairwise disjoint.
pub fn satisfies(g: &Admg, x: &[&str], y: &[&str], z: &[&str]) -> Result<bool, DSepError> {
    // 1. Z must avoid the descendants of X, X itself included.
    let forbidden: HashSet<&str> = g.descendants_of(x)?.into_iter().collect();
    if z.iter().any(|v| forbidden.contains(v)) {
        // Still resolve names so an unknown variable reports as such.
        for v in z {
            g.descendants_of(&[v])?;
        }
        return Ok(false);
    }
    // 2. No back-door path survives.
    g.cut_edges_out_of(x)?.is_d_separated(x, y, z)
}

/// Every admissible set for `(x, y)`, each containing all of `required`.
///
/// Mirrors the book's `listAdmissibleSets`. Passing a non-empty `required` asks
/// for the conditional back-door of Theorem 4.2.5 — the `W`-specific effect,
/// where the returned sets are the `Z ∪ W` that identify `P(y | do(x), w)`.
///
/// Candidates are drawn from the variables that are neither in `X` or `Y` nor
/// descendants of `X`, since condition 1 excludes those outright. Sets are
/// returned smallest first, and lexicographically within a size.
///
/// This enumerates the power set of the candidates, so it is exponential in
/// their number — fine for a textbook diagram, not for a large one.
///
/// # Errors
///
/// As [`satisfies`].
pub fn list_admissible_sets(
    g: &Admg,
    x: &[&str],
    y: &[&str],
    required: &[&str],
) -> Result<Vec<Vec<String>>, DSepError> {
    let excluded: HashSet<&str> = g
        .descendants_of(x)?
        .into_iter()
        .chain(y.iter().copied())
        .chain(required.iter().copied())
        .collect();

    let candidates: Vec<&str> = g
        .variables()
        .into_iter()
        .filter(|v| !excluded.contains(v))
        .collect();

    // A required variable that is itself barred by condition 1 makes the whole
    // question unanswerable, so report it rather than silently returning [].
    let forbidden: HashSet<&str> = g.descendants_of(x)?.into_iter().collect();
    if required.iter().any(|v| forbidden.contains(v)) {
        return Ok(Vec::new());
    }

    let mut found: Vec<Vec<String>> = Vec::new();
    for mask in 0..(1u64 << candidates.len()) {
        let mut z: Vec<&str> = required.to_vec();
        z.extend(
            candidates
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| *v),
        );
        if satisfies(g, x, y, &z)? {
            found.push(z.into_iter().map(ToString::to_string).collect());
        }
    }

    found.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    Ok(found)
}

/// The adjustment formula identifying the effect, as a displayable estimand.
///
/// Theorem 4.2.5: when `Z ∪ W` is admissible for `(X, Y)`,
///
/// ```text
/// P(y | do(x), w) = sum_z P(y | x, z, w) P(z | w)
/// ```
///
/// With `W` empty this degenerates to the usual back-door adjustment,
/// `P(y | do(x)) = sum_z P(y | x, z) P(z)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Estimand {
    /// Treatment.
    pub x: Vec<String>,
    /// Outcome.
    pub y: Vec<String>,
    /// Adjustment set, summed over.
    pub z: Vec<String>,
    /// Covariates the effect is specific to; conditioned on throughout.
    pub w: Vec<String>,
}

impl Estimand {
    /// Builds the estimand for adjusting over `z` with `w`-specific effect.
    ///
    /// # Errors
    ///
    /// [`DSepError`] if `z ∪ w` is not admissible for `(x, y)`, reported as
    /// [`DSepError::Overlap`]-style validation from [`satisfies`]; a
    /// non-admissible but well-formed set returns `Ok(None)`.
    pub fn new(
        g: &Admg,
        x: &[&str],
        y: &[&str],
        z: &[&str],
        w: &[&str],
    ) -> Result<Option<Self>, DSepError> {
        let mut adjustment: Vec<&str> = z.to_vec();
        adjustment.extend(w.iter().copied());
        if !satisfies(g, x, y, &adjustment)? {
            return Ok(None);
        }
        Ok(Some(Self {
            x: x.iter().map(ToString::to_string).collect(),
            y: y.iter().map(ToString::to_string).collect(),
            z: z.iter().map(ToString::to_string).collect(),
            w: w.iter().map(ToString::to_string).collect(),
        }))
    }

    /// The estimand as LaTeX, as the book's `ExpressionUtils::write` produces.
    #[must_use]
    pub fn to_latex(&self) -> String {
        let lower = |vs: &[String]| {
            vs.iter()
                .map(|v| v.to_lowercase())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let (x, y, z, w) = (
            lower(&self.x),
            lower(&self.y),
            lower(&self.z),
            lower(&self.w),
        );

        let mut lhs = format!("P({y} \\mid do({x})");
        if !w.is_empty() {
            lhs.push_str(&format!(", {w}"));
        }
        lhs.push(')');

        if self.z.is_empty() {
            // Nothing to adjust for: the effect is the conditional itself.
            let mut rhs = format!("P({y} \\mid {x}");
            if !w.is_empty() {
                rhs.push_str(&format!(", {w}"));
            }
            rhs.push(')');
            return format!("{lhs} = {rhs}");
        }

        let given_w = if w.is_empty() {
            String::new()
        } else {
            format!(" \\mid {w}")
        };
        let cond = if w.is_empty() {
            format!("{x}, {z}")
        } else {
            format!("{x}, {z}, {w}")
        };
        format!("{lhs} = \\sum_{{{z}}} P({y} \\mid {cond}) P({z}{given_w})")
    }
}

impl std::fmt::Display for Estimand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let join = |vs: &[String]| {
            vs.iter()
                .map(|v| v.to_lowercase())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let (x, y, z, w) = (join(&self.x), join(&self.y), join(&self.z), join(&self.w));

        let lhs = if w.is_empty() {
            format!("P({y} | do({x}))")
        } else {
            format!("P({y} | do({x}), {w})")
        };
        if self.z.is_empty() {
            let rhs = if w.is_empty() {
                format!("P({y} | {x})")
            } else {
                format!("P({y} | {x}, {w})")
            };
            return write!(f, "{lhs} = {rhs}");
        }
        let cond = if w.is_empty() {
            format!("{x}, {z}")
        } else {
            format!("{x}, {z}, {w}")
        };
        let zw = if w.is_empty() {
            z.clone()
        } else {
            format!("{z} | {w}")
        };
        write!(f, "{lhs} = sum_{{{z}}} P({y} | {cond}) P({zw})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The confounded triangle: Z -> X, Z -> Y, X -> Y.
    fn confounded() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("Z", "Y");
        g.directed("X", "Y");
        g
    }

    /// Example 4.4 — SAT results. Z1 is the school district, Z2 family income,
    /// X the type of school, Y the SAT result.
    fn example_4_4() -> Admg {
        let mut g = Admg::new();
        g.directed("Z1", "X");
        g.directed("Z2", "Z1");
        g.directed("Z2", "Y");
        g.directed("X", "Y");
        g.bidirected("Z1", "Z2");
        g.bidirected("X", "Z1");
        g
    }

    /// The nine-variable diagram of notebook 4-12.
    fn figure_4_12() -> Admg {
        let mut g = Admg::new();
        for (a, b) in [
            ("X", "Z1"),
            ("X", "Z4"),
            ("Z1", "Y"),
            ("Z2", "X"),
            ("Z2", "Z5"),
            ("Z3", "Y"),
            ("Z4", "Z3"),
            ("Z5", "Y"),
            ("Z6", "Z2"),
            ("Z6", "Z5"),
            ("Z7", "Z5"),
            ("Z7", "Y"),
        ] {
            g.directed(a, b);
        }
        g.bidirected("X", "Z6");
        g.bidirected("Z6", "Z7");
        g
    }

    #[test]
    fn the_empty_set_is_admissible_when_nothing_confounds() {
        let mut g = Admg::new();
        g.directed("X", "Y");
        assert!(satisfies(&g, &["X"], &["Y"], &[]).unwrap());
        assert_eq!(
            list_admissible_sets(&g, &["X"], &["Y"], &[]).unwrap(),
            vec![Vec::<String>::new()]
        );
    }

    #[test]
    fn a_confounder_must_be_adjusted_for() {
        let g = confounded();
        assert!(!satisfies(&g, &["X"], &["Y"], &[]).unwrap());
        assert!(satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap());
        assert_eq!(
            list_admissible_sets(&g, &["X"], &["Y"], &[]).unwrap(),
            vec![vec!["Z".to_string()]]
        );
    }

    #[test]
    fn a_descendant_of_the_treatment_is_never_admissible() {
        // X -> M -> Y with a confounder: adjusting for the mediator is barred by
        // condition 1 even though it would block the causal path.
        let mut g = Admg::new();
        g.directed("X", "M");
        g.directed("M", "Y");
        g.directed("Z", "X");
        g.directed("Z", "Y");
        assert!(!satisfies(&g, &["X"], &["Y"], &["M"]).unwrap());
        assert!(!satisfies(&g, &["X"], &["Y"], &["Z", "M"]).unwrap());
        assert!(satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap());
        // No admissible set may contain M.
        for set in list_admissible_sets(&g, &["X"], &["Y"], &[]).unwrap() {
            assert!(!set.contains(&"M".to_string()));
        }
    }

    #[test]
    fn conditioning_on_a_collider_opens_a_back_door() {
        // X -> Y with a collider C that is not a descendant of X.
        let mut g = Admg::new();
        g.directed("X", "Y");
        g.bidirected("X", "U1");
        g.bidirected("Y", "U2");
        g.directed("U1", "C");
        g.directed("U2", "C");
        assert!(satisfies(&g, &["X"], &["Y"], &[]).unwrap());
        assert!(!satisfies(&g, &["X"], &["Y"], &["C"]).unwrap());
    }

    #[test]
    fn example_4_4_conditional_backdoor() {
        let g = example_4_4();
        // The district Z1 alone does not close the back door: X <-> Z1 <-> Z2 -> Y
        // survives.
        assert!(!satisfies(&g, &["X"], &["Y"], &["Z1"]).unwrap());
        // Adding family income does.
        assert!(satisfies(&g, &["X"], &["Y"], &["Z1", "Z2"]).unwrap());

        // The book asks for sets containing Z1, the variable the query is
        // specific to.
        let sets = list_admissible_sets(&g, &["X"], &["Y"], &["Z1"]).unwrap();
        assert_eq!(sets, vec![vec!["Z1".to_string(), "Z2".to_string()]]);
    }

    #[test]
    fn figure_4_12_admissible_sets() {
        let g = figure_4_12();
        let sets = list_admissible_sets(&g, &["X"], &["Y"], &[]).unwrap();
        assert!(!sets.is_empty(), "the effect should be identifiable");
        // Every reported set really is admissible, contains no descendant of X,
        // and none is the empty set (X and Y are confounded through Z6).
        let descendants: HashSet<&str> = g.descendants_of(&["X"]).unwrap().into_iter().collect();
        for set in &sets {
            let refs: Vec<&str> = set.iter().map(String::as_str).collect();
            assert!(satisfies(&g, &["X"], &["Y"], &refs).unwrap(), "{set:?}");
            assert!(set.iter().all(|v| !descendants.contains(v.as_str())), "{set:?}");
            assert!(!set.is_empty());
        }
        // Sorted smallest first.
        assert!(sets.windows(2).all(|w| w[0].len() <= w[1].len()));
    }

    #[test]
    fn estimand_rendering() {
        let g = confounded();
        let e = Estimand::new(&g, &["X"], &["Y"], &["Z"], &[])
            .unwrap()
            .expect("Z is admissible");
        assert_eq!(e.to_string(), "P(y | do(x)) = sum_{z} P(y | x, z) P(z)");
        assert_eq!(
            e.to_latex(),
            "P(y \\mid do(x)) = \\sum_{z} P(y \\mid x, z) P(z)"
        );

        // The W-specific form of Example 4.4.
        let g = example_4_4();
        let e = Estimand::new(&g, &["X"], &["Y"], &["Z2"], &["Z1"])
            .unwrap()
            .expect("Z2 with Z1 is admissible");
        assert_eq!(
            e.to_string(),
            "P(y | do(x), z1) = sum_{z2} P(y | x, z2, z1) P(z2 | z1)"
        );

        // No adjustment needed at all.
        let mut g = Admg::new();
        g.directed("X", "Y");
        let e = Estimand::new(&g, &["X"], &["Y"], &[], &[])
            .unwrap()
            .expect("nothing to adjust for");
        assert_eq!(e.to_string(), "P(y | do(x)) = P(y | x)");
    }

    #[test]
    fn a_non_admissible_set_yields_no_estimand() {
        let g = confounded();
        assert_eq!(Estimand::new(&g, &["X"], &["Y"], &[], &[]).unwrap(), None);
    }
}
