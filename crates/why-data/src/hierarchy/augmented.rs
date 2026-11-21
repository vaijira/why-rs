//! Augmented DAGs of the decision-theoretic framework (Sec. 20.3.3):
//! Algorithm 72 (AG-CONSTRUCT).

use super::HierarchyError;
use crate::discovery::{Engine, Pag, UNKNOWN_SEPSET, subsets};

/// An extended conditional independence `A ⊥ B | C` among domain
/// variables, latents and regime indicators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eci {
    a: Vec<String>,
    b: Vec<String>,
    given: Vec<String>,
}

impl Eci {
    /// `a ⊥ b | given`.
    #[must_use]
    pub fn new(a: &[&str], b: &[&str], given: &[&str]) -> Self {
        let own = |s: &[&str]| s.iter().map(|v| (*v).to_string()).collect();
        Self {
            a: own(a),
            b: own(b),
            given: own(given),
        }
    }

    /// Whether this statement yields `x ⊥ y | z` by symmetry,
    /// decomposition and weak union.
    fn yields(&self, x: &str, y: &str, z: &[&str]) -> bool {
        let has = |s: &[String], v: &str| s.iter().any(|w| w == v);
        let sides = (has(&self.a, x) && has(&self.b, y)) || (has(&self.a, y) && has(&self.b, x));
        sides
            && self.given.iter().all(|g| z.contains(&g.as_str()))
            && z.iter().all(|w| {
                *w != x && *w != y && (has(&self.given, w) || has(&self.a, w) || has(&self.b, w))
            })
    }
}

/// Algorithm 72 (AG-CONSTRUCT): the augmented DAG of a DT causal model.
///
/// 1. Start from the complete circle graph over the domain variables `v`
///    and the latents `u`.
/// 2. Add `F_i → V_i` for each regime indicator, given as `(F_i, V_i)`.
/// 3. Run FCI with the ECIs as the oracle.
/// 4. Turn every `◦→` into `→`.
///
/// The oracle accepts `x ⊥ y | z` when one listed ECI gives it by
/// symmetry, decomposition and weak union. Other graphoid consequences
/// must be listed explicitly. A regime indicator is never adjacent to
/// anything but its target. When no separating set is found for such a
/// pair, the edge is still left out, but the pair then certifies no
/// collider. Edges that stay `◦–◦` are the ones on which compatible
/// augmented DAGs disagree (Ex. 20.43).
///
/// ```
/// use why_data::hierarchy::{Eci, ag_construct};
///
/// // Ex. 20.42: the output is Fig. 20.33(d).
/// let ecis = [
///     Eci::new(&["U", "Z"], &["FX"], &[]),
///     Eci::new(&["U"], &["Z"], &["FX"]),
///     Eci::new(&["Y"], &["FX"], &["X", "U"]),
///     Eci::new(&["Y"], &["Z"], &["X", "U", "FX"]),
/// ];
/// let g = ag_construct(&["Z", "X", "Y"], &["U"], &[("FX", "X")], &ecis).unwrap();
/// assert_eq!(g.edges(), ["FX --> X", "U --> X", "U --> Y", "X --> Y", "X <-- Z"]);
/// ```
///
/// # Errors
///
/// [`HierarchyError::InvalidInput`] for a regime indicator whose target is
/// not a domain variable, or a repeated name.
pub fn ag_construct(
    v: &[&str],
    u: &[&str],
    regimes: &[(&str, &str)],
    ecis: &[Eci],
) -> Result<Pag, HierarchyError> {
    let mut names: Vec<&str> = v.iter().chain(u).copied().collect();
    let first_f = names.len();
    for (f, target) in regimes {
        if !v.contains(target) {
            return Err(HierarchyError::InvalidInput(format!(
                "{f} targets {target}, which is not a domain variable"
            )));
        }
        names.push(f);
    }
    for (i, a) in names.iter().enumerate() {
        if names[..i].contains(a) {
            return Err(HierarchyError::InvalidInput(format!("{a} appears twice")));
        }
    }
    let n = names.len();
    let target = |i: usize| -> Option<usize> {
        (i >= first_f).then(|| {
            let t = regimes[i - first_f].1;
            names.iter().position(|w| *w == t).expect("validated")
        })
    };
    let fnodes = (first_f..n).map(|f| (f, None)).collect();
    let mut engine = Engine::skeleton(&names, fnodes, false, |i, j| {
        let (fi, fj) = (target(i), target(j));
        if fi.is_some() && fj.is_some() {
            return Ok(Some(Vec::new()));
        }
        if fi == Some(j) || fj == Some(i) {
            return Ok(None);
        }
        let others: Vec<usize> = (0..n).filter(|&k| k != i && k != j).collect();
        for z in subsets(&others) {
            let zs: Vec<&str> = z.iter().map(|&k| names[k]).collect();
            if ecis.iter().any(|e| e.yields(names[i], names[j], &zs)) {
                return Ok(Some(z));
            }
        }
        Ok((fi.is_some() || fj.is_some()).then(|| vec![UNKNOWN_SEPSET]))
    })?;
    engine.colliders();
    engine.fci_rules();
    let mut pag = engine.pag;
    pag.arrows_for_circles();
    Ok(pag)
}
