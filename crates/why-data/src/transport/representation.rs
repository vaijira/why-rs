//! Representations `R = φ(X)` and the variables they determine, constrain or
//! leave free (Def. 11.2.8).

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::TransportError;

/// The most points of `supp(X)` [`Representation::classify`] enumerates.
const SUPPORT_CAP: usize = 1 << 22;

/// How a representation treats one feature (Def. 11.2.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariableRole {
    /// Every value `r` pins the feature to a single value: `det(φ)`.
    Determined,
    /// Not determined, but some value `r` rules some value of it out:
    /// `cons(φ)`.
    Constrained,
    /// Changing the feature never changes `φ`: `free(φ)`.
    Free,
}

/// The roles of the features under a representation, decided by enumerating
/// `supp(X)`.
///
/// With `Z = det(φ)` and `Z† = cons(φ)`, Thm 11.2.11 rewrites the score
/// `E_{P*}[Y | φ(X) = r]` as `P*(y | z, r†)`, where `R† = φ†(Z†)` captures
/// what `r` says about `Z†`. [`SelectionDiagram::augment`](super::SelectionDiagram::augment)
/// adds `R†` to the diagram with arrows from `Z†`, after which
/// [`score_tr`](super::score_tr) decides transportability.
///
/// ```
/// use why_data::transport::{Representation, VariableRole};
///
/// // Ex. 11.18: R = ⟨β·X1:3, X1 + … + X6⟩ over nine binary features.
/// let names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"];
/// let supports = vec![vec![0, 1]; 9];
/// let phi = |x: &[i64]| (x[0] + 2 * x[1] + 4 * x[2], x[..6].iter().sum::<i64>());
/// let rep = Representation::classify(&names, &supports, phi).unwrap();
/// assert_eq!(rep.determined(), ["X1", "X2", "X3"]);
/// assert_eq!(rep.constrained(), ["X4", "X5", "X6"]);
/// assert_eq!(rep.free(), ["X7", "X8", "X9"]);
/// ```
#[derive(Clone, Debug)]
pub struct Representation {
    names: Vec<String>,
    roles: Vec<VariableRole>,
    values: usize,
}

impl Representation {
    /// Classifies each feature under `phi`, given each feature's finite
    /// support.
    ///
    /// For every value `r`, let `T^r = {x : φ(x) = r}`. A feature is
    /// determined when every `T^r` takes a single value on it, and free when
    /// every `T^r` is closed under changing it to any value of its support
    /// (Eqs. 11.100–11.101). It is constrained otherwise. A feature with a
    /// single possible value counts as determined.
    ///
    /// # Errors
    ///
    /// [`TransportError::InvalidInput`] for mismatched lengths or an empty
    /// support, and [`TransportError::TooLarge`] beyond 2²² points.
    pub fn classify<R, F>(names: &[&str], supports: &[Vec<i64>], phi: F) -> Result<Self, TransportError>
    where
        R: Eq + Hash,
        F: Fn(&[i64]) -> R,
    {
        if names.len() != supports.len() {
            return Err(TransportError::InvalidInput(format!(
                "{} names for {} supports",
                names.len(),
                supports.len()
            )));
        }
        if let Some(i) = supports.iter().position(Vec::is_empty) {
            return Err(TransportError::InvalidInput(format!(
                "`{}` has an empty support",
                names[i]
            )));
        }
        let size = supports
            .iter()
            .try_fold(1usize, |acc, s| acc.checked_mul(s.len()))
            .filter(|n| *n <= SUPPORT_CAP)
            .ok_or_else(|| {
                TransportError::TooLarge(format!("supp(X) exceeds {SUPPORT_CAP} points"))
            })?;
        let n = names.len();
        let mut classes: HashMap<R, HashSet<Vec<i64>>> = HashMap::new();
        let mut idx = vec![0usize; n];
        for _ in 0..size {
            let x: Vec<i64> = idx.iter().zip(supports).map(|(&i, s)| s[i]).collect();
            classes.entry(phi(&x)).or_default().insert(x);
            for k in 0..n {
                idx[k] += 1;
                if idx[k] < supports[k].len() {
                    break;
                }
                idx[k] = 0;
            }
        }
        let roles = (0..n)
            .map(|k| {
                let determined = classes.values().all(|t| {
                    let first = t.iter().next().expect("classes are non-empty")[k];
                    t.iter().all(|x| x[k] == first)
                });
                if determined {
                    return VariableRole::Determined;
                }
                let free = classes.values().all(|t| {
                    t.iter().all(|x| {
                        supports[k].iter().all(|&v| {
                            let mut moved = x.clone();
                            moved[k] = v;
                            t.contains(&moved)
                        })
                    })
                });
                if free {
                    VariableRole::Free
                } else {
                    VariableRole::Constrained
                }
            })
            .collect();
        Ok(Self {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            roles,
            values: classes.len(),
        })
    }

    /// The role of `name`.
    #[must_use]
    pub fn role(&self, name: &str) -> Option<VariableRole> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.roles[i])
    }

    fn with(&self, role: VariableRole) -> Vec<&str> {
        self.names
            .iter()
            .zip(&self.roles)
            .filter(|(_, r)| **r == role)
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// `det(φ)`, in input order.
    #[must_use]
    pub fn determined(&self) -> Vec<&str> {
        self.with(VariableRole::Determined)
    }

    /// `cons(φ)`, in input order.
    #[must_use]
    pub fn constrained(&self) -> Vec<&str> {
        self.with(VariableRole::Constrained)
    }

    /// `free(φ)`, in input order.
    #[must_use]
    pub fn free(&self) -> Vec<&str> {
        self.with(VariableRole::Free)
    }

    /// `|supp(R)|`: how many distinct values `φ` takes.
    #[must_use]
    pub fn support_size(&self) -> usize {
        self.values
    }
}
