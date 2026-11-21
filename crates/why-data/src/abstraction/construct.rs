//! The abstract invariance condition (Thm 15.2.2) and Algorithm 51.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use petgraph::visit::EdgeRef;

use super::consistency::{assignments, low_joint, names_in, supports};
use super::{Abstraction, AbstractionError, HighLevel};
use crate::graph::dseparation::Admg;
use crate::scm::Model;

/// A low-level assignment, as `(variable, value)` pairs.
type Assignment = Vec<(String, String)>;

/// The most low-level assignments of the other clusters the AIC check
/// enumerates for one cluster.
const MAX_ASSIGNMENTS: usize = 4096;

/// Two preimages of the same high-level inputs under which a cluster maps
/// to different levels (Thm 15.2.2 fails), or under which data disagree
/// (Def. 15.2.5 fails).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AicViolation {
    /// The cluster (or clusters) whose output differs.
    pub cluster: String,
    /// One low-level assignment of the inputs.
    pub first: Vec<(String, String)>,
    /// Another, with the same image under `τ`.
    pub second: Vec<(String, String)>,
    /// What differs: the cluster's level under each assignment, for some
    /// draw of the exogenous variables, or the high-level event whose
    /// probability differs.
    pub outputs: (String, String),
}

impl fmt::Display for AicViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show = |a: &[(String, String)]| {
            a.iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        write!(
            f,
            "{} gives {} under ({}) but {} under ({})",
            self.cluster,
            self.outputs.0,
            show(&self.first),
            self.outputs.1,
            show(&self.second)
        )
    }
}

/// Checks the abstract invariance condition (Thm 15.2.2) of `m` w.r.t. `τ`.
///
/// For each cluster `C_i`, the members of every other cluster are set to
/// two low-level values with the same image under `τ`. The cluster must then
/// map to the same level for every draw of `U` with positive probability.
/// Members of `C_i` are computed from one another, and projected-away
/// variables from `U` (see the module docs).
///
/// Returns the first violation found, or `None` when the condition holds.
///
/// # Errors
///
/// [`AbstractionError::UnknownVariable`] for a member the model lacks,
/// [`AbstractionError::TooLarge`] past 4096 assignments per cluster, and the
/// model's errors.
pub fn abstract_invariance(
    m: &mut Model,
    tau: &Abstraction,
) -> Result<Option<AicViolation>, AbstractionError> {
    for i in 0..tau.len() {
        let others: Vec<String> = (0..tau.len())
            .filter(|&j| j != i)
            .flat_map(|j| tau.members_at(j).to_vec())
            .collect();
        if others.is_empty() {
            continue;
        }
        let domains = supports(m, &others)?;
        let all = assignments(&others, &domains, MAX_ASSIGNMENTS)?;
        let mut groups: BTreeMap<Assignment, Vec<Assignment>> = BTreeMap::new();
        for a in all {
            let pairs: Vec<(&str, &str)> =
                a.iter().map(|(x, y)| (x.as_str(), y.as_str())).collect();
            groups.entry(tau.tau(&pairs)?).or_default().push(a);
        }
        let members = tau.members_at(i).to_vec();
        for group in groups.values() {
            for other in &group[1..] {
                let worlds = [group[0].clone(), other.clone()];
                let mut columns = Vec::new();
                for x in &worlds {
                    columns.extend(names_in(m, x, &members)?);
                }
                let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
                let table = m.world(m.observational()).probability_table(Some(&refs))?;
                let k = members.len();
                for (row, p) in &table.rows {
                    if *p <= 0.0 {
                        continue;
                    }
                    let text: Vec<String> = row.iter().map(|v| v.as_int().to_string()).collect();
                    let a: Vec<&str> = text[..k].iter().map(String::as_str).collect();
                    let b: Vec<&str> = text[k..].iter().map(String::as_str).collect();
                    let (ta, tb) = (tau.map_at(i, &a), tau.map_at(i, &b));
                    if ta != tb {
                        return Ok(Some(AicViolation {
                            cluster: tau.name_at(i).to_string(),
                            first: worlds[0].clone(),
                            second: worlds[1].clone(),
                            outputs: (ta, tb),
                        }));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// The causal diagram of `m`'s endogenous variables (directed edges only).
fn diagram(m: &Model) -> Admg {
    let world = m.world(m.observational()).world();
    let names = world.variables();
    let mut g = Admg::new();
    for n in names {
        g.add_node(n);
    }
    let graph = world.graph();
    for e in graph.edge_references() {
        let (a, b) = (&graph[e.source()], &graph[e.target()]);
        if names.contains(a) && names.contains(b) {
            g.directed(a, b);
        }
    }
    g
}

/// The high-level model Algorithm 51 builds from `M_L` and `τ`:
/// `U_H = U_L`, and each high-level variable is
/// `f^H_i(p̃a, u) = τ_{C_i}(f^L_V(p̃a_V, u_V) : V ∈ C_i)` for any preimage
/// `p̃a` of its high-level inputs. It is evaluated exactly that way: a
/// high-level intervention runs `M_L` under one preimage of it.
///
/// By Prop. 15.2.4 it is `L3`-τ consistent with `M_L`.
#[derive(Clone, Debug)]
pub struct Constructed {
    low: Model,
    tau: Abstraction,
    /// A low-level preimage of each (cluster, level).
    preimage: HashMap<(usize, String), Vec<(String, String)>>,
}

/// Algorithm 51: constructs `M_H` from `M_L` and admissible clusters that
/// satisfy the abstract invariance condition.
///
/// ```
/// use why_data::abstraction::{Abstraction, HighLevel, construct};
/// use why_data::scm::{expr, Dist, Model};
///
/// // Ex. 15.1: the total vote abstracts three voters.
/// let v = expr::var;
/// let m = Model::new(
///     [
///         ("X1".to_string(), v("U1")),
///         ("X2".to_string(), v("U2")),
///         ("X3".to_string(), v("U3")),
///         ("Y".to_string(), expr::ge(expr::add(expr::add(v("X1"), v("X2")), v("X3")), expr::int(2))),
///     ],
///     ["U1", "U2", "U3"].map(|u| (u.to_string(), Dist::Bernoulli(0.5))),
/// )
/// .unwrap();
/// let tau = Abstraction::new()
///     .cluster("XT", &["X1", "X2", "X3"], |x| {
///         x.iter().map(|b| b.parse::<u32>().unwrap()).sum::<u32>().to_string()
///     })
///     .unwrap()
///     .keep("Y")
///     .unwrap();
/// let mut high = construct(m, tau.clone()).unwrap();
/// // Forcing a total of two votes for Alice elects her.
/// let joint = high.joint(&tau, &[vec![("XT".into(), "2".into())]]).unwrap();
/// let p: f64 = joint.iter().filter(|(k, _)| k[1] == "1").map(|(_, p)| p).sum();
/// assert!((p - 1.0).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// [`AbstractionError::NotAdmissible`] when the clusters admit no
/// topological order, [`AbstractionError::Invariance`] when the AIC fails,
/// and the errors of [`abstract_invariance`].
pub fn construct(mut low: Model, tau: Abstraction) -> Result<Constructed, AbstractionError> {
    if let Some(c) = tau.inadmissible(&diagram(&low))? {
        return Err(AbstractionError::NotAdmissible(c));
    }
    if let Some(v) = abstract_invariance(&mut low, &tau)? {
        return Err(AbstractionError::Invariance(Box::new(v)));
    }
    let mut preimage = HashMap::new();
    for i in 0..tau.len() {
        let members = tau.members_at(i).to_vec();
        let domains = supports(&low, &members)?;
        for a in assignments(&members, &domains, usize::MAX)? {
            let values: Vec<&str> = a.iter().map(|(_, v)| v.as_str()).collect();
            preimage.entry((i, tau.map_at(i, &values))).or_insert(a);
        }
    }
    Ok(Constructed { low, tau, preimage })
}

impl Constructed {
    /// The low-level model.
    #[must_use]
    pub fn low(&self) -> &Model {
        &self.low
    }

    /// The abstraction function.
    #[must_use]
    pub fn tau(&self) -> &Abstraction {
        &self.tau
    }
}

impl HighLevel for Constructed {
    fn joint(
        &mut self,
        _tau: &Abstraction,
        worlds: &[Vec<(String, String)>],
    ) -> Result<BTreeMap<Vec<String>, f64>, AbstractionError> {
        let low_worlds = worlds
            .iter()
            .map(|x| {
                let mut out = Vec::new();
                for (c, level) in x {
                    let i = self.tau.index(c)?;
                    let pre = self.preimage.get(&(i, level.clone())).ok_or_else(|| {
                        AbstractionError::NoPreimage {
                            cluster: c.clone(),
                            level: level.clone(),
                        }
                    })?;
                    out.extend(pre.iter().cloned());
                }
                Ok(out)
            })
            .collect::<Result<Vec<_>, AbstractionError>>()?;
        low_joint(&mut self.low, &self.tau, &low_worlds)
    }
}
