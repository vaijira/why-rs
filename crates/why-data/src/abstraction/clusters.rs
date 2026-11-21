//! Constructive abstraction functions (Defs. 15.2.1, 15.2.2) and cluster
//! causal diagrams (Def. 15.3.1).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use super::AbstractionError;
use crate::crl::latent_projection;
use crate::fairness::Dataset;
use crate::graph::dseparation::{Admg, EdgeKind};

type Map = Arc<dyn Fn(&[&str]) -> String + Send + Sync>;

/// One intervariable cluster `C_i`, the high-level variable it becomes, and
/// its intravariable clustering `D_{C_i}`, as the function `τ_{C_i}` from the
/// members' values to the block (high-level level) they fall in.
#[derive(Clone)]
struct Cluster {
    name: String,
    members: Vec<String>,
    map: Map,
}

/// A constructive abstraction function `τ` (Def. 15.2.2): one cluster per
/// high-level variable, each with its own map `τ_{C_i}`. Low-level variables
/// in no cluster are projected away.
///
/// Values are strings, as in [`Dataset`]. Structural-model values enter as
/// integers (`0`/`1` for booleans).
///
/// ```
/// use why_data::abstraction::Abstraction;
///
/// // Ex. 15.5: X completes the treatment when both doses were taken;
/// // R is projected away.
/// let tau = Abstraction::new()
///     .cluster("X", &["A", "B"], |v| {
///         if v == ["1", "1"] { "1".into() } else { "0".into() }
///     })
///     .unwrap()
///     .keep("Y")
///     .unwrap();
/// let high = tau.tau(&[("R", "1"), ("A", "0"), ("B", "1"), ("Y", "1")]).unwrap();
/// assert_eq!(high, [("X".to_string(), "0".to_string()), ("Y".into(), "1".into())]);
/// ```
#[derive(Clone, Default)]
pub struct Abstraction {
    clusters: Vec<Cluster>,
}

impl fmt::Debug for Abstraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut m = f.debug_map();
        for c in &self.clusters {
            m.entry(&c.name, &c.members);
        }
        m.finish()
    }
}

impl Abstraction {
    /// An abstraction with no clusters yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the cluster `name` over `members`, mapped by `map`, which
    /// receives the members' values in the order given.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::BadCluster`] for an empty cluster or a name used
    /// twice, and [`AbstractionError::Overlap`] for a member already in
    /// another cluster.
    pub fn cluster(
        mut self,
        name: &str,
        members: &[&str],
        map: impl Fn(&[&str]) -> String + Send + Sync + 'static,
    ) -> Result<Self, AbstractionError> {
        if members.is_empty() || self.clusters.iter().any(|c| c.name == name) {
            return Err(AbstractionError::BadCluster(name.to_string()));
        }
        let mut seen = BTreeSet::new();
        for m in members {
            if !seen.insert(*m) || self.cluster_of(m).is_some() {
                return Err(AbstractionError::Overlap((*m).to_string()));
            }
        }
        self.clusters.push(Cluster {
            name: name.to_string(),
            members: members.iter().map(|s| (*s).to_string()).collect(),
            map: Arc::new(map),
        });
        Ok(self)
    }

    /// Keeps `var` as is: a cluster of its own, with the same name, whose
    /// levels are its own.
    ///
    /// # Errors
    ///
    /// Those of [`Abstraction::cluster`].
    pub fn keep(self, var: &str) -> Result<Self, AbstractionError> {
        self.cluster(var, &[var], |v| v[0].to_string())
    }

    /// The high-level variables, in the order the clusters were added.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.clusters.iter().map(|c| c.name.as_str()).collect()
    }

    /// The members of cluster `name`.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::UnknownCluster`] for a name that is not a cluster.
    pub fn members(&self, name: &str) -> Result<&[String], AbstractionError> {
        Ok(&self.clusters[self.index(name)?].members)
    }

    /// Every clustered low-level variable.
    #[must_use]
    pub fn clustered(&self) -> Vec<&str> {
        self.clusters
            .iter()
            .flat_map(|c| c.members.iter().map(String::as_str))
            .collect()
    }

    pub(crate) fn index(&self, name: &str) -> Result<usize, AbstractionError> {
        self.clusters
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| AbstractionError::UnknownCluster(name.to_string()))
    }

    pub(crate) fn len(&self) -> usize {
        self.clusters.len()
    }

    pub(crate) fn members_at(&self, i: usize) -> &[String] {
        &self.clusters[i].members
    }

    pub(crate) fn name_at(&self, i: usize) -> &str {
        &self.clusters[i].name
    }

    /// `τ_{C_i}` for cluster `i`.
    pub(crate) fn map_at(&self, i: usize, values: &[&str]) -> String {
        (self.clusters[i].map)(values)
    }

    /// The cluster that contains `var`.
    #[must_use]
    pub fn cluster_of(&self, var: &str) -> Option<&str> {
        self.clusters
            .iter()
            .find(|c| c.members.iter().any(|m| m == var))
            .map(|c| c.name.as_str())
    }

    /// `τ_{C}` of cluster `name` at its members' `values`.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::UnknownCluster`] for a name that is not a cluster,
    /// and [`AbstractionError::Data`] for a wrong number of values.
    pub fn apply(&self, name: &str, values: &[&str]) -> Result<String, AbstractionError> {
        let i = self.index(name)?;
        if values.len() != self.clusters[i].members.len() {
            return Err(AbstractionError::Data(format!(
                "{} values for the {} members of {name}",
                values.len(),
                self.clusters[i].members.len()
            )));
        }
        Ok(self.map_at(i, values))
    }

    /// `τ(w_L)` for a low-level assignment `w_L` that covers each cluster
    /// fully or not at all (Def. 15.2.2 (3)), in cluster order. Variables in
    /// no cluster are ignored.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::Partial`] when a cluster is only partly assigned.
    pub fn tau(
        &self,
        assignment: &[(&str, &str)],
    ) -> Result<Vec<(String, String)>, AbstractionError> {
        let given: BTreeMap<&str, &str> = assignment.iter().copied().collect();
        let mut out = Vec::new();
        for c in &self.clusters {
            let values: Vec<Option<&str>> = c
                .members
                .iter()
                .map(|m| given.get(m.as_str()).copied())
                .collect();
            match values.iter().filter(|v| v.is_some()).count() {
                0 => {}
                n if n == values.len() => {
                    let v: Vec<&str> = values.into_iter().map(|v| v.expect("all given")).collect();
                    out.push((c.name.clone(), (c.map)(&v)));
                }
                _ => return Err(AbstractionError::Partial(c.name.clone())),
            }
        }
        Ok(out)
    }

    /// `τ(P(V_L))`: the data set over the high-level variables, one row per
    /// low-level row, weights carried over.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::UnknownVariable`] when the data lack a clustered
    /// variable.
    pub fn transform(&self, data: &Dataset) -> Result<Dataset, AbstractionError> {
        for v in self.clustered() {
            if data.levels(v).is_none() {
                return Err(AbstractionError::UnknownVariable(v.to_string()));
            }
        }
        let rows: Vec<Vec<String>> = (0..data.len())
            .map(|r| {
                self.clusters
                    .iter()
                    .map(|c| {
                        let v: Vec<&str> = c
                            .members
                            .iter()
                            .map(|m| data.value(m, r).expect("checked"))
                            .collect();
                        (c.map)(&v)
                    })
                    .collect()
            })
            .collect();
        let out = Dataset::from_records(self.names(), rows)?;
        Ok(match data.weights() {
            Some(w) => out.with_weights(w.to_vec())?,
            None => out,
        })
    }

    fn check_graph(&self, g: &Admg) -> Result<(), AbstractionError> {
        for v in self.clustered() {
            if g.node(v).is_none() {
                return Err(AbstractionError::UnknownVariable(v.to_string()));
            }
        }
        Ok(())
    }

    /// Whether the clustering is admissible w.r.t. `g` (Def. 15.2.1): no
    /// descendant of a cluster outside it is also its ancestor, so the
    /// clusters keep a topological order.
    ///
    /// # Errors
    ///
    /// [`AbstractionError::UnknownVariable`] for a member `g` lacks.
    pub fn is_admissible(&self, g: &Admg) -> Result<bool, AbstractionError> {
        Ok(self.inadmissible(g)?.is_none())
    }

    /// The first cluster that breaks admissibility, if any.
    pub(crate) fn inadmissible(&self, g: &Admg) -> Result<Option<String>, AbstractionError> {
        self.check_graph(g)?;
        for c in &self.clusters {
            let members: Vec<&str> = c.members.iter().map(String::as_str).collect();
            let down: BTreeSet<&str> = g.descendants_of(&members)?.into_iter().collect();
            let up: BTreeSet<&str> = g.ancestors_of(&members)?.into_iter().collect();
            if down
                .intersection(&up)
                .any(|v| !c.members.iter().any(|m| m == v))
            {
                return Ok(Some(c.name.clone()));
            }
        }
        Ok(None)
    }

    /// The cluster causal diagram `G_C` (Def. 15.3.1) of `g`. Variables in no
    /// cluster are first projected out (the latent projection onto the
    /// clustered variables). Then `C_i → C_j` whenever some member of `C_i`
    /// points to one of `C_j`, and `C_i ↔ C_j` whenever some members share a
    /// bidirected edge.
    ///
    /// ```
    /// use why_data::abstraction::Abstraction;
    /// use why_data::graph::dseparation::Admg;
    ///
    /// // Ex. 15.9: projecting R away leaves X <-> Y.
    /// let mut g = Admg::new();
    /// for (a, b) in [("R", "A"), ("R", "B"), ("A", "B"), ("A", "Y"), ("B", "Y")] {
    ///     g.directed(a, b);
    /// }
    /// g.bidirected("R", "Y");
    /// let tau = Abstraction::new()
    ///     .cluster("X", &["A", "B"], |v| (v == ["1", "1"]).then_some("1").unwrap_or("0").into())
    ///     .unwrap()
    ///     .keep("Y")
    ///     .unwrap();
    /// let gc = tau.cdag(&g).unwrap();
    /// assert!(gc.is_d_separated(&["X"], &["Y"], &[]).is_ok());
    /// assert_eq!(gc.graph().edge_count(), 2); // X -> Y and X <-> Y
    /// ```
    ///
    /// # Errors
    ///
    /// [`AbstractionError::UnknownVariable`] for a member `g` lacks, and
    /// [`AbstractionError::NotAdmissible`] when the clusters admit no
    /// topological order.
    pub fn cdag(&self, g: &Admg) -> Result<Admg, AbstractionError> {
        if let Some(c) = self.inadmissible(g)? {
            return Err(AbstractionError::NotAdmissible(c));
        }
        let keep = self.clustered();
        let projected =
            latent_projection(g, &keep).map_err(|e| AbstractionError::Graph(e.to_string()))?;
        let mut out = Admg::new();
        for c in &self.clusters {
            out.add_node(&c.name);
        }
        let mut edges: BTreeSet<(String, String, bool)> = BTreeSet::new();
        let graph = projected.graph();
        for e in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(e).expect("valid edge");
            let (ca, cb) = (
                self.cluster_of(&graph[a])
                    .expect("projected onto clustered"),
                self.cluster_of(&graph[b])
                    .expect("projected onto clustered"),
            );
            if ca == cb {
                continue;
            }
            match graph[e] {
                EdgeKind::Directed => {
                    edges.insert((ca.to_string(), cb.to_string(), true));
                }
                EdgeKind::Bidirected => {
                    let (x, y) = if ca < cb { (ca, cb) } else { (cb, ca) };
                    edges.insert((x.to_string(), y.to_string(), false));
                }
            }
        }
        for (a, b, directed) in edges {
            if directed {
                out.directed(&a, &b);
            } else {
                out.bidirected(&a, &b);
            }
        }
        Ok(out)
    }
}
