//! d-separation over acyclic directed mixed graphs.
//!
//! Two complementary views of the same question, both from the Causal AI book's
//! chapter 2:
//!
//! - [`Admg::is_d_separated`] answers it in one shot with the ancestral
//!   moralization test, which never enumerates a path.
//! - [`Admg::paths`] enumerates every path between two variables and says which
//!   are blocked and why, following Definition 2.4.3 directly. This is what you
//!   want when explaining a result rather than just deciding it.
//!
//! The graph is *mixed*: alongside directed edges `V_i -> V_j` it carries
//! bidirected edges `V_i <-> V_j`, which stand for an unobserved common cause
//! and make the model non-Markovian. Bidirected edges never contribute
//! ancestry — only directed ones do.
//!
//! ```
//! use why_data::graph::dseparation::Admg;
//!
//! // The chain C -> S -> W, with C -> R -> W alongside it.
//! let mut g = Admg::new();
//! g.directed("C", "S");
//! g.directed("C", "R");
//! g.directed("S", "W");
//! g.directed("R", "W");
//! g.directed("W", "L");
//!
//! // R and L are connected through W, which blocks both paths once observed.
//! assert!(!g.is_d_separated(&["R"], &["L"], &[]).unwrap());
//! assert!(g.is_d_separated(&["R"], &["L"], &["W"]).unwrap());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use super::{DiGraph, EdgeRef, NodeIndex};
use petgraph::Direction::{Incoming, Outgoing};

/// How two variables are connected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// `from -> to`: a direct causal effect.
    Directed,
    /// `from <-> to`: an unobserved common cause. Symmetric despite being
    /// stored with an orientation, and never a source of ancestry.
    Bidirected,
}

/// What a d-separation query can get wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DSepError {
    /// A queried name is not in the graph.
    Unknown(String),
    /// `X`, `Y` and `Z` must be pairwise disjoint.
    Overlap(String),
    /// `X` and `Y` must be non-empty.
    Empty(&'static str),
    /// The directed edges contain a cycle, naming one variable on it.
    Cyclic(String),
}

impl fmt::Display for DSepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(v) => write!(f, "{v} is not a variable of this graph"),
            Self::Overlap(v) => write!(f, "{v} appears in more than one of X, Y and Z"),
            Self::Empty(s) => write!(f, "{s} must not be empty"),
            Self::Cyclic(v) => write!(f, "the directed edges are cyclic at {v}"),
        }
    }
}

impl std::error::Error for DSepError {}

/// One edge as traversed along a path, oriented relative to the direction of
/// travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathEdge {
    /// `left -> right`.
    Forward,
    /// `left <- right`.
    Backward,
    /// `left <-> right`.
    Bidirected,
}

impl PathEdge {
    /// Whether this edge puts an arrowhead on its right-hand endpoint.
    fn arrow_right(self) -> bool {
        matches!(self, Self::Forward | Self::Bidirected)
    }

    /// Whether this edge puts an arrowhead on its left-hand endpoint.
    fn arrow_left(self) -> bool {
        matches!(self, Self::Backward | Self::Bidirected)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Forward => " -> ",
            Self::Backward => " <- ",
            Self::Bidirected => " <-> ",
        }
    }
}

/// Why a path is blocked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Blocker {
    /// A chain or fork whose middle variable is in `Z`.
    NonCollider(String),
    /// A collider that is not in `Z` and has no descendant in `Z`.
    Collider(String),
}

impl fmt::Display for Blocker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCollider(v) => write!(f, "non-collider {v} is in Z"),
            Self::Collider(v) => write!(f, "collider {v} has no descendant in Z"),
        }
    }
}

/// A path between two variables, and whether `Z` blocks it.
#[derive(Clone, Debug)]
pub struct Path {
    /// The variables along the path, endpoints included.
    pub nodes: Vec<String>,
    /// The edges between them; one shorter than `nodes`.
    pub edges: Vec<PathEdge>,
    /// The first blocking variable found, or `None` if the path is open.
    pub blocked_by: Option<Blocker>,
}

impl Path {
    /// Whether `Z` leaves this path open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.blocked_by.is_none()
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, node) in self.nodes.iter().enumerate() {
            if i > 0 {
                f.write_str(self.edges[i - 1].as_str())?;
            }
            f.write_str(node)?;
        }
        Ok(())
    }
}

/// A validated query: the resolved `X`, `Y` and `Z`.
type Query = (Vec<NodeIndex>, Vec<NodeIndex>, Vec<NodeIndex>);

/// A vertex of the reachability graph built by [`Admg::is_d_separated`].
///
/// Step 4 of the algorithm joins every pair of variables colliding on a common
/// child, which is a clique over that child's parents and so `Theta(n^2)` edges
/// in the worst case. Line 5 only asks whether a path *exists*, and for
/// connectivity a star is interchangeable with a clique, so the parents are
/// instead wired to one [`Vertex::Hub`] apiece. Any two of them still reach each
/// other, now in two steps rather than one, and no pair becomes connected that
/// the clique would not have connected. That is what keeps step 4 — and the
/// whole procedure — linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Vertex {
    /// A variable of the graph.
    Var(NodeIndex),
    /// A stand-in for one run of colliders, adjacent to exactly the variables
    /// that collide on it.
    Hub(usize),
}

/// An edge seen from one of its endpoints.
#[derive(Clone, Copy, Debug)]
struct Incident {
    other: NodeIndex,
    /// Arrowhead at the endpoint we are looking from.
    arrow_at_this: bool,
    /// Arrowhead at the other endpoint.
    arrow_at_other: bool,
}

/// An acyclic directed mixed graph — a causal diagram that may have unobserved
/// common causes.
#[derive(Clone, Debug, Default)]
pub struct Admg {
    graph: DiGraph<String, EdgeKind>,
    index: HashMap<String, NodeIndex>,
}

impl Admg {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a variable, or returns the existing one.
    pub fn add_node(&mut self, name: &str) -> NodeIndex {
        if let Some(idx) = self.index.get(name) {
            return *idx;
        }
        let idx = self.graph.add_node(name.to_string());
        self.index.insert(name.to_string(), idx);
        idx
    }

    /// Adds an edge, creating either endpoint if it is new.
    pub fn add_edge(&mut self, from: &str, to: &str, kind: EdgeKind) {
        let a = self.add_node(from);
        let b = self.add_node(to);
        self.graph.add_edge(a, b, kind);
    }

    /// Adds `from -> to`.
    pub fn directed(&mut self, from: &str, to: &str) {
        self.add_edge(from, to, EdgeKind::Directed);
    }

    /// Adds `a <-> b`.
    pub fn bidirected(&mut self, a: &str, b: &str) {
        self.add_edge(a, b, EdgeKind::Bidirected);
    }

    /// The underlying graph.
    #[must_use]
    pub fn graph(&self) -> &DiGraph<String, EdgeKind> {
        &self.graph
    }

    /// The index of a variable.
    #[must_use]
    pub fn node(&self, name: &str) -> Option<NodeIndex> {
        self.index.get(name).copied()
    }

    /// The variables, in insertion order.
    #[must_use]
    pub fn variables(&self) -> Vec<&str> {
        self.graph.node_weights().map(String::as_str).collect()
    }

    fn resolve(&self, names: &[&str]) -> Result<Vec<NodeIndex>, DSepError> {
        names
            .iter()
            .map(|n| self.node(n).ok_or_else(|| DSepError::Unknown((*n).to_string())))
            .collect()
    }

    /// Every edge incident to `n`, from `n`'s point of view.
    ///
    /// A bidirected edge is stored with an orientation but is symmetric, so it
    /// is reported the same way from either end.
    fn incident(&self, n: NodeIndex) -> Vec<Incident> {
        let mut out = Vec::new();
        for e in self.graph.edges_directed(n, Outgoing) {
            out.push(match e.weight() {
                EdgeKind::Directed => Incident {
                    other: e.target(),
                    arrow_at_this: false,
                    arrow_at_other: true,
                },
                EdgeKind::Bidirected => Incident {
                    other: e.target(),
                    arrow_at_this: true,
                    arrow_at_other: true,
                },
            });
        }
        for e in self.graph.edges_directed(n, Incoming) {
            out.push(match e.weight() {
                EdgeKind::Directed => Incident {
                    other: e.source(),
                    arrow_at_this: true,
                    arrow_at_other: false,
                },
                EdgeKind::Bidirected => Incident {
                    other: e.source(),
                    arrow_at_this: true,
                    arrow_at_other: true,
                },
            });
        }
        out
    }

    /// `An(seeds)` — the seeds together with all their ancestors.
    ///
    /// Only directed edges carry ancestry, which is why this cannot reuse
    /// [`super::CausalGraphExt::ancestors`]: that walks every incoming edge and
    /// would follow bidirected ones too.
    fn ancestors(&self, seeds: impl IntoIterator<Item = NodeIndex>) -> HashSet<NodeIndex> {
        let mut seen = HashSet::new();
        let mut stack: Vec<NodeIndex> = seeds.into_iter().collect();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            for e in self.graph.edges_directed(n, Incoming) {
                if *e.weight() == EdgeKind::Directed {
                    stack.push(e.source());
                }
            }
        }
        seen
    }

    /// `An(names)` by name, for inspection.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] if a name is not in the graph.
    pub fn ancestors_of(&self, names: &[&str]) -> Result<Vec<&str>, DSepError> {
        let seeds = self.resolve(names)?;
        let set = self.ancestors(seeds);
        let mut out: Vec<&str> = set.iter().map(|n| self.graph[*n].as_str()).collect();
        out.sort_unstable();
        Ok(out)
    }

    /// `De(seeds)` — the seeds together with all their descendants, following
    /// directed edges only.
    fn descendants(&self, seeds: impl IntoIterator<Item = NodeIndex>) -> HashSet<NodeIndex> {
        let mut seen = HashSet::new();
        let mut stack: Vec<NodeIndex> = seeds.into_iter().collect();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            for e in self.graph.edges_directed(n, Outgoing) {
                if *e.weight() == EdgeKind::Directed {
                    stack.push(e.target());
                }
            }
        }
        seen
    }

    /// `De(names)` by name, including the variables themselves.
    ///
    /// A bidirected edge is not a causal path, so it contributes no descendants.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] if a name is not in the graph.
    pub fn descendants_of(&self, names: &[&str]) -> Result<Vec<&str>, DSepError> {
        let seeds = self.resolve(names)?;
        let set = self.descendants(seeds);
        let mut out: Vec<&str> = set.iter().map(|n| self.graph[*n].as_str()).collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Rebuilds the graph keeping only the edges `keep` accepts.
    ///
    /// Every variable is preserved, so node names stay valid across surgery.
    fn filter_edges(&self, keep: impl Fn(NodeIndex, NodeIndex, EdgeKind) -> bool) -> Self {
        let mut out = Self::new();
        // Insert in the original order so `variables()` is stable.
        for n in self.graph.node_indices() {
            out.add_node(&self.graph[n]);
        }
        for e in self.graph.edge_references() {
            if keep(e.source(), e.target(), *e.weight()) {
                out.add_edge(&self.graph[e.source()], &self.graph[e.target()], *e.weight());
            }
        }
        out
    }

    /// `G` with every arrowhead into `nodes` removed — the graph written
    /// `G` with `X` overlined, as produced by intervening on `X`.
    ///
    /// Bidirected edges touching `nodes` go too: `V <-> X` carries an arrowhead
    /// into `X` from an unobserved parent, and an intervention severs that just
    /// as it severs a directed one.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] if a name is not in the graph.
    pub fn cut_edges_into(&self, nodes: &[&str]) -> Result<Self, DSepError> {
        let cut: HashSet<NodeIndex> = self.resolve(nodes)?.into_iter().collect();
        Ok(self.filter_edges(|source, target, kind| match kind {
            EdgeKind::Directed => !cut.contains(&target),
            EdgeKind::Bidirected => !cut.contains(&source) && !cut.contains(&target),
        }))
    }

    /// `G` with every directed edge leaving `nodes` removed — the graph written
    /// `G` with `X` underlined.
    ///
    /// Bidirected edges stay: they have no tail at either endpoint, so nothing
    /// leaves `X` along one.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] if a name is not in the graph.
    pub fn cut_edges_out_of(&self, nodes: &[&str]) -> Result<Self, DSepError> {
        let cut: HashSet<NodeIndex> = self.resolve(nodes)?.into_iter().collect();
        Ok(self.filter_edges(|source, _, kind| {
            kind != EdgeKind::Directed || !cut.contains(&source)
        }))
    }

    /// The subgraph induced on `nodes`: those variables and only the edges with
    /// both endpoints among them.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] if a name is not in the graph.
    pub fn induced_subgraph(&self, nodes: &[&str]) -> Result<Self, DSepError> {
        let keep: HashSet<NodeIndex> = self.resolve(nodes)?.into_iter().collect();
        let mut out = Self::new();
        // Preserve the original relative order of the survivors.
        for n in self.graph.node_indices() {
            if keep.contains(&n) {
                out.add_node(&self.graph[n]);
            }
        }
        for e in self.graph.edge_references() {
            if keep.contains(&e.source()) && keep.contains(&e.target()) {
                out.add_edge(&self.graph[e.source()], &self.graph[e.target()], *e.weight());
            }
        }
        Ok(out)
    }

    /// A topological order of the variables, parents before children.
    ///
    /// Only directed edges constrain the order; bidirected ones express
    /// confounding, not precedence.
    ///
    /// # Errors
    ///
    /// [`DSepError::Cyclic`] if the directed edges contain a cycle.
    pub fn topological_order(&self) -> Result<Vec<String>, DSepError> {
        let mut directed = DiGraph::<String, ()>::new();
        let mut index = HashMap::new();
        for n in self.graph.node_indices() {
            index.insert(n, directed.add_node(self.graph[n].clone()));
        }
        for e in self.graph.edge_references() {
            if *e.weight() == EdgeKind::Directed {
                directed.add_edge(index[&e.source()], index[&e.target()], ());
            }
        }
        petgraph::algo::toposort(&directed, None)
            .map(|order| order.into_iter().map(|n| directed[n].clone()).collect())
            .map_err(|cycle| DSepError::Cyclic(directed[cycle.node_id()].clone()))
    }

    /// The c-components (districts) of the graph: the classes of variables
    /// connected to one another by bidirected edges.
    ///
    /// A variable touched by no bidirected edge forms a component of its own, so
    /// the components always partition the vertex set. `C(G) == {V}` — a single
    /// component covering everything — is the condition that makes the ID
    /// algorithm report a hedge.
    #[must_use]
    pub fn c_components(&self) -> Vec<Vec<String>> {
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        let mut out = Vec::new();
        for start in self.graph.node_indices() {
            if !seen.insert(start) {
                continue;
            }
            let mut component = vec![start];
            let mut stack = vec![start];
            while let Some(cur) = stack.pop() {
                for e in self
                    .graph
                    .edges_directed(cur, Outgoing)
                    .chain(self.graph.edges_directed(cur, Incoming))
                {
                    if *e.weight() != EdgeKind::Bidirected {
                        continue;
                    }
                    let other = if e.source() == cur { e.target() } else { e.source() };
                    if seen.insert(other) {
                        component.push(other);
                        stack.push(other);
                    }
                }
            }
            let mut names: Vec<String> =
                component.into_iter().map(|n| self.graph[n].clone()).collect();
            names.sort();
            out.push(names);
        }
        out.sort();
        out
    }

    fn check_query(&self, x: &[&str], y: &[&str], z: &[&str]) -> Result<Query, DSepError> {
        if x.is_empty() {
            return Err(DSepError::Empty("X"));
        }
        if y.is_empty() {
            return Err(DSepError::Empty("Y"));
        }
        let (xs, ys, zs) = (self.resolve(x)?, self.resolve(y)?, self.resolve(z)?);
        for (a, b) in [(&xs, &ys), (&xs, &zs), (&ys, &zs)] {
            if let Some(dup) = a.iter().find(|n| b.contains(n)) {
                return Err(DSepError::Overlap(self.graph[*dup].clone()));
            }
        }
        Ok((xs, ys, zs))
    }

    /// Whether `Z` d-separates `X` from `Y`.
    ///
    /// Implements the ancestral moralization test:
    ///
    /// 1. `A = An(X, Y, Z)`.
    /// 2. Build an undirected graph over `A`.
    /// 3. Join any two variables of `A \ Z` that are adjacent in the original
    ///    graph, by an edge of either kind.
    /// 4. Join `V_i` and `V_j` whenever they both point into some
    ///    `V_k in An(Z)` — that is, for every collider structure
    ///    `V_i -> V_k <- V_j`, `V_i <-> V_k <- V_j`, `V_i -> V_k <-> V_j` or
    ///    `V_i <-> V_k <-> V_j`.
    /// 5. `X` and `Y` are d-separated exactly when no variable of `Y` is
    ///    reachable from `X`.
    ///
    /// Two departures from that statement, both needed for correctness:
    ///
    /// **Step 4 is restricted to `V_i, V_j in A \ Z`**, not merely `A`.
    /// Admitting a `V_i in Z` would connect `X -> V_k <- z -> V_m <- Y` through
    /// `z`, but that path is blocked at `z`, a fork lying in `Z` — so the
    /// unrestricted form reports d-connection where the definition says
    /// d-separation.
    ///
    /// **Step 4 marries across whole collider paths**, not one `V_k` at a time.
    /// Marrying the parents of a single collider is complete for a DAG, where
    /// two colliders can never be adjacent on a path: the edge between them has
    /// one arrowhead, so at least one of the pair sees a tail. A bidirected edge
    /// has an arrowhead at *both* ends, so in a mixed graph colliders can be
    /// adjacent, and a run of them opens when each lies in `An(Z)`. The
    /// single-collider form misses `X -> W <-> Y <- Z` with `W, Y in Z`: both
    /// interior variables are open colliders, so the path is open, yet no single
    /// marriage connects `X` to `Z` once `W` and `Y` are removed. See
    /// `collider_runs`.
    ///
    /// A test cross-checks this against [`Admg::paths`] over every choice of
    /// `X`, `Y` and `Z` in three graphs, which is how the second departure was
    /// found.
    ///
    /// # Complexity
    ///
    /// `O(n + m)` time and space. Step 1 is one reverse traversal; step 3 is one
    /// pass over the edges; `collider_runs` is one more traversal; step
    /// 4 is one pass over the edges incident to `An(Z)`; step 5 is a BFS.
    ///
    /// Step 4 is the only step where that is not obvious, because the moral
    /// graph it describes is *not* linear in size: joining every pair of
    /// variables that collide on a common child is a clique over that child's
    /// parents, so the moral graph can have `Theta(n^2)` edges — a single
    /// variable with `n - 1` parents does it, from only `n - 1` input edges.
    ///
    /// Step 5 asks only whether a path *exists*, and for connectivity a star is
    /// interchangeable with a clique. So the clique is never materialised: the
    /// parents are wired to one `Vertex::Hub` instead, giving
    /// `sum deg(V_k) = O(m)` edges in place of `sum C(k, 2)`. Two parents still
    /// reach each other, in two hops rather than one, and the hub introduces no
    /// connection the clique would not have made, since any path through it
    /// enters and leaves at variables the clique joins directly.
    ///
    /// # Errors
    ///
    /// [`DSepError::Empty`] if `X` or `Y` is empty, [`DSepError::Overlap`] if
    /// the three sets are not pairwise disjoint, and [`DSepError::Unknown`] for
    /// a name the graph does not have.
    pub fn is_d_separated(&self, x: &[&str], y: &[&str], z: &[&str]) -> Result<bool, DSepError> {
        let (xs, ys, zs) = self.check_query(x, y, z)?;

        // 1. A = An(X, Y, Z)
        let seeds: Vec<NodeIndex> = xs.iter().chain(&ys).chain(&zs).copied().collect();
        let ancestral = self.ancestors(seeds);
        let z_set: HashSet<NodeIndex> = zs.iter().copied().collect();
        // 2. The moral graph's vertices: A minus the conditioning set.
        let keep: HashSet<NodeIndex> = ancestral.difference(&z_set).copied().collect();

        let mut adjacent: HashMap<Vertex, Vec<Vertex>> = HashMap::new();
        let link = |a: Vertex, b: Vertex, adjacent: &mut HashMap<Vertex, Vec<Vertex>>| {
            adjacent.entry(a).or_default().push(b);
            adjacent.entry(b).or_default().push(a);
        };

        // 3. Adjacency in the original graph, of either edge kind. O(m).
        for e in self.graph.edge_references() {
            let (u, v) = (e.source(), e.target());
            if u != v && keep.contains(&u) && keep.contains(&v) {
                link(Vertex::Var(u), Vertex::Var(v), &mut adjacent);
            }
        }

        // 4. Join variables that are collider-connected through An(Z): those
        // linked by a path whose every interior variable is a collider lying in
        // An(Z). A single interior collider is the case the algorithm states;
        // longer runs matter only once bidirected edges let colliders be
        // adjacent. Each run gets one hub rather than a clique, so this costs
        // one pass over the incident edges of An(Z): O(m).
        let an_z = self.ancestors(zs.iter().copied());
        for (i, run) in self.collider_runs(&an_z).into_iter().enumerate() {
            // Everything with an arrowhead into a member of the run. Members
            // themselves qualify — a run entered at one member can end at
            // another, which is what `X -> W <-> Y` does with the run {W, Y}.
            let hub = Vertex::Hub(i);
            let mut joined: HashSet<NodeIndex> = HashSet::new();
            for vk in &run {
                for inc in self.incident(*vk) {
                    if inc.arrow_at_this
                        && keep.contains(&inc.other)
                        && joined.insert(inc.other)
                    {
                        link(Vertex::Var(inc.other), hub, &mut adjacent);
                    }
                }
            }
        }

        // 5. Reachability from X to Y. O(n + m), the graph having O(n) vertices
        // and O(m) edges by construction.
        let targets: HashSet<NodeIndex> = ys.iter().copied().collect();
        let mut seen: HashSet<Vertex> = HashSet::new();
        let mut queue: VecDeque<Vertex> = xs
            .iter()
            .copied()
            .filter(|n| keep.contains(n))
            .map(Vertex::Var)
            .collect();
        seen.extend(queue.iter().copied());
        while let Some(v) = queue.pop_front() {
            if let Vertex::Var(n) = v
                && targets.contains(&n)
            {
                return Ok(false);
            }
            for next in adjacent.get(&v).into_iter().flatten() {
                if seen.insert(*next) {
                    queue.push_back(*next);
                }
            }
        }
        Ok(true)
    }

    /// Groups `An(Z)` into runs of colliders that a single path can cross
    /// end to end.
    ///
    /// Two colliders are adjacent on a path only if the edge between them
    /// carries an arrowhead at each end — that is, only if it is bidirected. So
    /// the runs are exactly the connected components of the bidirected edges
    /// whose endpoints both lie in `An(Z)`.
    ///
    /// Without bidirected edges every run is a singleton and step 4 reduces to
    /// marrying the parents of one collider, which is the form the algorithm
    /// states and is complete for a DAG.
    ///
    /// Computed once for the whole query, in `O(n + m)`.
    fn collider_runs(&self, an_z: &HashSet<NodeIndex>) -> Vec<HashSet<NodeIndex>> {
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        let mut out = Vec::new();
        for start in an_z {
            if !seen.insert(*start) {
                continue;
            }
            let mut run = HashSet::from([*start]);
            let mut stack = vec![*start];
            while let Some(cur) = stack.pop() {
                for e in self
                    .graph
                    .edges_directed(cur, Outgoing)
                    .chain(self.graph.edges_directed(cur, Incoming))
                {
                    if *e.weight() != EdgeKind::Bidirected {
                        continue;
                    }
                    let other = if e.source() == cur { e.target() } else { e.source() };
                    if an_z.contains(&other) && seen.insert(other) {
                        run.insert(other);
                        stack.push(other);
                    }
                }
            }
            out.push(run);
        }
        out
    }

    /// Every path between `x` and `y`, each labelled with what blocks it.
    ///
    /// Follows Definition 2.4.3: a path is blocked when a chain or fork has its
    /// middle variable in `Z`, or when a collider has neither itself nor any
    /// descendant in `Z`. Paths are simple — no variable repeats.
    ///
    /// This enumerates paths explicitly, so it is exponential in the worst
    /// case. Use [`Admg::is_d_separated`] to decide the question on a large
    /// graph; use this to explain the answer on a small one.
    ///
    /// # Errors
    ///
    /// As [`Admg::is_d_separated`].
    pub fn paths(&self, x: &str, y: &str, z: &[&str]) -> Result<Vec<Path>, DSepError> {
        let (xs, ys, zs) = self.check_query(&[x], &[y], z)?;
        let (start, goal) = (xs[0], ys[0]);
        let z_set: HashSet<NodeIndex> = zs.iter().copied().collect();
        // A collider opens when it, or any descendant, is observed — which is
        // the same as the collider being an ancestor of Z.
        let an_z = self.ancestors(zs.iter().copied());

        let mut out = Vec::new();
        let mut nodes = vec![start];
        let mut edges = Vec::new();
        let mut on_path: HashSet<NodeIndex> = HashSet::from([start]);
        self.walk(
            start,
            goal,
            &z_set,
            &an_z,
            &mut nodes,
            &mut edges,
            &mut on_path,
            &mut out,
        );
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        at: NodeIndex,
        goal: NodeIndex,
        z: &HashSet<NodeIndex>,
        an_z: &HashSet<NodeIndex>,
        nodes: &mut Vec<NodeIndex>,
        edges: &mut Vec<PathEdge>,
        on_path: &mut HashSet<NodeIndex>,
        out: &mut Vec<Path>,
    ) {
        if at == goal {
            out.push(Path {
                nodes: nodes.iter().map(|n| self.graph[*n].clone()).collect(),
                edges: edges.clone(),
                blocked_by: Self::first_blocker(&self.graph, nodes, edges, z, an_z),
            });
            return;
        }
        for inc in self.incident(at) {
            if !on_path.insert(inc.other) {
                continue;
            }
            let edge = match (inc.arrow_at_this, inc.arrow_at_other) {
                (true, true) => PathEdge::Bidirected,
                (false, true) => PathEdge::Forward,
                _ => PathEdge::Backward,
            };
            nodes.push(inc.other);
            edges.push(edge);
            self.walk(inc.other, goal, z, an_z, nodes, edges, on_path, out);
            nodes.pop();
            edges.pop();
            on_path.remove(&inc.other);
        }
    }

    /// The first interior variable that blocks the path, if any.
    fn first_blocker(
        graph: &DiGraph<String, EdgeKind>,
        nodes: &[NodeIndex],
        edges: &[PathEdge],
        z: &HashSet<NodeIndex>,
        an_z: &HashSet<NodeIndex>,
    ) -> Option<Blocker> {
        for i in 1..nodes.len().saturating_sub(1) {
            let node = nodes[i];
            let name = || graph[node].clone();
            // Arrowheads on both sides make it a collider.
            if edges[i - 1].arrow_right() && edges[i].arrow_left() {
                if !an_z.contains(&node) {
                    return Some(Blocker::Collider(name()));
                }
            } else if z.contains(&node) {
                return Some(Blocker::NonCollider(name()));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Figure 2.16's Markovian graph: C -> S, C -> R, S -> W, R -> W, W -> L.
    fn fig_2_16() -> Admg {
        let mut g = Admg::new();
        g.directed("C", "S");
        g.directed("C", "R");
        g.directed("S", "W");
        g.directed("R", "W");
        g.directed("W", "L");
        g
    }

    /// The non-Markovian variant: S <-> R, S -> W, R -> W, W -> L.
    fn fig_2_16_latent() -> Admg {
        let mut g = Admg::new();
        g.bidirected("S", "R");
        g.directed("S", "W");
        g.directed("R", "W");
        g.directed("W", "L");
        g
    }

    /// Figure 2.17, in which no set d-separates X from Y.
    fn fig_2_17() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "W");
        g.directed("T", "X");
        g.directed("T", "Z");
        g.directed("W", "Z");
        g.directed("Z", "Y");
        g.directed("R", "Y");
        g.directed("Y", "S");
        g.bidirected("W", "Y");
        g
    }

    #[test]
    fn chain_is_blocked_by_its_middle() {
        let mut g = Admg::new();
        g.directed("x", "z");
        g.directed("z", "y");
        assert!(!g.is_d_separated(&["x"], &["y"], &[]).unwrap());
        assert!(g.is_d_separated(&["x"], &["y"], &["z"]).unwrap());
    }

    #[test]
    fn fork_is_blocked_by_its_centre() {
        let mut g = Admg::new();
        g.directed("z", "x");
        g.directed("z", "y");
        assert!(!g.is_d_separated(&["x"], &["y"], &[]).unwrap());
        assert!(g.is_d_separated(&["x"], &["y"], &["z"]).unwrap());
    }

    #[test]
    fn collider_is_opened_by_conditioning() {
        let mut g = Admg::new();
        g.directed("x", "z");
        g.directed("y", "z");
        assert!(g.is_d_separated(&["x"], &["y"], &[]).unwrap());
        assert!(!g.is_d_separated(&["x"], &["y"], &["z"]).unwrap());
    }

    #[test]
    fn collider_is_opened_by_conditioning_on_a_descendant() {
        let mut g = Admg::new();
        g.directed("x", "z");
        g.directed("y", "z");
        g.directed("z", "w");
        assert!(g.is_d_separated(&["x"], &["y"], &[]).unwrap());
        assert!(!g.is_d_separated(&["x"], &["y"], &["w"]).unwrap());
    }

    #[test]
    fn bidirected_edges_open_paths_but_carry_no_ancestry() {
        let mut g = Admg::new();
        g.bidirected("x", "y");
        assert!(!g.is_d_separated(&["x"], &["y"], &[]).unwrap());

        // x <-> z <-> y is a collider at z, so it is closed until z is observed.
        let mut g = Admg::new();
        g.bidirected("x", "z");
        g.bidirected("y", "z");
        assert!(g.is_d_separated(&["x"], &["y"], &[]).unwrap());
        assert!(!g.is_d_separated(&["x"], &["y"], &["z"]).unwrap());

        // A bidirected edge is not a causal path, so z is not an ancestor of y.
        let mut g = Admg::new();
        g.bidirected("z", "y");
        assert_eq!(g.ancestors_of(&["y"]).unwrap(), vec!["y"]);
    }

    /// Example 2.15 — S and W are adjacent, so nothing separates them.
    #[test]
    fn example_2_15() {
        let g = fig_2_16();
        assert!(!g.is_d_separated(&["S"], &["W"], &[]).unwrap());
        let paths = g.paths("S", "W", &[]).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(Path::is_open));
    }

    /// Example 2.16 — W blocks both paths from R to L.
    #[test]
    fn example_2_16() {
        let g = fig_2_16();
        assert!(!g.is_d_separated(&["R"], &["L"], &[]).unwrap());
        assert!(g.is_d_separated(&["R"], &["L"], &["W"]).unwrap());

        let paths = g.paths("R", "L", &["W"]).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|p| !p.is_open()));
        assert!(
            paths
                .iter()
                .all(|p| matches!(p.blocked_by, Some(Blocker::NonCollider(ref v)) if v == "W"))
        );
    }

    /// Example 2.17 — the same question with a latent common cause of S and R.
    #[test]
    fn example_2_17() {
        let g = fig_2_16_latent();
        assert!(!g.is_d_separated(&["S"], &["L"], &[]).unwrap());
        let paths = g.paths("S", "L", &[]).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(Path::is_open));
        assert!(paths.iter().any(|p| p.to_string() == "S <-> R -> W -> L"));
    }

    /// Example 2.18 — no set d-separates X from Y in figure 2.17.
    #[test]
    fn example_2_18() {
        let g = fig_2_17();
        let candidates: &[&[&str]] = &[
            &[],
            &["T"],
            &["W"],
            &["Z"],
            &["R"],
            &["S"],
            &["T", "W"],
            &["T", "Z"],
            &["T", "R"],
            &["T", "W", "Z"],
            &["T", "W", "Z", "R"],
            &["T", "W", "Z", "R", "S"],
        ];
        for z in candidates {
            assert!(
                !g.is_d_separated(&["X"], &["Y"], z).unwrap(),
                "no set should d-separate X from Y, but {z:?} did"
            );
        }
    }

    /// The moralization test and the path enumeration must never disagree.
    #[test]
    fn the_two_methods_agree() {
        for g in [fig_2_16(), fig_2_16_latent(), fig_2_17()] {
            let names: Vec<String> = g.variables().iter().map(|s| (*s).to_string()).collect();
            for x in &names {
                for y in &names {
                    if x == y {
                        continue;
                    }
                    // Every conditioning set drawn from the remaining variables.
                    let rest: Vec<&str> = names
                        .iter()
                        .filter(|n| *n != x && *n != y)
                        .map(String::as_str)
                        .collect();
                    for mask in 0..(1u32 << rest.len()) {
                        let z: Vec<&str> = rest
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| mask & (1 << i) != 0)
                            .map(|(_, n)| *n)
                            .collect();
                        let separated = g.is_d_separated(&[x], &[y], &z).unwrap();
                        let no_open_path =
                            g.paths(x, y, &z).unwrap().iter().all(|p| !p.is_open());
                        assert_eq!(
                            separated, no_open_path,
                            "disagreement for X={x}, Y={y}, Z={z:?}"
                        );
                    }
                }
            }
        }
    }

    /// A wide collider is where the hub stands in for a clique: conditioning on
    /// the child must connect every pair of its parents, and conditioning on
    /// nothing must leave them all separated.
    #[test]
    fn a_wide_collider_connects_every_pair_of_parents() {
        const K: usize = 12;
        let parents: Vec<String> = (0..K).map(|i| format!("p{i}")).collect();
        let mut g = Admg::new();
        for p in &parents {
            g.directed(p, "c");
        }

        for (i, a) in parents.iter().enumerate() {
            for b in &parents[i + 1..] {
                assert!(
                    g.is_d_separated(&[a], &[b], &[]).unwrap(),
                    "{a} and {b} should be separated with nothing observed"
                );
                assert!(
                    !g.is_d_separated(&[a], &[b], &["c"]).unwrap(),
                    "{a} and {b} should be connected once c is observed"
                );
            }
        }

        // Two parents stay connected through the hub even when every other
        // parent is also conditioned on.
        let rest: Vec<&str> = parents[2..].iter().map(String::as_str).collect();
        assert!(
            !g.is_d_separated(&[&parents[0]], &[&parents[1]], &["c"]).unwrap()
        );
        let mut with_rest = vec!["c"];
        with_rest.extend(rest);
        assert!(
            !g.is_d_separated(&[&parents[0]], &[&parents[1]], &with_rest).unwrap()
        );
    }

    #[test]
    fn queries_are_validated() {
        let g = fig_2_16();
        assert_eq!(
            g.is_d_separated(&["S"], &["W"], &["nope"]).unwrap_err(),
            DSepError::Unknown("nope".to_string())
        );
        assert_eq!(
            g.is_d_separated(&["S"], &["W"], &["S"]).unwrap_err(),
            DSepError::Overlap("S".to_string())
        );
        assert_eq!(
            g.is_d_separated(&[], &["W"], &[]).unwrap_err(),
            DSepError::Empty("X")
        );
    }
}
