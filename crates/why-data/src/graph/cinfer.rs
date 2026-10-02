//! C-INFER: causal effect identification by causal factor trees.
//!
//! This is the identification engine of the Causal AI book's chapter 4,
//! section 4.4. Where [`super::identification`] runs the Shpitser-Pearl ID
//! recursion against observational data alone, C-INFER also accepts
//! *experimental* inputs `P(v | do(z))`, and it lays its reasoning out as data:
//!
//! - a **q-tree** ([`gen_query_tree`]) decomposes the query into the
//!   c-factors it needs;
//! - one **d-tree** per input ([`gen_input_tree`]) records which c-factors that
//!   input can produce;
//! - [`map_factors`] matches the q-tree's leaves to d-tree nodes, and
//!   [`compose_query`] reads the estimand off the matched paths.
//!
//! Every edge of either tree is one of four c-operators ([`Operator`]):
//!
//! | Operator | Rewrites | Licensed when |
//! | --- | --- | --- |
//! | `Σ` | `Q[T] -> Q[C] = sum_{T \ C} Q[T]` | `C` is ancestral in `G[T]` |
//! | `δ` | `Q[T] -> Q[C]` by Tian's factorisation | `C` is a c-component of `G[T]` |
//! | `σ` | `Q[T | do(z)] -> Q[T | do(x)]` | `T` meets neither `X` nor `Z` |
//! | `Γ` | `Q[A | do(x)] -> P(y | do(x), w)` | `A` as in [`gen_query_tree`] |
//!
//! With observational data alone this is Tian's identification algorithm and
//! succeeds exactly where [`super::identification::identify`] does; a test
//! checks that over hundreds of random graphs, and checks every estimand
//! numerically against the true effect of a random model. With experiments it
//! is the per-factor general identification (gID) of Lee, Correa and
//! Bareinboim, which identifies each target c-factor from whichever single
//! input can produce it.
//!
//! ```
//! use why_data::graph::cinfer::{Input, Query, c_infer};
//! use why_data::graph::dseparation::Admg;
//!
//! // The front-door graph.
//! let mut g = Admg::new();
//! g.directed("X", "Z");
//! g.directed("Z", "Y");
//! g.bidirected("X", "Y");
//!
//! let d = c_infer(&g, &Query::new(&["X"], &["Y"]), &[Input::observational()]).unwrap();
//! assert_eq!(
//!     d.estimand.to_string(),
//!     "sum_{z} P(z | x) sum_{x'} P(x') P(y | x', z)"
//! );
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use super::EdgeRef;
use super::dseparation::{Admg, DSepError};
use super::identification::{Dist, Formula};

/// A set of variable names.
pub type VarSet = BTreeSet<String>;

fn var_set(names: &[&str]) -> VarSet {
    names.iter().map(ToString::to_string).collect()
}

fn refs(s: &VarSet) -> Vec<&str> {
    s.iter().map(String::as_str).collect()
}

fn join(s: &VarSet) -> String {
    s.iter().cloned().collect::<Vec<_>>().join(", ")
}

/// A c-factor `Q[scope | do(doing)]`: the joint of `scope` given its parents,
/// in the model where `doing` has been intervened on, with the unobserved
/// confounders of `scope` summed out.
///
/// `Q[V] = P(V)`, and `Q[V \ Z | do(z)] = P(V \ Z | do(z))`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CFactor {
    /// The variables the factor is a joint over.
    pub scope: VarSet,
    /// The intervention defining the regime; empty for observational data.
    pub doing: VarSet,
}

impl fmt::Display for CFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.doing.is_empty() {
            write!(f, "Q[{}]", join(&self.scope))
        } else {
            write!(f, "Q[{} | do({})]", join(&self.scope), join(&self.doing))
        }
    }
}

/// The target `P(y | do(x), w)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// The outcomes.
    pub y: VarSet,
    /// The intervention.
    pub x: VarSet,
    /// What is conditioned on; empty for an unconditional effect.
    pub w: VarSet,
}

impl Query {
    /// `P(y | do(x))`.
    #[must_use]
    pub fn new(x: &[&str], y: &[&str]) -> Self {
        Self {
            y: var_set(y),
            x: var_set(x),
            w: VarSet::new(),
        }
    }

    /// The same query, additionally conditioned on `w`.
    #[must_use]
    pub fn given(mut self, w: &[&str]) -> Self {
        self.w = var_set(w);
        self
    }

    fn validate(&self, g: &Admg) -> Result<(), DSepError> {
        if self.y.is_empty() {
            return Err(DSepError::Empty("Y"));
        }
        for v in self.y.iter().chain(&self.x).chain(&self.w) {
            if g.node(v).is_none() {
                return Err(DSepError::Unknown(v.clone()));
            }
        }
        for (a, b) in [(&self.x, &self.y), (&self.x, &self.w), (&self.y, &self.w)] {
            if let Some(v) = a.intersection(b).next() {
                return Err(DSepError::Overlap(v.clone()));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut cond = Vec::new();
        if !self.x.is_empty() {
            cond.push(format!("do({})", join(&self.x)));
        }
        if !self.w.is_empty() {
            cond.push(join(&self.w));
        }
        if cond.is_empty() {
            write!(f, "P({})", join(&self.y))
        } else {
            write!(f, "P({} | {})", join(&self.y), cond.join(", "))
        }
    }
}

/// An available distribution: `P(V)` when `doing` is empty, the experimental
/// `P(V \ Z | do(z))` otherwise.
///
/// An experiment is taken to be available at every value of `Z`, as in gID: an
/// estimand may sum over a variable that one of its terms intervenes on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Input {
    /// The intervened variables `Z`.
    pub doing: VarSet,
}

impl Input {
    /// The observational distribution `P(V)`.
    #[must_use]
    pub fn observational() -> Self {
        Self::default()
    }

    /// The experimental distribution `P(V | do(z))`.
    #[must_use]
    pub fn experimental(z: &[&str]) -> Self {
        Self { doing: var_set(z) }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.doing.is_empty() {
            f.write_str("P(V)")
        } else {
            write!(f, "P(V | do({}))", join(&self.doing))
        }
    }
}

/// A c-operator: the label on an edge of a causal factor tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operator {
    /// `Σ`, marginalisation: the child is the parent with `sum_out` summed
    /// away, which is sound because what is left is ancestral in the parent's
    /// subgraph.
    Sigma {
        /// The variables summed out.
        sum_out: VarSet,
    },
    /// `δ`, c-component factorisation: `components` partitions the parent's
    /// scope into c-components and the child is one of them.
    Delta {
        /// The full partition the child was taken from.
        components: Vec<VarSet>,
    },
    /// `σ`, regime invariance: the same c-factor read in another regime.
    Regime {
        /// The intervention of the input.
        from: VarSet,
        /// The intervention of the query.
        to: VarSet,
    },
    /// `Γ`, unconditioning: the query `P(y | do(x), w)` is the child c-factor
    /// normalised over `y`.
    Gamma {
        /// The conditioning variables that remain relevant to `y`; the rest of
        /// `w` is dropped.
        given: VarSet,
    },
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sigma { sum_out } => write!(f, "Σ_{{{}}}", join(sum_out)),
            Self::Delta { .. } => f.write_str("δ"),
            Self::Regime { from, to } => {
                let regime = |s: &VarSet| {
                    if s.is_empty() {
                        "∅".to_string()
                    } else {
                        join(s)
                    }
                };
                write!(f, "σ do({}) → do({})", regime(from), regime(to))
            }
            Self::Gamma { given } => write!(f, "Γ | {}", join(given)),
        }
    }
}

/// What a tree node stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    /// The query, at the root of a q-tree.
    Query(Query),
    /// A c-factor.
    Factor(CFactor),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query(q) => q.fmt(f),
            Self::Factor(c) => c.fmt(f),
        }
    }
}

/// A node of a [`CTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CTreeNode {
    /// What the node stands for.
    pub term: Term,
    /// The operator on the edge from the parent; `None` at the root.
    pub op_from_parent: Option<Operator>,
    /// The parent's index; `None` at the root.
    pub parent: Option<usize>,
    /// The children's indices.
    pub children: Vec<usize>,
}

/// A causal factor tree.
///
/// In a q-tree the root is the query and evaluation flows *up*: each node is
/// computed from its children, by `Σ` or `Γ` from its single child or as the
/// product of its `δ` children. In a d-tree the root is an input and
/// evaluation flows *down*: each node is its parent with its edge's operator
/// applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CTree {
    /// Every node, the root first.
    pub nodes: Vec<CTreeNode>,
    /// The root's index.
    pub root: usize,
}

impl CTree {
    fn new(term: Term) -> Self {
        Self {
            nodes: vec![CTreeNode {
                term,
                op_from_parent: None,
                parent: None,
                children: Vec::new(),
            }],
            root: 0,
        }
    }

    /// Adds `term` under `parent`, or returns the existing child that already
    /// stands for it, so that d-tree paths for different leaves share their
    /// common prefix.
    fn add_child(&mut self, parent: usize, term: Term, op: Operator) -> usize {
        if let Some(&existing) = self.nodes[parent]
            .children
            .iter()
            .find(|&&c| self.nodes[c].term == term)
        {
            return existing;
        }
        let idx = self.nodes.len();
        self.nodes.push(CTreeNode {
            term,
            op_from_parent: Some(op),
            parent: Some(parent),
            children: Vec::new(),
        });
        self.nodes[parent].children.push(idx);
        idx
    }

    /// The c-factor at `idx`, if it holds one rather than the query.
    #[must_use]
    pub fn factor(&self, idx: usize) -> Option<&CFactor> {
        match &self.nodes[idx].term {
            Term::Factor(c) => Some(c),
            Term::Query(_) => None,
        }
    }

    /// The nodes without children, in insertion order.
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].children.is_empty())
            .collect()
    }

    /// The first node standing for `factor`.
    #[must_use]
    pub fn find(&self, factor: &CFactor) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| matches!(&n.term, Term::Factor(c) if c == factor))
    }

    /// The indices from the root down to `idx`, both included.
    #[must_use]
    pub fn path_to(&self, idx: usize) -> Vec<usize> {
        let mut path = vec![idx];
        let mut cur = idx;
        while let Some(p) = self.nodes[cur].parent {
            path.push(p);
            cur = p;
        }
        path.reverse();
        path
    }

    fn write_node(
        &self,
        f: &mut fmt::Formatter<'_>,
        idx: usize,
        prefix: &str,
        last: bool,
    ) -> fmt::Result {
        let node = &self.nodes[idx];
        let child_prefix = match &node.op_from_parent {
            None => {
                writeln!(f, "{}", node.term)?;
                String::new()
            }
            Some(op) => {
                let branch = if last { "└─" } else { "├─" };
                writeln!(f, "{prefix}{branch} [{op}] {}", node.term)?;
                format!("{prefix}{}", if last { "   " } else { "│  " })
            }
        };
        for (i, &c) in node.children.iter().enumerate() {
            self.write_node(f, c, &child_prefix, i + 1 == node.children.len())?;
        }
        Ok(())
    }
}

impl fmt::Display for CTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_node(f, self.root, "", true)
    }
}

/// Where a q-tree leaf's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The leaf is `Q[X_i | do(x)]` for an intervened `X_i`, which is the
    /// indicator `1[X_i = x_i]`: it fixes `X_i` to the query's value.
    Indicator,
    /// The leaf is node `node` of the d-tree of input `input`.
    Input {
        /// The index into the inputs.
        input: usize,
        /// The index into that input's d-tree.
        node: usize,
    },
}

/// A successful identification, with the trees that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Derivation {
    /// The q-tree.
    pub query_tree: CTree,
    /// One d-tree per input, in the order the inputs were given.
    pub input_trees: Vec<CTree>,
    /// The source of every q-tree leaf, keyed by the leaf's index.
    pub mapping: BTreeMap<usize, Source>,
    /// The estimand.
    pub estimand: Formula,
}

/// Why C-INFER failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CInferError {
    /// Some target c-factor appears in no d-tree.
    NotIdentifiable {
        /// The q-tree.
        query_tree: CTree,
        /// The d-trees, which show how far each input got.
        input_trees: Vec<CTree>,
        /// The q-tree leaves no input could produce.
        unmapped: Vec<CFactor>,
    },
    /// The query, an input or the graph is malformed.
    Graph(DSepError),
}

impl From<DSepError> for CInferError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl fmt::Display for CInferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotIdentifiable { unmapped, .. } => {
                let names: Vec<String> = unmapped.iter().map(ToString::to_string).collect();
                write!(f, "not identifiable: no input yields {}", names.join(", "))
            }
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CInferError {}

/// `An(seeds)` in `g`, as names.
fn ancestors(g: &Admg, seeds: &VarSet) -> Result<VarSet, DSepError> {
    Ok(g.ancestors_of(&refs(seeds))?
        .into_iter()
        .map(ToString::to_string)
        .collect())
}

/// The variables joined to `seeds` by a path of edges of either kind, taken in
/// either direction.
fn connected(g: &Admg, seeds: &VarSet) -> VarSet {
    let graph = g.graph();
    let mut adjacent: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in graph.edge_references() {
        let (a, b) = (graph[e.source()].as_str(), graph[e.target()].as_str());
        adjacent.entry(a).or_default().push(b);
        adjacent.entry(b).or_default().push(a);
    }
    let mut seen = VarSet::new();
    let mut stack: Vec<&str> = seeds.iter().map(String::as_str).collect();
    while let Some(v) = stack.pop() {
        if !seen.insert(v.to_string()) {
            continue;
        }
        stack.extend(adjacent.get(v).into_iter().flatten().copied());
    }
    seen
}

fn c_components(g: &Admg) -> Vec<VarSet> {
    g.c_components()
        .into_iter()
        .map(|c| c.into_iter().collect())
        .collect()
}

/// `GENQUERYTREE`: decomposes the query into the c-factors it needs.
///
/// For `P(y | do(x))`, with `D = An(Y)` in `G` with `X` overlined:
///
/// ```text
/// P(y | do(x)) <-Σ_{D \ Y}- Q[D | do(x)] <-δ- { Q[D_i | do(x)] }
/// ```
///
/// where the `D_i` are the c-components of that graph restricted to `D`.
/// Overlining `X` cuts every edge into it, bidirected ones included, so each
/// `X_i` in `D` is a component of its own; those leaves are indicators.
///
/// For `P(y | do(x), w)` the `Σ` is replaced by `Γ`. With `D = An(Y ∪ W)` in
/// `G` with `X` overlined, `W_Y` the part of `W` connected to `Y` once the
/// edges out of `W` are also cut and the graph is restricted to `D \ X`, and
/// `A = An(Y ∪ W_Y)` in `G` with `X` overlined and `W` underlined,
///
/// ```text
/// P(y | do(x), w) = sum_{A \ (Y ∪ W)} Q[A | do(x)] / sum_{A \ W} Q[A | do(x)]
/// ```
///
/// and `Q[A | do(x)]` splits by `δ` as before. Variables of `W` outside `W_Y`
/// carry no information about `Y` and drop out; those whose only link to `Y`
/// runs out of them stay in as parents, where they act as interventions —
/// which is rule 2 of the do-calculus.
///
/// The connectivity is taken in `D \ X`, not in `D` as the book states it.
/// An intervened `X_i` has only tails once its incoming edges are cut, so on
/// any path it is a non-collider, and fixed — it blocks the path. Counting
/// paths through it marks a harmless `W` as relevant: in
/// `X -> W -> Y, X -> Y, X <-> W`, the path `W <- X -> Y` would keep `W`, the
/// δ-step would then need `Q[W]`, which `X <-> W` makes unidentifiable, and
/// `P(y | do(x), w) = P(y | x, w)` would be reported as not identifiable.
///
/// # Errors
///
/// [`DSepError::Unknown`] for a name not in the graph, [`DSepError::Overlap`]
/// if `X`, `Y` and `W` are not pairwise disjoint, and [`DSepError::Empty`] for
/// an empty `Y`.
pub fn gen_query_tree(g: &Admg, query: &Query) -> Result<CTree, DSepError> {
    query.validate(g)?;
    let gx = g.cut_edges_into(&refs(&query.x))?;
    let mut tree = CTree::new(Term::Query(query.clone()));

    let (scope, op) = if query.w.is_empty() {
        let d = ancestors(&gx, &query.y)?;
        let sum_out = d.difference(&query.y).cloned().collect();
        (d, Operator::Sigma { sum_out })
    } else {
        let yw: VarSet = query.y.union(&query.w).cloned().collect();
        let d = ancestors(&gx, &yw)?;
        let gxw = gx.cut_edges_out_of(&refs(&query.w))?;
        let d_minus_x: VarSet = d.difference(&query.x).cloned().collect();
        let reach = connected(&gxw.induced_subgraph(&refs(&d_minus_x))?, &query.y);
        let wy: VarSet = query.w.intersection(&reach).cloned().collect();
        let a = ancestors(&gxw, &query.y.union(&wy).cloned().collect())?;
        (a, Operator::Gamma { given: wy })
    };

    let top = tree.add_child(
        tree.root,
        Term::Factor(CFactor {
            scope: scope.clone(),
            doing: query.x.clone(),
        }),
        op,
    );
    let components = c_components(&gx.induced_subgraph(&refs(&scope))?);
    // A single component would make δ the identity; leave it out.
    if components.len() > 1 {
        for c in &components {
            tree.add_child(
                top,
                Term::Factor(CFactor {
                    scope: c.clone(),
                    doing: query.x.clone(),
                }),
                Operator::Delta {
                    components: components.clone(),
                },
            );
        }
    }
    Ok(tree)
}

/// The q-tree leaves that must come from an input: every leaf that is not an
/// indicator.
fn target_leaves<'a>(query: &Query, t_q: &'a CTree) -> Vec<(usize, &'a CFactor)> {
    t_q.leaves()
        .into_iter()
        .filter_map(|i| t_q.factor(i).map(|c| (i, c)))
        .filter(|(_, c)| !c.scope.is_subset(&query.x))
        .collect()
}

/// `GENINPUTTREE`: what `input` can produce towards the q-tree's leaves.
///
/// The root is `Q[V \ Z | do(z)]`, the input itself. For each target leaf
/// `Q[D | do(x)]` with `D` disjoint from `Z`, starting from the root with
/// current scope `T`, repeat:
///
/// 1. `σ` — if `T = D`, read it in the query's regime and stop.
/// 2. `Σ` — if `A = An(D)` in `G[T]` is smaller than `T`, move to `Q[A]`.
/// 3. `δ` — if the c-component of `G[T]` containing `D` is smaller than `T`,
///    move to it.
/// 4. Otherwise `T` is a hedge for `D`: this input cannot produce it.
///
/// Each step strictly shrinks `T` and keeps `D` inside it, so the loop ends.
/// `D` cannot meet `X`, because only indicator leaves do, so step 1's
/// condition `D ∩ (X ∪ Z) = ∅` always holds by then.
///
/// # Errors
///
/// [`DSepError::Unknown`] if the input names a variable not in the graph.
pub fn gen_input_tree(
    g: &Admg,
    input: &Input,
    query: &Query,
    t_q: &CTree,
) -> Result<CTree, DSepError> {
    for z in &input.doing {
        if g.node(z).is_none() {
            return Err(DSepError::Unknown(z.clone()));
        }
    }
    let z = &input.doing;
    let v: VarSet = g.variables().into_iter().map(ToString::to_string).collect();
    let mut tree = CTree::new(Term::Factor(CFactor {
        scope: v.difference(z).cloned().collect(),
        doing: z.clone(),
    }));

    for (_, leaf) in target_leaves(query, t_q) {
        let d = &leaf.scope;
        if !d.is_disjoint(z) {
            continue;
        }
        let mut cur = tree.root;
        loop {
            let t = tree
                .factor(cur)
                .expect("d-trees hold only c-factors")
                .scope
                .clone();
            if t == *d {
                tree.add_child(
                    cur,
                    Term::Factor(leaf.clone()),
                    Operator::Regime {
                        from: z.clone(),
                        to: query.x.clone(),
                    },
                );
                break;
            }
            let sub = g.induced_subgraph(&refs(&t))?;
            let a = ancestors(&sub, d)?;
            if a != t {
                let sum_out = t.difference(&a).cloned().collect();
                cur = tree.add_child(
                    cur,
                    Term::Factor(CFactor {
                        scope: a,
                        doing: z.clone(),
                    }),
                    Operator::Sigma { sum_out },
                );
                continue;
            }
            let components = c_components(&sub);
            match components.iter().find(|c| d.is_subset(c)) {
                Some(c) if *c != t => {
                    let c = c.clone();
                    cur = tree.add_child(
                        cur,
                        Term::Factor(CFactor {
                            scope: c,
                            doing: z.clone(),
                        }),
                        Operator::Delta { components },
                    );
                }
                _ => break,
            }
        }
    }
    Ok(tree)
}

/// `MAPFACTORS`: a source for each q-tree leaf that has one.
///
/// Indicator leaves always map. Any other leaf maps to the first input whose
/// d-tree reached it, so inputs listed earlier are preferred. A target leaf
/// absent from the result is one no input can produce.
#[must_use]
pub fn map_factors(query: &Query, t_q: &CTree, t_ps: &[CTree]) -> BTreeMap<usize, Source> {
    let mut mapping = BTreeMap::new();
    for i in t_q.leaves() {
        let Some(leaf) = t_q.factor(i) else { continue };
        if leaf.scope.is_subset(&query.x) {
            mapping.insert(i, Source::Indicator);
            continue;
        }
        if let Some((input, node)) = t_ps
            .iter()
            .enumerate()
            .find_map(|(k, t)| t.find(leaf).map(|n| (k, n)))
        {
            mapping.insert(i, Source::Input { input, node });
        }
    }
    mapping
}

/// The directed parents of `set` in `g`.
fn parents(g: &Admg, set: &VarSet) -> VarSet {
    let graph = g.graph();
    graph
        .edge_references()
        .filter(|e| {
            *e.weight() == super::dseparation::EdgeKind::Directed
                && set.contains(&graph[e.target()])
        })
        .map(|e| graph[e.source()].clone())
        .collect()
}

/// For each `order[i]`, the predecessors it must be conditioned on in Tian's
/// factorisation of the c-factor over `order`'s variables.
///
/// The textbook form conditions `v_i` on every predecessor. Tian and Pearl
/// show `(T_i ∪ pa(T_i)) \ {v_i}` is enough, where `T_i` is the c-component
/// of `v_i` in the subgraph over `v_i` and its predecessors. The two agree in
/// value, but the full form mentions variables `Q[C]` does not depend on,
/// and when nothing sums those out they are left free in the estimand.
fn minimal_givens(g: &Admg, order: &[String]) -> Result<Vec<Vec<String>>, DSepError> {
    (0..order.len())
        .map(|i| {
            let prefix: Vec<&str> = order[..=i].iter().map(String::as_str).collect();
            let district = c_components(&g.induced_subgraph(&prefix)?)
                .into_iter()
                .find(|c| c.contains(&order[i]))
                .expect("the c-components partition the variables");
            let mut blanket = parents(g, &district);
            blanket.extend(district);
            Ok(order[..i]
                .iter()
                .filter(|v| blanket.contains(*v))
                .cloned()
                .collect())
        })
        .collect()
}

/// The formula for d-tree node `node`, by applying each operator along the
/// path from the input at the root.
fn evaluate(
    g: &Admg,
    t_p: &CTree,
    node: usize,
    input: &Input,
    order: &[String],
) -> Result<Formula, DSepError> {
    let within =
        |s: &VarSet| -> Vec<String> { order.iter().filter(|v| s.contains(*v)).cloned().collect() };
    let path = t_p.path_to(node);
    let root = t_p.factor(t_p.root).expect("d-trees hold only c-factors");
    let doing: Vec<String> = input.doing.iter().cloned().collect();
    let root_order = within(&root.scope);
    let givens = minimal_givens(g, &root_order)?;
    let mut dist = Dist::chain_with(&root_order, &doing, |i| givens[i].clone());
    for &n in &path[1..] {
        let scope = &t_p.factor(n).expect("d-trees hold only c-factors").scope;
        match &t_p.nodes[n].op_from_parent {
            Some(Operator::Sigma { .. }) => dist = dist.marginal(scope),
            Some(Operator::Delta { .. }) => {
                let current = within(&dist.domain);
                let givens = minimal_givens(g, &current)?;
                dist = dist.restrict_with(scope, &current, |i| givens[i].clone());
            }
            // The same function, read in the query's regime.
            Some(Operator::Regime { .. }) => {}
            Some(Operator::Gamma { .. }) | None => {
                unreachable!("Γ only appears in q-trees, and only the root lacks an operator")
            }
        }
    }
    Ok(dist.formula)
}

/// `COMPOSEQUERY`: the estimand, given a source for every leaf.
///
/// Each leaf's c-factor is evaluated along its d-tree path, the target factor
/// is their product, and the root's operator finishes the job: `Σ` sums out
/// `D \ Y`, `Γ` divides by the sum over `Y`. Indicator leaves contribute no
/// factor; they fix `X` at the query's value, which removes `X` from every
/// summation.
///
/// Returns `None` if some target leaf has no source.
///
/// # Errors
///
/// [`DSepError::Cyclic`] if the graph's directed edges contain a cycle.
pub fn compose_query(
    g: &Admg,
    query: &Query,
    t_q: &CTree,
    inputs: &[Input],
    t_ps: &[CTree],
    mapping: &BTreeMap<usize, Source>,
) -> Result<Option<Formula>, DSepError> {
    let order = g.topological_order()?;
    let mut factors = Vec::new();
    for i in t_q.leaves() {
        match mapping.get(&i) {
            Some(Source::Indicator) => {}
            Some(&Source::Input { input, node }) => {
                factors.push(evaluate(g, &t_ps[input], node, &inputs[input], &order)?);
            }
            None => return Ok(None),
        }
    }
    // Plain terms first and sums last, as the book prints them; the sort is
    // stable, so each group keeps the leaves' order.
    factors.sort_by_key(|f| !matches!(f, Formula::P { .. }));
    let product = Formula::product(factors);

    let top = t_q.nodes[t_q.root].children[0];
    let scope = &t_q
        .factor(top)
        .expect("the root's child is a c-factor")
        .scope;
    let without = |drop: &[&VarSet]| -> Vec<String> {
        scope
            .iter()
            .filter(|v| !drop.iter().any(|s| s.contains(*v)))
            .cloned()
            .collect()
    };
    let formula = match &t_q.nodes[top].op_from_parent {
        Some(Operator::Gamma { .. }) => {
            // `Formula::sum` cancels `sum_v P(v | ...)` only inside a product,
            // so a lone term is wrapped in one.
            let product = match product {
                p @ Formula::Product(_) => p,
                other => Formula::Product(vec![other]),
            };
            let num = Formula::sum(without(&[&query.y, &query.w, &query.x]), product.clone());
            let den = Formula::sum(without(&[&query.w, &query.x]), product);
            // Every factor of the denominator can sum to one, leaving nothing.
            if matches!(&den, Formula::Product(f) if f.is_empty()) {
                num
            } else {
                Formula::Ratio {
                    num: Box::new(num),
                    den: Box::new(den),
                }
            }
        }
        _ => Formula::sum(without(&[&query.y, &query.x]), product),
    };
    let mut formula = formula.simplify();
    // The query's variables are free, and so may be others: an experiment's
    // `z` that nothing sums over stays as `do(z)`. A sum elsewhere must not
    // reuse any of their names.
    let mut free = formula.free_vars();
    free.extend(query.x.iter().chain(&query.y).chain(&query.w).cloned());
    formula.avoid_capture(&free);
    Ok(Some(formula))
}

/// `C-INFER`: identifies `query` from `inputs` and the diagram `g`.
///
/// Builds the q-tree, a d-tree per input, maps the leaves and composes the
/// estimand. Fails exactly when some target c-factor is out of every input's
/// reach.
///
/// # Errors
///
/// [`CInferError::NotIdentifiable`] when a target leaf has no source, carrying
/// the trees and the leaves that were missed; [`CInferError::Graph`] for a
/// malformed query or input, or a cyclic diagram.
pub fn c_infer(g: &Admg, query: &Query, inputs: &[Input]) -> Result<Derivation, CInferError> {
    let query_tree = gen_query_tree(g, query)?;
    let input_trees = inputs
        .iter()
        .map(|p| gen_input_tree(g, p, query, &query_tree))
        .collect::<Result<Vec<_>, _>>()?;
    let mapping = map_factors(query, &query_tree, &input_trees);

    let unmapped: Vec<CFactor> = target_leaves(query, &query_tree)
        .into_iter()
        .filter(|(i, _)| !mapping.contains_key(i))
        .map(|(_, c)| c.clone())
        .collect();
    if !unmapped.is_empty() {
        return Err(CInferError::NotIdentifiable {
            query_tree,
            input_trees,
            unmapped,
        });
    }

    let estimand = compose_query(g, query, &query_tree, inputs, &input_trees, &mapping)?
        .expect("every leaf was mapped");
    Ok(Derivation {
        query_tree,
        input_trees,
        mapping,
        estimand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::identification::{identify, identify_conditional};

    fn obs() -> Vec<Input> {
        vec![Input::observational()]
    }

    /// X -> Z -> Y with X <-> Y.
    fn front_door() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "Z");
        g.directed("Z", "Y");
        g.bidirected("X", "Y");
        g
    }

    /// Z -> X, Z -> Y, X -> Y.
    fn back_door() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("Z", "Y");
        g.directed("X", "Y");
        g
    }

    /// X <-> Z, Z -> Y, X -> Y, X <-> Y: the bow arc X -> Y, X <-> Y with a
    /// confounded extra parent of Y, which does not rescue it.
    fn confounded_bow() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "Y");
        g.directed("X", "Y");
        g.bidirected("X", "Z");
        g.bidirected("X", "Y");
        g
    }

    /// W -> Z -> X -> Y with W <-> X and W <-> Y.
    fn napkin() -> Admg {
        let mut g = Admg::new();
        g.directed("W", "Z");
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.bidirected("W", "X");
        g.bidirected("W", "Y");
        g
    }

    #[test]
    fn front_door_benchmark() {
        let d = c_infer(&front_door(), &Query::new(&["X"], &["Y"]), &obs()).unwrap();
        assert_eq!(
            d.estimand.to_string(),
            "sum_{z} P(z | x) sum_{x'} P(x') P(y | x', z)"
        );
        assert_eq!(
            d.query_tree.to_string(),
            "P(Y | do(X))\n\
             └─ [Σ_{X, Z}] Q[X, Y, Z | do(X)]\n   \
                ├─ [δ] Q[X | do(X)]\n   \
                ├─ [δ] Q[Y | do(X)]\n   \
                └─ [δ] Q[Z | do(X)]\n"
        );
        // Q[Y] needs δ to its district {X, Y} and then Σ over X; Q[Z] needs Σ
        // over Y and then δ. Both start from P(V), and σ hands them over.
        assert_eq!(
            d.input_trees[0].to_string(),
            "Q[X, Y, Z]\n\
             ├─ [δ] Q[X, Y]\n\
             │  └─ [Σ_{X}] Q[Y]\n\
             │     └─ [σ do(∅) → do(X)] Q[Y | do(X)]\n\
             └─ [Σ_{Y}] Q[X, Z]\n   \
                └─ [δ] Q[Z]\n      \
                   └─ [σ do(∅) → do(X)] Q[Z | do(X)]\n"
        );
        assert_eq!(d.mapping.len(), 3);
        assert_eq!(d.mapping[&2], Source::Indicator);
    }

    #[test]
    fn back_door_benchmark() {
        let d = c_infer(&back_door(), &Query::new(&["X"], &["Y"]), &obs()).unwrap();
        assert_eq!(d.estimand.to_string(), "sum_{z} P(y | z, x) P(z)");
    }

    #[test]
    fn unidentifiable_benchmark() {
        let err = c_infer(&confounded_bow(), &Query::new(&["X"], &["Y"]), &obs()).unwrap_err();
        let CInferError::NotIdentifiable {
            unmapped,
            input_trees,
            ..
        } = err
        else {
            panic!("expected a failure to identify, got {err}");
        };
        assert_eq!(
            unmapped,
            vec![CFactor {
                scope: var_set(&["Y"]),
                doing: var_set(&["X"]),
            }]
        );
        // Q[Z] is still reachable; the d-tree records that much.
        assert!(
            input_trees[0]
                .find(&CFactor {
                    scope: var_set(&["Z"]),
                    doing: var_set(&["X"]),
                })
                .is_some()
        );
        assert!(identify(&confounded_bow(), &["X"], &["Y"]).is_err());
    }

    #[test]
    fn the_napkin_matches_id() {
        let d = c_infer(&napkin(), &Query::new(&["X"], &["Y"]), &obs()).unwrap();
        assert_eq!(
            d.estimand,
            identify(&napkin(), &["X"], &["Y"]).unwrap(),
            "{}",
            d.estimand
        );
    }

    #[test]
    fn a_surrogate_experiment_rescues_a_hedge() {
        // Z -> X -> Y, X <-> Z <-> Y. Every confounding path runs through Z,
        // so randomising Z — not X — is enough to read the effect off.
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.bidirected("X", "Z");
        g.bidirected("Z", "Y");
        let q = Query::new(&["X"], &["Y"]);

        assert!(c_infer(&g, &q, &obs()).is_err());
        assert!(identify(&g, &["X"], &["Y"]).is_err());

        let inputs = [Input::observational(), Input::experimental(&["Z"])];
        let d = c_infer(&g, &q, &inputs).unwrap();
        assert_eq!(d.estimand.to_string(), "P(y | x, do(z))");
        assert_eq!(d.estimand.to_latex(), "P(y \\mid x, \\mathrm{do}(z))");
        let leaf = d.query_tree.leaves()[1];
        assert!(matches!(d.mapping[&leaf], Source::Input { input: 1, .. }));
    }

    #[test]
    fn an_experiment_on_the_treatment_answers_directly() {
        let mut g = Admg::new();
        g.directed("X", "Y");
        g.bidirected("X", "Y");
        let d = c_infer(
            &g,
            &Query::new(&["X"], &["Y"]),
            &[Input::observational(), Input::experimental(&["X"])],
        )
        .unwrap();
        assert_eq!(d.estimand.to_string(), "P(y | do(x))");
    }

    #[test]
    fn observational_inputs_are_preferred_in_order() {
        // Back-door is identifiable from P(V), so the experiment, listed
        // second, is never consulted.
        let d = c_infer(
            &back_door(),
            &Query::new(&["X"], &["Y"]),
            &[Input::observational(), Input::experimental(&["X"])],
        )
        .unwrap();
        assert_eq!(d.estimand.to_string(), "sum_{z} P(y | z, x) P(z)");
    }

    /// The notebook's `G4`, as in the ID tests.
    fn ch4_g4() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "Z1");
        g.directed("Z1", "Y");
        g.directed("Z2", "Y");
        g.bidirected("X", "Z2");
        g.bidirected("Z1", "Z2");
        g
    }

    #[test]
    fn conditional_queries_go_through_gamma() {
        let q = Query::new(&["X"], &["Y"]).given(&["Z2"]);
        assert!(c_infer(&ch4_g4(), &q, &obs()).is_err());

        // ID prints sum_{z1} P(y | z2, x, z1) P(z1 | x). Y's district is just
        // {Y}, with parents Z1 and Z2, so conditioning on X adds nothing.
        let joint = c_infer(&ch4_g4(), &Query::new(&["X", "Z2"], &["Y"]), &obs()).unwrap();
        assert_eq!(
            joint.estimand.to_string(),
            "sum_{z1} P(y | z2, z1) P(z1 | x)"
        );

        // In the back-door graph conditioning on Z is as good as setting it:
        // Γ's denominator sums to one and no ratio is left.
        let d = c_infer(
            &back_door(),
            &Query::new(&["X"], &["Y"]).given(&["Z"]),
            &obs(),
        )
        .unwrap();
        assert_eq!(d.estimand.to_string(), "P(y | z, x)");
        assert!(matches!(
            d.query_tree.nodes[1].op_from_parent,
            Some(Operator::Gamma { .. })
        ));
    }

    #[test]
    fn queries_and_inputs_are_validated() {
        let g = back_door();
        let bad = |q: Query, inputs: &[Input]| c_infer(&g, &q, inputs).unwrap_err();
        assert_eq!(
            bad(Query::new(&["nope"], &["Y"]), &obs()),
            CInferError::Graph(DSepError::Unknown("nope".into()))
        );
        assert_eq!(
            bad(Query::new(&["X"], &["X"]), &obs()),
            CInferError::Graph(DSepError::Overlap("X".into()))
        );
        assert_eq!(
            bad(Query::new(&["X"], &["Y"]).given(&["Y"]), &obs()),
            CInferError::Graph(DSepError::Overlap("Y".into()))
        );
        assert_eq!(
            bad(Query::new(&["X"], &[]), &obs()),
            CInferError::Graph(DSepError::Empty("Y"))
        );
        assert_eq!(
            bad(Query::new(&["X"], &["Y"]), &[Input::experimental(&["Q"])]),
            CInferError::Graph(DSepError::Unknown("Q".into()))
        );
    }

    /// Deterministic randomness, so failures reproduce.
    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(h: u64) -> f64 {
        // 53 random mantissa bits; the cast is exact.
        #[allow(clippy::cast_precision_loss)]
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        u
    }

    /// A random binary model over a random ADMG: one binary latent per
    /// bidirected edge, and strictly positive conditional tables.
    struct Model {
        names: Vec<String>,
        parents: Vec<Vec<usize>>,
        /// Per variable, the latents it depends on.
        latents_of: Vec<Vec<usize>>,
        latent_p: Vec<f64>,
        seed: u64,
    }

    impl Model {
        fn random(n: usize, seed: u64) -> (Admg, Self) {
            let names: Vec<String> = (0..n).map(|i| format!("V{i}")).collect();
            let mut g = Admg::new();
            for v in &names {
                g.add_node(v);
            }
            let mut parents = vec![Vec::new(); n];
            let mut latents_of = vec![Vec::new(); n];
            let mut latent_p = Vec::new();
            let mut r = seed;
            let mut next = || {
                r = splitmix(r);
                unit(r)
            };
            for j in 0..n {
                for i in 0..j {
                    if next() < 0.45 {
                        g.directed(&names[i], &names[j]);
                        parents[j].push(i);
                    }
                    if next() < 0.3 {
                        g.bidirected(&names[i], &names[j]);
                        latents_of[i].push(latent_p.len());
                        latents_of[j].push(latent_p.len());
                        latent_p.push(0.15 + 0.7 * next());
                    }
                }
            }
            let model = Self {
                names,
                parents,
                latents_of,
                latent_p,
                seed,
            };
            (g, model)
        }

        fn index(&self, name: &str) -> usize {
            let base = name.trim_end_matches('\'');
            self.names.iter().position(|n| n == base).expect("known")
        }

        /// `P(V_i = 1 | pa, u)`, a fixed pseudo-random number in (0.1, 0.9).
        fn p_one(&self, i: usize, v: u64, u: u64) -> f64 {
            let mut key = self.seed ^ (i as u64) << 56;
            for &p in &self.parents[i] {
                key = splitmix(key ^ ((v >> p) & 1) << 1 ^ p as u64);
            }
            for &l in &self.latents_of[i] {
                key = splitmix(key ^ ((u >> l) & 1) << 1 ^ (l as u64) << 8);
            }
            0.1 + 0.8 * unit(splitmix(key))
        }

        /// The joint over V, indexed by bitmask, after setting `fixed`.
        fn joint(&self, fixed: &[(usize, u64)]) -> Vec<f64> {
            let n = self.names.len();
            let mut table = vec![0.0; 1 << n];
            for u in 0..(1u64 << self.latent_p.len()) {
                let mut pu = 1.0;
                for (l, p) in self.latent_p.iter().enumerate() {
                    pu *= if (u >> l) & 1 == 1 { *p } else { 1.0 - p };
                }
                for v in 0..(1u64 << n) {
                    let mut p = pu;
                    for i in 0..n {
                        let bit = (v >> i) & 1;
                        if let Some(&(_, val)) = fixed.iter().find(|(f, _)| *f == i) {
                            if bit != val {
                                p = 0.0;
                            }
                            continue;
                        }
                        let one = self.p_one(i, v, u);
                        p *= if bit == 1 { one } else { 1.0 - one };
                    }
                    table[v as usize] += p;
                }
            }
            table
        }
    }

    fn marginal(table: &[f64], assignment: &[(usize, u64)]) -> f64 {
        table
            .iter()
            .enumerate()
            .filter(|(v, _)| assignment.iter().all(|&(i, b)| ((*v as u64) >> i) & 1 == b))
            .map(|(_, p)| p)
            .sum()
    }

    /// Evaluates an estimand against a model, caching each regime's joint.
    struct Evaluator<'a> {
        model: &'a Model,
        cache: HashMap<Vec<(usize, u64)>, Vec<f64>>,
    }

    impl Evaluator<'_> {
        fn eval(&mut self, f: &Formula, env: &mut HashMap<String, u64>) -> f64 {
            match f {
                Formula::P { vars, given, doing } => {
                    let mut fixed: Vec<(usize, u64)> = doing
                        .iter()
                        .map(|d| (self.model.index(d), env[d]))
                        .collect();
                    fixed.sort_unstable();
                    let model = self.model;
                    let table = self
                        .cache
                        .entry(fixed)
                        .or_insert_with_key(|k| model.joint(k));
                    let bind = |names: &[String]| -> Vec<(usize, u64)> {
                        names.iter().map(|n| (model.index(n), env[n])).collect()
                    };
                    let g = bind(given);
                    let mut joint = bind(vars);
                    joint.extend(g.iter().copied());
                    marginal(table, &joint) / marginal(table, &g)
                }
                Formula::Product(fs) => fs.iter().map(|f| self.eval(f, env)).product(),
                Formula::Ratio { num, den } => self.eval(num, env) / self.eval(den, env),
                Formula::Sum { over, body } => {
                    let shadowed: Vec<Option<u64>> =
                        over.iter().map(|v| env.get(v).copied()).collect();
                    let mut total = 0.0;
                    for bits in 0..(1u64 << over.len()) {
                        for (k, v) in over.iter().enumerate() {
                            env.insert(v.clone(), (bits >> k) & 1);
                        }
                        total += self.eval(body, env);
                    }
                    for (v, old) in over.iter().zip(shadowed) {
                        match old {
                            Some(b) => env.insert(v.clone(), b),
                            None => env.remove(v),
                        };
                    }
                    total
                }
            }
        }
    }

    /// Checks `estimand` against the true `P(y | do(x), w)` at every binary
    /// assignment of its free variables.
    ///
    /// Those are normally the query's own. Any others — an experiment's `z`
    /// that nothing sums over, say — must be ones the effect does not depend
    /// on, and enumerating them checks exactly that.
    fn assert_sound(model: &Model, q: &Query, estimand: &Formula) {
        let mut ev = Evaluator {
            model,
            cache: HashMap::new(),
        };
        let mut free = estimand.free_vars();
        free.extend(q.x.iter().chain(&q.y).chain(&q.w).cloned());
        let free: Vec<&String> = free.iter().collect();
        for bits in 0..(1u64 << free.len()) {
            let mut env: HashMap<String, u64> = free
                .iter()
                .enumerate()
                .map(|(k, v)| ((*v).clone(), (bits >> k) & 1))
                .collect();
            let at = |s: &VarSet| -> Vec<(usize, u64)> {
                s.iter().map(|v| (model.index(v), env[v])).collect()
            };
            let truth = {
                let table = model.joint(&at(&q.x));
                let w = at(&q.w);
                let mut yw = at(&q.y);
                yw.extend(w.iter().copied());
                marginal(&table, &yw) / marginal(&table, &w)
            };
            let got = ev.eval(estimand, &mut env);
            assert!(
                (got - truth).abs() < 1e-9,
                "{q} at {env:?}: estimand {estimand} gives {got}, truth is {truth}"
            );
        }
    }

    /// Over random graphs: C-INFER on P(V) identifies exactly what ID and IDC
    /// do — both are complete, so any disagreement is a bug — and every
    /// estimand it returns, with or without an extra experiment, evaluates to
    /// the true effect of a random model on that graph.
    #[test]
    fn agrees_with_id_and_is_numerically_sound() {
        let mut identified = 0;
        let mut rescued = 0;
        for seed in 0..120u64 {
            let n = if seed % 3 == 0 { 5 } else { 4 };
            let (g, model) = Model::random(n, seed);
            let names = model.names.clone();
            for x in &names {
                for y in names.iter().filter(|y| *y != x) {
                    let q = Query::new(&[x], &[y]);
                    let ours = c_infer(&g, &q, &obs());
                    let theirs = identify(&g, &[x], &[y]);
                    assert_eq!(ours.is_ok(), theirs.is_ok(), "seed {seed}: {q}");
                    if let Ok(d) = &ours {
                        identified += 1;
                        assert_sound(&model, &q, &d.estimand);
                    }

                    // Add one experiment on a third variable.
                    let z = &names[(splitmix(seed ^ 0xabc) as usize) % n];
                    if z != y {
                        let inputs = [Input::observational(), Input::experimental(&[z])];
                        match c_infer(&g, &q, &inputs) {
                            Ok(d) => {
                                if ours.is_err() {
                                    rescued += 1;
                                }
                                assert_sound(&model, &q, &d.estimand);
                            }
                            Err(_) => assert!(ours.is_err(), "an extra input lost {q}"),
                        }
                    }

                    for w in names.iter().filter(|w| *w != x && *w != y) {
                        let q = Query::new(&[x], &[y]).given(&[w]);
                        let ours = c_infer(&g, &q, &obs());
                        let theirs = identify_conditional(&g, &[x], &[y], &[w]);
                        assert_eq!(ours.is_ok(), theirs.is_ok(), "seed {seed}: {q}");
                        if let Ok(d) = &ours {
                            assert_sound(&model, &q, &d.estimand);
                        }
                    }
                }
            }
        }
        // Guard against a generator that only ever produces trivial cases.
        assert!(identified > 500, "only {identified} identified");
        assert!(rescued > 5, "only {rescued} rescued by an experiment");
    }
}
