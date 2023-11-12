/// Common data structures for dagitty interaction.
pub mod dagitty;

use std::collections::HashSet;
use std::fmt::Debug;

pub use petgraph::{
    graph::{DiGraph, UnGraph},
    graph::{EdgeIndex, NodeIndex},
    graph::{IndexType, WalkNeighbors},
    stable_graph::DefaultIx,
    Directed,
    Direction::Incoming,
    EdgeType, Graph, Undirected,
};

/// Causal Graph
pub enum CausalGraph<N, E, Ix = DefaultIx> {
    /// Dag
    Dag(Graph<N, E, Directed, Ix>),
    /// Ungraph
    Ungraph(Graph<N, E, Undirected, Ix>),
}

impl<N, E, Ix: IndexType> Debug for CausalGraph<N, E, Ix> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Causal Graph")
    }
}

impl<N, E, Ix: IndexType> CausalGraph<N, E, Ix> {
    /// Add new node to the graph.
    pub fn add_node(&mut self, n: N) -> NodeIndex<Ix> {
        match self {
            Self::Dag(g) => g.add_node(n),
            Self::Ungraph(g) => g.add_node(n),
        }
    }

    /// Add new edge to the graph.
    pub fn add_edge(&mut self, left: NodeIndex<Ix>, right: NodeIndex<Ix>, e: E) -> EdgeIndex<Ix> {
        match self {
            Self::Dag(g) => g.add_edge(left, right, e),
            Self::Ungraph(g) => g.add_edge(left, right, e),
        }
    }
}

/// Extend Graph with new calls needed by causal graph algorithms.
pub trait CausalGraphExt<'a, N, E, Ty: EdgeType, Ix: IndexType> {
    /// Return all ancestors from a given node.
    fn ancestors(&'a self, node: NodeIndex<Ix>) -> Ancestors<'a, N, E, Ty, Ix>;

    /// Add edges
    fn add_edges(&mut self, edges: impl Iterator<Item = (NodeIndex<Ix>, NodeIndex<Ix>, E)>);
}

/// Ancestors is an structure containing all nodes that are ancestors of a particular one.
pub struct Ancestors<'a, N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    g: &'a Graph<N, E, Ty, Ix>,
    visited: HashSet<NodeIndex<Ix>>,
    pending_neighbors: Vec<WalkNeighbors<Ix>>,
}

impl<'a, N, E, Ty: EdgeType, Ix: IndexType> Ancestors<'a, N, E, Ty, Ix> {
    fn new(g: &'a Graph<N, E, Ty, Ix>, node: NodeIndex<Ix>) -> Self {
        Self {
            g,
            visited: HashSet::from([node]),
            pending_neighbors: vec![g.neighbors_directed(node, Incoming).detach()],
        }
    }
}

impl<'a, N, E, Ty: EdgeType, Ix: IndexType> Debug for Ancestors<'a, N, E, Ty, Ix> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ancestors visited: {:?}", self.visited)
    }
}

impl<'a, N, E, Ty: EdgeType, Ix: IndexType> Iterator for Ancestors<'a, N, E, Ty, Ix> {
    type Item = NodeIndex<Ix>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pending_neighbors.is_empty() {
            None
        } else {
            let mut found = None;
            while let Some(mut neighbors) = self.pending_neighbors.pop() {
                while let Some(edge) = neighbors.next(self.g) {
                    if !self.visited.contains(&edge.1) {
                        found = Some(edge.1);
                        self.visited.insert(edge.1);
                        break;
                    }
                }
                if let Some(node) = found {
                    self.pending_neighbors
                        .push(self.g.neighbors_directed(node, Incoming).detach());
                    self.pending_neighbors.insert(0, neighbors);
                    break;
                }
            }
            found
        }
    }
}

impl<'a, N, E, Ty, Ix> CausalGraphExt<'a, N, E, Ty, Ix> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn ancestors(&'a self, node: NodeIndex<Ix>) -> Ancestors<'a, N, E, Ty, Ix> {
        Ancestors::new(self, node)
    }

    fn add_edges(&mut self, edges: impl Iterator<Item = (NodeIndex<Ix>, NodeIndex<Ix>, E)>) {
        for edge in edges {
            self.add_edge(edge.0, edge.1, edge.2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ancestors() {
        let mut g = Graph::<&str, &str>::new();

        let a = g.add_node("A");
        let b = g.add_node("B");
        let d = g.add_node("D");
        let e = g.add_node("E");
        let z = g.add_node("Z");

        g.add_edge(a, e, "");
        g.add_edge(a, z, "");
        g.add_edge(b, d, "");
        g.add_edge(b, z, "");
        g.add_edge(e, d, "");

        let mut iter = g.ancestors(a).into_iter();

        assert_eq!(None, iter.next());

        let iter = g.ancestors(z).into_iter();
        let nodes = iter.collect::<HashSet<NodeIndex>>();
        assert_eq!(HashSet::from([a, b]), nodes);
    }
}
