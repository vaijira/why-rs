//! Partial mixed graphs (Def. 16.1.7): MAGs and PAGs.

use std::fmt;

use super::DiscoveryError;
use crate::graph::dseparation::{Admg, EdgeKind};

/// An edge mark at one end: arrowhead, tail or circle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mark {
    /// `−`.
    Tail,
    /// `>`.
    Arrow,
    /// `◦`, a mark the equivalence class does not fix.
    Circle,
}

/// A partial mixed graph (Def. 16.1.7): every pair of nodes is joined by at
/// most one edge, and each end of an edge carries a [`Mark`]. A MAG is one
/// with no circles. A PAG (Def. 16.1.8) is one whose marks are those common
/// to a Markov equivalence class of MAGs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pag {
    names: Vec<String>,
    /// `marks[i][j]`: the mark at `j` on the edge between `i` and `j`.
    marks: Vec<Vec<Option<Mark>>>,
}

impl Pag {
    /// A graph over `names` with no edges.
    #[must_use]
    pub fn new(names: &[&str]) -> Self {
        let n = names.len();
        Self {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            marks: vec![vec![None; n]; n],
        }
    }

    /// The complete graph of `o-o` edges, where FCI starts.
    #[must_use]
    pub fn complete(names: &[&str]) -> Self {
        let mut p = Self::new(names);
        for (i, row) in p.marks.iter_mut().enumerate() {
            for (j, m) in row.iter_mut().enumerate() {
                if i != j {
                    *m = Some(Mark::Circle);
                }
            }
        }
        p
    }

    /// A MAG read off a mixed graph: `→` becomes tail-arrow and `↔`
    /// arrow-arrow.
    #[must_use]
    pub fn from_admg(g: &Admg) -> Self {
        let names = g.variables();
        let mut p = Self::new(&names);
        let graph = g.graph();
        for e in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(e).expect("valid edge");
            let (i, j) = (
                p.index(&graph[a]).expect("a node"),
                p.index(&graph[b]).expect("a node"),
            );
            match graph[e] {
                EdgeKind::Directed => p.set(i, j, Mark::Tail, Mark::Arrow),
                EdgeKind::Bidirected => p.set(i, j, Mark::Arrow, Mark::Arrow),
            }
        }
        p
    }

    /// The nodes.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The index of `name`.
    #[must_use]
    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn adj(&self, i: usize, j: usize) -> bool {
        self.marks[i][j].is_some()
    }

    /// The mark at `j` on the edge `i`–`j`.
    pub(crate) fn at(&self, i: usize, j: usize) -> Option<Mark> {
        self.marks[i][j]
    }

    /// Sets the edge `i`–`j` with `mi` at `i` and `mj` at `j`.
    pub(crate) fn set(&mut self, i: usize, j: usize, mi: Mark, mj: Mark) {
        self.marks[j][i] = Some(mi);
        self.marks[i][j] = Some(mj);
    }

    /// Sets the mark at `j` on the existing edge `i`–`j`.
    pub(crate) fn set_at(&mut self, i: usize, j: usize, m: Mark) {
        debug_assert!(self.adj(i, j));
        self.marks[i][j] = Some(m);
    }

    pub(crate) fn remove(&mut self, i: usize, j: usize) {
        self.marks[i][j] = None;
        self.marks[j][i] = None;
    }

    pub(crate) fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(move |&j| self.adj(i, j))
    }

    /// `i → j`: tail at `i`, arrowhead at `j`.
    pub(crate) fn directed(&self, i: usize, j: usize) -> bool {
        self.at(j, i) == Some(Mark::Tail) && self.at(i, j) == Some(Mark::Arrow)
    }

    fn idx(&self, name: &str) -> Result<usize, DiscoveryError> {
        self.index(name)
            .ok_or_else(|| DiscoveryError::UnknownVariable(name.to_string()))
    }

    /// The edge between `a` and `b` as `(mark at a, mark at b)`, if any.
    ///
    /// # Errors
    ///
    /// [`DiscoveryError::UnknownVariable`] for a name not in the graph.
    pub fn edge(&self, a: &str, b: &str) -> Result<Option<(Mark, Mark)>, DiscoveryError> {
        let (i, j) = (self.idx(a)?, self.idx(b)?);
        Ok(self.at(j, i).zip(self.at(i, j)))
    }

    /// Turns every `◦→` into `→` (the last step of AG-CONSTRUCT,
    /// Algorithm 72).
    pub(crate) fn arrows_for_circles(&mut self) {
        for i in 0..self.len() {
            for j in 0..self.len() {
                if self.adj(i, j)
                    && self.at(i, j) == Some(Mark::Arrow)
                    && self.at(j, i) == Some(Mark::Circle)
                {
                    self.set_at(j, i, Mark::Tail);
                }
            }
        }
    }

    /// Every edge as text, such as `A o-> B`, `B --> C`, `X <-> Y` or
    /// `X o-o Y`, with the endpoints in name order, sorted.
    #[must_use]
    pub fn edges(&self) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..self.len() {
            for j in 0..self.len() {
                if i < j && self.adj(i, j) {
                    let (a, b) = if self.names[i] <= self.names[j] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    let left = match self.at(b, a).expect("adjacent") {
                        Mark::Tail => '-',
                        Mark::Arrow => '<',
                        Mark::Circle => 'o',
                    };
                    let right = match self.at(a, b).expect("adjacent") {
                        Mark::Tail => '-',
                        Mark::Arrow => '>',
                        Mark::Circle => 'o',
                    };
                    out.push(format!(
                        "{} {left}-{right} {}",
                        self.names[a], self.names[b]
                    ));
                }
            }
        }
        out.sort();
        out
    }

    /// Whether the directed edge `a → b` is visible (Lemma 16.1.4): some
    /// node `c` not adjacent to `b` has an edge into `a`, or reaches `a`
    /// through a collider path into `a` whose every inner node is a parent
    /// of `b`. A visible edge is unconfounded in every diagram of the class.
    ///
    /// # Errors
    ///
    /// [`DiscoveryError::UnknownVariable`] for a name not in the graph, and
    /// [`DiscoveryError::InvalidInput`] when `a → b` is not an edge.
    pub fn is_visible(&self, a: &str, b: &str) -> Result<bool, DiscoveryError> {
        let (a, b) = (self.idx(a)?, self.idx(b)?);
        if !self.directed(a, b) {
            return Err(DiscoveryError::InvalidInput(format!(
                "{} -> {} is not an edge",
                self.names[a], self.names[b]
            )));
        }
        // Walk collider paths backwards from a: each step goes from node x
        // (into which the path arrives) to a neighbor y with an arrowhead at
        // x; inner nodes must be colliders and parents of b.
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(a, vec![a])];
        while let Some((x, path)) = stack.pop() {
            for y in self.neighbors(x) {
                if path.contains(&y) || y == b || self.at(y, x) != Some(Mark::Arrow) {
                    continue;
                }
                if !self.adj(y, b) {
                    return Ok(true);
                }
                // y can continue the path only as a parent of b and a
                // collider: the edge to x is into y here, and the edge from
                // y's next neighbor is checked into y when expanding y.
                if self.directed(y, b) && self.at(x, y) == Some(Mark::Arrow) {
                    let mut next = path.clone();
                    next.push(y);
                    stack.push((y, next));
                }
            }
        }
        Ok(false)
    }
}

impl fmt::Display for Pag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.edges().join(", "))
    }
}
