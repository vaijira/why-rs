//! MAG marginalization (Algorithms 53/55) and Markov equivalence of MAGs
//! (Prop. 16.1.1).

use std::collections::BTreeSet;

use super::{DiscoveryError, Mark, Pag};
use crate::graph::dseparation::{Admg, EdgeKind};

/// The MAG over `observed` of the causal diagram `g` (Algorithm 53).
///
/// Each bidirected edge of `g` stands for a latent common cause, and every
/// variable of `g` outside `observed` is latent too. Two observed variables
/// are adjacent when an inducing path joins them relative to the latent
/// ones (Def. 16.1.4). The edge points from one to the other when it is an
/// ancestor of the other, and is bidirected when neither is.
///
/// ```
/// use why_data::discovery::mag;
/// use why_data::graph::dseparation::Admg;
///
/// // Ex. 16.4 (Fig. 16.4(a)): A -> B -> C -> D with B <-> D.
/// let mut g = Admg::new();
/// for (a, b) in [("A", "B"), ("B", "C"), ("C", "D")] {
///     g.directed(a, b);
/// }
/// g.bidirected("B", "D");
/// let m = mag(&g, &["A", "B", "C", "D"]).unwrap();
/// assert_eq!(m.edges(), ["A --> B", "A --> D", "B --> C", "B --> D", "C --> D"]);
/// ```
///
/// # Errors
///
/// [`DiscoveryError::UnknownVariable`] for an observed name `g` lacks.
pub fn mag(g: &Admg, observed: &[&str]) -> Result<Pag, DiscoveryError> {
    let names: Vec<String> = g.variables().iter().map(|s| (*s).to_string()).collect();
    for o in observed {
        if !names.iter().any(|n| n == o) {
            return Err(DiscoveryError::UnknownVariable((*o).to_string()));
        }
    }
    // The DAG with explicit latents: parents[i], and each bidirected edge a
    // new node pointing to both ends.
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
    let index = |n: &str| names.iter().position(|m| m == n).expect("a node");
    let graph = g.graph();
    let mut latent_parents: Vec<(usize, usize)> = Vec::new();
    for e in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(e).expect("valid edge");
        let (a, b) = (index(&graph[a]), index(&graph[b]));
        match graph[e] {
            EdgeKind::Directed => parents[b].push(a),
            EdgeKind::Bidirected => latent_parents.push((a, b)),
        }
    }
    for (a, b) in latent_parents {
        let u = parents.len();
        parents.push(Vec::new());
        parents[a].push(u);
        parents[b].push(u);
    }
    let n = parents.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, ps) in parents.iter().enumerate() {
        for &p in ps {
            children[p].push(c);
        }
    }
    let obs: Vec<usize> = observed.iter().map(|o| index(o)).collect();
    let latent: Vec<bool> = (0..n).map(|i| !obs.contains(&i)).collect();
    // ancestors[i]: proper ancestors of i.
    let ancestors: Vec<BTreeSet<usize>> = (0..n)
        .map(|i| {
            let mut seen = BTreeSet::new();
            let mut stack = parents[i].clone();
            while let Some(p) = stack.pop() {
                if seen.insert(p) {
                    stack.extend(parents[p].iter().copied());
                }
            }
            seen
        })
        .collect();
    // Undirected adjacency with the arrowhead side: (neighbor, arrow at me).
    let adjacency: Vec<Vec<(usize, bool)>> = (0..n)
        .map(|i| {
            let mut v: Vec<(usize, bool)> = parents[i].iter().map(|&p| (p, true)).collect();
            v.extend(children[i].iter().map(|&c| (c, false)));
            v
        })
        .collect();
    let mut out = Pag::new(observed);
    for (a, &x) in obs.iter().enumerate() {
        for (b, &y) in obs.iter().enumerate().skip(a + 1) {
            let anc = |v: usize| ancestors[x].contains(&v) || ancestors[y].contains(&v);
            if inducing(&adjacency, &latent, &anc, x, y) {
                if ancestors[y].contains(&x) {
                    out.set(a, b, Mark::Tail, Mark::Arrow);
                } else if ancestors[x].contains(&y) {
                    out.set(a, b, Mark::Arrow, Mark::Tail);
                } else {
                    out.set(a, b, Mark::Arrow, Mark::Arrow);
                }
            }
        }
    }
    Ok(out)
}

/// Whether an inducing path joins `x` and `y` relative to the latent nodes:
/// every inner observed node is a collider, and every collider is an
/// ancestor of `x` or `y` (`anc`).
fn inducing(
    adjacency: &[Vec<(usize, bool)>],
    latent: &[bool],
    anc: &dyn Fn(usize) -> bool,
    x: usize,
    y: usize,
) -> bool {
    // Depth-first over simple paths: (node, arrowhead into it on the last
    // edge, path so far).
    let mut on_path = vec![false; latent.len()];
    on_path[x] = true;
    fn go(
        adjacency: &[Vec<(usize, bool)>],
        latent: &[bool],
        anc: &dyn Fn(usize) -> bool,
        y: usize,
        c: usize,
        into_c: Option<bool>,
        on_path: &mut [bool],
    ) -> bool {
        for &(next, arrow_at_c) in &adjacency[c] {
            if on_path[next] {
                continue;
            }
            if let Some(into) = into_c {
                let collider = into && arrow_at_c;
                if (!latent[c] && !collider) || (collider && !anc(c)) {
                    continue;
                }
            }
            if next == y {
                return true;
            }
            // The arrowhead at `next` on the edge c–next.
            let arrow_at_next = adjacency[next]
                .iter()
                .find(|(m, _)| *m == c)
                .is_some_and(|(_, a)| *a);
            on_path[next] = true;
            if go(
                adjacency,
                latent,
                anc,
                y,
                next,
                Some(arrow_at_next),
                on_path,
            ) {
                return true;
            }
            on_path[next] = false;
        }
        false
    }
    go(adjacency, latent, anc, y, x, None, &mut on_path)
}

/// Every discriminating path for `b` ending in `c` (Def. 16.1.6), as node
/// sequences `⟨w, v1, …, vk, b, c⟩`: `k ≥ 1`, each `vi` is a collider on the
/// path and a parent of `c`, and `w` is not adjacent to `c`. The marks are
/// read as they stand, so this works on PAGs too (rule R4).
pub(crate) fn discriminating_paths(p: &Pag, b: usize, c: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if !p.adj(b, c) {
        return out;
    }
    // Grow backwards from b: `path` is [v_j, …, v_k, b] reversed order kept
    // as a stack from the inner node nearest w.
    fn grow(p: &Pag, c: usize, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let v = *path.last().expect("non-empty");
        for x in p.neighbors(v) {
            if x == c || path.contains(&x) || p.at(x, v) != Some(Mark::Arrow) {
                continue;
            }
            if !p.adj(x, c) {
                let mut found: Vec<usize> = path.iter().rev().copied().collect();
                found.insert(0, x);
                found.push(c);
                out.push(found);
            } else if p.directed(x, c) && p.at(v, x) == Some(Mark::Arrow) {
                path.push(x);
                grow(p, c, path, out);
                path.pop();
            }
        }
    }
    for v in p.neighbors(b) {
        // v is the inner node next to b: a collider (arrowhead from b) and a
        // parent of c.
        if v == c || !p.directed(v, c) || p.at(b, v) != Some(Mark::Arrow) {
            continue;
        }
        let mut path = vec![b, v];
        grow(p, c, &mut path, &mut out);
    }
    out
}

/// Whether two MAGs over the same variables are Markov equivalent
/// (Prop. 16.1.1): same adjacencies, same unshielded colliders, and the
/// same collider status for every node on a path that is discriminating in
/// both.
///
/// # Errors
///
/// [`DiscoveryError::Mismatch`] when the variables differ.
pub fn markov_equivalent(m1: &Pag, m2: &Pag) -> Result<bool, DiscoveryError> {
    let a: BTreeSet<&String> = m1.names().iter().collect();
    let b: BTreeSet<&String> = m2.names().iter().collect();
    if a != b {
        return Err(DiscoveryError::Mismatch);
    }
    // Re-index m2 onto m1's order.
    let map: Vec<usize> = m1
        .names()
        .iter()
        .map(|n| m2.index(n).expect("same names"))
        .collect();
    let n = m1.len();
    let at2 = |i: usize, j: usize| m2.at(map[i], map[j]);
    for i in 0..n {
        for j in 0..n {
            if i != j && m1.adj(i, j) != m2.adj(map[i], map[j]) {
                return Ok(false);
            }
        }
    }
    let collider1 = |x: usize, y: usize, z: usize| {
        m1.at(x, y) == Some(Mark::Arrow) && m1.at(z, y) == Some(Mark::Arrow)
    };
    let collider2 = |x: usize, y: usize, z: usize| {
        at2(x, y) == Some(Mark::Arrow) && at2(z, y) == Some(Mark::Arrow)
    };
    for y in 0..n {
        for x in m1.neighbors(y) {
            for z in m1.neighbors(y) {
                if x < z && !m1.adj(x, z) && collider1(x, y, z) != collider2(x, y, z) {
                    return Ok(false);
                }
            }
        }
    }
    // Discriminating paths of m1 that are discriminating in m2 too.
    let m2_reindexed = {
        let names: Vec<&str> = m1.names().iter().map(String::as_str).collect();
        let mut p = Pag::new(&names);
        for i in 0..n {
            for j in 0..n {
                if i < j && m1.adj(i, j) {
                    p.set(
                        i,
                        j,
                        at2(j, i).expect("adjacent"),
                        at2(i, j).expect("adjacent"),
                    );
                }
            }
        }
        p
    };
    for y in 0..n {
        for z in m1.neighbors(y) {
            let in_two = discriminating_paths(&m2_reindexed, y, z);
            for path in discriminating_paths(m1, y, z) {
                if in_two.contains(&path) {
                    let w = path[path.len() - 3];
                    if collider1(w, y, z) != collider2(w, y, z) {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}
