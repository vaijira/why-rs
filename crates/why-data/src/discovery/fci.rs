//! The FCI engine (Algorithm 54) shared by FCI, I-FCI and Ψ-FCI, and the
//! PC-style orientation of Ψ-PC.

use std::collections::{HashMap, HashSet};

use super::mag::discriminating_paths;
use super::{DiscoveryError, Independence, Mark, Pag};

/// Every subset of `items`, smallest first, each in the items' order.
pub(crate) fn subsets(items: &[usize]) -> Vec<Vec<usize>> {
    fn combos(
        items: &[usize],
        size: usize,
        from: usize,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for i in from..items.len() {
            cur.push(items[i]);
            combos(items, size, i + 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    for size in 0..=items.len() {
        combos(items, size, 0, &mut Vec::new(), &mut out);
    }
    out
}

/// A separating set the skeleton callback returns for an edge removed
/// without a known separating set (by background knowledge). Phase II and
/// R4 leave such pairs alone.
pub(crate) const UNKNOWN_SEPSET: usize = usize::MAX;

/// The state of a constraint-based learner after Phase I.
pub(crate) struct Engine {
    pub pag: Pag,
    sepsets: HashMap<(usize, usize), Vec<usize>>,
    /// Pairs removed without a known separating set.
    unknown: HashSet<(usize, usize)>,
    /// F-nodes and the symmetric difference each stands for (`None` when
    /// unknown, as in Ψ-FCI).
    fnodes: Vec<(usize, Option<Vec<usize>>)>,
    /// Whether R12 applies (I-FCI only: it needs the targets).
    r12: bool,
}

impl Engine {
    /// Phase I: starts from the complete circle graph over `names` and
    /// removes each edge `separate` finds a separating set for.
    pub fn skeleton(
        names: &[&str],
        fnodes: Vec<(usize, Option<Vec<usize>>)>,
        r12: bool,
        mut separate: impl FnMut(usize, usize) -> Result<Option<Vec<usize>>, DiscoveryError>,
    ) -> Result<Self, DiscoveryError> {
        let mut pag = Pag::complete(names);
        let mut sepsets = HashMap::new();
        let mut unknown = HashSet::new();
        for i in 0..names.len() {
            for j in i + 1..names.len() {
                if let Some(s) = separate(i, j)? {
                    pag.remove(i, j);
                    if s == [UNKNOWN_SEPSET] {
                        unknown.insert((i, j));
                        unknown.insert((j, i));
                    } else {
                        sepsets.insert((i, j), s.clone());
                        sepsets.insert((j, i), s);
                    }
                }
            }
        }
        Ok(Self {
            pag,
            sepsets,
            unknown,
            fnodes,
            r12,
        })
    }

    /// Non-adjacent with a known separating set: only then does an
    /// unshielded triple certify a non-collider.
    fn apart(&self, x: usize, y: usize) -> bool {
        !self.pag.adj(x, y) && !self.unknown.contains(&(x, y))
    }

    fn in_sepset(&self, x: usize, y: usize, z: usize) -> bool {
        self.sepsets.get(&(x, y)).is_some_and(|s| s.contains(&z))
    }

    /// Sets the mark at `j` on edge `i`–`j` when it is a circle.
    fn orient(&mut self, i: usize, j: usize, m: Mark) -> bool {
        if self.pag.at(i, j) == Some(Mark::Circle) {
            self.pag.set_at(i, j, m);
            true
        } else {
            false
        }
    }

    /// Phase II (R0): `x *-> z <-* y` for every unshielded triple whose
    /// middle is outside the separating set.
    pub fn colliders(&mut self) {
        let n = self.pag.len();
        for z in 0..n {
            let nb: Vec<usize> = self.pag.neighbors(z).collect();
            for (a, &x) in nb.iter().enumerate() {
                for &y in &nb[a + 1..] {
                    if !self.pag.adj(x, y)
                        && !self.unknown.contains(&(x, y))
                        && !self.in_sepset(x, y, z)
                    {
                        self.orient(x, z, Mark::Arrow);
                        self.orient(y, z, Mark::Arrow);
                    }
                }
            }
        }
    }

    /// R11 / R+: every edge at an F-node points out of it.
    fn out_of_f(&mut self) -> bool {
        let mut changed = false;
        for (f, _) in self.fnodes.clone() {
            for y in self.pag.neighbors(f).collect::<Vec<_>>() {
                changed |= self.orient(y, f, Mark::Tail);
                changed |= self.orient(f, y, Mark::Arrow);
            }
        }
        changed
    }

    /// R12: an F-node with `K = {x}` adjacent to `y ∉ K` makes `x → y`.
    fn r12(&mut self) -> bool {
        let mut changed = false;
        for (f, k) in self.fnodes.clone() {
            let Some(k) = k else { continue };
            if k.len() != 1 {
                continue;
            }
            let x = k[0];
            for y in self.pag.neighbors(f).collect::<Vec<_>>() {
                if y != x && self.pag.adj(x, y) {
                    changed |= self.orient(y, x, Mark::Tail);
                    changed |= self.orient(x, y, Mark::Arrow);
                }
            }
        }
        changed
    }

    /// Phase III: R11 (or R+), then R1-R4 and R8-R10 (and R12 for I-FCI)
    /// until none applies.
    pub fn fci_rules(&mut self) {
        loop {
            let mut changed = self.out_of_f();
            if self.r12 {
                changed |= self.r12();
            }
            changed |= self.r1() | self.r2() | self.r3() | self.r4() | self.r8() | self.r9_r10();
            if !changed {
                break;
            }
        }
    }

    fn r1(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for b in 0..n {
            for a in self.pag.neighbors(b).collect::<Vec<_>>() {
                if self.pag.at(a, b) != Some(Mark::Arrow) {
                    continue;
                }
                for c in self.pag.neighbors(b).collect::<Vec<_>>() {
                    if c != a && self.apart(a, c) && self.pag.at(c, b) == Some(Mark::Circle) {
                        changed |= self.orient(c, b, Mark::Tail);
                        changed |= self.orient(b, c, Mark::Arrow);
                    }
                }
            }
        }
        changed
    }

    fn r2(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for a in 0..n {
            for c in self.pag.neighbors(a).collect::<Vec<_>>() {
                if self.pag.at(a, c) != Some(Mark::Circle) {
                    continue;
                }
                let found = self.pag.neighbors(a).any(|b| {
                    b != c
                        && self.pag.adj(b, c)
                        && ((self.pag.directed(a, b) && self.pag.at(b, c) == Some(Mark::Arrow))
                            || (self.pag.at(a, b) == Some(Mark::Arrow) && self.pag.directed(b, c)))
                });
                if found {
                    changed |= self.orient(a, c, Mark::Arrow);
                }
            }
        }
        changed
    }

    fn r3(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for b in 0..n {
            let into: Vec<usize> = self
                .pag
                .neighbors(b)
                .filter(|&a| self.pag.at(a, b) == Some(Mark::Arrow))
                .collect();
            for (k, &a) in into.iter().enumerate() {
                for &c in &into[k + 1..] {
                    if !self.apart(a, c) {
                        continue;
                    }
                    for d in self.pag.neighbors(b).collect::<Vec<_>>() {
                        if d != a
                            && d != c
                            && self.pag.adj(a, d)
                            && self.pag.adj(c, d)
                            && self.pag.at(a, d) == Some(Mark::Circle)
                            && self.pag.at(c, d) == Some(Mark::Circle)
                            && self.pag.at(d, b) == Some(Mark::Circle)
                        {
                            changed |= self.orient(d, b, Mark::Arrow);
                        }
                    }
                }
            }
        }
        changed
    }

    fn r4(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for b in 0..n {
            for c in self.pag.neighbors(b).collect::<Vec<_>>() {
                if self.pag.at(c, b) != Some(Mark::Circle) {
                    continue;
                }
                if let Some(path) = discriminating_paths(&self.pag, b, c)
                    .into_iter()
                    .find(|p| !self.unknown.contains(&(p[0], c)))
                {
                    let w = path[0];
                    let a = path[path.len() - 3];
                    if self.in_sepset(w, c, b) {
                        changed |= self.orient(c, b, Mark::Tail);
                        changed |= self.orient(b, c, Mark::Arrow);
                    } else {
                        changed |= self.orient(a, b, Mark::Arrow);
                        changed |= self.orient(b, a, Mark::Arrow);
                        changed |= self.orient(c, b, Mark::Arrow);
                        changed |= self.orient(b, c, Mark::Arrow);
                    }
                }
            }
        }
        changed
    }

    fn r8(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for a in 0..n {
            for c in self.pag.neighbors(a).collect::<Vec<_>>() {
                // a o-> c
                if self.pag.at(c, a) != Some(Mark::Circle) || self.pag.at(a, c) != Some(Mark::Arrow)
                {
                    continue;
                }
                let found = self.pag.neighbors(a).any(|b| {
                    b != c
                        && self.pag.directed(b, c)
                        && (self.pag.directed(a, b)
                            || (self.pag.at(b, a) == Some(Mark::Tail)
                                && self.pag.at(a, b) == Some(Mark::Circle)))
                });
                if found {
                    changed |= self.orient(c, a, Mark::Tail);
                }
            }
        }
        changed
    }

    /// Whether the edge from `u` to `v` can lie on a potentially directed
    /// path: no arrowhead at `u`, no tail at `v`.
    fn pd(&self, u: usize, v: usize) -> bool {
        self.pag.adj(u, v)
            && self.pag.at(v, u) != Some(Mark::Arrow)
            && self.pag.at(u, v) != Some(Mark::Tail)
    }

    /// The second nodes of every uncovered potentially directed path from
    /// `a` to `target` that avoids `avoid`.
    fn updp_seconds(&self, a: usize, target: usize, avoid: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut path = vec![a];
        self.updp(target, avoid, &mut path, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn updp(&self, target: usize, avoid: usize, path: &mut Vec<usize>, out: &mut Vec<usize>) {
        let u = *path.last().expect("non-empty");
        for v in self.pag.neighbors(u).collect::<Vec<_>>() {
            if v == avoid || path.contains(&v) || !self.pd(u, v) {
                continue;
            }
            // Uncovered: the node two back is not adjacent to v.
            if path.len() >= 2 && !self.apart(path[path.len() - 2], v) {
                continue;
            }
            if v == target {
                out.push(*path.get(1).unwrap_or(&v));
                continue;
            }
            path.push(v);
            self.updp(target, avoid, path, out);
            path.pop();
        }
    }

    fn r9_r10(&mut self) -> bool {
        let n = self.pag.len();
        let mut changed = false;
        for a in 0..n {
            for c in self.pag.neighbors(a).collect::<Vec<_>>() {
                if self.pag.at(c, a) != Some(Mark::Circle) || self.pag.at(a, c) != Some(Mark::Arrow)
                {
                    continue;
                }
                // R9: an uncovered p.d. path a, b, …, c with b not adjacent to c.
                let r9 = self.pag.neighbors(a).any(|b| {
                    b != c && self.apart(b, c) && self.pd(a, b) && {
                        let mut path = vec![a, b];
                        let mut out = Vec::new();
                        self.updp(c, usize::MAX, &mut path, &mut out);
                        !out.is_empty()
                    }
                });
                if r9 {
                    changed |= self.orient(c, a, Mark::Tail);
                    continue;
                }
                // R10: b → c ← d, with uncovered p.d. paths from a to b and to d
                // whose first steps are distinct and non-adjacent.
                let parents: Vec<usize> = self
                    .pag
                    .neighbors(c)
                    .filter(|&b| b != a && self.pag.directed(b, c))
                    .collect();
                let mut r10 = false;
                'outer: for (k, &b) in parents.iter().enumerate() {
                    for &d in &parents[k + 1..] {
                        let mu = self.updp_seconds(a, b, c);
                        let omega = self.updp_seconds(a, d, c);
                        for &m in &mu {
                            for &w in &omega {
                                if m != w && self.apart(m, w) {
                                    r10 = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
                if r10 {
                    changed |= self.orient(c, a, Mark::Tail);
                }
            }
        }
        changed
    }

    /// Ψ-PC's orientation: colliders keep tails at their ends, F-node edges
    /// point out, then Meek's rules R1-R3 (Fig. 16.17) until none applies.
    /// Undirected edges stay `o-o`.
    pub fn pc_rules(&mut self) {
        let n = self.pag.len();
        // Arrowheads from Phase II get tails at their other end.
        for i in 0..n {
            for j in 0..n {
                if self.pag.at(i, j) == Some(Mark::Arrow) && self.pag.at(j, i) == Some(Mark::Circle)
                {
                    self.pag.set_at(j, i, Mark::Tail);
                }
            }
        }
        let undirected = |p: &Pag, a: usize, b: usize| {
            p.at(a, b) == Some(Mark::Circle) && p.at(b, a) == Some(Mark::Circle)
        };
        loop {
            let mut changed = self.out_of_f();
            for b in 0..n {
                for c in self.pag.neighbors(b).collect::<Vec<_>>() {
                    if !undirected(&self.pag, b, c) {
                        continue;
                    }
                    // R1: a → b o-o c, a not adjacent to c.
                    let r1 = self
                        .pag
                        .neighbors(b)
                        .any(|a| a != c && self.pag.directed(a, b) && self.apart(a, c));
                    // R2: b → a → c with b o-o c.
                    let r2 = self
                        .pag
                        .neighbors(b)
                        .any(|a| a != c && self.pag.directed(b, a) && self.pag.directed(a, c));
                    // R3 (orienting b → c): b o-o x, b o-o y, x → c ← y, x, y
                    // non-adjacent.
                    let both: Vec<usize> = self
                        .pag
                        .neighbors(b)
                        .filter(|&x| {
                            x != c && undirected(&self.pag, b, x) && self.pag.directed(x, c)
                        })
                        .collect();
                    let r3 = both
                        .iter()
                        .enumerate()
                        .any(|(k, &x)| both[k + 1..].iter().any(|&y| self.apart(x, y)));
                    if r1 || r2 || r3 {
                        self.pag.set(b, c, Mark::Tail, Mark::Arrow);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// Algorithm 54 (FCI): the PAG of the Markov equivalence class that the
/// independence tests imply. Phase I searches every subset for a separating
/// set, smallest first. Phase II orients unshielded colliders, and Phase III
/// applies rules R1-R4 and R8-R10 (Fig. 16.6) until none applies.
///
/// ```
/// use why_data::discovery::{GraphOracle, fci};
/// use why_data::graph::dseparation::Admg;
///
/// // Fig. 16.5(a): A -> B <- D, B -> C, D <-> C. Ex. 16.7 learns
/// // Fig. 16.5(d).
/// let mut g = Admg::new();
/// for (a, b) in [("A", "B"), ("D", "B"), ("B", "C")] {
///     g.directed(a, b);
/// }
/// g.bidirected("D", "C");
/// let pag = fci(&["A", "B", "C", "D"], &mut GraphOracle::new(g)).unwrap();
/// assert_eq!(pag.edges(), ["A o-> B", "B --> C", "B <-o D", "C <-- D"]);
/// ```
///
/// # Errors
///
/// Whatever the tests report.
pub fn fci(names: &[&str], test: &mut impl Independence) -> Result<Pag, DiscoveryError> {
    let n = names.len();
    let mut engine = Engine::skeleton(names, Vec::new(), false, |i, j| {
        let others: Vec<usize> = (0..n).filter(|&k| k != i && k != j).collect();
        for z in subsets(&others) {
            let zs: Vec<&str> = z.iter().map(|&k| names[k]).collect();
            if test.independent(names[i], names[j], &zs)? {
                return Ok(Some(z));
            }
        }
        Ok(None)
    })?;
    engine.colliders();
    engine.fci_rules();
    Ok(engine.pag)
}
