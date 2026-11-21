//! Linear structural causal models (Secs. 19.2-19.5): covariances, treks,
//! regression, and the single-door, back-door, instrumental-variable and
//! instrumental-set criteria.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::ParametricError;
use crate::discovery::subsets;
use crate::graph::dseparation::{Admg, EdgeKind};

/// Edges as `(from, to)` name pairs.
type EdgeList = Vec<(String, String)>;

/// The directed and bidirected edges of `g`, by name.
pub(crate) fn edges(g: &Admg) -> (EdgeList, EdgeList) {
    let graph = g.graph();
    let (mut d, mut b) = (Vec::new(), Vec::new());
    for e in graph.edge_indices() {
        let (x, y) = graph.edge_endpoints(e).expect("valid edge");
        let pair = (graph[x].clone(), graph[y].clone());
        match graph[e] {
            EdgeKind::Directed => d.push(pair),
            EdgeKind::Bidirected => b.push(pair),
        }
    }
    (d, b)
}

/// `g` without the directed edge `x → y`.
fn without_edge(g: &Admg, x: &str, y: &str) -> Admg {
    let (d, b) = edges(g);
    let mut out = Admg::new();
    for v in g.variables() {
        out.add_node(v);
    }
    for (a, c) in d {
        if !(a == x && c == y) {
            out.directed(&a, &c);
        }
    }
    for (a, c) in b {
        out.bidirected(&a, &c);
    }
    out
}

/// Solves `a · x = b` by Gaussian elimination with partial pivoting;
/// `None` when `a` is singular.
pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// A linear SCM `V = Λᵀ V + U` with `Cov(U) = Ω` (Eqs. 19.3, 19.5).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearScm {
    names: Vec<String>,
    /// `lambda[i][j]`: the coefficient of `V_i → V_j`.
    lambda: Vec<Vec<f64>>,
    /// `omega[i][j]`: the error covariance `ε_ij`.
    omega: Vec<Vec<f64>>,
}

impl LinearScm {
    /// The linear SCM over `g`'s variables with the given coefficients
    /// (`(from, to, λ)` for directed edges), error variances (`(v, ε_vv)`;
    /// 1 when omitted) and error covariances (`(a, b, ε_ab)` for bidirected
    /// edges).
    ///
    /// # Errors
    ///
    /// [`ParametricError::NotAnEdge`] for a coefficient or covariance on a
    /// pair `g` does not join with that kind of edge, and
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn new(
        g: &Admg,
        coefficients: &[(&str, &str, f64)],
        variances: &[(&str, f64)],
        covariances: &[(&str, &str, f64)],
    ) -> Result<Self, ParametricError> {
        let mut names: Vec<String> = g.variables().iter().map(|v| (*v).to_string()).collect();
        names.sort();
        let n = names.len();
        let idx = |v: &str| {
            names
                .iter()
                .position(|m| m == v)
                .ok_or_else(|| ParametricError::UnknownVariable(v.to_string()))
        };
        let (d, b) = edges(g);
        let has_d = |x: &str, y: &str| d.iter().any(|(a, c)| a == x && c == y);
        let has_b = |x: &str, y: &str| {
            b.iter()
                .any(|(a, c)| (a == x && c == y) || (a == y && c == x))
        };
        let mut lambda = vec![vec![0.0; n]; n];
        let mut omega = vec![vec![0.0; n]; n];
        for (i, row) in omega.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for (x, y, l) in coefficients {
            if !has_d(x, y) {
                return Err(ParametricError::NotAnEdge(format!("{x} -> {y}")));
            }
            lambda[idx(x)?][idx(y)?] = *l;
        }
        for (v, e) in variances {
            let i = idx(v)?;
            omega[i][i] = *e;
        }
        for (x, y, e) in covariances {
            if !has_b(x, y) {
                return Err(ParametricError::NotAnEdge(format!("{x} <-> {y}")));
            }
            let (i, j) = (idx(x)?, idx(y)?);
            omega[i][j] = *e;
            omega[j][i] = *e;
        }
        Ok(Self {
            names,
            lambda,
            omega,
        })
    }

    /// The variables, in name order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    fn idx(&self, v: &str) -> Result<usize, ParametricError> {
        self.names
            .iter()
            .position(|m| m == v)
            .ok_or_else(|| ParametricError::UnknownVariable(v.to_string()))
    }

    /// The structural coefficient `λ_xy`.
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn coefficient(&self, x: &str, y: &str) -> Result<f64, ParametricError> {
        Ok(self.lambda[self.idx(x)?][self.idx(y)?])
    }

    /// `(I − Λ)⁻¹`, whose `(i, j)` entry sums the directed paths from `i`
    /// to `j` (Eq. 19.35).
    fn paths(&self) -> Vec<Vec<f64>> {
        let n = self.names.len();
        // Columns of the inverse of I − Λ, by solving column by column.
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| f64::from(u8::from(i == j)) - self.lambda[i][j])
                    .collect()
            })
            .collect();
        let mut inv = vec![vec![0.0; n]; n];
        for j in 0..n {
            let e: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i == j))).collect();
            let col = solve(a.clone(), e).expect("I − Λ is unit triangular after reordering");
            for (row, c) in inv.iter_mut().zip(col) {
                row[j] = c;
            }
        }
        inv
    }

    /// The observational covariance `Σ = (I − Λ)⁻ᵀ Ω (I − Λ)⁻¹` (Eq. 19.22).
    #[must_use]
    pub fn covariance(&self) -> Covariance {
        let n = self.names.len();
        let p = self.paths();
        // Σ = Pᵀ Ω P.
        let mut op = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                op[i][j] = (0..n).map(|k| self.omega[i][k] * p[k][j]).sum();
            }
        }
        let mut sigma = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                sigma[i][j] = (0..n).map(|k| p[k][i] * op[k][j]).sum();
            }
        }
        Covariance {
            names: self.names.clone(),
            sigma,
        }
    }

    /// The total effect `δ_xy`, the sum of directed-path products from `x`
    /// to `y` (Eq. 19.16).
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn total_effect(&self, x: &str, y: &str) -> Result<f64, ParametricError> {
        Ok(self.paths()[self.idx(x)?][self.idx(y)?])
    }

    /// `σ_vw` by the trek rule (Eq. 19.38): the sum over the treks between
    /// `v` and `w` of their monomials (Def. 19.2.2). [`LinearScm::covariance`]
    /// computes the same number by matrix algebra.
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn trek_covariance(&self, v: &str, w: &str) -> Result<f64, ParametricError> {
        let (v, w) = (self.idx(v)?, self.idx(w)?);
        let n = self.names.len();
        // The two sides of a trek are directed paths from its top,
        // enumerated one by one: p[k][t] sums the coefficient products of
        // every directed path k -> … -> t (the empty path when k = t).
        fn walk(lambda: &[Vec<f64>], at: usize, product: f64, sums: &mut [f64]) {
            sums[at] += product;
            for (next, &l) in lambda[at].iter().enumerate() {
                if l != 0.0 {
                    walk(lambda, next, product * l, sums);
                }
            }
        }
        let p: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let mut sums = vec![0.0; n];
                walk(&self.lambda, k, 1.0, &mut sums);
                sums
            })
            .collect();
        // A trek: top k with ε_kk, or a bidirected edge a <-> b with ε_ab;
        // then directed paths to v and to w.
        let mut total = 0.0;
        for k in 0..n {
            total += p[k][v] * self.omega[k][k] * p[k][w];
        }
        for a in 0..n {
            for b in 0..n {
                if a != b && self.omega[a][b] != 0.0 {
                    total += p[a][v] * self.omega[a][b] * p[b][w];
                }
            }
        }
        Ok(total)
    }
}

/// An observed covariance matrix over named variables.
#[derive(Clone, Debug, PartialEq)]
pub struct Covariance {
    names: Vec<String>,
    sigma: Vec<Vec<f64>>,
}

impl Covariance {
    /// A covariance matrix from its variables and entries.
    ///
    /// # Errors
    ///
    /// [`ParametricError::InvalidInput`] when the matrix is not square, of
    /// the variables' size, and symmetric.
    pub fn new(names: &[&str], sigma: Vec<Vec<f64>>) -> Result<Self, ParametricError> {
        let n = names.len();
        if sigma.len() != n || sigma.iter().any(|r| r.len() != n) {
            return Err(ParametricError::InvalidInput(
                "Σ must be square, one row per variable".into(),
            ));
        }
        for i in 0..n {
            for j in 0..n {
                if (sigma[i][j] - sigma[j][i]).abs() > 1e-9 {
                    return Err(ParametricError::InvalidInput("Σ must be symmetric".into()));
                }
            }
        }
        Ok(Self {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            sigma,
        })
    }

    fn idx(&self, v: &str) -> Result<usize, ParametricError> {
        self.names
            .iter()
            .position(|m| m == v)
            .ok_or_else(|| ParametricError::UnknownVariable(v.to_string()))
    }

    /// `σ_vw`.
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn get(&self, v: &str, w: &str) -> Result<f64, ParametricError> {
        Ok(self.sigma[self.idx(v)?][self.idx(w)?])
    }

    /// The coefficient of `x` when `y` is regressed on `x` and `z`,
    /// `β_{yx.z}` (Eq. 19.93).
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name, and
    /// [`ParametricError::Singular`] when the regressors are collinear.
    pub fn regression(&self, y: &str, x: &str, z: &[&str]) -> Result<f64, ParametricError> {
        let mut regressors = vec![self.idx(x)?];
        for v in z {
            regressors.push(self.idx(v)?);
        }
        let y = self.idx(y)?;
        let a: Vec<Vec<f64>> = regressors
            .iter()
            .map(|&i| regressors.iter().map(|&j| self.sigma[i][j]).collect())
            .collect();
        let b: Vec<f64> = regressors.iter().map(|&i| self.sigma[i][y]).collect();
        Ok(solve(a, b).ok_or(ParametricError::Singular)?[0])
    }

    /// Solves `Σ_{Z,X} λ = Σ_{Z,y}` for the coefficients of the edges
    /// `X → y` (an instrumental set, Def. 19.5.3), in the order of `xs`.
    ///
    /// # Errors
    ///
    /// [`ParametricError::Singular`] when the system is degenerate.
    pub fn instrumental_set(
        &self,
        y: &str,
        xs: &[&str],
        zs: &[&str],
    ) -> Result<Vec<f64>, ParametricError> {
        let y = self.idx(y)?;
        let xi = xs
            .iter()
            .map(|v| self.idx(v))
            .collect::<Result<Vec<_>, _>>()?;
        let zi = zs
            .iter()
            .map(|v| self.idx(v))
            .collect::<Result<Vec<_>, _>>()?;
        let a: Vec<Vec<f64>> = zi
            .iter()
            .map(|&z| xi.iter().map(|&x| self.sigma[z][x]).collect())
            .collect();
        let b: Vec<f64> = zi.iter().map(|&z| self.sigma[z][y]).collect();
        solve(a, b).ok_or(ParametricError::Singular)
    }
}

/// A set `Z` that satisfies the single-door criterion for `λ_xy`
/// (Thm 19.3.1): no descendant of `y`, and `Z` d-separates `x` from `y` in
/// `G` without `x → y`. Then `λ_xy = β_{yx.Z}`. The smallest such set is
/// returned, if any.
///
/// # Errors
///
/// [`ParametricError::NotAnEdge`] when `x → y` is not an edge.
pub fn single_door(g: &Admg, x: &str, y: &str) -> Result<Option<Vec<String>>, ParametricError> {
    let (d, _) = edges(g);
    if !d.iter().any(|(a, b)| a == x && b == y) {
        return Err(ParametricError::NotAnEdge(format!("{x} -> {y}")));
    }
    let cut = without_edge(g, x, y);
    let desc: BTreeSet<&str> = g.descendants_of(&[y])?.into_iter().collect();
    let candidates: Vec<&str> = g
        .variables()
        .into_iter()
        .filter(|v| *v != x && *v != y && !desc.contains(v))
        .collect();
    search(&candidates, |z| Ok(cut.is_d_separated(&[x], &[y], z)?))
}

/// A set `Z` that satisfies the back-door criterion for the total effect
/// `δ_xy` (Thm 19.3.2): no descendant of `x`, and `Z` d-separates `x` from
/// `y` once the edges out of `x` are removed. Then `δ_xy = β_{yx.Z}`.
///
/// # Errors
///
/// [`ParametricError::UnknownVariable`] for an unknown name.
pub fn back_door(g: &Admg, x: &str, y: &str) -> Result<Option<Vec<String>>, ParametricError> {
    let cut = g.cut_edges_out_of(&[x])?;
    let desc: BTreeSet<&str> = g.descendants_of(&[x])?.into_iter().collect();
    let candidates: Vec<&str> = g
        .variables()
        .into_iter()
        .filter(|v| *v != x && *v != y && !desc.contains(v))
        .collect();
    search(&candidates, |z| Ok(cut.is_d_separated(&[x], &[y], z)?))
}

/// The smallest subset of `candidates` passing `test`.
fn search(
    candidates: &[&str],
    mut test: impl FnMut(&[&str]) -> Result<bool, ParametricError>,
) -> Result<Option<Vec<String>>, ParametricError> {
    let idx: Vec<usize> = (0..candidates.len()).collect();
    for s in subsets(&idx) {
        let z: Vec<&str> = s.iter().map(|&i| candidates[i]).collect();
        if test(&z)? {
            return Ok(Some(z.iter().map(|v| (*v).to_string()).collect()));
        }
    }
    Ok(None)
}

/// A conditional instrument `(z, W)` for `λ_xy` (Def. 19.4.2): `z` is
/// d-connected to `x` given `W`, `W` d-separates `z` from `y` in `G` without
/// `x → y`, and `W` has no descendant of `y`. Then
/// `λ_xy = β_{yz.W} / β_{xz.W}` (Thm 19.4.2). An unconditional instrument is
/// the case `W = ∅`.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::parametric::conditional_instrument;
///
/// // Ex. 19.19 (Fig. 19.17): W -> Z -> X -> Y, X <-> Y, W <-> Y.
/// let mut g = Admg::new();
/// for (a, b) in [("W", "Z"), ("Z", "X"), ("X", "Y")] {
///     g.directed(a, b);
/// }
/// g.bidirected("X", "Y");
/// g.bidirected("W", "Y");
/// let (z, w) = conditional_instrument(&g, "X", "Y").unwrap().unwrap();
/// assert_eq!((z.as_str(), w), ("Z", vec!["W".to_string()]));
/// ```
///
/// # Errors
///
/// [`ParametricError::NotAnEdge`] when `x → y` is not an edge.
pub fn conditional_instrument(
    g: &Admg,
    x: &str,
    y: &str,
) -> Result<Option<(String, Vec<String>)>, ParametricError> {
    let (d, _) = edges(g);
    if !d.iter().any(|(a, b)| a == x && b == y) {
        return Err(ParametricError::NotAnEdge(format!("{x} -> {y}")));
    }
    let cut = without_edge(g, x, y);
    let desc: BTreeSet<&str> = g.descendants_of(&[y])?.into_iter().collect();
    let mut zs: Vec<&str> = g
        .variables()
        .into_iter()
        .filter(|v| *v != x && *v != y)
        .collect();
    zs.sort_unstable();
    for z in zs {
        let candidates: Vec<&str> = g
            .variables()
            .into_iter()
            .filter(|v| *v != x && *v != y && *v != z && !desc.contains(v))
            .collect();
        if let Some(w) = search(&candidates, |w| {
            Ok(!g.is_d_separated(&[z], &[x], w)? && cut.is_d_separated(&[z], &[y], w)?)
        })? {
            return Ok(Some((z.to_string(), w)));
        }
    }
    Ok(None)
}

/// The flow graph (Def. 19.5.1): top copies `0..n` and bottom copies
/// `n..2n`. Edges: `j → i` (top) and `i′ → j′` (bottom) for `i → j`;
/// `i → i′`; `i → j′` and `j → i′` for `i ↔ j`.
struct FlowGraph {
    names: Vec<String>,
    adj: Vec<Vec<usize>>,
}

impl FlowGraph {
    fn new(g: &Admg) -> Self {
        let mut names: Vec<String> = g.variables().iter().map(|v| (*v).to_string()).collect();
        names.sort();
        let n = names.len();
        let idx = |v: &str| names.iter().position(|m| m == v).expect("a node");
        let mut adj = vec![Vec::new(); 2 * n];
        let (d, b) = edges(g);
        for (x, y) in &d {
            let (i, j) = (idx(x), idx(y));
            adj[j].push(i);
            adj[n + i].push(n + j);
        }
        for (i, row) in adj.iter_mut().enumerate().take(n) {
            row.push(n + i);
        }
        for (x, y) in &b {
            let (i, j) = (idx(x), idx(y));
            adj[i].push(n + j);
            adj[j].push(n + i);
        }
        Self { names, adj }
    }

    fn idx(&self, v: &str) -> Result<usize, ParametricError> {
        self.names
            .iter()
            .position(|m| m == v)
            .ok_or_else(|| ParametricError::UnknownVariable(v.to_string()))
    }

    /// Nodes reachable from `from`, with the edges in `banned` removed.
    fn reach(&self, from: usize, banned: &BTreeSet<(usize, usize)>) -> BTreeSet<usize> {
        let mut seen = BTreeSet::from([from]);
        let mut queue = VecDeque::from([from]);
        while let Some(u) = queue.pop_front() {
            for &v in &self.adj[u] {
                if !banned.contains(&(u, v)) && seen.insert(v) {
                    queue.push_back(v);
                }
            }
        }
        seen
    }

    /// A vertex-disjoint maximum flow from `sources` to `sinks` (unit
    /// vertex capacities, Thm 19.5.2): the flow value and the sinks and
    /// sources it uses.
    fn max_flow(&self, sources: &[usize], sinks: &[usize]) -> (usize, Vec<usize>, Vec<usize>) {
        // Split each vertex v into v_in = 2v, v_out = 2v + 1; super source
        // and sink at the end.
        let m = self.adj.len();
        let (s, t) = (2 * m, 2 * m + 1);
        let size = 2 * m + 2;
        let mut cap: BTreeMap<(usize, usize), i32> = BTreeMap::new();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); size];
        let mut add = |a: usize, b: usize, cap: &mut BTreeMap<(usize, usize), i32>| {
            if !cap.contains_key(&(a, b)) {
                out[a].push(b);
                out[b].push(a);
            }
            *cap.entry((a, b)).or_insert(0) += 1;
            cap.entry((b, a)).or_insert(0);
        };
        for (v, ws) in self.adj.iter().enumerate() {
            add(2 * v, 2 * v + 1, &mut cap);
            for &w in ws {
                add(2 * v + 1, 2 * w, &mut cap);
            }
        }
        for &v in sources {
            add(s, 2 * v, &mut cap);
        }
        for &v in sinks {
            add(2 * v + 1, t, &mut cap);
        }
        let mut flow = 0;
        loop {
            let mut prev = vec![usize::MAX; size];
            prev[s] = s;
            let mut queue = VecDeque::from([s]);
            while let Some(u) = queue.pop_front() {
                for &v in &out[u] {
                    if prev[v] == usize::MAX && cap[&(u, v)] > 0 {
                        prev[v] = u;
                        queue.push_back(v);
                    }
                }
            }
            if prev[t] == usize::MAX {
                break;
            }
            let mut v = t;
            while v != s {
                let u = prev[v];
                *cap.get_mut(&(u, v)).expect("edge") -= 1;
                *cap.get_mut(&(v, u)).expect("edge") += 1;
                v = u;
            }
            flow += 1;
        }
        let used_sinks = sinks
            .iter()
            .copied()
            .filter(|&v| cap[&(2 * v + 1, t)] == 0)
            .collect();
        let used_sources = sources
            .iter()
            .copied()
            .filter(|&v| cap[&(s, 2 * v)] == 0)
            .collect();
        (flow, used_sinks, used_sources)
    }
}

/// The rank of `Σ_{S,T}` for generic parameters (Thm 19.5.2): the vertex
/// max-flow from the top copies of `S` to the bottom copies of `T` in the
/// flow graph.
///
/// # Errors
///
/// [`ParametricError::UnknownVariable`] for an unknown name.
pub fn generic_rank(g: &Admg, s: &[&str], t: &[&str]) -> Result<usize, ParametricError> {
    let fg = FlowGraph::new(g);
    let n = fg.names.len();
    let si = s.iter().map(|v| fg.idx(v)).collect::<Result<Vec<_>, _>>()?;
    let ti = t
        .iter()
        .map(|v| Ok(n + fg.idx(v)?))
        .collect::<Result<Vec<_>, ParametricError>>()?;
    Ok(fg.max_flow(&si, &ti).0)
}

/// The largest instrumental set for edges into `y` (Def. 19.5.3), found by
/// Algorithm 66 (maximal match-block) in the flow graph: candidates are the
/// variables whose every trek to `y` ends with a directed edge into `y`, and
/// sinks are `y`'s parents. Returns `(Z, X)` with `|Z| = |X|`, `X ⊆ Pa(y)`.
/// Then `Σ_{Z,X} λ = Σ_{Z,y}` determines `λ_{x y}` for every `x ∈ X`.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::parametric::instrumental_set;
///
/// // Ex. 19.20 (Fig. 19.18): Z1, Z2 -> X1, X2 -> Y with X1 <-> Y, X2 <-> Y.
/// let mut g = Admg::new();
/// for z in ["Z1", "Z2"] {
///     for x in ["X1", "X2"] {
///         g.directed(z, x);
///     }
/// }
/// for x in ["X1", "X2"] {
///     g.directed(x, "Y");
///     g.bidirected(x, "Y");
/// }
/// let (z, x) = instrumental_set(&g, "Y").unwrap();
/// assert_eq!((z, x), (vec!["Z1".to_string(), "Z2".into()], vec!["X1".to_string(), "X2".into()]));
/// ```
///
/// # Errors
///
/// [`ParametricError::UnknownVariable`] for an unknown `y`.
pub fn instrumental_set(g: &Admg, y: &str) -> Result<(Vec<String>, Vec<String>), ParametricError> {
    let fg = FlowGraph::new(g);
    let n = fg.names.len();
    let yi = fg.idx(y)?;
    let (d, _) = edges(g);
    let parents: Vec<usize> = d
        .iter()
        .filter(|(_, c)| c == y)
        .map(|(p, _)| fg.idx(p).expect("a node"))
        .collect();
    // Edges x′ → y′ for every parent.
    let into_y: BTreeSet<(usize, usize)> = parents.iter().map(|&p| (n + p, n + yi)).collect();
    let mut sources: Vec<usize> = (0..n)
        .filter(|&z| z != yi && !fg.reach(z, &into_y).contains(&(n + yi)))
        .collect();
    let mut sinks: Vec<usize> = parents.iter().map(|&p| n + p).collect();
    loop {
        let (_, used_sinks, used_sources) = fg.max_flow(&sources, &sinks);
        let dead: Vec<usize> = sinks
            .iter()
            .copied()
            .filter(|t| !used_sinks.contains(t))
            .collect();
        if dead.is_empty() {
            let mut z: Vec<String> = used_sources.iter().map(|&s| fg.names[s].clone()).collect();
            let mut x: Vec<String> = sinks.iter().map(|&t| fg.names[t - n].clone()).collect();
            z.sort();
            x.sort();
            return Ok((z, x));
        }
        sinks.retain(|t| !dead.contains(t));
        sources.retain(|&s| {
            let r = fg.reach(s, &BTreeSet::new());
            !dead.iter().any(|t| r.contains(t))
        });
    }
}
