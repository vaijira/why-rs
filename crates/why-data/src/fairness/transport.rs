//! Task 2, fair prediction by causally aware pre-processing: Causal
//! Individual Fairness (Def. 6.1.34, Algorithm 16), and the exact discrete
//! optimal-transport solver it runs on.
//!
//! Joint optimal transport of `(W, Y) | x1` onto `(W, Y) | x0` fails
//! (Ex. 6.24): the map mixes the noise of `W` and `Y`, which breaks
//! `Ỹ_{x0,w} ⊥ W̃_{x1}`. Causal IF instead transports one variable at a time,
//! in topological order `Z → W → Y`, each conditionally on its already
//! transported parents. Theorem 6.1.33 then gives `x-DE^sym(ỹ) = 0`, plus
//! `x-IE^sym(ỹ) = 0` when `W ∉ BN` and `x-SE(ỹ) = 0` when `Z ∉ BN`.
//!
//! Only the `x0` rows move; `x1` is the reference group. Optimal-transport
//! plans between discrete distributions split mass, so the map is stochastic.
//! [`CausalIf::transform`] returns the exact transformed distribution as a
//! weighted table, while [`CausalIf::transform_sampled`] draws one transported
//! value per row.

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use super::cookbook::BusinessNecessity;
use super::dataset::Column;
use super::estimate::{Sfm, contrast_codes, named_columns, numeric_levels};
use super::{Dataset, FairnessError, SplitMix64};

/// A conditional coupling: for each source point, `(target, P(target | source))`.
type Coupling<K> = HashMap<K, Vec<(K, f64)>>;

/// Exact optimal transport between two discrete distributions, returned as
/// the conditional `P(target j | source i)` for each source `i`. Both inputs
/// are normalised. Solved as a min-cost flow by successive shortest paths
/// with Dijkstra potentials. `cost` must be non-negative.
pub(crate) fn transport_plan(
    a: &[f64],
    b: &[f64],
    cost: &dyn Fn(usize, usize) -> f64,
) -> Result<Vec<Vec<(usize, f64)>>, FairnessError> {
    let (n, m) = (a.len(), b.len());
    let (sa, sb) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    if n == 0 || m == 0 || sa <= 0.0 || sb <= 0.0 {
        return Err(FairnessError::InvalidInput("transport between empty distributions".into()));
    }
    let nodes = n + m + 2;
    let (s, t) = (0, n + m + 1);
    struct Graph {
        to: Vec<usize>,
        cap: Vec<f64>,
        cst: Vec<f64>,
        adj: Vec<Vec<usize>>,
    }
    impl Graph {
        fn add(&mut self, u: usize, v: usize, c: f64, w: f64) -> usize {
            let e = self.to.len();
            self.adj[u].push(e);
            self.to.push(v);
            self.cap.push(c);
            self.cst.push(w);
            self.adj[v].push(e + 1);
            self.to.push(u);
            self.cap.push(0.0);
            self.cst.push(-w);
            e
        }
    }
    let mut gr = Graph {
        to: Vec::new(),
        cap: Vec::new(),
        cst: Vec::new(),
        adj: vec![Vec::new(); nodes],
    };
    for (i, ai) in a.iter().enumerate() {
        gr.add(s, 1 + i, ai / sa, 0.0);
    }
    let mut pair_edge = vec![vec![0usize; m]; n];
    for (i, row) in pair_edge.iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = gr.add(1 + i, 1 + n + j, 2.0, cost(i, j));
        }
    }
    for (j, bj) in b.iter().enumerate() {
        gr.add(1 + n + j, t, bj / sb, 0.0);
    }
    let Graph { to, mut cap, cst, adj } = gr;

    const EPS: f64 = 1e-13;
    let mut h = vec![0.0; nodes];
    let mut flow = 0.0;
    let mut guard = 0;
    while flow < 1.0 - 1e-12 {
        guard += 1;
        if guard > 4 * (n + m) * (n + m) + 100 {
            return Err(FairnessError::InvalidInput("optimal transport did not converge".into()));
        }
        // Dense Dijkstra on reduced costs.
        let mut dist = vec![f64::INFINITY; nodes];
        let mut prev = vec![usize::MAX; nodes];
        let mut done = vec![false; nodes];
        dist[s] = 0.0;
        for _ in 0..nodes {
            let Some(u) = (0..nodes)
                .filter(|&u| !done[u] && dist[u].is_finite())
                .min_by(|&x, &y| dist[x].total_cmp(&dist[y]))
            else {
                break;
            };
            done[u] = true;
            for &e in &adj[u] {
                if cap[e] > EPS {
                    let v = to[e];
                    let nd = dist[u] + (cst[e] + h[u] - h[v]).max(0.0);
                    if nd < dist[v] {
                        dist[v] = nd;
                        prev[v] = e;
                    }
                }
            }
        }
        if !dist[t].is_finite() {
            break;
        }
        for v in 0..nodes {
            if dist[v].is_finite() {
                h[v] += dist[v];
            }
        }
        let mut push = f64::INFINITY;
        let mut v = t;
        while v != s {
            let e = prev[v];
            push = push.min(cap[e]);
            v = to[e ^ 1];
        }
        let mut v = t;
        while v != s {
            let e = prev[v];
            cap[e] -= push;
            cap[e ^ 1] += push;
            v = to[e ^ 1];
        }
        flow += push;
    }

    Ok((0..n)
        .map(|i| {
            let row: Vec<(usize, f64)> = (0..m)
                .map(|j| (j, cap[pair_edge[i][j] ^ 1]))
                .filter(|&(_, f)| f > 1e-12)
                .collect();
            let total: f64 = row.iter().map(|(_, f)| f).sum();
            row.into_iter().map(|(j, f)| (j, f / total)).collect()
        })
        .collect())
}

/// The monotone (quantile) coupling of two 1-D distributions, which is
/// optimal for every convex cost of `|u − v|`. `O(n log n + m log m)`.
fn monotone_plan(av: &[f64], a: &[f64], bv: &[f64], b: &[f64]) -> Vec<Vec<(usize, f64)>> {
    let (sa, sb) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    let mut ia: Vec<usize> = (0..a.len()).collect();
    let mut ib: Vec<usize> = (0..b.len()).collect();
    ia.sort_by(|&x, &y| av[x].total_cmp(&av[y]));
    ib.sort_by(|&x, &y| bv[x].total_cmp(&bv[y]));
    let mut plan = vec![Vec::new(); a.len()];
    let (mut p, mut q) = (0, 0);
    let (mut ra, mut rb) = (a[ia[0]] / sa, b[ib[0]] / sb);
    loop {
        let mass = ra.min(rb);
        if mass > 0.0 {
            plan[ia[p]].push((ib[q], mass));
        }
        ra -= mass;
        rb -= mass;
        if ra <= 1e-15 {
            p += 1;
            if p == ia.len() {
                break;
            }
            ra = a[ia[p]] / sa;
        }
        if rb <= 1e-15 {
            q += 1;
            if q == ib.len() {
                // Rounding left a sliver of source mass: give it to the last
                // target.
                if ra > 0.0 {
                    plan[ia[p]].push((ib[q - 1], ra));
                }
                for &i in &ia[p + 1..] {
                    plan[i].push((ib[q - 1], a[i] / sa));
                }
                break;
            }
            rb = b[ib[q]] / sb;
        }
    }
    for (i, row) in plan.iter_mut().enumerate() {
        let total = a[i] / sa;
        for (_, f) in row.iter_mut() {
            *f /= total;
        }
    }
    plan
}

/// Ground cost between two points of a (possibly multi-column) variable:
/// `|u − v|` on numeric columns, `1(u ≠ v)` on categorical ones, summed.
struct Metric<'a> {
    cols: Vec<(String, &'a Column, Vec<Option<f64>>)>,
}

impl<'a> Metric<'a> {
    fn new(data: &'a Dataset, names: &[String]) -> Result<Self, FairnessError> {
        Ok(Self {
            cols: named_columns(data, names)?
                .into_iter()
                .map(|(n, c)| {
                    let v = numeric_levels(c);
                    (n, c, v)
                })
                .collect(),
        })
    }

    fn cost(&self, a: &[u32], b: &[u32]) -> f64 {
        self.cols
            .iter()
            .zip(a.iter().zip(b))
            .map(|((_, _, num), (&u, &v))| match (num[u as usize], num[v as usize]) {
                (Some(x), Some(y)) => (x - y).abs(),
                _ => f64::from(u8::from(u != v)),
            })
            .sum()
    }

    /// The numeric value, when the variable is a single numeric column.
    fn scalar(&self, key: &[u32]) -> Option<f64> {
        match self.cols.as_slice() {
            [(_, _, num)] => num[key[0] as usize],
            _ => None,
        }
    }

    fn label(&self, key: &[u32]) -> String {
        if self.cols.is_empty() {
            return "∅".into();
        }
        self.cols
            .iter()
            .zip(key)
            .map(|((n, c, _), &k)| format!("{n}={}", c.levels[k as usize]))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn strings(&self, key: &[u32]) -> Vec<String> {
        self.cols
            .iter()
            .zip(key)
            .map(|((_, c, _), &k)| c.levels[k as usize].clone())
            .collect()
    }
}

/// Optimal coupling of two weighted point sets keyed by column codes.
fn couple<K: Clone + Eq + Hash + Ord + AsRef<[u32]>>(
    src: &HashMap<K, f64>,
    tgt: &HashMap<K, f64>,
    metric: &Metric<'_>,
) -> Result<Coupling<K>, FairnessError> {
    let mut sk: Vec<&K> = src.keys().collect();
    let mut tk: Vec<&K> = tgt.keys().collect();
    sk.sort();
    tk.sort();
    let a: Vec<f64> = sk.iter().map(|k| src[*k]).collect();
    let b: Vec<f64> = tk.iter().map(|k| tgt[*k]).collect();
    let sv: Option<Vec<f64>> = sk.iter().map(|k| metric.scalar(k.as_ref())).collect();
    let tv: Option<Vec<f64>> = tk.iter().map(|k| metric.scalar(k.as_ref())).collect();
    let plan = match (sv, tv) {
        (Some(sv), Some(tv)) => monotone_plan(&sv, &a, &tv, &b),
        _ => transport_plan(&a, &b, &|i, j| metric.cost(sk[i].as_ref(), tk[j].as_ref()))?,
    };
    Ok(sk
        .into_iter()
        .zip(plan)
        .map(|(k, row)| (k.clone(), row.into_iter().map(|(j, p)| (tk[j].clone(), p)).collect()))
        .collect())
}

type Key = Vec<u32>;

/// Algorithm 16, Causal Individual Fairness.
///
/// For `V ∈ {Z, W, Y}` in turn:
///
/// * if `V ∉ BN`, transport `V | x0, pa(V)` onto `V | x1, τ(pa(V))`;
/// * if `V ∈ BN`, transport `V | x0, pa(V)` onto `V | x0, τ(pa(V))`. Only the
///   parents change, so the group's own mechanism for `V` is kept.
///
/// `Y` is never in the business-necessity set. Here `pa(Z) = ∅`,
/// `pa(W) = Z` and `pa(Y) = (Z, W)`, with `X` held apart.
///
/// The ground cost is `|u − v|` on numeric columns and `1(u ≠ v)` on
/// categorical ones, summed over the columns of `Z` or `W`. A single numeric
/// column uses the monotone coupling, so the map preserves ranks within each
/// stratum.
///
/// ```
/// use why_data::fairness::{CausalIf, Dataset, Sfm, estimate};
///
/// let csv = "x,z,w,y\n\
///            a,0,0,0\na,0,1,1\na,1,0,0\na,1,1,1\na,1,1,0\n\
///            b,0,0,1\nb,0,1,1\nb,1,1,1\nb,1,0,0\nb,0,0,0\n";
/// let data = Dataset::from_csv(csv).unwrap();
/// let sfm = Sfm::new("x", ["z"], ["w"], "y");
/// let fair = CausalIf::new(sfm.clone(), "a", "b").transform(&data).unwrap();
/// let m = estimate(&fair, &sfm, "a", "b").unwrap();
/// assert_eq!((m.de_sym, m.ie_sym, m.se), (0.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone)]
pub struct CausalIf {
    /// SFM projection; `sfm.y` is the outcome transported.
    pub sfm: Sfm,
    /// The group whose values are transported.
    pub x0: String,
    /// The reference group.
    pub x1: String,
    /// Permitted pathways.
    pub business_necessity: BusinessNecessity,
}

struct Fitted<'a> {
    zm: Metric<'a>,
    wm: Metric<'a>,
    ym: Metric<'a>,
    plan_z: Coupling<Key>,
    plan_w: HashMap<(Key, Key), Coupling<Key>>,
    plan_y: HashMap<(Key, Key, Key, Key), Coupling<Key>>,
    /// `(group, z, w, y)` and weight, per dataset row in the contrast.
    rows: Vec<(usize, usize, Key, Key, Key, f64)>,
}

impl CausalIf {
    /// Causal IF with an empty business-necessity set.
    pub fn new(sfm: Sfm, x0: impl Into<String>, x1: impl Into<String>) -> Self {
        Self {
            sfm,
            x0: x0.into(),
            x1: x1.into(),
            business_necessity: BusinessNecessity::NONE,
        }
    }

    /// Sets the business-necessity set.
    pub fn with_business_necessity(mut self, bn: BusinessNecessity) -> Self {
        self.business_necessity = bn;
        self
    }

    fn fit<'a>(&self, data: &'a Dataset) -> Result<Fitted<'a>, FairnessError> {
        let xcol = data.column(&self.sfm.x)?;
        let (c0, c1) = contrast_codes(xcol, &self.sfm.x, &self.x0, &self.x1)?;
        let zm = Metric::new(data, &self.sfm.z)?;
        let wm = Metric::new(data, &self.sfm.w)?;
        let ym = Metric::new(data, std::slice::from_ref(&self.sfm.y))?;
        if ym.cols[0].2.iter().any(Option::is_none) {
            let col = ym.cols[0].1;
            let bad = col.levels.iter().zip(&ym.cols[0].2).find(|(_, v)| v.is_none()).unwrap().0;
            return Err(FairnessError::NonNumericOutcome {
                column: self.sfm.y.clone(),
                value: bad.clone(),
            });
        }
        let levels = [self.x0.clone(), self.x1.clone()];

        let mut rows = Vec::new();
        let mut z_dist: [HashMap<Key, f64>; 2] = Default::default();
        let mut w_dist: [HashMap<Key, HashMap<Key, f64>>; 2] = Default::default();
        let mut y_dist: [HashMap<(Key, Key), HashMap<Key, f64>>; 2] = Default::default();
        for i in 0..data.len() {
            let xc = xcol.codes[i] as usize;
            let g = match xc {
                _ if xc == c0 => 0,
                _ if xc == c1 => 1,
                _ => continue,
            };
            let w = data.weight(i);
            let zk: Key = zm.cols.iter().map(|(_, c, _)| c.codes[i]).collect();
            let wk: Key = wm.cols.iter().map(|(_, c, _)| c.codes[i]).collect();
            let yk: Key = vec![ym.cols[0].1.codes[i]];
            if w > 0.0 {
                *z_dist[g].entry(zk.clone()).or_default() += w;
                *w_dist[g].entry(zk.clone()).or_default().entry(wk.clone()).or_default() += w;
                *y_dist[g]
                    .entry((zk.clone(), wk.clone()))
                    .or_default()
                    .entry(yk.clone())
                    .or_default() += w;
            }
            rows.push((i, g, zk, wk, yk, w));
        }
        for (g, level) in levels.iter().enumerate() {
            if z_dist[g].is_empty() {
                return Err(FairnessError::EmptyGroup {
                    level: level.clone(),
                });
            }
        }
        let bn = self.business_necessity;
        let missing = |g: usize, stratum: String| FairnessError::Positivity {
            level: levels[g].clone(),
            stratum,
        };

        let plan_z = if bn.z {
            z_dist[0].keys().map(|k| (k.clone(), vec![(k.clone(), 1.0)])).collect()
        } else {
            couple(&z_dist[0], &z_dist[1], &zm)?
        };

        let wg = if bn.w { 0 } else { 1 };
        let mut plan_w = HashMap::new();
        for (z, targets) in &plan_z {
            for (zt, _) in targets {
                if plan_w.contains_key(&(z.clone(), zt.clone())) {
                    continue;
                }
                let tgt = w_dist[wg].get(zt).ok_or_else(|| missing(wg, zm.label(zt)))?;
                plan_w.insert((z.clone(), zt.clone()), couple(&w_dist[0][z], tgt, &wm)?);
            }
        }

        let mut plan_y = HashMap::new();
        for ((z, w), src) in &y_dist[0] {
            for (zt, _) in &plan_z[z] {
                for (wt, _) in &plan_w[&(z.clone(), zt.clone())][w] {
                    let key = (z.clone(), w.clone(), zt.clone(), wt.clone());
                    if plan_y.contains_key(&key) {
                        continue;
                    }
                    let tgt = y_dist[1].get(&(zt.clone(), wt.clone())).ok_or_else(|| {
                        missing(1, format!("{}; {}", zm.label(zt), wm.label(wt)))
                    })?;
                    plan_y.insert(key, couple(src, tgt, &ym)?);
                }
            }
        }

        Ok(Fitted {
            zm,
            wm,
            ym,
            plan_z,
            plan_w,
            plan_y,
            rows,
        })
    }

    /// The exact transformed distribution as a weighted table with columns
    /// `X, Z…, W…, Y`. Rows with `X ∉ {x0, x1}` are dropped, and identical
    /// rows are merged.
    pub fn transform(&self, data: &Dataset) -> Result<Dataset, FairnessError> {
        let fit = self.fit(data)?;
        let mut out: BTreeMap<(usize, Key, Key, Key), f64> = BTreeMap::new();
        for (_, g, z, w, y, weight) in &fit.rows {
            if *weight == 0.0 {
                continue;
            }
            if *g == 1 {
                *out.entry((1, z.clone(), w.clone(), y.clone())).or_default() += weight;
                continue;
            }
            for (zt, pz) in &fit.plan_z[z] {
                for (wt, pw) in &fit.plan_w[&(z.clone(), zt.clone())][w] {
                    let py_plan = &fit.plan_y[&(z.clone(), w.clone(), zt.clone(), wt.clone())][y];
                    for (yt, py) in py_plan {
                        *out.entry((0, zt.clone(), wt.clone(), yt.clone())).or_default() +=
                            weight * pz * pw * py;
                    }
                }
            }
        }
        let header: Vec<String> = std::iter::once(self.sfm.x.clone())
            .chain(self.sfm.z.iter().cloned())
            .chain(self.sfm.w.iter().cloned())
            .chain(std::iter::once(self.sfm.y.clone()))
            .collect();
        let levels = [&self.x0, &self.x1];
        let (records, weights): (Vec<Vec<String>>, Vec<f64>) = out
            .into_iter()
            .map(|((g, z, w, y), weight)| {
                let mut rec = vec![levels[g].clone()];
                rec.extend(fit.zm.strings(&z));
                rec.extend(fit.wm.strings(&w));
                rec.extend(fit.ym.strings(&y));
                (rec, weight)
            })
            .unzip();
        Dataset::from_records(header, records)?.with_weights(weights)
    }

    /// One draw of the stochastic transport per row. Returns all of
    /// `data`'s columns and rows, with `Z`, `W` and `Y` replaced on the `x0`
    /// rows. Weights are kept.
    pub fn transform_sampled(&self, data: &Dataset, seed: u64) -> Result<Dataset, FairnessError> {
        let fit = self.fit(data)?;
        let mut rng = SplitMix64::new(seed);
        let draw = |rng: &mut SplitMix64, options: &[(Key, f64)]| -> Key {
            let u = rng.unit();
            let mut acc = 0.0;
            for (k, p) in options {
                acc += p;
                if u < acc {
                    return k.clone();
                }
            }
            options.last().unwrap().0.clone()
        };
        let names = data.column_names().to_vec();
        let mut records: Vec<Vec<String>> = (0..data.len())
            .map(|i| names.iter().map(|n| data.value(n, i).unwrap().to_string()).collect())
            .collect();
        let pos = |name: &String| names.iter().position(|n| n == name).unwrap();
        let zpos: Vec<usize> = self.sfm.z.iter().map(pos).collect();
        let wpos: Vec<usize> = self.sfm.w.iter().map(pos).collect();
        let ypos = pos(&self.sfm.y);
        for (i, g, z, w, y, weight) in &fit.rows {
            if *g == 1 || *weight == 0.0 {
                continue;
            }
            let zt = draw(&mut rng, &fit.plan_z[z]);
            let wt = draw(&mut rng, &fit.plan_w[&(z.clone(), zt.clone())][w]);
            let yt = draw(&mut rng, &fit.plan_y[&(z.clone(), w.clone(), zt.clone(), wt.clone())][y]);
            for (p, v) in zpos.iter().zip(fit.zm.strings(&zt)) {
                records[*i][*p] = v;
            }
            for (p, v) in wpos.iter().zip(fit.wm.strings(&wt)) {
                records[*i][*p] = v;
            }
            records[*i][ypos] = fit.ym.strings(&yt).remove(0);
        }
        let out = Dataset::from_records(names, records)?;
        match data.weights() {
            Some(w) => out.with_weights(w.to_vec()),
            None => Ok(out),
        }
    }
}

/// Test hook: couples two distributions of a single column exactly as
/// Causal IF would.
#[cfg(test)]
pub(crate) fn couple_for_tests(
    data: &Dataset,
    column: &str,
    src: &HashMap<Key, f64>,
    tgt: &HashMap<Key, f64>,
) -> Coupling<Key> {
    let metric = Metric::new(data, &[column.to_string()]).unwrap();
    couple(src, tgt, &metric).unwrap()
}
