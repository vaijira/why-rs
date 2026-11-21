//! Tests for chapter 12: the book's examples, and random multi-domain SCMs
//! that check every estimand against the true target quantity.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use super::*;
use crate::graph::cinfer::counterfactual::{Counterfactual, CtfQuery, Setting, ctf_infer};
use crate::graph::cinfer::{Input, Query, c_infer};
use crate::graph::dseparation::{Admg, EdgeKind};
use crate::transport::subsets;
use petgraph::visit::EdgeRef;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| splitmix(h ^ u64::from(b)))
}

/// How a regime sets one variable in the simulator.
#[derive(Clone, Debug)]
enum Act {
    /// Fixed to a value: a query's atomic intervention.
    Fixed(u64),
    /// A fair coin: an input's atomic intervention, read as a trial at every
    /// level.
    Coin,
    /// A stochastic policy of these parents; `salt` picks its table.
    Policy { parents: Vec<usize>, salt: u64 },
}

/// A random binary SCM compatible with a selection diagram. Each variable is
/// a response function of its parents, the binary latents behind its
/// bidirected edges, and its own `k`-valued noise: 0 and 1 force the
/// output, and every other value selects a pseudo-random function. Noise is
/// shared across worlds, which gives counterfactuals their meaning, and the
/// forcing values keep every conditional strictly positive.
///
/// Random boolean functions often ignore an input (a function of one bit is
/// constant half the time), which makes the model unfaithful to its
/// diagram. So with `k = 0` the model is instead a table of strictly
/// positive random conditionals `P(V = 1 | pa, u)`, generically faithful,
/// which lets the completeness tests check that a criterion's failure shows
/// up numerically. Counterfactual tests need the response functions.
/// Distributions are computed from the conditionals either way; only
/// counterfactuals enumerate the noise.
///
/// A source's mechanism and noise distribution for `V` differ from the
/// target's exactly when `V ∈ Δ^{k,*}`; a latent's distribution differs when
/// all its children do.
struct Model {
    k: usize,
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    latents_of: Vec<Vec<usize>>,
    latent_children: Vec<Vec<usize>>,
    seed: u64,
    /// The target first, then each source with its discrepancies.
    domains: Vec<(Domain, BTreeSet<usize>)>,
    graph: Admg,
}

impl Model {
    fn new(d: &SelectionDiagram, seed: u64) -> Self {
        Self::with_noise(d, seed, 0)
    }

    fn with_noise(d: &SelectionDiagram, seed: u64, k: usize) -> Self {
        let names = d.graph().topological_order().expect("acyclic");
        let index = |n: &str| names.iter().position(|m| m == n).expect("known");
        let n = names.len();
        let mut parents = vec![Vec::new(); n];
        let mut latents_of = vec![Vec::new(); n];
        let mut latent_children = Vec::new();
        let g = d.graph().graph();
        for e in g.edge_references() {
            let (a, b) = (index(&g[e.source()]), index(&g[e.target()]));
            match e.weight() {
                EdgeKind::Directed => parents[b].push(a),
                EdgeKind::Bidirected => {
                    latents_of[a].push(latent_children.len());
                    latents_of[b].push(latent_children.len());
                    latent_children.push(vec![a, b]);
                }
            }
        }
        let mut domains = vec![(Domain::Target, BTreeSet::new())];
        for s in d.sources() {
            let delta = d.discrepancies(s).expect("source").iter().map(|v| index(v)).collect();
            domains.push((Domain::source(s), delta));
        }
        Self {
            k,
            names,
            parents,
            latents_of,
            latent_children,
            seed,
            domains,
            graph: d.graph().clone(),
        }
    }

    fn index(&self, name: &str) -> usize {
        let base = name.trim_end_matches('\'');
        self.names.iter().position(|n| n == base).expect("known")
    }

    fn salt(&self, dom: usize, differs: bool) -> u64 {
        if differs {
            splitmix(self.seed ^ 0xd0_0000 ^ dom as u64)
        } else {
            0
        }
    }

    fn noise_p(&self, dom: usize, i: usize) -> Vec<f64> {
        let s = self.salt(dom, self.domains[dom].1.contains(&i));
        let mut r = splitmix(self.seed ^ 0x0004_015e ^ (i as u64) << 20 ^ s);
        let w: Vec<f64> = (0..self.k)
            .map(|_| {
                r = splitmix(r);
                unit(r) + 0.1
            })
            .collect();
        let t: f64 = w.iter().sum();
        w.into_iter().map(|x| x / t).collect()
    }

    fn latent_p(&self, dom: usize, l: usize) -> f64 {
        let differs = dom != 0 && self.latent_children[l].iter().all(|c| self.domains[dom].1.contains(c));
        0.15 + 0.7 * unit(splitmix(self.seed ^ 0x1a7 ^ l as u64 ^ self.salt(dom, differs)))
    }

    fn latent_weight(&self, dom: usize, u: u64) -> f64 {
        (0..self.latent_children.len())
            .map(|l| {
                let p = self.latent_p(dom, l);
                if (u >> l) & 1 == 1 { p } else { 1.0 - p }
            })
            .product()
    }

    /// Every exogenous state of domain `dom`: probability, latent bits and
    /// noises.
    fn units(&self, dom: usize) -> Vec<(f64, u64, Vec<usize>)> {
        assert!(self.k >= 3, "counterfactuals need response functions");
        let n = self.names.len();
        let noise: Vec<Vec<f64>> = (0..n).map(|i| self.noise_p(dom, i)).collect();
        let mut out = Vec::new();
        for u in 0..(1u64 << self.latent_children.len()) {
            let pu = self.latent_weight(dom, u);
            for code in 0..self.k.pow(n as u32) {
                let eps: Vec<usize> = (0..n).map(|i| code / self.k.pow(i as u32) % self.k).collect();
                let pe: f64 = eps.iter().enumerate().map(|(i, &e)| noise[i][e]).product();
                out.push((pu * pe, u, eps));
            }
        }
        out
    }

    /// `P(V_i = 1 | v, u)` in domain `dom`, the noise summed out.
    fn p_one(&self, dom: usize, i: usize, v: u64, u: u64) -> f64 {
        if self.k == 0 {
            let mut key = self.seed ^ 0xc9_7000 ^ (i as u64) << 56
                ^ self.salt(dom, self.domains[dom].1.contains(&i));
            for &p in &self.parents[i] {
                key = splitmix(key ^ ((v >> p) & 1) << 1 ^ p as u64);
            }
            for &l in &self.latents_of[i] {
                key = splitmix(key ^ ((u >> l) & 1) << 1 ^ (l as u64) << 8);
            }
            return 0.1 + 0.8 * unit(splitmix(key));
        }
        self.noise_p(dom, i)
            .iter()
            .enumerate()
            .map(|(e, p)| p * self.f(dom, i, v, u, e) as f64)
            .sum()
    }

    fn f(&self, dom: usize, i: usize, v: u64, u: u64, e: usize) -> u64 {
        match e {
            0 => 0,
            1 => 1,
            _ => {
                let mut key = self.seed ^ (i as u64) << 56 ^ (e as u64) << 48
                    ^ self.salt(dom, self.domains[dom].1.contains(&i));
                for &p in &self.parents[i] {
                    key = splitmix(key ^ ((v >> p) & 1) << 1 ^ p as u64);
                }
                for &l in &self.latents_of[i] {
                    key = splitmix(key ^ ((u >> l) & 1) << 1 ^ (l as u64) << 8);
                }
                splitmix(key) & 1
            }
        }
    }

    fn policy_p1(&self, salt: u64, i: usize, parents: &[usize], v: u64) -> f64 {
        let mut key = splitmix(self.seed ^ salt ^ (i as u64) << 40);
        for &p in parents {
            key = splitmix(key ^ ((v >> p) & 1) << 1 ^ p as u64);
        }
        0.15 + 0.7 * unit(key)
    }

    /// The order to solve a regime in: the topological order of `G_σ`.
    fn order(&self, regime: &Regime) -> Vec<usize> {
        regime
            .apply(&self.graph)
            .expect("valid regime")
            .topological_order()
            .expect("acyclic")
            .iter()
            .map(|v| self.index(v))
            .collect()
    }

    /// The acts of a regime, with atomic interventions as coins (inputs) or
    /// fixed at `fixed` (queries), and policies salted by `salt`.
    fn acts(&self, regime: &Regime, fixed: Option<&HashMap<String, u64>>, salt: u64) -> HashMap<usize, Act> {
        regime
            .iter()
            .map(|(v, i)| {
                let act = match i {
                    Intervention::Atomic => match fixed {
                        Some(env) => Act::Fixed(env[v]),
                        None => Act::Coin,
                    },
                    Intervention::Policy { parents } => {
                        let mut ps: Vec<usize> = parents.iter().map(|p| self.index(p)).collect();
                        ps.sort_unstable();
                        Act::Policy { parents: ps, salt }
                    }
                };
                (self.index(v), act)
            })
            .collect()
    }

    /// The joint over V in domain `dom` under `acts`, indexed by bitmask.
    fn table(&self, dom: usize, order: &[usize], acts: &HashMap<usize, Act>) -> Vec<f64> {
        let mut table = vec![0.0; 1 << self.names.len()];
        for u in 0..(1u64 << self.latent_children.len()) {
            let mut states = vec![(self.latent_weight(dom, u), 0u64)];
            for &i in order {
                let mut next = Vec::with_capacity(states.len() * 2);
                for (w, v) in states {
                    let p1 = match acts.get(&i) {
                        None => self.p_one(dom, i, v, u),
                        Some(Act::Fixed(b)) => *b as f64,
                        Some(Act::Coin) => 0.5,
                        Some(Act::Policy { parents, salt }) => self.policy_p1(*salt, i, parents, v),
                    };
                    if p1 < 1.0 {
                        next.push((w * (1.0 - p1), v));
                    }
                    if p1 > 0.0 {
                        next.push((w * p1, v | 1 << i));
                    }
                }
                states = next;
            }
            for (w, v) in states {
                table[v as usize] += w;
            }
        }
        table
    }

    /// The target policy's tables, as [`TargetPolicy`] holds them.
    fn target_policy(&self, regime: &Regime) -> TargetPolicy {
        let mut pi = TargetPolicy::new();
        for (v, i) in regime.iter() {
            let Intervention::Policy { parents } = i else { continue };
            let mut ps: Vec<usize> = parents.iter().map(|p| self.index(p)).collect();
            ps.sort_unstable();
            let x = self.index(v);
            for bits in 0..(1u64 << ps.len()) {
                let mut vbits = 0;
                for (k, &p) in ps.iter().enumerate() {
                    vbits |= ((bits >> k) & 1) << p;
                }
                let p1 = self.policy_p1(TARGET_SALT, x, &ps, vbits);
                let assignment: Vec<(String, String)> = ps
                    .iter()
                    .map(|&p| (self.names[p].clone(), ((vbits >> p) & 1).to_string()))
                    .collect();
                let refs: Vec<(&str, &str)> =
                    assignment.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
                pi.set(v, &refs, "1", p1);
                pi.set(v, &refs, "0", 1.0 - p1);
            }
        }
        pi
    }

    /// The world where `fixed` was set, for one exogenous state of the
    /// target.
    fn solve(&self, u: u64, eps: &[usize], fixed: &[(usize, u64)]) -> u64 {
        let mut v = 0;
        for (i, &e) in eps.iter().enumerate() {
            let bit = match fixed.iter().find(|(f, _)| *f == i) {
                Some(&(_, b)) => b,
                None => self.f(0, i, v, u, e),
            };
            v |= bit << i;
        }
        v
    }

    fn value_of(&self, c: &Counterfactual, u: u64, eps: &[usize], env: &HashMap<String, u64>) -> u64 {
        let fixed: Vec<(usize, u64)> = c
            .subscript
            .iter()
            .map(|(k, s)| {
                let b = match s {
                    Setting::Value(p) => env[&format!("{k}{}", "'".repeat(*p))],
                    Setting::Nested(inner) => self.value_of(inner, u, eps, env),
                };
                (self.index(k), b)
            })
            .collect();
        (self.solve(u, eps, &fixed) >> self.index(&c.var)) & 1
    }

    /// The true `P*(events | given)` of a counterfactual query.
    fn ctf_truth(&self, q: &CtfQuery, env: &HashMap<String, u64>) -> Option<f64> {
        let holds = |e: &crate::graph::cinfer::counterfactual::Event, u: u64, eps: &[usize]| {
            self.value_of(&e.ctf, u, eps, env) == env[&format!("{}{}", e.ctf.var, "'".repeat(e.value))]
        };
        let (mut num, mut den) = (0.0, 0.0);
        for (p, u, eps) in self.units(0) {
            if q.given.iter().all(|e| holds(e, u, &eps)) {
                den += p;
                if q.events.iter().all(|e| holds(e, u, &eps)) {
                    num += p;
                }
            }
        }
        (den > 1e-12).then(|| num / den)
    }
}

const TARGET_SALT: u64 = 0x7a_2637;

fn marginal(table: &[f64], assignment: &[(usize, u64)]) -> f64 {
    table
        .iter()
        .enumerate()
        .filter(|(v, _)| assignment.iter().all(|&(i, b)| ((*v as u64) >> i) & 1 == b))
        .map(|(_, p)| p)
        .sum()
}

/// The exact distribution of every domain under every regime an estimand
/// asks for, computed on demand.
struct Exact<'a> {
    model: &'a Model,
    cache: RefCell<HashMap<Domain, Vec<f64>>>,
}

impl<'a> Exact<'a> {
    fn new(model: &'a Model) -> Self {
        Self {
            model,
            cache: RefCell::new(HashMap::new()),
        }
    }

    fn table(&self, domain: &Domain) -> Result<Vec<f64>, TransportError> {
        if let Some(t) = self.cache.borrow().get(domain) {
            return Ok(t.clone());
        }
        let m = self.model;
        let (base, regime) = match domain {
            Domain::Under(b, r) => (&**b, r.clone()),
            d => (d, Regime::new()),
        };
        let dom = m
            .domains
            .iter()
            .position(|(d, _)| d == base)
            .ok_or_else(|| TransportError::MissingData(format!("no domain {domain}")))?;
        let salt = hash_str(&format!("{domain:?}"));
        let t = m.table(dom, &m.order(&regime), &m.acts(&regime, None, salt));
        self.cache.borrow_mut().insert(domain.clone(), t.clone());
        Ok(t)
    }

    /// The true `P*(y | w; σ_X)` at `env`.
    fn truth(&self, q: &PolicyQuery, env: &HashMap<String, u64>) -> f64 {
        let m = self.model;
        let t = m.table(0, &m.order(&q.regime), &m.acts(&q.regime, Some(env), TARGET_SALT));
        let bind = |s: &VarSet| -> Vec<(usize, u64)> { s.iter().map(|v| (m.index(v), env[v])).collect() };
        let w = bind(&q.w);
        let mut yw = bind(&q.y);
        yw.extend(w.iter().copied());
        marginal(&t, &yw) / marginal(&t, &w)
    }
}

type VarSet = BTreeSet<String>;

impl Distributions for Exact<'_> {
    fn levels(&self, var: &str) -> Option<Vec<String>> {
        self.model
            .names
            .iter()
            .any(|n| n == var)
            .then(|| vec!["0".to_string(), "1".to_string()])
    }

    fn probability(&self, domain: &Domain, event: &[(&str, &str)]) -> Result<f64, TransportError> {
        let t = self.table(domain)?;
        let a: Vec<(usize, u64)> = event
            .iter()
            .map(|(n, v)| (self.model.index(n), u64::from(*v == "1")))
            .collect();
        Ok(marginal(&t, &a))
    }
}

/// Evaluates `e` at `env` against the exact distributions and `policy`.
fn eval(exact: &Exact<'_>, policy: &TargetPolicy, e: &Estimand, env: &HashMap<String, u64>) -> f64 {
    let vals: Vec<(String, String)> = e
        .free_vars()
        .into_iter()
        .map(|v| {
            let b = env.get(&v).unwrap_or_else(|| panic!("{e}: free {v} has no value"));
            (v, b.to_string())
        })
        .collect();
    let at: Vec<(&str, &str)> = vals.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    e.evaluate(&WithPolicy::new(exact, policy), &at)
        .unwrap_or_else(|err| panic!("{e} fails to evaluate: {err}"))
}

/// Every binary assignment of `names`.
fn assignments(names: &[String]) -> Vec<HashMap<String, u64>> {
    (0..(1u64 << names.len()))
        .map(|bits| {
            names
                .iter()
                .enumerate()
                .map(|(k, v)| (v.clone(), (bits >> k) & 1))
                .collect()
        })
        .collect()
}

/// A random selection diagram over `V0..Vn` with at most four bidirected
/// edges and one or two sources.
fn random_diagram(n: usize, seed: u64) -> Option<SelectionDiagram> {
    let names: Vec<String> = (0..n).map(|i| format!("V{i}")).collect();
    let mut g = Admg::new();
    for v in &names {
        g.add_node(v);
    }
    let mut r = seed;
    let mut next = || {
        r = splitmix(r);
        unit(r)
    };
    let mut latents = 0;
    for j in 0..n {
        for i in 0..j {
            if next() < 0.45 {
                g.directed(&names[i], &names[j]);
            }
            if next() < 0.25 {
                g.bidirected(&names[i], &names[j]);
                latents += 1;
            }
        }
    }
    if latents > 4 {
        return None;
    }
    let mut d = SelectionDiagram::new(g);
    for s in 0..1 + (seed % 2) as usize {
        let delta: Vec<&str> = names.iter().filter(|_| next() < 0.3).map(String::as_str).collect();
        d = d.with_source(&(s + 1).to_string(), &delta).expect("fresh");
    }
    Some(d)
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| (*x).to_string()).collect()
}

// ---------------------------------------------------------------------------
// Adjustment (Sec. 12.1, 12.1.1)
// ---------------------------------------------------------------------------

/// Fig. 12.3, reconstructed from the derivation in Eqs. 12.7–12.14: `Z1`
/// and `Z3` are pre-mediators, `Z2` and `Z4` share a latent with `Y`, so
/// `Z_p = {Z2, Z4}`. Panel (a) adds the direct effects `X1 → Z2` and
/// `X2 → Z4` and a latent between `Z2` and `Z4`.
fn figure_12_3(a: bool) -> Admg {
    let mut g = Admg::new();
    g.directed("X1", "Z1");
    g.directed("Z1", "Z2");
    g.directed("X2", "Z3");
    g.directed("Z3", "Z4");
    g.directed("X1", "Y");
    g.directed("X2", "Y");
    g.bidirected("Z2", "Y");
    g.bidirected("Z4", "Y");
    if a {
        g.directed("X1", "Z2");
        g.directed("X2", "Z4");
        g.bidirected("Z2", "Z4");
    }
    g
}

#[test]
fn example_12_3_orders_figure_12_3_b_but_not_a() {
    let z = ["Z1", "Z2", "Z3", "Z4"];
    let x = ["X1", "X2"];
    let order = e_adjustment_order(&figure_12_3(false), &x, &["Y"], &z).unwrap();
    let order = order.expect("Fig. 12.3(b) has a suitable order");
    // Every prefix must satisfy (12.15) or (12.16) for its last member; the
    // order found is one of those the text allows.
    assert_eq!(order.len(), 4);
    assert!(!is_e_admissible(&figure_12_3(true), &x, &["Y"], &z).unwrap());
    // Without the mediators' descendants, adjustment is trivially fine.
    assert!(is_e_admissible(&figure_12_3(true), &x, &["Y"], &[]).unwrap());
}

/// Example 12.2 (Fig. 12.2): baseline risk `B` and background `E` both
/// affect treatment and outcome; only `E` differs between the trials and the
/// clinical population.
#[test]
fn example_12_2_adjusts_for_baseline_and_background() {
    let mut g = Admg::new();
    for v in ["B", "E"] {
        g.directed(v, "X");
        g.directed(v, "Y");
    }
    g.directed("X", "Y");
    let d = SelectionDiagram::new(g).with_source("FDA", &["E"]).unwrap();
    assert!(is_tr_admissible(&d, "FDA", &["X"], &["Y"], &["B", "E"]).unwrap());
    assert!(is_tr_admissible_singleton(&d, "FDA", "X", &["Y"], &["B", "E"]).unwrap());
    assert!(!is_tr_admissible(&d, "FDA", &["X"], &["Y"], &["B"]).unwrap());
    let e = tr_adjustment(&d, "FDA", &["X"], &["Y"], &["B", "E"]).unwrap();
    assert_eq!(e.to_string(), "sum_{b, e} P^FDA(y | b, e; do(x)) P^*(b, e)");
    // Eq. 12.20's surrogate trap: B alone leaves S → E → Y open.
    assert_eq!(find_tr_adjustment(&d, "FDA", &["X"], &["Y"]).unwrap(), Some(s(&["E"])));
}

/// Examples 12.5 and 12.6: a proxy `Z` of an unmeasured age, and a
/// mediator. In the proxy case the z-specific effect is not invariant and
/// adjusting for `Z` fails; with no discrepancy on the outcome's side the
/// plain experimental effect transports (Eq. 12.20).
#[test]
fn examples_12_5_and_12_6_reject_proxies_and_mediators() {
    // Fig. 12.1(b): Age → X, Age → Y, Age → Z, X → Y, with Age latent.
    let mut proxy = Admg::new();
    proxy.directed("X", "Y");
    proxy.bidirected("Z", "X");
    proxy.bidirected("Z", "Y");
    proxy.bidirected("X", "Y");
    let d = SelectionDiagram::new(proxy).with_source("LA", &["Z"]).unwrap();
    assert!(!is_tr_admissible(&d, "LA", &["X"], &["Y"], &["Z"]).unwrap());
    // S → Z <-> Y is closed when Z is not conditioned on: Eq. 12.20.
    assert_eq!(find_tr_adjustment(&d, "LA", &["X"], &["Y"]).unwrap(), Some(vec![]));

    // Fig. 12.1(c): X → Z → Y, X → Y, X <-> Y, with S → Z.
    let mut med = Admg::new();
    med.directed("X", "Z");
    med.directed("Z", "Y");
    med.directed("X", "Y");
    med.bidirected("X", "Y");
    let d = SelectionDiagram::new(med).with_source("LA", &["Z"]).unwrap();
    assert!(!is_tr_admissible(&d, "LA", &["X"], &["Y"], &["Z"]).unwrap());
    assert_eq!(find_tr_adjustment(&d, "LA", &["X"], &["Y"]).unwrap(), None);
}

/// Over random diagrams and SCMs: `Z` satisfies e-adjustment exactly when
/// `P(y | do(x)) = Σ_z P(y | do(x), z) P(z)` holds numerically (Thm 12.1.2,
/// both directions), including multiple treatments and post-treatment `Z`.
#[test]
fn e_adjustment_holds_exactly_when_the_formula_does() {
    let (mut valid, mut invalid, mut post) = (0, 0, 0);
    for seed in 0..60u64 {
        let n = if seed % 3 == 0 { 5 } else { 4 };
        let Some(d) = random_diagram(n, seed) else { continue };
        let m = Model::new(&d, seed);
        let g = d.graph();
        let names = m.names.clone();
        let natural = m.table(0, &(0..n).collect::<Vec<_>>(), &HashMap::new());
        for xs in subsets(&names).unwrap().into_iter().filter(|x| (1..=2).contains(&x.len())) {
            let y = names.iter().rev().find(|v| !xs.contains(v)).unwrap().clone();
            let rest: Vec<String> = names.iter().filter(|v| !xs.contains(v) && **v != y).cloned().collect();
            let xr: Vec<&str> = xs.iter().map(String::as_str).collect();
            let de: BTreeSet<&str> = g.descendants_of(&xr).unwrap().into_iter().collect();
            for z in subsets(&rest).unwrap() {
                let zr: Vec<&str> = z.iter().map(String::as_str).collect();
                let ok = is_e_admissible(g, &xr, &[&y], &zr).unwrap();
                let mut equal = true;
                for env in assignments(&xs) {
                    let fixed: Vec<(usize, u64)> = xs.iter().map(|x| (m.index(x), env[x])).collect();
                    let mut acts = HashMap::new();
                    for &(i, b) in &fixed {
                        acts.insert(i, Act::Fixed(b));
                    }
                    let t = m.table(0, &(0..n).collect::<Vec<_>>(), &acts);
                    let yi = m.index(&y);
                    let truth = marginal(&t, &[(yi, 1)]);
                    let zi: Vec<usize> = z.iter().map(|v| m.index(v)).collect();
                    let mut adj = 0.0;
                    for zb in 0..(1u64 << zi.len()) {
                        let za: Vec<(usize, u64)> = zi.iter().enumerate().map(|(k, &i)| (i, (zb >> k) & 1)).collect();
                        let mut yz = za.clone();
                        yz.push((yi, 1));
                        adj += marginal(&t, &yz) / marginal(&t, &za) * marginal(&natural, &za);
                    }
                    if (adj - truth).abs() > 1e-9 {
                        equal = false;
                    }
                }
                assert_eq!(ok, equal, "seed {seed}: X = {xs:?}, Y = {y}, Z = {z:?}");
                if ok {
                    valid += 1;
                    if z.iter().any(|v| de.contains(v.as_str())) {
                        post += 1;
                    }
                } else {
                    invalid += 1;
                }
            }
        }
    }
    assert!(valid > 300 && invalid > 300 && post > 50, "{valid} valid ({post} post-treatment), {invalid} invalid");
}

/// Over random selection diagrams: the tr-adjustment criterion holds exactly
/// when the transport formula is numerically right (Thm 12.1.3), the
/// estimand [`tr_adjustment`] builds evaluates to the truth, Def. 12.1.1
/// agrees with Def. 12.1.3 for one treatment (Lemma 12.1.4), and a
/// pre-treatment admissible subset always exists (Thm 12.1.6).
#[test]
fn tr_adjustment_is_sound_and_complete() {
    let (mut valid, mut invalid) = (0, 0);
    for seed in 0..60u64 {
        let n = if seed % 3 == 0 { 5 } else { 4 };
        let Some(d) = random_diagram(n, seed) else { continue };
        let m = Model::new(&d, seed);
        let exact = Exact::new(&m);
        let names = m.names.clone();
        let src = d.sources().next().unwrap().to_string();
        let none = TargetPolicy::new();
        for xs in subsets(&names).unwrap().into_iter().filter(|x| (1..=2).contains(&x.len())) {
            let y = names.iter().rev().find(|v| !xs.contains(v)).unwrap().clone();
            let rest: Vec<String> = names.iter().filter(|v| !xs.contains(v) && **v != y).cloned().collect();
            let xr: Vec<&str> = xs.iter().map(String::as_str).collect();
            let regime = Regime::new().atomic(&xr);
            for z in subsets(&rest).unwrap() {
                let zr: Vec<&str> = z.iter().map(String::as_str).collect();
                let ok = is_tr_admissible(&d, &src, &xr, &[&y], &zr).unwrap();
                if xs.len() == 1 {
                    assert_eq!(
                        ok,
                        is_tr_admissible_singleton(&d, &src, xr[0], &[&y], &zr).unwrap(),
                        "seed {seed}: Lemma 12.1.4 fails for X = {xs:?}, Y = {y}, Z = {z:?}"
                    );
                }
                let formula = super::adjustment::adjustment_estimand(&Domain::source(&src), &Domain::Target, &xr, &[&y], &zr);
                let q = PolicyQuery::new(&[&y], regime.clone());
                let mut free = xs.clone();
                free.push(y.clone());
                let mut equal = true;
                for env in assignments(&free) {
                    let truth = exact.truth(&q, &env);
                    if (eval(&exact, &none, &formula, &env) - truth).abs() > 1e-9 {
                        equal = false;
                    }
                }
                assert_eq!(ok, equal, "seed {seed}: X = {xs:?}, Y = {y}, Z = {z:?}: {formula}");
                if ok {
                    valid += 1;
                    let pre = pre_treatment_subset(&d, &src, &xr, &[&y], &zr).unwrap().expect("Thm 12.1.6");
                    let pr: Vec<&str> = pre.iter().map(String::as_str).collect();
                    assert!(is_tr_admissible(&d, &src, &xr, &[&y], &pr).unwrap());
                } else {
                    invalid += 1;
                }
            }
        }
    }
    assert!(valid > 200 && invalid > 200, "{valid} valid, {invalid} invalid");
}

// ---------------------------------------------------------------------------
// σ-TR (Sec. 12.1.2)
// ---------------------------------------------------------------------------

/// Examples 12.7 and 12.13 (Figs. 12.5 and 12.13): a sequential protocol,
/// with `Y` differing in the first hospital and `Z` in the second. The edge
/// list is read off Eq. 12.67; the figure shows no bidirected edge.
fn sequential_treatment() -> SelectionDiagram {
    let mut g = Admg::new();
    g.directed("X1", "Z");
    g.directed("X1", "X2");
    g.directed("Z", "X2");
    g.directed("X1", "Y");
    g.directed("X2", "Y");
    g.directed("Z", "Y");
    SelectionDiagram::new(g)
        .with_source("1", &["Y"])
        .unwrap()
        .with_source("2", &["Z"])
        .unwrap()
}

#[test]
fn example_12_13_combines_two_soft_experiments() {
    let d = sequential_treatment();
    let target = Regime::new().policy("X1", &[]).policy("X2", &["X1", "Z"]);
    let inputs = [
        DomainInput::new(Domain::source("1"), Regime::new().atomic(&["X1", "X2"])),
        DomainInput::new(Domain::source("2"), Regime::new().policy("X2", &["X1", "Z"])),
    ];
    let q = PolicyQuery::new(&["Y"], target.clone());
    let t = sigma_tr(&d, &q, &inputs).unwrap();
    // Eq. 12.66, with σ_{X1} = P̂(X1) summed rather than fixed.
    assert_eq!(
        t.estimand.to_string(),
        "sum_{x1, x2, z} P̂(x1) P^1(z; do(x1, x2)) P̂(x2 | x1, z) P^2(y | x1, z, x2; σ_{X2})"
    );
    assert_eq!(
        t.query_tree.to_string(),
        "P*(Y; σ_{X1, X2})\n\
         └─ [Σ_{X1, X2, Z}] Q^*[X1, X2, Y, Z; σ_{X1, X2}]\n   \
            ├─ [δ] Q^*[X1; σ_{X1, X2}]\n   \
            ├─ [δ] Q^*[X2; σ_{X1, X2}]\n   \
            ├─ [δ] Q^*[Y; σ_{X1, X2}]\n   \
            └─ [δ] Q^*[Z; σ_{X1, X2}]\n"
    );
    // Neither hospital suffices alone (Exercise 12.4).
    for one in &inputs {
        assert!(matches!(
            sigma_tr(&d, &q, std::slice::from_ref(one)),
            Err(TrError::NotTransportable { .. })
        ));
    }
    let m = Model::new(&d, 7);
    assert_policy_sound(&m, &q, &t.estimand, "Ex. 12.13");
}

/// Examples 12.8 and 12.14 (Figs. 12.6 and 12.14): the loan policy. The
/// bidirected edges are reconstructed from the trees of Fig. 12.14: `R ↔ Y`
/// (the c-component `{R, Y}` of the q-tree), `W ↔ Z` (named in the text),
/// and `R ↔ W`, `W ↔ X`, which make `{R, W, X, Y}` the c-component of
/// `G_{σ_Z}` that Fig. 12.14(f) splits off.
fn loan_study() -> SelectionDiagram {
    let mut g = Admg::new();
    g.directed("W", "X");
    g.directed("X", "Z");
    g.directed("R", "Z");
    g.directed("Z", "Y");
    g.bidirected("R", "Y");
    g.bidirected("R", "W");
    g.bidirected("W", "X");
    g.bidirected("W", "Z");
    SelectionDiagram::new(g)
        .with_source("1", &["W"])
        .unwrap()
        .with_source("2", &["R"])
        .unwrap()
}

#[test]
fn example_12_14_transports_the_loan_policy() {
    let d = loan_study();
    let inputs = [
        DomainInput::observational(Domain::Target),
        DomainInput::new(Domain::source("1"), Regime::new().policy("Z", &["X", "R"])),
        DomainInput::new(Domain::source("2"), Regime::new().policy("W", &[])),
    ];
    let q = PolicyQuery::new(&["Y"], Regime::new().policy("X", &["R"]));
    let t = sigma_tr(&d, &q, &inputs).unwrap();
    // Eq. 12.75, with P¹(r, w; σ_Z) P¹(x' | w, r; σ_Z) merged by the chain rule.
    assert_eq!(
        t.estimand.to_string(),
        "sum_{r, x, z} P̂(x | r) P^2(z | r, x; σ_{W}) \
         sum_{w, x'} P^1(r, w, x'; σ_{Z}) P^1(y | r, w, x', z; σ_{Z})"
    );
    // Q*[R, Y] from π¹ and Q*[Z] from π².
    assert_eq!(
        t.mapping.values().filter(|s| matches!(s, TrSource::Input { input: 1, .. })).count(),
        1
    );
    assert_eq!(
        t.mapping.values().filter(|s| matches!(s, TrSource::Input { input: 2, .. })).count(),
        1
    );
    assert_eq!(
        t.input_trees[2].to_string(),
        "Q^2[R, W, X, Y, Z; σ_{W}]\n\
         └─ [Σ_{Y}] Q^2[R, W, X, Z; σ_{W}]\n   \
            └─ [δ] Q^2[Z; σ_{W}]\n      \
               └─ [π, σ] Q^*[Z; σ_{X}]\n"
    );
    // The target's own data cannot produce either factor (Ex. 12.14).
    let Err(TrError::NotTransportable { unmapped, .. }) = sigma_tr(&d, &q, &inputs[..1]) else {
        panic!("P*(V) alone transports the loan policy")
    };
    assert_eq!(unmapped.len(), 2);
    let m = Model::new(&d, 11);
    assert_policy_sound(&m, &q, &t.estimand, "Ex. 12.14");
}

/// Checks `e` against the true `P*(y | w; σ_X)` at every binary assignment of
/// its free variables.
fn assert_policy_sound(m: &Model, q: &PolicyQuery, e: &Estimand, label: &str) {
    let exact = Exact::new(m);
    let policy = m.target_policy(&q.regime);
    let mut allowed: BTreeSet<String> = q.y.union(&q.w).cloned().collect();
    allowed.extend(q.regime.atomic_variables());
    assert!(e.free_vars().is_subset(&allowed), "{label}: {e} is free in more than {allowed:?}");
    let names: Vec<String> = allowed.into_iter().collect();
    for env in assignments(&names) {
        let truth = exact.truth(q, &env);
        let got = eval(&exact, &policy, e, &env);
        assert!((got - truth).abs() < 1e-9, "{label}: {e} gives {got} at {env:?}, truth is {truth}");
    }
}

/// A random regime on one variable other than `avoid`: atomic, or a policy
/// reading some of its non-descendants.
fn random_regime(g: &Admg, names: &[String], avoid: &str, next: &mut impl FnMut() -> f64) -> Regime {
    let cands: Vec<&String> = names.iter().filter(|v| *v != avoid).collect();
    let x = cands[(next() * cands.len() as f64) as usize % cands.len()];
    if next() < 0.5 {
        return Regime::new().atomic(&[x]);
    }
    let de: BTreeSet<&str> = g.descendants_of(&[x]).unwrap().into_iter().collect();
    let parents: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|v| !de.contains(v) && next() < 0.5)
        .collect();
    Regime::new().policy(x, &parents)
}

/// Over random selection diagrams, inputs and regimes: every σ-TR estimand
/// evaluates to the true target quantity, and with only the target's natural
/// and atomic inputs σ-TR decides exactly what layer-2 C-INFER does.
#[test]
fn sigma_tr_is_sound_and_extends_c_infer() {
    let (mut transported, mut refused, mut soft, mut cross) = (0, 0, 0, 0);
    for seed in 0..120u64 {
        let n = if seed % 3 == 0 { 5 } else { 4 };
        let Some(d) = random_diagram(n, seed) else { continue };
        let m = Model::new(&d, seed);
        let g = d.graph();
        let names = m.names.clone();
        let mut r = splitmix(seed ^ 0x51_9a);
        let mut next = || {
            r = splitmix(r);
            unit(r)
        };
        let domains: Vec<Domain> = std::iter::once(Domain::Target)
            .chain(d.sources().map(Domain::source))
            .collect();
        for trial in 0..12 {
            let y = names[names.len() - 1 - trial % 2].clone();
            let regime = random_regime(g, &names, &y, &mut next);
            if regime.apply(g).is_err() {
                continue;
            }
            let x = regime.variables().next().unwrap().to_string();
            let w: Vec<&str> = names
                .iter()
                .map(String::as_str)
                .filter(|v| *v != y && *v != x && next() < 0.15)
                .take(1)
                .collect();
            let q = PolicyQuery::new(&[&y], regime.clone()).given(&w);
            let k = 1 + (next() * 3.0) as usize;
            let inputs: Vec<DomainInput> = (0..k)
                .map(|_| {
                    let dom = domains[(next() * domains.len() as f64) as usize % domains.len()].clone();
                    let reg = if next() < 0.4 {
                        Regime::new()
                    } else {
                        random_regime(g, &names, "", &mut next)
                    };
                    DomainInput::new(dom, reg)
                })
                .collect();
            let label = format!("seed {seed}: {q} from {inputs:?}");
            match sigma_tr(&d, &q, &inputs) {
                Ok(t) => {
                    transported += 1;
                    if matches!(regime.get(&x), Some(Intervention::Policy { .. })) {
                        soft += 1;
                    }
                    assert_policy_sound(&m, &q, &t.estimand, &label);
                }
                Err(TrError::NotTransportable { .. }) => refused += 1,
                Err(e) => panic!("{label}: {e}"),
            }

            // The layer-2 special case: target-only, atomic, unconditional.
            if regime.get(&x) == Some(&Intervention::Atomic) {
                let layer2: Vec<(DomainInput, Input)> = vec![
                    (DomainInput::observational(Domain::Target), Input::observational()),
                    (
                        DomainInput::new(Domain::Target, Regime::new().atomic(&[&names[0]])),
                        Input::experimental(&[&names[0]]),
                    ),
                ];
                let sub = &layer2[..1 + trial % 2];
                let (ours, theirs): (Vec<_>, Vec<_>) = sub.iter().cloned().unzip();
                let q2 = PolicyQuery::new(&[&y], regime.clone());
                if names[0] == x && sub.len() == 2 {
                    continue;
                }
                let a = sigma_tr(&d, &q2, &ours).is_ok();
                let b = c_infer(g, &Query::new(&[&x], &[&y]), &theirs).is_ok();
                assert_eq!(a, b, "seed {seed}: σ-TR and C-INFER disagree on {q2} from {ours:?}");
                cross += 1;
            }
        }
    }
    assert!(
        transported > 300 && refused > 100 && soft > 100 && cross > 200,
        "{transported} transported ({soft} soft), {refused} refused, {cross} cross-checked"
    );
}

// ---------------------------------------------------------------------------
// ctf-TR (Sec. 12.2)
// ---------------------------------------------------------------------------

/// Example 12.16/12.20 (Figs. 12.16 and 12.19): degree `X`, occupation `W`,
/// socio-economics `Z`, earnings `Y`. Read off the q-tree of Fig. 12.19(b):
/// `{X, Z}` and `{W, Y}` are c-components, and `Y_{xzw}`, `W_x`, `X_z` give
/// the parents.
fn earnings() -> SelectionDiagram {
    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("Z", "Y");
    g.directed("X", "W");
    g.directed("X", "Y");
    g.directed("W", "Y");
    g.bidirected("Z", "X");
    g.bidirected("W", "Y");
    SelectionDiagram::new(g)
        .with_source("1", &["Z"])
        .unwrap()
        .with_source("2", &["W"])
        .unwrap()
}

#[test]
fn example_12_20_transports_the_ett_from_a_scholarship_program() {
    let d = earnings();
    let inputs = [
        DomainInput::new(Domain::source("1"), Regime::new().policy("X", &[])),
        DomainInput::observational(Domain::source("2")),
    ];
    let m = Model::with_noise(&d, 3, 4);
    for text in ["P(y_x | x')", "P(y_x | x)"] {
        let q = CtfQuery::parse(d.graph(), text).unwrap();
        let t = ctf_tr(&d, &q, &inputs).unwrap();
        assert_ctf_sound(&m, &q, &t, text);
        // Q*[W, Y] from the scholarship program, Q*[X, Z] from π².
        let sources: BTreeSet<usize> = t.mapping.values().map(|(k, _)| *k).collect();
        assert_eq!(sources, BTreeSet::from([0, 1]), "{text}: {}", t.estimand);
    }
    let q = CtfQuery::parse(d.graph(), "P(y_x | x')").unwrap();
    assert_eq!(
        ctf_tr(&d, &q, &inputs).unwrap().estimand.to_string(),
        "[sum_{w, z} P^2(z, x') P^1(w | x; σ_{X}) P^1(y | z, x, w; σ_{X})] / [P^2(x')]"
    );
    // Swapping the domains breaks it: Z ∈ Δ¹ and W ∈ Δ² (Ex. 12.17).
    let swapped = [
        DomainInput::observational(Domain::source("1")),
        DomainInput::new(Domain::source("2"), Regime::new().policy("X", &[])),
    ];
    assert!(matches!(ctf_tr(&d, &q, &swapped), Err(TrError::NotTransportable { .. })));
}

/// Example 12.18: `P(y_x, x')` is inconsistent in the bow arc, whatever the
/// data.
#[test]
fn example_12_18_inconsistent_factors_are_not_transportable() {
    let mut g = Admg::new();
    g.directed("X", "Y");
    g.bidirected("X", "Y");
    let d = SelectionDiagram::new(g).with_source("1", &[]).unwrap();
    let q = CtfQuery::parse(d.graph(), "P(y_x, x')").unwrap();
    let all = [
        DomainInput::observational(Domain::Target),
        DomainInput::new(Domain::source("1"), Regime::new().atomic(&["X"])),
    ];
    assert!(matches!(ctf_tr(&d, &q, &all), Err(TrError::Inconsistent { .. })));
}

#[test]
fn example_12_15_natural_direct_effect_is_numerically_right() {
    let mut g = Admg::new();
    g.directed("X", "Z");
    g.directed("Z", "Y");
    g.directed("X", "Y");
    g.bidirected("X", "Z");
    let d = SelectionDiagram::new(g).with_source("AU", &["Y"]).unwrap();
    let inputs = [
        DomainInput::observational(Domain::Target),
        DomainInput::new(Domain::source("AU"), Regime::new().atomic(&["X"])),
    ];
    let m = Model::with_noise(&d, 5, 4);
    for text in ["P(y_{x', Z_x})", "P(y_{x, Z_x})"] {
        let q = CtfQuery::parse(d.graph(), text).unwrap();
        let t = ctf_tr(&d, &q, &inputs).unwrap();
        assert_ctf_sound(&m, &q, &t, text);
    }
    // Neither country's data suffice alone.
    let q = CtfQuery::parse(d.graph(), "P(y_{x', Z_x})").unwrap();
    for one in &inputs {
        assert!(ctf_tr(&d, &q, std::slice::from_ref(one)).is_err());
    }
}

/// Checks a counterfactual transport against the truth at every binary
/// assignment of the query's values, skipping the zeros that rest on
/// different primes meaning different values.
fn assert_ctf_sound(m: &Model, q: &CtfQuery, t: &CtfTrDerivation, label: &str) {
    let exact = Exact::new(m);
    let none = TargetPolicy::new();
    let mut free = crate::graph::cinfer::counterfactual::query_values(q);
    if let CtfTrEstimand::Estimand(e) = &t.estimand {
        free.extend(e.free_vars());
    }
    let free: Vec<String> = free.into_iter().collect();
    for env in assignments(&free) {
        let distinct = free.iter().enumerate().all(|(i, a)| {
            free[..i]
                .iter()
                .all(|b| a.trim_end_matches('\'') != b.trim_end_matches('\'') || env[a] != env[b])
        });
        let Some(truth) = m.ctf_truth(q, &env) else { continue };
        let got = match &t.estimand {
            CtfTrEstimand::Zero if !distinct => continue,
            CtfTrEstimand::Zero => 0.0,
            CtfTrEstimand::One => 1.0,
            CtfTrEstimand::Estimand(e) => eval(&exact, &none, e, &env),
        };
        assert!((got - truth).abs() < 1e-9, "{label}: {} gives {got} at {env:?}, truth {truth}", t.estimand);
    }
}

/// Over random selection diagrams and inputs: every counterfactual transport
/// evaluates to the true counterfactual of the target, and with only the
/// target's own natural and atomic data ctf-TR decides exactly what
/// chapter 5's counterfactual C-INFER does.
#[test]
fn ctf_tr_is_sound_and_extends_ctf_infer() {
    let (mut transported, mut refused, mut cross) = (0, 0, 0);
    for seed in 0..50u64 {
        let Some(d) = random_diagram(4, seed) else { continue };
        let m = Model::with_noise(&d, seed, 4);
        let g = d.graph();
        let names = m.names.clone();
        let mut r = splitmix(seed ^ 0xc7f);
        let mut next = || {
            r = splitmix(r);
            unit(r)
        };
        let domains: Vec<Domain> = std::iter::once(Domain::Target)
            .chain(d.sources().map(Domain::source))
            .collect();
        let (x, z, y) = (&names[0], &names[2], &names[3]);
        let lx = x.to_lowercase();
        let queries = [
            format!("P({}_{lx})", y.to_lowercase()),
            format!("P({}_{lx} | {lx}')", y.to_lowercase()),
            format!("P({}_{lx}, {}_{lx}')", y.to_lowercase(), z.to_lowercase()),
            format!("P({}_{{{lx}', {}_{lx}}})", y.to_lowercase(), z.to_uppercase()),
        ];
        for text in &queries {
            let Ok(q) = CtfQuery::parse(g, text) else { panic!("{text} does not parse") };
            for _ in 0..3 {
                let k = 1 + (next() * 3.0) as usize;
                let inputs: Vec<DomainInput> = (0..k)
                    .map(|_| {
                        let dom = domains[(next() * domains.len() as f64) as usize % domains.len()].clone();
                        let reg = if next() < 0.4 {
                            Regime::new()
                        } else {
                            random_regime(g, &names, "", &mut next)
                        };
                        DomainInput::new(dom, reg)
                    })
                    .filter(|i| i.regime.apply(g).is_ok())
                    .collect();
                let label = format!("seed {seed}: {text} from {inputs:?}");
                match ctf_tr(&d, &q, &inputs) {
                    Ok(t) => {
                        transported += 1;
                        assert_ctf_sound(&m, &q, &t, &label);
                    }
                    Err(TrError::NotTransportable { .. } | TrError::Inconsistent { .. }) => refused += 1,
                    Err(e) => panic!("{label}: {e}"),
                }
            }
            let ours = [
                DomainInput::observational(Domain::Target),
                DomainInput::new(Domain::Target, Regime::new().atomic(&[z])),
            ];
            let theirs = [Input::observational(), Input::experimental(&[z])];
            assert_eq!(
                ctf_tr(&d, &q, &ours).is_ok(),
                ctf_infer(g, &q, &theirs).is_ok(),
                "seed {seed}: ctf-TR and ctf C-INFER disagree on {text}"
            );
            cross += 1;
        }
    }
    assert!(transported > 150 && refused > 50 && cross > 100, "{transported} transported, {refused} refused, {cross} cross-checked");
}


