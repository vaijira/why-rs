use std::collections::HashMap;

use super::*;
use crate::graph::EdgeRef;
use crate::graph::dseparation::EdgeKind;
use crate::graph::cinfer::tests::{splitmix, unit};
use crate::graph::cinfer::{Query, c_infer};

fn obs() -> Vec<Input> {
    vec![Input::observational()]
}

fn parse(g: &Admg, s: &str) -> CtfQuery {
    CtfQuery::parse(g, s).unwrap()
}

fn estimand(g: &Admg, s: &str) -> String {
    match ctf_infer(g, &parse(g, s), &obs()) {
        Ok(d) => d.estimand.to_string(),
        Err(e) => panic!("{s}: {e}"),
    }
}

/// X -> Y.
fn edge() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "Y");
    g
}

/// X -> Y with X <-> Y.
fn bow() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "Y");
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

/// X -> M -> Y with X <-> Y.
fn front_door() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "M");
    g.directed("M", "Y");
    g.bidirected("X", "Y");
    g
}

/// X -> M -> Y and X -> Y.
fn mediation() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "M");
    g.directed("M", "Y");
    g.directed("X", "Y");
    g
}

#[test]
fn parses_textbook_notation() {
    let g = mediation();
    for s in [
        "P(y_x | x')",
        "P(y_{m, x'}, m')",
        "P(y_{M_{x'}, x})",
        "P(y'_x, y_x')",
        "P(y | m_x, x')",
    ] {
        assert_eq!(parse(&g, s).to_string(), s);
    }
    // Without the P( ), with spacing and case freedom.
    let q = parse(&g, " Y_X , m ");
    assert_eq!(q.to_string(), "P(y_x, m)");
    // And built by hand.
    let built = CtfQuery::new(vec![
        Counterfactual::new("Y")
            .set("X", 0)
            .nest(Counterfactual::new("M").set("X", 1))
            .is(0),
    ]);
    assert_eq!(built, parse(&g, "y_{x, M_{x'}}"));
}

#[test]
fn parse_errors_point_at_the_problem() {
    let g = mediation();
    let err = CtfQuery::parse(&g, "P(y_q)").unwrap_err();
    assert_eq!(err.at, 4);
    assert!(err.message.contains("not a variable"));
    assert!(CtfQuery::parse(&g, "P(y_x").is_err());
    assert!(CtfQuery::parse(&g, "y_M_x").is_err(), "nesting needs braces");
    assert!(CtfQuery::parse(&g, "y_{x, x'}").is_err(), "X set twice");
    assert!(CtfQuery::parse(&g, "y_{M'_x}").is_err());
}

#[test]
fn ett_without_confounding_is_the_conditional() {
    assert_eq!(estimand(&edge(), "P(y_x | x')"), "P(y | x)");
}

#[test]
fn ett_with_a_back_door() {
    // sum_z P(y | x, z) P(z | x'), the textbook form, written as a ratio.
    assert_eq!(
        estimand(&back_door(), "P(y_x | x')"),
        "[sum_{z} P(z, x') P(y | z, x)] / [P(x')]"
    );
}

#[test]
fn ett_through_the_front_door() {
    let g = front_door();
    let d = ctf_infer(&g, &parse(&g, "P(y_x | x')"), &obs()).unwrap();
    assert_eq!(d.estimand.to_string(), "sum_{m} P(m | x) P(y | x', m)");
    assert_eq!(
        d.query_tree.to_string(),
        "P(y_x | x')\n\
         └─ [Γ Σ_{m}] P(y_x, x', m_x)\n   \
            ├─ [δ] P(m_x)\n   \
            │  └─ [Ξ] Q[M | do(X)](m; x)\n   \
            └─ [δ] P(y_m, x')\n      \
               └─ [Ξ] Q[X, Y | do(M)](x', y; m)\n"
    );
}

#[test]
fn ett_in_the_bow_arc_is_inconsistent() {
    let g = bow();
    let err = ctf_infer(&g, &parse(&g, "P(y_x | x')"), &obs()).unwrap_err();
    let CtfError::Inconsistent { factor, .. } = &err else {
        panic!("expected an inconsistent factor, got {err}");
    };
    let factor: Vec<String> = factor.iter().map(ToString::to_string).collect();
    assert_eq!(factor, ["y_x", "x'"]);
    // No experiment helps: Lemma 5.5.1.
    let inputs = [Input::observational(), Input::experimental(&["X"])];
    assert!(matches!(
        ctf_infer(&g, &parse(&g, "P(y_x | x')"), &inputs),
        Err(CtfError::Inconsistent { .. })
    ));
}

#[test]
fn probability_of_necessity_and_sufficiency_is_not_identifiable() {
    // Y in two worlds shares its noise, and its parent takes two values.
    let err = ctf_infer(&edge(), &parse(&edge(), "P(y_x, y'_{x'})"), &obs()).unwrap_err();
    assert!(matches!(err, CtfError::Inconsistent { .. }), "{err}");
}

#[test]
fn natural_direct_effect_term_unnests() {
    let g = mediation();
    let d = ctf_infer(&g, &parse(&g, "P(y_{x, M_{x'}})"), &obs()).unwrap();
    assert_eq!(d.estimand.to_string(), "sum_{m} P(m | x') P(y | x, m)");
    assert!(
        d.query_tree.to_string().starts_with(
            "P(y_{M_{x'}, x})\n└─ [CUT Σ_{m}] P(y_{m, x}, m_x')\n"
        ),
        "{}",
        d.query_tree
    );
}

#[test]
fn interventional_queries_reduce_to_layer_two() {
    let g = front_door();
    assert_eq!(
        estimand(&g, "P(y_x)"),
        c_infer(&g, &Query::new(&["X"], &["Y"]), &obs())
            .unwrap()
            .estimand
            .to_string()
    );
}

#[test]
fn conditioning_on_a_confounded_treatment_still_works() {
    // P(y | x) at layer 1, asked as a counterfactual: the composition rewrite
    // turns y into y_x, and the bidirected edge keeps x in the same factor.
    assert_eq!(estimand(&bow(), "P(y | x)"), "P(y | x)");
}

#[test]
fn contradictions_and_certainties() {
    let g = edge();
    let value = |s: &str| ctf_infer(&g, &parse(&g, s), &obs()).unwrap().estimand;
    assert_eq!(value("P(x'_x)"), CtfEstimand::Zero);
    assert_eq!(value("P(y_x, y'_x)"), CtfEstimand::Zero);
    assert_eq!(value("P(x_x)"), CtfEstimand::One);
    assert_eq!(value("P(y_x | y_x)"), CtfEstimand::One);
    // Y does not cause X, so X_y is just X.
    assert_eq!(value("P(x_y)").to_string(), "P(x)");
    assert_eq!(
        ctf_infer(&g, &parse(&g, "P(y | x_x')"), &obs()).unwrap_err(),
        CtfError::ImpossibleCondition
    );
}

#[test]
fn an_experiment_rescues_a_factor() {
    // X -> Y -> W, with Y <-> W: Q[W] is a hedge from P(V) — the bow arc on
    // Y -> W — but an experiment on Y produces it.
    let mut g = Admg::new();
    g.directed("X", "Y");
    g.directed("Y", "W");
    g.bidirected("Y", "W");
    let q = parse(&g, "P(w_y | x)");
    let err = ctf_infer(&g, &q, &obs()).unwrap_err();
    assert!(matches!(err, CtfError::NotIdentifiable { .. }), "{err}");
    let d = ctf_infer(&g, &q, &[Input::observational(), Input::experimental(&["Y"])]).unwrap();
    assert_eq!(d.estimand.to_string(), "P(w | do(y))");
}

#[test]
fn unknown_names_are_reported() {
    let q = CtfQuery::new(vec![Counterfactual::new("Y").set("Q", 0).is(0)]);
    assert_eq!(
        ctf_infer(&edge(), &q, &obs()).unwrap_err(),
        CtfError::Graph(DSepError::Unknown("Q".into()))
    );
    let q = CtfQuery::new(vec![Counterfactual::new("Y").is(0)]);
    assert_eq!(
        ctf_infer(&edge(), &q, &[Input::experimental(&["Q"])]).unwrap_err(),
        CtfError::Graph(DSepError::Unknown("Q".into()))
    );
}

/// A random binary SCM over a random ADMG. Every variable is a response
/// function of its parents, the latents behind its bidirected edges, and its
/// own noise, and the noise is shared across worlds — which is what gives
/// counterfactuals their meaning. Noise values 0 and 1 force the output to 0
/// and 1, so every interventional conditional is strictly positive.
struct Scm {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    latents_of: Vec<Vec<usize>>,
    seed: u64,
    /// Every exogenous state: its probability, latent bits and noises. A
    /// noise of 2 selects a pseudo-random response function.
    units: Vec<(f64, u64, Vec<usize>)>,
}

impl Scm {
    /// A random ADMG over `V0..Vn` with at most four bidirected edges, and a
    /// random SCM on it.
    fn random(n: usize, seed: u64) -> Option<(Admg, Self)> {
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
                if next() < 0.5 {
                    g.directed(&names[i], &names[j]);
                }
                if next() < 0.3 {
                    g.bidirected(&names[i], &names[j]);
                    latents += 1;
                }
            }
        }
        if latents > 4 {
            return None;
        }
        let scm = Self::from_graph(&g, seed);
        Some((g, scm))
    }

    /// A random SCM on `g`: one binary latent per bidirected edge.
    fn from_graph(g: &Admg, seed: u64) -> Self {
        let names = g.topological_order().expect("acyclic");
        let n = names.len();
        let at = |v: &str| names.iter().position(|n| n == v).expect("known");
        let mut parents = vec![Vec::new(); n];
        let mut latents_of = vec![Vec::new(); n];
        let mut latent_p = Vec::new();
        let mut r = splitmix(seed ^ 0x5eed);
        let mut next = || {
            r = splitmix(r);
            unit(r)
        };
        let graph = g.graph();
        for e in graph.edge_references() {
            let (a, b) = (at(&graph[e.source()]), at(&graph[e.target()]));
            match e.weight() {
                EdgeKind::Directed => parents[b].push(a),
                EdgeKind::Bidirected => {
                    latents_of[a].push(latent_p.len());
                    latents_of[b].push(latent_p.len());
                    latent_p.push(0.15 + 0.7 * next());
                }
            }
        }
        let noise_p: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                let w = [next() + 0.1, next() + 0.1, next() + 0.1];
                let s: f64 = w.iter().sum();
                [w[0] / s, w[1] / s, w[2] / s]
            })
            .collect();
        let mut units = Vec::new();
        for u in 0..(1u64 << latent_p.len()) {
            let mut pu = 1.0;
            for (l, p) in latent_p.iter().enumerate() {
                pu *= if (u >> l) & 1 == 1 { *p } else { 1.0 - p };
            }
            for code in 0..3usize.pow(u32::try_from(n).expect("small")) {
                let eps: Vec<usize> = (0..n).map(|i| code / 3usize.pow(i as u32) % 3).collect();
                let pe: f64 = eps.iter().enumerate().map(|(i, &e)| noise_p[i][e]).product();
                units.push((pu * pe, u, eps));
            }
        }
        Self {
            names,
            parents,
            latents_of,
            seed,
            units,
        }
    }

    fn index(&self, name: &str) -> usize {
        let base = name.trim_end_matches('\'');
        self.names.iter().position(|n| n == base).expect("known")
    }

    fn f(&self, i: usize, v: u64, u: u64, e: usize) -> u64 {
        match e {
            0 => 0,
            1 => 1,
            _ => {
                let mut key = self.seed ^ (i as u64) << 56;
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

    /// The world where `fixed` was set, for one exogenous state.
    fn solve(&self, u: u64, eps: &[usize], fixed: &[(usize, u64)]) -> u64 {
        let mut v = 0;
        for (i, &e) in eps.iter().enumerate() {
            let bit = match fixed.iter().find(|(f, _)| *f == i) {
                Some(&(_, b)) => b,
                None => self.f(i, v, u, e),
            };
            v |= bit << i;
        }
        v
    }

    fn joint(&self, fixed: &[(usize, u64)]) -> Vec<f64> {
        let mut table = vec![0.0; 1 << self.names.len()];
        for (p, u, eps) in &self.units {
            table[self.solve(*u, eps, fixed) as usize] += p;
        }
        table
    }

    fn value_of(&self, c: &Counterfactual, u: u64, eps: &[usize], env: &HashMap<String, u64>) -> u64 {
        let fixed: Vec<(usize, u64)> = c
            .subscript
            .iter()
            .map(|(k, s)| {
                let b = match s {
                    Setting::Value(p) => env[&value_name(k, *p)],
                    Setting::Nested(inner) => self.value_of(inner, u, eps, env),
                };
                (self.index(k), b)
            })
            .collect();
        (self.solve(u, eps, &fixed) >> self.index(&c.var)) & 1
    }

    /// The true `P(events | given)`, or `None` when `given` is impossible.
    fn truth(&self, q: &CtfQuery, env: &HashMap<String, u64>) -> Option<f64> {
        let holds = |e: &Event, u: u64, eps: &[usize]| {
            self.value_of(&e.ctf, u, eps, env) == env[&value_name(&e.ctf.var, e.value)]
        };
        let (mut num, mut den) = (0.0, 0.0);
        for (p, u, eps) in &self.units {
            if q.given.iter().all(|e| holds(e, *u, eps)) {
                den += p;
                if q.events.iter().all(|e| holds(e, *u, eps)) {
                    num += p;
                }
            }
        }
        (den > 1e-12).then(|| num / den)
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

struct Evaluator<'a> {
    scm: &'a Scm,
    cache: HashMap<Vec<(usize, u64)>, Vec<f64>>,
}

impl Evaluator<'_> {
    fn eval(&mut self, f: &Formula, env: &mut HashMap<String, u64>) -> f64 {
        match f {
            Formula::P { vars, given, doing } => {
                let scm = self.scm;
                let mut fixed: Vec<(usize, u64)> =
                    doing.iter().map(|d| (scm.index(d), env[d])).collect();
                fixed.sort_unstable();
                let table = self.cache.entry(fixed).or_insert_with_key(|k| scm.joint(k));
                let bind = |names: &[String]| -> Vec<(usize, u64)> {
                    names.iter().map(|n| (scm.index(n), env[n])).collect()
                };
                let g = bind(given);
                let mut joint = bind(vars);
                joint.extend(g.iter().copied());
                marginal(table, &joint) / marginal(table, &g)
            }
            Formula::Product(fs) => fs.iter().map(|f| self.eval(f, env)).product(),
            Formula::Ratio { num, den } => self.eval(num, env) / self.eval(den, env),
            Formula::Sum { over, body } => {
                let shadowed: Vec<Option<u64>> = over.iter().map(|v| env.get(v).copied()).collect();
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

/// Checks `d`'s estimand against the truth at every binary assignment of the
/// query's values and of any other free name in the estimand. A zero rests on
/// different primes meaning different values, so it is checked only where
/// they do.
fn assert_sound(scm: &Scm, q: &CtfQuery, d: &CtfDerivation) {
    let mut ev = Evaluator {
        scm,
        cache: HashMap::new(),
    };
    let mut free = query_values(q);
    if let CtfEstimand::Formula(f) = &d.estimand {
        free.extend(f.free_vars());
    }
    let free: Vec<String> = free.into_iter().collect();
    for bits in 0..(1u64 << free.len()) {
        let mut env: HashMap<String, u64> = free
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), (bits >> k) & 1))
            .collect();
        let distinct = free.iter().enumerate().all(|(i, a)| {
            free[..i].iter().all(|b| {
                a.trim_end_matches('\'') != b.trim_end_matches('\'') || env[a] != env[b]
            })
        });
        let Some(truth) = scm.truth(q, &env) else { continue };
        let got = match &d.estimand {
            CtfEstimand::Zero if !distinct => continue,
            CtfEstimand::Zero => 0.0,
            CtfEstimand::One => 1.0,
            CtfEstimand::Formula(f) => ev.eval(f, &mut env),
        };
        assert!(
            (got - truth).abs() < 1e-9,
            "{q} at {env:?}: estimand {} gives {got}, truth is {truth}",
            d.estimand
        );
    }
}

/// Over random graphs and SCMs: every counterfactual estimand C-INFER returns
/// evaluates to the true counterfactual probability, and on `P(y_x)` it
/// identifies exactly what layer-2 C-INFER does.
#[test]
fn estimands_are_numerically_sound() {
    let mut identified = 0;
    let mut refused = 0;
    let mut tried = 0;
    for seed in 0..40u64 {
        let n = if seed % 4 == 0 { 5 } else { 4 };
        let Some((g, scm)) = Scm::random(n, seed) else { continue };
        let names = scm.names.clone();
        for (xi, x) in names.iter().enumerate() {
            for y in &names[xi + 1..] {
                let others: Vec<&String> = names.iter().filter(|v| *v != x && *v != y).collect();
                let z = others[(splitmix(seed ^ xi as u64) as usize) % others.len()];
                let ctf = |v: &str| Counterfactual::new(v);
                let queries = [
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0)]).given(vec![ctf(x).is(1)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0), ctf(x).is(1)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0), ctf(y).set(x, 1).is(1)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0)]).given(vec![ctf(z).is(0)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0), ctf(z).set(x, 1).is(0)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).nest(ctf(z).set(x, 1)).is(0)]),
                    CtfQuery::new(vec![ctf(y).set(x, 0).is(0)])
                        .given(vec![ctf(x).is(1), ctf(z).is(1)]),
                    CtfQuery::new(vec![ctf(y).is(0)]).given(vec![ctf(z).set(x, 1).is(0)]),
                ];
                for (k, q) in queries.iter().enumerate() {
                    tried += 1;
                    let ours = ctf_infer(&g, q, &obs());
                    if k == 0 {
                        let l2 = c_infer(&g, &Query::new(&[x], &[y]), &obs());
                        assert_eq!(ours.is_ok(), l2.is_ok(), "seed {seed}: {q} vs layer 2");
                    }
                    match &ours {
                        Ok(d) => {
                            identified += 1;
                            assert_sound(&scm, q, d);
                        }
                        Err(_) => refused += 1,
                    }
                    // An experiment on z can only help.
                    let more = [Input::observational(), Input::experimental(&[z])];
                    match ctf_infer(&g, q, &more) {
                        Ok(d) => assert_sound(&scm, q, &d),
                        Err(e) => assert!(ours.is_err(), "seed {seed}: do({z}) lost {q}: {e}"),
                    }
                }
            }
        }
    }
    // Guard against a generator that only produces trivial cases.
    assert!(identified > 200, "only {identified} of {tried} identified");
    assert!(refused > 40, "only {refused} of {tried} refused");
}


// The chapter 5 benchmarks. Each checks the q-tree's operators, the estimand,
// and — on a random SCM over the same graph — that the estimand is the true
// counterfactual probability and equals the book's expected result.

fn p(vars: &[&str], given: &[&str], doing: &[&str]) -> Formula {
    let own = |v: &[&str]| v.iter().map(ToString::to_string).collect();
    Formula::P {
        vars: own(vars),
        given: own(given),
        doing: own(doing),
    }
}

fn sum(over: &[&str], factors: Vec<Formula>) -> Formula {
    Formula::Sum {
        over: over.iter().map(ToString::to_string).collect(),
        body: Box::new(Formula::Product(factors)),
    }
}

/// Checks `d` against the truth of `q` on a random SCM over `g`, and that it
/// agrees with `expected` at every assignment of the values involved.
fn assert_benchmark(g: &Admg, q: &CtfQuery, d: &CtfDerivation, expected: &Formula) {
    let scm = Scm::from_graph(g, 7);
    assert_sound(&scm, q, d);
    let CtfEstimand::Formula(got) = &d.estimand else {
        panic!("{q}: expected a formula, got {}", d.estimand);
    };
    let mut ev = Evaluator {
        scm: &scm,
        cache: HashMap::new(),
    };
    let mut free = query_values(q);
    free.extend(got.free_vars());
    free.extend(expected.free_vars());
    let free: Vec<String> = free.into_iter().collect();
    for bits in 0..(1u64 << free.len()) {
        let mut env: HashMap<String, u64> = free
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), (bits >> k) & 1))
            .collect();
        let (a, b) = (ev.eval(got, &mut env), ev.eval(expected, &mut env));
        assert!((a - b).abs() < 1e-9, "{q} at {env:?}: {got} gives {a}, {expected} gives {b}");
    }
}

/// Benchmark 1: the effect of treatment on the treated in the back-door
/// graph, from P(V). Γ makes the ratio, the ancestral set adds Z, δ splits
/// it into three singletons, and Ξ collapses each.
#[test]
fn benchmark_ett_back_door() {
    let g = back_door();
    let q = parse(&g, "P(y_x | x')");
    let d = ctf_infer(&g, &q, &obs()).unwrap();
    assert_eq!(
        d.query_tree.to_string(),
        "P(y_x | x')\n\
         └─ [Γ Σ_{z}] P(y_x, x', z)\n   \
            ├─ [δ] P(x'_z)\n   \
            │  └─ [Ξ] Q[X | do(Z)](x'; z)\n   \
            ├─ [δ] P(y_{x, z})\n   \
            │  └─ [Ξ] Q[Y | do(X, Z)](y; x, z)\n   \
            └─ [δ] P(z)\n      \
               └─ [Ξ] Q[Z](z)\n"
    );
    assert_eq!(
        d.estimand.to_string(),
        "[sum_{z} P(z, x') P(y | z, x)] / [P(x')]"
    );
    // The book's sum_z P(y | x, z) P(z | x').
    let book = sum(&["Z"], vec![p(&["Y"], &["X", "Z"], &[]), p(&["Z"], &["X'"], &[])]);
    assert_benchmark(&g, &q, &d, &book);
}

/// X -> Z -> Y with X <-> Y.
fn front_door_xzy() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "Z");
    g.directed("Z", "Y");
    g.bidirected("X", "Y");
    g
}

/// Benchmark 2: the nested P(Y_{x1, Z_{x0}} = y) in the front-door graph,
/// written with x1 = x' and x0 = x. CUT sums over z; the rewrite drops x'
/// from Y's subscript, since X reaches Y only through Z; δ separates Y from Z,
/// which share no bidirected edge; Ξ collapses both.
#[test]
fn benchmark_front_door_nested() {
    let g = front_door_xzy();
    let q = parse(&g, "P(y_{x', Z_x})");
    let d = ctf_infer(&g, &q, &obs()).unwrap();
    assert_eq!(
        d.query_tree.to_string(),
        "P(y_{x', Z_{x}})\n\
         └─ [CUT Σ_{z}] P(y_z, z_x)\n   \
            └─ [Σ_{z}] P(y_z, z_x)\n      \
               ├─ [δ] P(y_z)\n      \
               │  └─ [Ξ] Q[Y | do(Z)](y; z)\n      \
               └─ [δ] P(z_x)\n         \
                  └─ [Ξ] Q[Z | do(X)](z; x)\n"
    );
    // Q[Y | do(Z)] is itself the front-door adjustment; its index is not
    // named x', which the query uses for another value.
    assert_eq!(
        d.estimand.to_string(),
        "sum_{z} P(z | x) sum_{x''} P(x'') P(y | x'', z)"
    );
    // The book's sum_z P(y | do(x1, z)) P(z | do(x0)).
    let book = sum(&["Z"], vec![p(&["Y"], &[], &["X'", "Z"]), p(&["Z"], &[], &["X"])]);
    assert_benchmark(&g, &q, &d, &book);
}

/// R -> T -> X -> Z -> Y and W -> X.
fn chains() -> Admg {
    let mut g = Admg::new();
    for (a, b) in [("R", "T"), ("T", "X"), ("X", "Z"), ("Z", "Y"), ("W", "X")] {
        g.directed(a, b);
    }
    g
}

/// Benchmark 3: P(Y_{x1, Z_{x0}} = y1 | X_{w0} = x1, T_{r1} = t1, R = r0),
/// written with index 1 as a prime. CUT unnests Z_x; composition turns X_w
/// into X_{r, w} on the event R = r; Γ keeps only the factors linked to Y.
///
/// This graph has no bidirected edge, so the conditioning events share no
/// noise with Y's or Z's and cancel. P(V) is enough, and given first it is
/// used. Given alone, the experiment P(V | do(W, T)) yields both c-factors
/// through σ.
#[test]
fn benchmark_conditional_mediated() {
    let g = chains();
    let q = parse(&g, "P(y'_{x', Z_x} | x'_w, t'_{r'}, r)");
    let wt = Input::experimental(&["W", "T"]);
    let tree = "P(y'_{x', Z_{x}} | x'_w, t'_r', r)\n\
                └─ [CUT Σ_{z}] P(y'_z, z_x | x'_{r, w}, t'_r', r)\n   \
                   └─ [Γ Σ_{z}] P(y'_z, z_x)\n      \
                      ├─ [δ] P(y'_z)\n      \
                      │  └─ [Ξ] Q[Y | do(Z)](y'; z)\n      \
                      └─ [δ] P(z_x)\n         \
                         └─ [Ξ] Q[Z | do(X)](z; x)\n";
    // The book's answer reduces to sum_z P(y1 | do(z)) P(z | do(x0)).
    let book = sum(&["Z"], vec![p(&["Y'"], &[], &["Z"]), p(&["Z"], &[], &["X"])]);

    let d = ctf_infer(&g, &q, &[Input::observational(), wt.clone()]).unwrap();
    assert_eq!(d.query_tree.to_string(), tree);
    assert_eq!(d.estimand.to_string(), "sum_{z} P(z | x) P(y' | z)");
    assert!(d.mapping.values().all(|s| matches!(s, Source::Input { input: 0, .. })));
    assert_benchmark(&g, &q, &d, &book);

    let d = ctf_infer(&g, &q, &[wt]).unwrap();
    assert_eq!(d.query_tree.to_string(), tree);
    assert_eq!(
        d.estimand.to_string(),
        "sum_{z} P(z | x, do(t, w)) P(y' | z, do(t, w))"
    );
    assert_eq!(
        d.input_trees[0].to_string(),
        "Q[R, X, Y, Z | do(T, W)]\n\
         ├─ [Σ_{R}] Q[X, Y, Z | do(T, W)]\n\
         │  └─ [δ] Q[Y | do(T, W)]\n\
         │     └─ [σ do(T, W) → do(Z)] Q[Y | do(Z)]\n\
         └─ [Σ_{R, Y}] Q[X, Z | do(T, W)]\n   \
            └─ [δ] Q[Z | do(T, W)]\n      \
               └─ [σ do(T, W) → do(X)] Q[Z | do(X)]\n"
    );
    assert_benchmark(&g, &q, &d, &book);
}

/// X -> W -> Z -> Y with X <-> W.
fn confounded_chain() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "W");
    g.directed("W", "Z");
    g.directed("Z", "Y");
    g.bidirected("X", "W");
    g
}

/// Benchmark 4: P(W_x = w, W_{x'} = w') is not identifiable. Both events are
/// W, so they share W's noise and form one ctf-factor, in which W's parent X
/// takes both x and x' — inconsistent, so Ξ cannot apply.
#[test]
fn benchmark_inconsistent_factor() {
    let g = confounded_chain();
    let q = parse(&g, "P(w_x, w'_{x'})");
    let err = ctf_infer(&g, &q, &obs()).unwrap_err();
    let CtfError::Inconsistent { query_tree, factor } = &err else {
        panic!("expected an inconsistent factor, got {err}");
    };
    let factor: Vec<String> = factor.iter().map(ToString::to_string).collect();
    assert_eq!(factor, ["w_x", "w'_x'"]);
    assert_eq!(query_tree.to_string(), "P(w_x, w'_x')\n└─ [Σ_{}] P(w_x, w'_x')\n");

    // Lemma 5.5.1: no experiment rescues it, not even on X.
    let inputs = [Input::observational(), Input::experimental(&["X"])];
    assert!(matches!(ctf_infer(&g, &q, &inputs), Err(CtfError::Inconsistent { .. })));
    // And the confounding is not what makes it fail: the shared noise of W is.
    let mut plain = Admg::new();
    for (a, b) in [("X", "W"), ("W", "Z"), ("Z", "Y")] {
        plain.directed(a, b);
    }
    assert!(matches!(
        ctf_infer(&plain, &q, &obs()),
        Err(CtfError::Inconsistent { .. })
    ));
}
