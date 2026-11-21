//! Tests for chapter 11: the book's examples, and random multi-domain models
//! that check every estimand against the true target quantity.

use std::collections::{BTreeSet, HashMap};

use super::*;
use crate::fairness::SplitMix64;
use crate::graph::dseparation::Admg;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// A random binary model compatible with a selection diagram: one binary
/// latent per bidirected edge and strictly positive conditional tables. A
/// source's table for `V` differs from the target's exactly when
/// `V ∈ Δ^{k,*}`; a latent's distribution differs when all its children do.
struct Model {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    latents_of: Vec<Vec<usize>>,
    latent_children: Vec<Vec<usize>>,
    latent_p: Vec<f64>,
    seed: u64,
    domains: Vec<(Domain, BTreeSet<usize>)>,
}

impl Model {
    fn from_diagram(d: &SelectionDiagram, seed: u64) -> Self {
        let names: Vec<String> = d.graph().variables().into_iter().map(str::to_string).collect();
        let index = |n: &str| names.iter().position(|m| m == n).expect("known");
        let n = names.len();
        let mut parents = vec![Vec::new(); n];
        let mut latents_of = vec![Vec::new(); n];
        let mut latent_children = Vec::new();
        let mut latent_p = Vec::new();
        let g = d.graph().graph();
        for e in g.edge_indices() {
            let (a, b) = g.edge_endpoints(e).expect("edge");
            let (a, b) = (index(&g[a]), index(&g[b]));
            match g[e] {
                EdgeKind::Directed => parents[b].push(a),
                EdgeKind::Bidirected => {
                    let l = latent_p.len();
                    latents_of[a].push(l);
                    latents_of[b].push(l);
                    latent_children.push(vec![a, b]);
                    latent_p.push(0.15 + 0.7 * unit(splitmix(seed ^ 0x5eed ^ l as u64)));
                }
            }
        }
        let mut domains = vec![(Domain::Target, BTreeSet::new())];
        for s in d.sources() {
            let delta = d.discrepancies(s).expect("source").iter().map(|v| index(v)).collect();
            domains.push((Domain::source(s), delta));
        }
        Self {
            names,
            parents,
            latents_of,
            latent_children,
            latent_p,
            seed,
            domains,
        }
    }

    fn index(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).expect("known")
    }

    fn salt(&self, dom: usize, differs: bool) -> u64 {
        if differs {
            splitmix(self.seed ^ 0xd0_0000 ^ dom as u64)
        } else {
            0
        }
    }

    fn p_one(&self, dom: usize, i: usize, v: u64, u: u64) -> f64 {
        let delta = &self.domains[dom].1;
        let mut key = self.seed ^ (i as u64) << 56 ^ self.salt(dom, delta.contains(&i));
        for &p in &self.parents[i] {
            key = splitmix(key ^ ((v >> p) & 1) << 1 ^ p as u64);
        }
        for &l in &self.latents_of[i] {
            key = splitmix(key ^ ((u >> l) & 1) << 1 ^ (l as u64) << 8);
        }
        0.1 + 0.8 * unit(splitmix(key))
    }

    fn latent(&self, dom: usize, l: usize) -> f64 {
        let delta = &self.domains[dom].1;
        if self.latent_children[l].iter().all(|c| delta.contains(c)) && dom != 0 {
            0.15 + 0.7 * unit(splitmix(self.seed ^ self.salt(dom, true) ^ l as u64))
        } else {
            self.latent_p[l]
        }
    }

    /// The joint over V in domain `dom`, indexed by bitmask.
    fn joint(&self, dom: usize) -> Vec<f64> {
        let n = self.names.len();
        let mut table = vec![0.0; 1 << n];
        for u in 0..(1u64 << self.latent_p.len()) {
            let mut pu = 1.0;
            for l in 0..self.latent_p.len() {
                let p = self.latent(dom, l);
                pu *= if (u >> l) & 1 == 1 { p } else { 1.0 - p };
            }
            for v in 0..(1u64 << n) {
                let mut p = pu;
                for i in 0..n {
                    let one = self.p_one(dom, i, v, u);
                    p *= if (v >> i) & 1 == 1 { one } else { 1.0 - one };
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

/// The exact distributions of every domain, with the target restricted to
/// the measured variables when `measured` is set.
struct Exact<'a> {
    model: &'a Model,
    tables: HashMap<Domain, Vec<f64>>,
    measured: Option<BTreeSet<String>>,
}

impl<'a> Exact<'a> {
    fn new(model: &'a Model) -> Self {
        let tables = (0..model.domains.len())
            .map(|d| (model.domains[d].0.clone(), model.joint(d)))
            .collect();
        Self {
            model,
            tables,
            measured: None,
        }
    }

    fn measuring(&self, w: &[&str]) -> Self {
        Self {
            model: self.model,
            tables: self.tables.clone(),
            measured: Some(w.iter().map(|s| (*s).to_string()).collect()),
        }
    }

    fn truth(&self, y: &str, yv: u64, z: &[(&str, u64)]) -> f64 {
        let t = &self.tables[&Domain::Target];
        let zs: Vec<(usize, u64)> = z.iter().map(|(n, v)| (self.model.index(n), *v)).collect();
        let mut yz = zs.clone();
        yz.push((self.model.index(y), yv));
        marginal(t, &yz) / marginal(t, &zs)
    }
}

impl Distributions for Exact<'_> {
    fn levels(&self, var: &str) -> Option<Vec<String>> {
        self.model
            .names
            .iter()
            .any(|n| n == var)
            .then(|| vec!["0".to_string(), "1".to_string()])
    }

    fn probability(&self, domain: &Domain, event: &[(&str, &str)]) -> Result<f64, TransportError> {
        if *domain == Domain::Target
            && let Some(w) = &self.measured
            && let Some((v, _)) = event.iter().find(|(v, _)| !w.contains(*v))
        {
            return Err(TransportError::MissingData(format!("target does not measure {v}")));
        }
        let table = self
            .tables
            .get(domain)
            .ok_or_else(|| TransportError::MissingData(format!("no domain {domain}")))?;
        let a: Vec<(usize, u64)> = event
            .iter()
            .map(|(n, v)| (self.model.index(n), u64::from(*v == "1")))
            .collect();
        Ok(marginal(table, &a))
    }
}

/// Checks `e` against the true `P*(y | z)` at every binary assignment.
fn assert_sound(exact: &Exact<'_>, e: &Estimand, y: &str, z: &[&str], label: &str) {
    let mut allowed: BTreeSet<String> = z.iter().map(|s| (*s).to_string()).collect();
    allowed.insert(y.to_string());
    assert!(
        e.free_vars().is_subset(&allowed),
        "{label}: {e} has free variables beyond {allowed:?}"
    );
    for bits in 0..(1u64 << (z.len() + 1)) {
        let zv: Vec<(&str, u64)> = z.iter().enumerate().map(|(k, n)| (*n, (bits >> k) & 1)).collect();
        let yv = (bits >> z.len()) & 1;
        let truth = exact.truth(y, yv, &zv);
        let ys = yv.to_string();
        let mut at: Vec<(&str, &str)> = zv
            .iter()
            .map(|(n, v)| (*n, if *v == 1 { "1" } else { "0" }))
            .collect();
        at.push((y, &ys));
        let got = e
            .evaluate(exact, &at)
            .unwrap_or_else(|err| panic!("{label}: {e} fails to evaluate: {err}"));
        assert!(
            (got - truth).abs() < 1e-9,
            "{label}: {e} gives {got} at {at:?}, truth is {truth}"
        );
    }
}

fn random_diagram(n: usize, seed: u64) -> SelectionDiagram {
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
    for j in 0..n {
        for i in 0..j {
            if next() < 0.45 {
                g.directed(&names[i], &names[j]);
            }
            if next() < 0.25 {
                g.bidirected(&names[i], &names[j]);
            }
        }
    }
    let k = 1 + (seed % 2) as usize;
    let mut deltas = Vec::new();
    for _ in 0..k {
        let delta: Vec<&str> = names
            .iter()
            .filter(|_| next() < 0.35)
            .map(String::as_str)
            .collect();
        deltas.push(delta);
    }
    let mut d = SelectionDiagram::new(g);
    for (s, delta) in deltas.iter().enumerate() {
        d = d.with_source(&(s + 1).to_string(), delta).expect("fresh source");
    }
    d
}

/// Over random selection diagrams: Score-TR and the statistical
/// transportability product with nothing measured in the target decide the
/// same queries (both are complete), direct transportability implies both,
/// and every estimand evaluates to the true target conditional.
#[test]
fn score_tr_and_the_product_agree_and_are_sound() {
    let mut transported = 0;
    let mut beyond_direct = 0;
    let mut refused = 0;
    for seed in 0..150u64 {
        let n = if seed % 3 == 0 { 5 } else { 4 };
        let d = random_diagram(n, seed);
        let model = Model::from_diagram(&d, seed);
        let exact = Exact::new(&model);
        let no_target = exact.measuring(&[]);
        let names: Vec<String> = model.names.clone();
        for y in &names {
            let others: Vec<&str> = names.iter().filter(|v| *v != y).map(String::as_str).collect();
            for z in subsets(&others).expect("small") {
                let label = format!("seed {seed}: P*({y} | {z:?})");
                let st = score_tr(&d, &z, y);
                let pr = statistical_transport(&d, &z, y, &[]);
                assert_eq!(
                    st.is_ok(),
                    pr.is_ok(),
                    "{label}: Score-TR {:?} vs product {:?}",
                    st.as_ref().map(|s| s.estimand.to_string()),
                    pr.as_ref().map(ToString::to_string)
                );
                let direct = d
                    .sources()
                    .any(|s| is_directly_transportable(&d, s, y, &z).expect("valid"));
                if direct {
                    assert!(st.is_ok(), "{label}: directly transportable but refused");
                }
                match (st, pr) {
                    (Ok(s), Ok(p)) => {
                        transported += 1;
                        if !direct {
                            beyond_direct += 1;
                        }
                        assert_sound(&no_target, &s.estimand, y, &z, &label);
                        assert_sound(&no_target, &p, y, &z, &label);
                    }
                    _ => refused += 1,
                }
            }
        }
    }
    assert!(transported > 1500, "only {transported} transported");
    assert!(beyond_direct > 100, "only {beyond_direct} beyond direct transport");
    assert!(refused > 500, "only {refused} refused");
}

/// Over random diagrams and random target measurements `W`: every estimand
/// of Thm 11.3.4 reads the target only through `P*(W)` and is sound;
/// Thms 11.3.1–11.3.3 are sound and never succeed where Thm 11.3.4 fails;
/// and measuring more never loses a query.
#[test]
fn the_product_is_sound_and_subsumes_the_scenarios() {
    let mut rescued = 0;
    let mut scenarios = 0;
    for seed in 0..120u64 {
        let n = if seed % 3 == 0 { 5 } else { 4 };
        let d = random_diagram(n, seed ^ 0xabc);
        let model = Model::from_diagram(&d, seed);
        let exact = Exact::new(&model);
        let names: Vec<String> = model.names.clone();
        let mut r = seed;
        for y in &names {
            let others: Vec<&str> = names.iter().filter(|v| *v != y).map(String::as_str).collect();
            for x in subsets(&others).expect("small").into_iter().skip(1) {
                r = splitmix(r);
                let w: Vec<&str> = names
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (r >> i) & 1 == 1)
                    .map(|(_, v)| v.as_str())
                    .collect();
                let label = format!("seed {seed}: P*({y} | {x:?}) with W = {w:?}");
                let measured = exact.measuring(&w);
                let without = statistical_transport(&d, &x, y, &[]);
                let with = statistical_transport(&d, &x, y, &w);
                if without.is_ok() {
                    assert!(with.is_ok(), "{label}: measuring more lost the query");
                }
                if let Ok(e) = &with {
                    if without.is_err() {
                        rescued += 1;
                    }
                    assert!(
                        e.target_variables().iter().all(|v| w.contains(&v.as_str())),
                        "{label}: {e} reads unmeasured target variables"
                    );
                    assert_sound(&measured, e, y, &x, &label);
                }
                for s in d.sources() {
                    let found = [
                        features_measured(&d, s, &x, y, &w),
                        outcome_measured(&d, s, &x, y, &w),
                        auxiliary_measured(&d, s, &x, y, &w),
                    ];
                    for f in found.into_iter().filter_map(|f| f.expect("valid")) {
                        scenarios += 1;
                        assert!(with.is_ok(), "{label}: a scenario theorem applies but Thm 11.3.4 fails");
                        assert_sound(&measured, &f.estimand, y, &x, &label);
                    }
                }
            }
        }
    }
    assert!(rescued > 100, "only {rescued} rescued by target data");
    assert!(scenarios > 300, "only {scenarios} scenario formulas");
}

fn diagram(directed: &[(&str, &str)], bidirected: &[(&str, &str)]) -> Admg {
    let mut g = Admg::new();
    for (a, b) in directed {
        g.directed(a, b);
    }
    for (a, b) in bidirected {
        g.bidirected(a, b);
    }
    g
}

#[test]
fn figures_11_6_and_11_7() {
    let transportable = [
        (diagram(&[("X", "Z"), ("Z", "Y")], &[]), "X"),
        (diagram(&[("X", "Z"), ("Y", "Z")], &[]), "X"),
        (diagram(&[("X", "Z"), ("Y", "Z")], &[]), "Z"),
    ];
    for (g, s) in transportable {
        let d = SelectionDiagram::new(g).with_source("1", &[s]).unwrap();
        assert!(is_directly_transportable(&d, "1", "Y", &["X"]).unwrap());
    }
    let not = [
        (diagram(&[("X", "Y")], &[]), "Y"),
        (diagram(&[("X", "Y")], &[("X", "Y")]), "X"),
        (diagram(&[("Y", "X")], &[]), "X"),
    ];
    for (g, s) in not {
        let d = SelectionDiagram::new(g).with_source("1", &[s]).unwrap();
        assert!(!is_directly_transportable(&d, "1", "Y", &["X"]).unwrap());
    }
}

/// Ex. 11.4. `P^{2*}(Y = 1 | X = 1)` is 3/4: the book's 3/8 is the joint
/// `P(Y = 1, X = 1)`. Either way it differs from the sources' 1/2.
#[test]
fn example_11_4_differs_from_the_source() {
    let (mut joint, mut px) = (0.0f64, 0.0f64);
    for ux in 0..2 {
        for uxy in 0..2 {
            let p = if ux == 1 { 0.75 } else { 0.25 } * 0.5;
            let x = ux ^ uxy;
            let y = x ^ uxy;
            if x == 1 {
                px += p;
                if y == 1 {
                    joint += p;
                }
            }
        }
    }
    assert!((joint - 3.0 / 8.0).abs() < 1e-12);
    assert!((joint / px - 0.75).abs() < 1e-12);
}

/// Ex. 11.5 on Fig. 11.15(b). The figure's edges are not in the text; this
/// reconstruction (S → X → Y ← Z, Y → R ← S) reproduces Fig. 11.9: only
/// `P(Y | X, Z)` and `P(Y | X)` are directly transportable.
#[test]
fn example_11_5_lists_two_distributions() {
    let g = diagram(&[("X", "Y"), ("Z", "Y"), ("Y", "R")], &[]);
    let d = SelectionDiagram::new(g).with_source("1", &["X", "R"]).unwrap();
    let mut found = list_dtr_dist(&d, "1", "Y", &["X", "Z", "R"]).unwrap();
    found.sort();
    assert_eq!(found, vec![vec!["X"], vec!["X", "Z"]]);
}

/// Fig. 11.8: `k` disjoint two-node paths from S to Y have `3^k` separators.
#[test]
fn figure_11_8_has_exponentially_many_separators() {
    let mut g = Admg::new();
    let mut delta = Vec::new();
    let mut names = Vec::new();
    for i in 1..=3 {
        let (a, b) = (format!("X{i}1"), format!("X{i}2"));
        g.directed(&a, &b);
        g.directed(&b, "Y");
        names.push(a.clone());
        names.push(b);
        delta.push(a);
    }
    let d = SelectionDiagram::new(g)
        .with_source("1", &as_refs(&delta))
        .unwrap();
    let all = list_dtr_dist(&d, "1", "Y", &as_refs(&names)).unwrap();
    assert_eq!(all.len(), 27);
    for x in &all {
        assert!(is_directly_transportable(&d, "1", "Y", &as_refs(x)).unwrap());
    }
}

/// Fig. 11.12(a), reconstructed from Exs. 11.10–11.11: S → X1 → Y, and
/// S → X3 ↔ X2 ← Y with X3 → X4.
fn figure_11_12() -> Admg {
    diagram(&[("X1", "Y"), ("Y", "X2"), ("X3", "X4")], &[("X3", "X2")])
}

#[test]
fn examples_11_10_and_11_11_maximal_sets() {
    let d = SelectionDiagram::new(figure_11_12())
        .with_source("1", &["X1", "X3"])
        .unwrap();
    let sets = transportable_feature_sets(&d, &["X1", "X2", "X3", "X4"], "Y").unwrap();
    assert!(sets.transportable.contains(&vec!["X1".into()]));
    assert!(sets.transportable.contains(&vec!["X1".into(), "X2".into()]));
    assert!(!sets.transportable.contains(&vec!["X1".into(), "X2".into(), "X3".into(), "X4".into()]));
    let mut maximal = sets.maximal.clone();
    maximal.sort();
    assert_eq!(maximal, vec![vec!["X1", "X2"], vec!["X1", "X3", "X4"]]);
    for s in &sets.transportable {
        assert!(s.contains(&"X1".to_string()), "{s:?} leaves S → X1 → Y open");
    }
}

/// Ex. 11.12 on Fig. 11.13, two overlaid copies of Fig. 11.12(a): four
/// maximal transportable sets, and the causal set {X1, X5} is transportable.
#[test]
fn example_11_12_four_maximal_sets() {
    let g = diagram(
        &[
            ("X1", "Y"),
            ("Y", "X2"),
            ("X3", "X4"),
            ("X5", "Y"),
            ("Y", "X6"),
            ("X7", "X8"),
        ],
        &[("X3", "X2"), ("X7", "X6")],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X1", "X3", "X5", "X7"])
        .unwrap();
    let features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"];
    let sets = transportable_feature_sets(&d, &features, "Y").unwrap();
    let mut maximal = sets.maximal.clone();
    maximal.sort();
    let expect: Vec<Vec<&str>> = vec![
        vec!["X1", "X2", "X5", "X6"],
        vec!["X1", "X2", "X5", "X7", "X8"],
        vec!["X1", "X3", "X4", "X5", "X6"],
        vec!["X1", "X3", "X4", "X5", "X7", "X8"],
    ];
    assert_eq!(maximal, expect);
    assert!(sets.transportable.contains(&vec!["X1".into(), "X5".into()]));
}

/// Ex. 11.8: covariate shift, with two sources differing on X only.
#[test]
fn example_11_8_structural_covariate_shift() {
    let g = diagram(&[("X1", "Y"), ("X2", "Y")], &[("X1", "X2")]);
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X1", "X2"])
        .unwrap()
        .with_source("2", &["X1", "X2"])
        .unwrap();
    assert!(structural_covariate_shift(&d, &["X1", "X2"], "Y").unwrap());
    assert!(!structural_covariate_shift(&d, &["X1"], "Y").unwrap());
    let s = score_tr(&d, &["X1", "X2"], "Y").unwrap();
    assert_eq!(s.estimand.to_string(), "P^1(y | x1, x2)");

    let shifted = d.clone().with_source("3", &["Y"]).unwrap();
    assert!(!structural_covariate_shift(&shifted, &["X1", "X2"], "Y").unwrap());
}

/// Exs. 11.31–11.32 (Figs. 11.32(a) and 11.31(b)).
#[test]
fn examples_11_31_and_11_32() {
    let g = diagram(&[("X", "Z"), ("Z", "R"), ("R", "Y")], &[]);
    let d = SelectionDiagram::new(g).with_source("1", &["R"]).unwrap();
    let e = statistical_transport(&d, &["X"], "Y", &["Z", "R"]).unwrap();
    assert_eq!(e.to_string(), "sum_{r, z} P^1(z | x) P^*(r | z) P^1(y | r)");
    // Ex. 11.30 says Thm 11.3.3 cannot see this, but Z = {Z, R} separates S
    // from X and Y: X → Z → R ← S is blocked at Z.
    let f = auxiliary_measured(&d, "1", &["X"], "Y", &["Z", "R"]).unwrap().unwrap();
    assert_eq!(f.z, ["Z", "R"]);
    let model = Model::from_diagram(&d, 5);
    assert_sound(&Exact::new(&model).measuring(&["Z", "R"]), &f.estimand, "Y", &["X"], "Ex. 11.30");

    let g = diagram(&[("X", "Y"), ("Z", "Y")], &[("X", "Z")]);
    let d = SelectionDiagram::new(g).with_source("1", &["X"]).unwrap();
    for w in [&[][..], &["Z"]] {
        assert!(matches!(
            statistical_transport(&d, &["X"], "Y", w),
            Err(TransportError::NotTransportable(_))
        ));
    }
    let e = statistical_transport(&d, &["X"], "Y", &["X", "Z"]).unwrap();
    assert_eq!(e.to_string(), "sum_{z} P^*(z | x) P^1(y | x, z)");
    let f = features_measured(&d, "1", &["X"], "Y", &["X", "Z"]).unwrap().unwrap();
    assert_eq!(f.estimand.to_string(), "sum_{z} P^1(y | x, z) P^*(z | x)");
}

/// Ex. 11.33 on Fig. 11.33(c), reconstructed from Eq. 11.239.
#[test]
fn example_11_33() {
    let g = diagram(
        &[("A", "Z"), ("A", "X1"), ("Z", "Y"), ("X1", "Y"), ("X2", "Y")],
        &[],
    );
    let d = SelectionDiagram::new(g.clone()).with_source("1", &["A"]).unwrap();
    let e = statistical_transport(&d, &["X1", "X2"], "Y", &["A"]).unwrap();
    assert_eq!(
        e.to_string(),
        "[sum_{a, z} P^*(a) P^1(z | a) P^1(x1 | a) P^1(y | z, x1, x2)] / [sum_{a} P^*(a) P^1(x1 | a)]"
    );
    let model = Model::from_diagram(&d, 7);
    let exact = Exact::new(&model).measuring(&["A"]);
    assert_sound(&exact, &e, "Y", &["X1", "X2"], "Ex. 11.33");
}

/// Fig. 11.34(a), reconstructed from Eqs. 11.241–11.244, with
/// `Δ¹ = {Z, A}` and `Δ² = {X2, Z, Y}`.
fn figure_11_34() -> SelectionDiagram {
    let g = diagram(
        &[
            ("X2", "Z"),
            ("Z", "R"),
            ("R", "X1"),
            ("X2", "X1"),
            ("X1", "Y"),
            ("A", "Y"),
        ],
        &[("X1", "A")],
    );
    SelectionDiagram::new(g)
        .with_source("1", &["Z", "A"])
        .unwrap()
        .with_source("2", &["X2", "Z", "Y"])
        .unwrap()
}

/// Ex. 11.34: transport needs both sources at once. Tian's factor for `A` is
/// `P²(a | x1, x2, r)`; the printed `P²(a | x1)` gives a wrong answer.
#[test]
fn example_11_34_needs_tians_conditioning_set() {
    let d = figure_11_34();
    let model = Model::from_diagram(&d, 11);
    let exact = Exact::new(&model);

    let e = statistical_transport(&d, &["X1", "X2"], "Y", &["Z", "X2"]).unwrap();
    assert_eq!(
        e.to_string(),
        "[sum_{a, r, z} P^*(z | x2) P^1(r | z) P^2(x1 | x2, r) P^2(a | x2, r, x1) P^1(y | x1, a)] \
         / [sum_{r, z} P^*(z | x2) P^1(r | z) P^2(x1 | x2, r)]"
    );
    assert_sound(&exact.measuring(&["Z", "X2"]), &e, "Y", &["X1", "X2"], "Ex. 11.34");

    // No single source suffices: each differs on a component the query needs.
    for s in ["1", "2"] {
        let mut single = SelectionDiagram::new(d.graph().clone());
        single = single
            .with_source(s, &as_refs(&d.discrepancies(s).unwrap().iter().cloned().collect::<Vec<_>>()))
            .unwrap();
        assert!(statistical_transport(&single, &["X1", "X2"], "Y", &["Z", "X2"]).is_err());
    }

    // The projected case, W = {R, X2}.
    let projected = statistical_transport(&d, &["X1", "X2"], "Y", &["R", "X2"]).unwrap();
    assert_eq!(projected.target_variables(), BTreeSet::from(["R".into(), "X2".into()]));
    assert_sound(&exact.measuring(&["R", "X2"]), &projected, "Y", &["X1", "X2"], "Ex. 11.34, W = {R, X2}");

    // Eq. 11.241 as printed.
    let p = |dom: &str, v: &[&str], g: &[&str]| {
        let domain = if dom == "*" { Domain::Target } else { Domain::source(dom) };
        Estimand::p(domain, v, g)
    };
    let prod = Estimand::Product(vec![
        p("1", &["Y"], &["X1", "A"]),
        p("2", &["A"], &["X1"]),
        p("2", &["X1"], &["X2", "R"]),
        p("1", &["R"], &["Z"]),
        p("*", &["Z"], &["X2"]),
    ]);
    let s = |over: &[&str]| Estimand::Sum {
        over: over.iter().map(|v| (*v).to_string()).collect(),
        body: Box::new(prod.clone()),
    };
    let printed = Estimand::Ratio {
        num: Box::new(s(&["A", "R", "Z"])),
        den: Box::new(s(&["Y", "A", "R", "Z"])),
    };
    // It cannot depend on x2: summing out r and z leaves
    // sum_a P¹(y | x1, a) P²(a | x1). The truth does, through A ↔ X1 ← X2.
    let measured = exact.measuring(&["Z", "X2"]);
    let at = |x2: &'static str| [("X1", "0"), ("X2", x2), ("Y", "1")];
    let printed0 = printed.evaluate(&measured, &at("0")).unwrap();
    let printed1 = printed.evaluate(&measured, &at("1")).unwrap();
    let truth0 = exact.truth("Y", 1, &[("X1", 0), ("X2", 0)]);
    let truth1 = exact.truth("Y", 1, &[("X1", 0), ("X2", 1)]);
    assert!((printed0 - printed1).abs() < 1e-12);
    assert!((truth0 - truth1).abs() > 1e-4, "{truth0} vs {truth1}");
    assert!((printed0 - truth0).abs() > 1e-5);
}

/// Ex. 11.20 and Thm 11.2.11: X1 and X3 are determined, X2 and X4
/// constrained through `R† = X2 / X4`, and the representation's score is
/// transportable through the augmented diagram although covariate shift
/// fails. The graph follows the remark after Eq. 11.127.
#[test]
fn example_11_20_transportable_representation() {
    fn ratio(n: i64, d: i64) -> (i64, i64) {
        fn gcd(a: i64, b: i64) -> i64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        let g = gcd(n, d);
        (n / g, d / g)
    }
    let phi = |x: &[i64]| {
        (
            ratio(x[0] * x[1] * x[2], x[3]),
            ratio(x[0] * x[1], x[2] * x[3]),
            ratio(x[1] * x[2], x[0] * x[3]),
        )
    };
    let names = ["X1", "X2", "X3", "X4"];
    let rep = Representation::classify(&names, &vec![vec![1, 2, 3]; 4], phi).unwrap();
    assert_eq!(rep.determined(), ["X1", "X3"]);
    assert_eq!(rep.constrained(), ["X2", "X4"]);
    assert!(rep.free().is_empty());

    let g = diagram(&[("X1", "Y"), ("X3", "Y"), ("Y", "X2")], &[("X4", "Y")]);
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X4"])
        .unwrap()
        .with_source("2", &["X2"])
        .unwrap();
    assert!(!structural_covariate_shift(&d, &names, "Y").unwrap());
    let aug = d.augment("R+", &rep.constrained()).unwrap();
    let mut z = rep.determined();
    z.push("R+");
    let s = score_tr(&aug, &z, "Y").unwrap();
    let mut doms: Vec<Domain> = s.components.iter().map(|(_, dom)| dom.clone()).collect();
    doms.sort();
    doms.dedup();
    assert_eq!(doms, [Domain::source("1"), Domain::source("2")]);
    // As in Eq. 11.127, it takes both sources: neither suffices alone.
    for (src, delta) in [("1", "X4"), ("2", "X2")] {
        let single = SelectionDiagram::new(d.graph().clone())
            .with_source(src, &[delta])
            .unwrap()
            .augment("R+", &rep.constrained())
            .unwrap();
        assert!(score_tr(&single, &z, "Y").is_err(), "source {src} alone");
    }

    // φ_sum of Ex. 11.18 constrains every feature and determines none.
    let rep = Representation::classify(&names, &vec![vec![0, 1]; 4], |x: &[i64]| x.iter().sum::<i64>())
        .unwrap();
    assert!(rep.determined().is_empty());
    assert_eq!(rep.constrained(), names);
    assert_eq!(rep.support_size(), 5);
}

#[test]
fn causal_mechanistic_stability() {
    assert!(cms_holds(&[&["X1", "X2"], &["X3"]], &[&["X1"], &["X3"]]));
    assert!(!cms_holds(&[&["X1"]], &[&["Y"]]));
    assert!(cms_holds(&[], &[]));
}

#[test]
fn chi2_survival_matches_tables() {
    use super::invariance::chi2_survival;
    assert!((chi2_survival(3.841_458_820_694_124, 1.0) - 0.05).abs() < 1e-9);
    assert!((chi2_survival(9.487_729_036_781_154, 4.0) - 0.05).abs() < 1e-9);
    assert!((chi2_survival(2.0, 2.0) - (-1.0f64).exp()).abs() < 1e-12);
}

/// The exact distribution of `(X1, X2, Y)` in a source where
/// `Y ← X1 ⊕ U_Y` is shared and `P(X1)`, `X2 ← Y ⊕ U_2` differ, as a
/// weighted table.
fn invariance_source(p_x1: f64, p_u2: f64) -> Dataset {
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    for x1 in 0..2 {
        for uy in 0..2 {
            for u2 in 0..2 {
                let y = x1 ^ uy;
                let x2 = y ^ u2;
                let p = if x1 == 1 { p_x1 } else { 1.0 - p_x1 }
                    * if uy == 1 { 0.2 } else { 0.8 }
                    * if u2 == 1 { p_u2 } else { 1.0 - p_u2 };
                rows.push(vec![x1.to_string(), x2.to_string(), y.to_string()]);
                weights.push(p * 10_000.0);
            }
        }
    }
    Dataset::from_records(["X1", "X2", "Y"], rows)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

/// Def. 11.2.11 on exact data: only `E[Y | X1]` is invariant across the two
/// sources, so it is the minimal invariant set of Thm 11.2.9.
#[test]
fn source_invariance_singles_out_the_parent() {
    let s1 = invariance_source(0.3, 0.1);
    let s2 = invariance_source(0.6, 0.4);
    let sets = invariant_feature_sets(&[&s1, &s2], &["X1", "X2"], "Y", 0.05).unwrap();
    let names: Vec<Vec<String>> = sets.iter().map(|(s, _)| s.clone()).collect();
    assert_eq!(names, vec![vec!["X1".to_string()]]);
    let r = &sets[0].1;
    assert!(r.max_gap < 1e-12);
    let r = source_invariance(&[&s1, &s2], &["X1", "X2"], "Y").unwrap();
    assert!(r.max_gap > 0.1 && r.p_value < 1e-6 && r.df == 4);
}

fn sample(header: &[&str], n: usize, seed: u64, mut row: impl FnMut(&mut SplitMix64) -> Vec<i64>) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let rows: Vec<Vec<String>> = (0..n)
        .map(|_| row(&mut rng).into_iter().map(|v| v.to_string()).collect())
        .collect();
    Dataset::from_records(header.iter().copied(), rows).unwrap()
}

/// Exs. 11.35–11.37 with a ten-valued `X`: one source shares the target's
/// mechanism for Y, the other has `Y ⊥ X`. Algorithm 34 pools the first;
/// Algorithm 35 finds it from held-out target data, which is too small to
/// fit twenty cells well alone.
#[test]
fn algorithms_34_and_35_find_the_matching_source() {
    let target_y = |x: i64, rng: &mut SplitMix64| i64::from(x >= 5) ^ i64::from(rng.unit() > 0.9);
    let target = sample(&["X", "Y"], 120, 1, |r| {
        let x = r.below(10) as i64;
        vec![x, target_y(x, r)]
    });
    let same = sample(&["X", "Y"], 5000, 2, |r| {
        let x = (r.below(10) as i64).min(r.below(10) as i64);
        vec![x, target_y(x, r)]
    });
    let other = sample(&["X", "Y"], 5000, 3, |r| {
        vec![r.below(10) as i64, i64::from(r.unit() > 0.5)]
    });
    let sources = [&same, &other];

    let (model, j) =
        structure_informed_da(&sources, &target, &[&[], &["Y"]], &["X"], "Y", 0.5).unwrap();
    assert_eq!(j, [0]);
    for x in 0..10 {
        let want = if x >= 5 { 0.9 } else { 0.1 };
        let got = model.probability(&[&x.to_string()], "1");
        assert!((got - want).abs() < 0.08, "x = {x}: {got}");
    }

    let choice = structure_agnostic_da(&sources, &target, &["X"], "Y", 0.5, 9).unwrap();
    assert_eq!(choice.sources, [0]);
    assert_eq!(choice.losses.len(), 4);
}

/// Exs. 11.38–11.39: the parents of Y differ (X1, X2 in the source, X3, X2 in
/// the target) but the noisy subtraction is shared.
#[test]
fn algorithms_36_and_37_transport_a_modular_mechanism() {
    let noise = |r: &mut SplitMix64| {
        if r.unit() < 0.91 { 0 } else { 1 + r.below(9) as i64 }
    };
    let source = sample(&["X1", "X2", "X3", "Y"], 20_000, 4, |r| {
        let x: Vec<i64> = (0..3).map(|_| r.below(10) as i64).collect();
        let y = (x[0] - x[1] + noise(r)).rem_euclid(10);
        vec![x[0], x[1], x[2], y]
    });
    let target = sample(&["X1", "X2", "X3", "Y"], 400, 5, |r| {
        let x: Vec<i64> = (0..3).map(|_| r.below(10) as i64).collect();
        let y = (x[2] - x[1] + noise(r)).rem_euclid(10);
        vec![x[0], x[1], x[2], y]
    });

    let (model, j) = multi_cause_da(
        &[&source],
        &target,
        &[&[]],
        &[&["X1", "X2"]],
        &["X3", "X2"],
        "Y",
        0.5,
    )
    .unwrap();
    assert_eq!(j, [0]);
    for a in 0..10i64 {
        for b in 0..10i64 {
            let (sa, sb) = (a.to_string(), b.to_string());
            let want = (a - b).rem_euclid(10).to_string();
            assert_eq!(model.predict(&[&sa, &sb]), want);
        }
    }

    let choice = multi_cause_agnostic_da(&[&source], &target, &["X1", "X2", "X3"], "Y", 0.5, 3).unwrap();
    assert_eq!(choice.candidates, 98);
    // Reading both lists in reverse order is the same model, so either wins.
    let pair = (
        choice.target_parents.join(","),
        choice.source_parents.as_ref().map(|p| p.join(",")),
    );
    assert!(
        pair == ("X3,X2".into(), Some("X1,X2".into())) || pair == ("X2,X3".into(), Some("X2,X1".into())),
        "selected {pair:?}"
    );
    assert_eq!(choice.sources, [0]);
}

/// Algorithm 38 on a sequential instance in the spirit of Ex. 11.40 (whose
/// equations disagree with each other): unary doubling `g1` and binary
/// addition `g2`, mod 10. The source runs `V2 = g1(V1)`, `V3 = g2(V1, V2)`,
/// `V4 = g1(V3)`; the target adds `V5 = g2(V4, V2)`, which no source
/// position computes from those inputs but whose mechanism one source
/// position shares. Without any target data the predictor recovers
/// `V5 = 8·V1 mod 10`.
#[test]
fn algorithm_38_composes_modules_zero_shot() {
    let noise = |r: &mut SplitMix64| {
        if r.unit() < 0.91 { 0 } else { 1 + r.below(9) as i64 }
    };
    let src = sample(&["V1", "V2", "V3", "V4"], 20_000, 6, |r| {
        let v1 = r.below(10) as i64;
        let v2 = (2 * v1 + noise(r)).rem_euclid(10);
        let v3 = (v1 + v2 + noise(r)).rem_euclid(10);
        let v4 = (2 * v3 + noise(r)).rem_euclid(10);
        vec![v1, v2, v3, v4]
    });
    let empty = Dataset::from_records(["V1", "V2", "V3", "V4", "V5"], Vec::<Vec<String>>::new()).unwrap();
    let names = |k: usize| (1..=k).map(|i| format!("V{i}")).collect::<Vec<_>>();
    let labels = |l: &[&str]| l.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
    let source = SequenceDomain {
        data: &src,
        positions: names(4),
        parents: vec![vec![], vec![0], vec![0, 1], vec![2]],
        mechanisms: labels(&["root", "g1", "g2", "g1"]),
    };
    let target = SequenceDomain {
        data: &empty,
        positions: names(5),
        parents: vec![vec![], vec![0], vec![1, 0], vec![2], vec![3, 1]],
        mechanisms: labels(&["root", "g1", "g2", "g1", "g2"]),
    };
    let pred = sequential_da(&[source], &target, 1, 0.01).unwrap();
    assert_eq!(pred.pooled(3), [(Some(0), 1), (Some(0), 3), (None, 3)]);
    assert_eq!(pred.pooled(4), [(Some(0), 2), (None, 4)]);
    for v1 in 0..10i64 {
        let s = v1.to_string();
        let dist = pred.distribution(&[&s]).unwrap();
        let k = pred.alphabet().iter().position(|a| *a == (8 * v1).rem_euclid(10).to_string()).unwrap();
        let best = dist.iter().copied().fold(0.0, f64::max);
        assert!((dist[k] - best).abs() < 1e-12 && dist[k] > 0.5, "v1 = {v1}: {dist:?}");
        assert!((dist.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn empirical_plug_in_matches_the_formula() {
    // Ex. 11.31 again, on exact weighted tables: the plug-in estimate of the
    // transport formula equals the target's conditional.
    let g = diagram(&[("X", "Z"), ("Z", "R"), ("R", "Y")], &[]);
    let d = SelectionDiagram::new(g).with_source("1", &["R"]).unwrap();
    let model = Model::from_diagram(&d, 3);
    let exact = Exact::new(&model);
    let table = |dom: &Domain, keep: &[&str]| {
        let t = &exact.tables[dom];
        let idx: Vec<usize> = keep.iter().map(|v| model.index(v)).collect();
        let mut rows = Vec::new();
        let mut w = Vec::new();
        for (mask, p) in t.iter().enumerate() {
            rows.push(idx.iter().map(|i| ((mask >> i) & 1).to_string()).collect::<Vec<_>>());
            w.push(*p);
        }
        Dataset::from_records(keep.iter().copied(), rows).unwrap().with_weights(w).unwrap()
    };
    let src = table(&Domain::source("1"), &["X", "Z", "R", "Y"]);
    let tgt = table(&Domain::Target, &["Z", "R"]);
    let dists = EmpiricalDistributions::new()
        .with(Domain::source("1"), &src)
        .with(Domain::Target, &tgt);
    let e = statistical_transport(&d, &["X"], "Y", &["Z", "R"]).unwrap();
    for x in 0..2u64 {
        let xs = x.to_string();
        let got = e.evaluate(&dists, &[("X", &xs), ("Y", "1")]).unwrap();
        let truth = exact.truth("Y", 1, &[("X", x)]);
        assert!((got - truth).abs() < 1e-9);
    }
    // P*(Y | X) itself needs unmeasured target columns.
    let direct = Estimand::p(Domain::Target, &["Y"], &["X"]);
    assert!(matches!(
        direct.evaluate(&dists, &[("X", "1"), ("Y", "1")]),
        Err(TransportError::MissingData(_))
    ));
}

#[test]
fn errors_are_reported() {
    let g = diagram(&[("X", "Y")], &[]);
    let d = SelectionDiagram::new(g.clone()).with_source("1", &["X"]).unwrap();
    assert!(matches!(d.clone().with_source("1", &[]), Err(TransportError::DuplicateSource(_))));
    assert!(matches!(d.clone().with_source("*", &[]), Err(TransportError::DuplicateSource(_))));
    assert!(matches!(
        SelectionDiagram::new(g).with_source("1", &["Q"]),
        Err(TransportError::Unknown(_))
    ));
    assert!(matches!(score_tr(&d, &["Y"], "Y"), Err(TransportError::InvalidInput(_))));
    assert!(matches!(list_dtr_dist(&d, "2", "Y", &["X"]), Err(TransportError::Unknown(_))));
    assert!(matches!(d.augment("X", &[]), Err(TransportError::InvalidInput(_))));
}

/// Another source's S-node is not a common cause of its children: with
/// `Δ¹ = {A}` and `Δ² = {A, B}`, conditioning on `A` must not open
/// `S_1 → A ← S_2 → B → Y`. `P*(y | a) = P¹(y | a)` holds, and does so
/// numerically.
#[test]
fn other_sources_s_nodes_do_not_connect_their_children() {
    let mut g = Admg::new();
    g.directed("A", "Y");
    g.directed("B", "Y");
    let d = SelectionDiagram::new(g)
        .with_source("1", &["A"])
        .unwrap()
        .with_source("2", &["A", "B"])
        .unwrap();
    assert!(d.is_s_admissible("1", &["Y"], &["A"]).unwrap());
    assert!(is_directly_transportable(&d, "1", "Y", &["A"]).unwrap());
    assert!(list_dtr_dist(&d, "1", "Y", &["A", "B"]).unwrap().contains(&vec!["A".to_string()]));
    let model = Model::from_diagram(&d, 17);
    let exact = Exact::new(&model);
    let e = Estimand::p(Domain::source("1"), &["Y"], &["A"]);
    assert_sound(&exact, &e, "Y", &["A"], "P*(y | a) from π¹");
}
