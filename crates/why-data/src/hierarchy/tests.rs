//! Chapter 20: the book's examples, and properties on random graphs.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::graph::backdoor;
use crate::graph::dseparation::Admg;

fn graph(directed: &[(&str, &str)], bidirected: &[(&str, &str)]) -> Admg {
    let mut g = Admg::new();
    for (a, b) in directed {
        g.directed(a, b);
    }
    for (a, b) in bidirected {
        g.bidirected(a, b);
    }
    g
}

fn t(var: &str, sub: &[(&str, u8)]) -> Term {
    Term::new(var, sub)
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Defs. 20.1.11-20.1.12 and Lemma 20.1.4 on Exs. 20.6, 20.12-20.14,
/// 20.17 and 20.30.
#[test]
fn layers_of_the_book_examples() {
    let fork = graph(&[("X", "Z"), ("X", "Y")], &[]);
    let x1 = [("X", 1)];
    let x0 = [("X", 0)];
    // Ex. 20.6.
    let q = [t("X", &[]), t("Y", &x1), t("Z", &x1)];
    assert_eq!(layer(&fork, &q).unwrap(), Layer::L225);
    let q = [t("X", &[]), t("Y", &x1), t("Z", &x0)];
    assert_eq!(layer(&fork, &q).unwrap(), Layer::L25);
    assert_eq!(
        layer(&fork, &[t("Y", &x1), t("Y", &[])]).unwrap(),
        Layer::L3
    );
    // Exs. 20.12-20.13.
    assert_eq!(
        layer(&fork, &[t("Z", &x1), t("Y", &x0)]).unwrap(),
        Layer::L25
    );
    // L1 and L2.
    assert_eq!(
        layer(&fork, &[t("Y", &[]), t("Z", &[])]).unwrap(),
        Layer::L1
    );
    assert_eq!(
        layer(&fork, &[t("Y", &x1), t("Z", &x1)]).unwrap(),
        Layer::L2
    );
    // A subscript outside the ancestors is dropped: P(X_y) = P(X).
    assert_eq!(layer(&fork, &[t("X", &[("Y", 1)])]).unwrap(), Layer::L1);
    // Ex. 20.14: the ETT in Fig. 20.8(a) (Z -> X -> Y).
    let g = graph(&[("Z", "X"), ("X", "Y")], &[]);
    assert_eq!(layer(&g, &[t("Y", &x1), t("X", &[])]).unwrap(), Layer::L225);
    // Ex. 20.30: the chain X -> Z -> Y, by Lemma 20.1.4 and by Prop. 20.3.4.
    let chain = graph(&[("X", "Z"), ("Z", "Y")], &[]);
    let cases: [(&[Term], bool); 4] = [
        (&[t("Z", &x1), t("Y", &x0)], false),
        (&[t("Y", &x1), t("Y", &[("Z", 1)])], false),
        (&[t("Z", &[]), t("Y", &x1)], false),
        (&[t("Z", &[]), t("Y", &[("Z", 1)])], true),
    ];
    for (q, inside) in cases {
        assert_eq!(layer(&chain, q).unwrap() <= Layer::L225, inside, "{q:?}");
        let mut all = BTreeMap::new();
        let mut ok = true;
        for term in q {
            for (k, v) in &term.sub {
                ok &= *all.entry(k.as_str()).or_insert(*v) == *v;
            }
        }
        let x: Vec<(&str, u8)> = all.into_iter().collect();
        let s = swig(&chain, &x).unwrap();
        assert_eq!(ok && s.holds(q, &chain), inside, "{q:?}");
    }
    // The chain's P(Z_x, Y_z) is in L2.25 once the reachability sets merge.
    let crs = reachability_set(&chain, &[t("Z", &x1), t("Y", &[("Z", 0)])]).unwrap();
    let shown: Vec<String> = crs.iter().map(ToString::to_string).collect();
    assert_eq!(shown, ["X", "Y_{Z=0}", "Z_{X=1}"]);
}

/// Algorithms 68-69 on Exs. 20.19-20.20, Fig. 20.14, Experiment 1 and the
/// gap in COMPATIBLE.
#[test]
fn realizability_of_the_book_examples() {
    let xy = graph(&[("X", "Y")], &[]);
    let ett = [t("Y", &[("X", 1)]), t("X", &[])];
    let plan = ctf_realize(&xy, &ett, &maximal_actions(&xy).unwrap()).unwrap();
    assert_eq!(plan.interventions().len(), 1);
    assert!(ctf_realize(&xy, &ett, &fisher_actions(&xy).unwrap()).is_err());
    let ps = [t("Y", &[("X", 1)]), t("X", &[]), t("Y", &[])];
    assert!(ctf_realize(&xy, &ps, &maximal_actions(&xy).unwrap()).is_err());
    // Fig. 20.14: P(Z_x, W_t).
    let q = [t("Z", &[("X", 1)]), t("W", &[("T", 1)])];
    let g1 = graph(&[("T", "A"), ("A", "Z"), ("A", "W"), ("X", "Z")], &[]);
    let g2 = graph(&[("T", "Z"), ("T", "W"), ("X", "Z")], &[]);
    assert!(ctf_realize(&g1, &q, &maximal_actions(&g1).unwrap()).is_err());
    let shown: Vec<String> = ancestor_set(&g1, &q)
        .unwrap()
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(shown, ["A", "A_{T=1}", "T", "W_{T=1}", "Z_{X=1}"]);
    assert!(ctf_realize(&g2, &q, &maximal_actions(&g2).unwrap()).is_ok());
    // Experiment 1: P(Y_x, Z_x') with Y <-> Z needs per-child randomization.
    let e1 = graph(&[("X", "Y"), ("X", "Z")], &[("Y", "Z")]);
    let q = [t("Y", &[("X", 1)]), t("Z", &[("X", 0)])];
    assert!(ctf_realize(&e1, &q, &maximal_actions(&e1).unwrap()).is_ok());
    assert!(ctf_realize(&e1, &q, &layer225_actions(&e1).unwrap()).is_err());
    // Cor. 20.2.7: P(Y_x, Y_x') is never realizable.
    let q = [t("Y", &[("X", 1)]), t("Y", &[("X", 0)])];
    assert!(ctf_realize(&e1, &q, &maximal_actions(&e1).unwrap()).is_err());
    // The gap in Algorithm 69: RAND(X) alone cannot give P(Y_x, Z).
    let q = [t("Y", &[("X", 1)]), t("Z", &[])];
    assert!(ctf_realize(&e1, &q, &fisher_actions(&e1).unwrap()).is_err());
    assert!(ctf_realize(&e1, &q, &maximal_actions(&e1).unwrap()).is_ok());
}

const NAMES: [&str; 6] = ["A", "B", "C", "D", "E", "F"];

fn random_graph(seed: u64, n: usize, bidirected: bool) -> Admg {
    let mut g = Admg::new();
    for v in &NAMES[..n] {
        g.add_node(v);
    }
    let mut r = seed;
    for (i, a) in NAMES[..n].iter().enumerate() {
        for b in &NAMES[i + 1..n] {
            r = splitmix(r);
            if r % 100 < 40 {
                g.directed(a, b);
            } else if bidirected && r % 100 < 50 {
                g.bidirected(a, b);
            }
        }
    }
    g
}

fn random_query(seed: u64, n: usize) -> Vec<Term> {
    let mut r = splitmix(seed ^ 0x51);
    let mut next = || {
        r = splitmix(r);
        r
    };
    let len = 1 + (next() % 3) as usize;
    let max_sub = 1 + (next() % 3);
    (0..len)
        .map(|_| {
            let vi = (next() % n as u64) as usize;
            let var = NAMES[vi];
            let k = (next() % (max_sub + 1)) as usize;
            let mut sub = Vec::new();
            for _ in 0..k {
                // Mostly earlier variables, which can be ancestors.
                let s = if vi > 0 && next() % 4 != 0 {
                    NAMES[(next() % vi as u64) as usize]
                } else {
                    NAMES[(next() % n as u64) as usize]
                };
                if s != var && !sub.iter().any(|(w, _): &(&str, u8)| *w == s) {
                    sub.push((s, (next() % 2) as u8));
                }
            }
            Term::new(var, &sub)
        })
        .collect()
}

/// Cor. 20.2.6, Thm 20.1.6 and the regimes behind each layer: on random
/// graphs and queries, CTF-REALIZE succeeds with
/// * no action exactly on L1;
/// * `RAND` of every variable exactly on L2;
/// * one-value counterfactual randomization exactly on L2.25;
/// * per-child counterfactual randomization (`A†`) exactly on L2.5, which is
///   Cor. 20.2.6's ancestor criterion.
///
/// Prop. 20.3.4's SWIG test implies L2.25.
#[test]
fn layers_match_realizability_on_random_graphs() {
    let mut counts = BTreeMap::new();
    for seed in 0..3000u64 {
        let n = 3 + (seed % 4) as usize;
        let g = random_graph(splitmix(seed), n, seed % 2 == 0);
        let q = random_query(seed, n);
        let l = layer(&g, &q).unwrap();
        *counts.entry(l).or_insert(0) += 1;
        let ok = |a: Vec<Action>| ctf_realize(&g, &q, &a).is_ok();
        assert_eq!(ok(Vec::new()), l == Layer::L1, "seed {seed}: {q:?}");
        assert_eq!(
            ok(fisher_actions(&g).unwrap()),
            l <= Layer::L2,
            "seed {seed}: {q:?}"
        );
        assert_eq!(
            ok(layer225_actions(&g).unwrap()),
            l <= Layer::L225,
            "seed {seed}: {q:?}"
        );
        assert_eq!(
            ok(maximal_actions(&g).unwrap()),
            l <= Layer::L25,
            "seed {seed}: {q:?}"
        );
        let an = ancestor_set(&g, &q).unwrap();
        let repeated = an
            .iter()
            .any(|a| an.iter().any(|b| a.var == b.var && a.sub != b.sub));
        assert_eq!(!repeated, l <= Layer::L25, "seed {seed}");
        let mut x = BTreeMap::new();
        let mut consistent = true;
        for term in &q {
            for (k, v) in &term.sub {
                consistent &= *x.entry(k.as_str()).or_insert(*v) == *v;
            }
        }
        if consistent {
            let x: Vec<(&str, u8)> = x.into_iter().collect();
            if swig(&g, &x).unwrap().holds(&q, &g) {
                assert!(l <= Layer::L225, "seed {seed}: {q:?}");
            }
        }
    }
    for l in [Layer::L1, Layer::L2, Layer::L225, Layer::L25, Layer::L3] {
        assert!(counts.get(&l).copied().unwrap_or(0) > 60, "{counts:?}");
    }
}

/// A random binary SCM: each variable reads its parents, two private
/// exogenous bits and one bit per bidirected edge it is on.
struct RandomScm {
    names: Vec<String>,
    order: Vec<usize>,
    parents: Vec<Vec<usize>>,
    /// Per variable: the indices of the exogenous bits it reads.
    bits: Vec<Vec<usize>>,
    nbits: usize,
    key: u64,
}

impl RandomScm {
    fn new(g: &Admg, key: u64) -> Self {
        let names: Vec<String> = g.topological_order().unwrap();
        let idx = |v: &str| names.iter().position(|w| w == v).unwrap();
        let n = names.len();
        let mut parents = vec![Vec::new(); n];
        let mut bits: Vec<Vec<usize>> = (0..n).map(|i| vec![2 * i, 2 * i + 1]).collect();
        let mut nbits = 2 * n;
        let graph = g.graph();
        for e in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(e).unwrap();
            let (a, b) = (idx(&graph[a]), idx(&graph[b]));
            match graph[e] {
                crate::graph::dseparation::EdgeKind::Directed => parents[b].push(a),
                crate::graph::dseparation::EdgeKind::Bidirected => {
                    bits[a].push(nbits);
                    bits[b].push(nbits);
                    nbits += 1;
                }
            }
        }
        Self {
            names,
            order: (0..n).collect(),
            parents,
            bits,
            nbits,
            key,
        }
    }

    fn f(&self, v: usize, pa: &[u8], u: u64) -> u8 {
        let mut h = self.key ^ (v as u64) << 40;
        for (i, p) in pa.iter().enumerate() {
            h = splitmix(h ^ u64::from(*p) << i);
        }
        for &b in &self.bits[v] {
            h = splitmix(h ^ ((u >> b) & 1) << 20);
        }
        (splitmix(h) & 1) as u8
    }

    /// `W_t(u)`.
    fn response(&self, term: &Term, u: u64) -> u8 {
        let mut val = vec![0u8; self.names.len()];
        for &v in &self.order {
            val[v] = match term.sub.get(&self.names[v]) {
                Some(x) => *x,
                None => {
                    let pa: Vec<u8> = self.parents[v].iter().map(|&p| val[p]).collect();
                    self.f(v, &pa, u)
                }
            };
        }
        val[self.names.iter().position(|w| *w == term.var).unwrap()]
    }

    /// The unit's mechanisms, by name.
    fn mechanism(&self, u: u64) -> impl Fn(&str, &BTreeMap<String, u8>) -> u8 + '_ {
        move |v, pa| {
            let i = self.names.iter().position(|w| w == v).unwrap();
            let vals: Vec<u8> = self.parents[i]
                .iter()
                .map(|&p| pa[&self.names[p]])
                .collect();
            self.f(i, &vals, u)
        }
    }
}

/// Thm 20.2.5: every plan CTF-REALIZE returns gives each unit exactly its
/// potential responses `(W_t(u))_t`, so the samples follow `P(W*)` in every
/// SCM compatible with the graph.
#[test]
fn plans_read_the_potential_responses() {
    let mut checked = 0;
    for seed in 0..600u64 {
        let n = 3 + (seed % 4) as usize;
        let g = random_graph(splitmix(seed ^ 0x77), n, true);
        let q = random_query(seed ^ 0x99, n);
        let scm = RandomScm::new(&g, splitmix(seed));
        for actions in [
            fisher_actions(&g).unwrap(),
            layer225_actions(&g).unwrap(),
            maximal_actions(&g).unwrap(),
        ] {
            let Ok(plan) = ctf_realize(&g, &q, &actions) else {
                continue;
            };
            checked += 1;
            // 256 units with random exogenous bits.
            for k in 0..256u64 {
                let u = splitmix(seed << 8 ^ k) & ((1u64 << scm.nbits) - 1);
                let got = plan.sample(scm.mechanism(u));
                let want: Vec<u8> = q.iter().map(|term| scm.response(term, u)).collect();
                assert_eq!(got, want, "seed {seed}: {q:?} with {actions:?}");
            }
        }
    }
    assert!(checked > 600, "{checked}");
}

/// Def. 20.3.9 on random graphs: with factual covariates, the
/// counterfactual adjustment criterion agrees with the back-door
/// criterion.
#[test]
fn counterfactual_adjustment_is_the_back_door() {
    let mut agree = [0, 0];
    for seed in 0..800u64 {
        let n = 4 + (seed % 3) as usize;
        let g = random_graph(splitmix(seed ^ 0x3), n, true);
        let mut r = splitmix(seed);
        let x = NAMES[(r % n as u64) as usize];
        r = splitmix(r);
        let y = NAMES[(r % n as u64) as usize];
        if x == y {
            continue;
        }
        let others: Vec<&str> = NAMES[..n]
            .iter()
            .copied()
            .filter(|v| *v != x && *v != y)
            .collect();
        let s = swig(&g, &[(x, 1)]).unwrap();
        for mask in 0..1u32 << others.len() {
            let l: Vec<&str> = others
                .iter()
                .enumerate()
                .filter(|(i, _)| mask >> i & 1 == 1)
                .map(|(_, v)| *v)
                .collect();
            let a = s.adjusts(y, &l).unwrap();
            assert_eq!(
                a,
                backdoor::satisfies(&g, &[x], &[y], &l).unwrap(),
                "seed {seed}: {x} {y} {l:?}"
            );
            agree[usize::from(a)] += 1;
        }
    }
    assert!(agree[0] > 100 && agree[1] > 100, "{agree:?}");
}

/// A model with the averages of Table 20.5 (Experiment 3). Units with
/// `x' = 0` include a 10% subgroup that `D_{x''=1}` singles out and that
/// only `x = 0` rewards.
fn experiment3() -> MabScm {
    let u = |p, x, d: [u8; 2], y: [u8; 2]| MabUnit { p, z: 0, x, d, y };
    MabScm::new(vec![
        u(0.05, 0, [0, 1], [1, 0]),
        u(0.10, 0, [0, 0], [0, 1]),
        u(0.25, 0, [0, 0], [1, 1]),
        u(0.10, 0, [0, 0], [0, 0]),
        u(0.25, 1, [0, 0], [1, 1]),
        u(0.15, 1, [0, 0], [1, 0]),
        u(0.10, 1, [0, 0], [0, 1]),
    ])
    .unwrap()
}

/// Thm 20.2.8 and Cor. 20.2.9: the strategy values are ordered on random
/// models; Experiment 3's are 0.65, 0.70, 0.75 and 0.80; and MAB-OPT
/// learns past the best L2 strategy.
#[test]
fn counterfactual_strategies_dominate() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    let s = experiment3().strategies();
    assert!(
        close(s.obs, 0.65) && close(s.int, 0.70) && close(s.ett, 0.75) && close(s.opt, 0.80),
        "{s:?}"
    );
    assert_eq!(experiment3().opt_probe(0, 0), 1);
    for seed in 0..300u64 {
        let mut r = splitmix(seed);
        let mut next = || {
            r = splitmix(r);
            r
        };
        let mut units = Vec::new();
        for _ in 0..8 {
            let b = next();
            units.push(MabUnit {
                p: (1 + next() % 10) as f64,
                z: (b & 1) as u8,
                x: (b >> 1 & 1) as u8,
                d: [(b >> 2 & 1) as u8, (b >> 3 & 1) as u8],
                y: [(b >> 4 & 1) as u8, (b >> 5 & 1) as u8],
            });
        }
        let total: f64 = units.iter().map(|u| u.p).sum();
        for u in &mut units {
            u.p /= total;
        }
        let s = MabScm::new(units).unwrap().strategies();
        assert!(
            s.opt >= s.ett - 1e-12 && s.ett >= s.int - 1e-12 && s.ett >= s.obs - 1e-12,
            "{s:?}"
        );
    }
    let (mut opt, mut int) = (0.0, 0.0);
    for seed in 0..10 {
        opt += mab_opt(&experiment3(), 6000, seed).mean(3000, 6000) / 10.0;
        int += mab_int(&experiment3(), 6000, seed).mean(3000, 6000) / 10.0;
    }
    assert!(opt > 0.77 && int < 0.72, "{opt} {int}");
}

/// Algorithm 72 on Ex. 20.43: two augmented DAGs fit, so `Z – W` stays
/// unoriented.
#[test]
fn ag_construct_leaves_ambiguous_edges() {
    let ecis = [
        Eci::new(&["Z"], &["FX"], &[]),
        Eci::new(&["W"], &["X"], &["Z", "FX"]),
        Eci::new(&["Y"], &["Z"], &["X", "W", "FX"]),
        Eci::new(&["Y"], &["FX"], &["X", "W"]),
        Eci::new(&["W"], &["FX"], &["Z"]),
    ];
    let g = ag_construct(&["X", "W", "Y", "Z"], &[], &[("FX", "X")], &ecis).unwrap();
    assert_eq!(
        g.edges(),
        ["FX --> X", "W --> Y", "W o-o Z", "X --> Y", "X <-- Z"]
    );
    let fx: BTreeSet<_> = g.edges().into_iter().filter(|e| e.contains("FX")).collect();
    assert_eq!(fx.len(), 1);
}
