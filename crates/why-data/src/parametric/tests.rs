//! Tests for chapter 19: the book's linear examples, random linear SCMs
//! checking every criterion against the true coefficients, the fairness SCM
//! of Sec. 19.7.6.2, and random monotone SCMs checking M-ID against the
//! true counterfactuals.

use std::collections::BTreeMap;

use super::*;
use crate::fairness::Dataset;
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

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Random coefficients in ±[0.3, 1.2], unit-ish variances, and covariances
/// small enough to keep Ω positive definite.
fn random_scm(g: &Admg, seed: u64) -> LinearScm {
    let mut r = seed;
    let mut next = || {
        r = splitmix(r);
        (r >> 11) as f64 / (1u64 << 53) as f64
    };
    let (d, b) = linear::edges(g);
    let coef: Vec<(String, String, f64)> = d
        .iter()
        .map(|(x, y)| {
            let m = 0.3 + 0.9 * next();
            (x.clone(), y.clone(), if next() < 0.5 { m } else { -m })
        })
        .collect();
    let cov: Vec<(String, String, f64)> = b
        .iter()
        .map(|(x, y)| (x.clone(), y.clone(), 0.4 * next() - 0.2))
        .collect();
    // Diagonal dominance keeps Ω positive definite.
    let vars: Vec<(String, f64)> = g
        .variables()
        .iter()
        .map(|v| (v.to_string(), 1.0 + 0.5 * next()))
        .collect();
    let c: Vec<(&str, &str, f64)> = coef
        .iter()
        .map(|(a, b, l)| (a.as_str(), b.as_str(), *l))
        .collect();
    let o: Vec<(&str, &str, f64)> = cov
        .iter()
        .map(|(a, b, l)| (a.as_str(), b.as_str(), *l))
        .collect();
    let v: Vec<(&str, f64)> = vars.iter().map(|(a, l)| (a.as_str(), *l)).collect();
    LinearScm::new(g, &c, &v, &o).unwrap()
}

#[test]
fn covariance_and_the_trek_rule() {
    // Ex. 19.4: the bow, σ_xy = λ_xy + ε_xy with unit variances.
    let g = graph(&[("X", "Y")], &[("X", "Y")]);
    let m = LinearScm::new(&g, &[("X", "Y", 0.7)], &[], &[("X", "Y", 0.2)]).unwrap();
    assert!(close(m.covariance().get("X", "Y").unwrap(), 0.7 + 0.2));
    // Ex. 19.5: the chain with X <-> Z, σ_xy = λ_zy (λ_xz + ε_xz).
    let g = graph(&[("X", "Z"), ("Z", "Y")], &[("X", "Z")]);
    let m = LinearScm::new(
        &g,
        &[("X", "Z", 0.5), ("Z", "Y", 0.8)],
        &[],
        &[("X", "Z", 0.3)],
    )
    .unwrap();
    assert!(close(
        m.covariance().get("X", "Y").unwrap(),
        0.8 * (0.5 + 0.3)
    ));
    // Ex. 19.7 (Fig. 19.8), and random graphs: the trek rule matches Σ.
    let g = graph(
        &[("W", "X"), ("W", "V"), ("V", "Y"), ("X", "Z"), ("Z", "Y")],
        &[("X", "Z")],
    );
    for seed in 0..20u64 {
        let m = random_scm(&g, seed);
        let s = m.covariance();
        for a in m.names() {
            for b in m.names() {
                assert!(
                    close(s.get(a, b).unwrap(), m.trek_covariance(a, b).unwrap()),
                    "{a}{b}"
                );
            }
        }
    }
}

#[test]
fn single_door_and_back_door() {
    // Fig. 19.13 (Ex. 19.16): W -> X -> Y, W -> Y, X <-> T <-> Y. Only {W}
    // works; conditioning on T would open X <-> T <-> Y.
    let g = graph(
        &[("W", "X"), ("W", "Y"), ("X", "Y")],
        &[("X", "T"), ("T", "Y")],
    );
    assert_eq!(
        single_door(&g, "X", "Y").unwrap(),
        Some(vec!["W".to_string()])
    );
    let m = random_scm(&g, 3);
    let beta = m.covariance().regression("Y", "X", &["W"]).unwrap();
    assert!(close(beta, m.coefficient("X", "Y").unwrap()));

    // Fig. 19.15 (Ex. 19.17): the back door through S identifies
    // δ_xy = λ_xw λ_wy + λ_xy, though W <-> Y blocks the single door.
    let g = graph(
        &[("S", "X"), ("S", "Y"), ("X", "W"), ("W", "Y"), ("X", "Y")],
        &[("S", "X"), ("W", "Y")],
    );
    assert_eq!(
        back_door(&g, "X", "Y").unwrap(),
        Some(vec!["S".to_string()])
    );
    let m = random_scm(&g, 5);
    let beta = m.covariance().regression("Y", "X", &["S"]).unwrap();
    assert!(close(beta, m.total_effect("X", "Y").unwrap()));
}

#[test]
fn instrumental_variables_and_sets() {
    // Ex. 19.19: a conditional instrument.
    let g = graph(
        &[("W", "Z"), ("Z", "X"), ("X", "Y")],
        &[("X", "Y"), ("W", "Y")],
    );
    let m = random_scm(&g, 7);
    let s = m.covariance();
    let iv = s.regression("Y", "Z", &["W"]).unwrap() / s.regression("X", "Z", &["W"]).unwrap();
    assert!(close(iv, m.coefficient("X", "Y").unwrap()));
    // Ex. 19.20: the instrumental set {Z1, Z2} for {X1, X2}.
    let ex20 = graph(
        &[
            ("Z1", "X1"),
            ("Z1", "X2"),
            ("Z2", "X1"),
            ("Z2", "X2"),
            ("X1", "Y"),
            ("X2", "Y"),
        ],
        &[("X1", "Y"), ("X2", "Y")],
    );
    assert_eq!(
        generic_rank(&ex20, &["Z1", "Z2"], &["X1", "X2"]).unwrap(),
        2
    );
    let m = random_scm(&ex20, 9);
    let l = m
        .covariance()
        .instrumental_set("Y", &["X1", "X2"], &["Z1", "Z2"])
        .unwrap();
    assert!(close(l[0], m.coefficient("X1", "Y").unwrap()));
    assert!(close(l[1], m.coefficient("X2", "Y").unwrap()));
    // Ex. 19.21: Z1 -> Z2 -> X1, X2: rank 1, no instrumental set.
    let ex21 = graph(
        &[
            ("Z1", "Z2"),
            ("Z2", "X1"),
            ("Z2", "X2"),
            ("X1", "Y"),
            ("X2", "Y"),
        ],
        &[("X1", "Y"), ("X2", "Y")],
    );
    assert_eq!(
        generic_rank(&ex21, &["Z1", "Z2"], &["X1", "X2"]).unwrap(),
        1
    );
    assert!(instrumental_set(&ex21, "Y").unwrap().1.len() < 2);
    // Ex. 19.28: adding Z2 <-> X2 helps.
    let mut ex28 = ex21.clone();
    ex28.bidirected("Z2", "X2");
    assert_eq!(
        generic_rank(&ex28, &["Z1", "Z2"], &["X1", "X2"]).unwrap(),
        2
    );
    let (z, x) = instrumental_set(&ex28, "Y").unwrap();
    assert_eq!((z.len(), x.len()), (2, 2));
    let m = random_scm(&ex28, 11);
    let zs: Vec<&str> = z.iter().map(String::as_str).collect();
    let xs: Vec<&str> = x.iter().map(String::as_str).collect();
    let l = m.covariance().instrumental_set("Y", &xs, &zs).unwrap();
    for (xi, li) in xs.iter().zip(&l) {
        assert!(close(*li, m.coefficient(xi, "Y").unwrap()));
    }
}

/// Every criterion that claims identification on a random diagram returns
/// the true coefficient (or total effect) from Σ alone.
#[test]
fn every_criterion_is_sound_on_random_linear_scms() {
    const NAMES: [&str; 5] = ["A", "B", "C", "D", "E"];
    let (mut sd, mut civ, mut is, mut bd) = (0, 0, 0, 0);
    for seed in 0..60u64 {
        let mut r = splitmix(seed ^ 0x19);
        let mut next = || {
            r = splitmix(r);
            r % 100
        };
        let mut g = Admg::new();
        for n in NAMES {
            g.add_node(n);
        }
        for (j, b) in NAMES.iter().enumerate() {
            for a in &NAMES[..j] {
                if next() < 40 {
                    g.directed(a, b);
                }
                if next() < 20 {
                    g.bidirected(a, b);
                }
            }
        }
        let m = random_scm(&g, splitmix(seed));
        let s = m.covariance();
        let (d, _) = linear::edges(&g);
        for (x, y) in &d {
            if let Some(z) = single_door(&g, x, y).unwrap() {
                let zs: Vec<&str> = z.iter().map(String::as_str).collect();
                assert!(
                    close(
                        s.regression(y, x, &zs).unwrap(),
                        m.coefficient(x, y).unwrap()
                    ),
                    "seed {seed}"
                );
                sd += 1;
            }
            if let Some((z, w)) = conditional_instrument(&g, x, y).unwrap() {
                let ws: Vec<&str> = w.iter().map(String::as_str).collect();
                let est = s.regression(y, &z, &ws).unwrap() / s.regression(x, &z, &ws).unwrap();
                assert!(
                    (est - m.coefficient(x, y).unwrap()).abs() < 1e-7,
                    "seed {seed}: {z} | {ws:?}"
                );
                civ += 1;
            }
        }
        for y in NAMES {
            let (z, x) = instrumental_set(&g, y).unwrap();
            if x.is_empty() {
                continue;
            }
            let zs: Vec<&str> = z.iter().map(String::as_str).collect();
            let xs: Vec<&str> = x.iter().map(String::as_str).collect();
            let l = s.instrumental_set(y, &xs, &zs).unwrap();
            for (xi, li) in xs.iter().zip(&l) {
                assert!(
                    (li - m.coefficient(xi, y).unwrap()).abs() < 1e-7,
                    "seed {seed}: {zs:?} -> {xs:?} -> {y}"
                );
            }
            is += 1;
        }
        for x in NAMES {
            for y in NAMES {
                if x != y
                    && let Some(z) = back_door(&g, x, y).unwrap()
                {
                    let zs: Vec<&str> = z.iter().map(String::as_str).collect();
                    assert!(close(
                        s.regression(y, x, &zs).unwrap(),
                        m.total_effect(x, y).unwrap()
                    ));
                    bd += 1;
                }
            }
        }
    }
    assert!(
        sd > 30 && civ > 30 && is > 30 && bd > 30,
        "{sd} {civ} {is} {bd}"
    );
}

// ---------------------------------------------------------------------------
// Monotonicity.

/// A random binary SCM: each variable is a threshold of a weighted sum of
/// its parents, its latents and a 4-valued noise. A parent declared
/// monotone (positive) enters with a positive weight. Any other parent
/// enters xor-ed with a noise bit, so it is genuinely non-monotone.
struct Monotone {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    monotone: Vec<Vec<bool>>,
    latents: Vec<(usize, usize)>,
    weights: Vec<Vec<f64>>,
    latent_w: Vec<f64>,
    noise: Vec<[f64; 4]>,
    flip_p: f64,
}

impl Monotone {
    fn new(g: &Admg, positive: &[(&str, &str)], seed: u64) -> Self {
        let names = g.topological_order().unwrap();
        let idx = |v: &str| names.iter().position(|n| n == v).unwrap();
        let (d, b) = linear::edges(g);
        let n = names.len();
        let mut parents = vec![Vec::new(); n];
        for (x, y) in &d {
            parents[idx(y)].push(idx(x));
        }
        let mut r = seed;
        let mut next = || {
            r = splitmix(r);
            (r >> 11) as f64 / (1u64 << 53) as f64
        };
        let monotone = (0..n)
            .map(|i| {
                parents[i]
                    .iter()
                    .map(|&p| positive.iter().any(|(a, c)| idx(a) == p && idx(c) == i))
                    .collect()
            })
            .collect();
        let weights = (0..n)
            .map(|i| parents[i].iter().map(|_| 0.5 + next()).collect())
            .collect();
        let latents: Vec<(usize, usize)> = b.iter().map(|(x, y)| (idx(x), idx(y))).collect();
        let latent_w = latents.iter().map(|_| 0.5 + next()).collect();
        let noise = (0..n)
            .map(|_| {
                let w = [0.2 + next(), 0.2 + next(), 0.2 + next(), 0.2 + next()];
                let t: f64 = w.iter().sum();
                w.map(|x| x / t)
            })
            .collect();
        Self {
            names,
            parents,
            monotone,
            latents,
            weights,
            latent_w,
            noise,
            flip_p: 0.3,
        }
    }

    /// Every exogenous state: probability, latent bits, noises, flip bits.
    fn units(&self) -> Vec<(f64, u64, Vec<usize>, u64)> {
        let n = self.names.len();
        let mut out = Vec::new();
        for lat in 0..(1u64 << self.latents.len()) {
            let pl = 0.5f64.powi(self.latents.len() as i32);
            for code in 0..4usize.pow(n as u32) {
                let e: Vec<usize> = (0..n).map(|i| code / 4usize.pow(i as u32) % 4).collect();
                let pe: f64 = e
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| self.noise[i][k])
                    .product();
                for flips in 0..(1u64 << n) {
                    let pf: f64 = (0..n)
                        .map(|i| {
                            if flips >> i & 1 == 1 {
                                self.flip_p
                            } else {
                                1.0 - self.flip_p
                            }
                        })
                        .product();
                    out.push((pl * pe * pf, lat, e.clone(), flips));
                }
            }
        }
        out
    }

    fn solve(&self, lat: u64, e: &[usize], flips: u64, fixed: &BTreeMap<usize, u8>) -> Vec<u8> {
        let n = self.names.len();
        let mut v = vec![0u8; n];
        for i in 0..n {
            if let Some(x) = fixed.get(&i) {
                v[i] = *x;
                continue;
            }
            let mut s = e[i] as f64 * 0.6 - 1.0;
            for (k, &p) in self.parents[i].iter().enumerate() {
                let input = if self.monotone[i][k] {
                    v[p]
                } else {
                    v[p] ^ u8::from(flips >> i & 1 == 1)
                };
                s += self.weights[i][k] * f64::from(input);
            }
            for (l, (a, b)) in self.latents.iter().enumerate() {
                if *a == i || *b == i {
                    s += self.latent_w[l] * f64::from(u8::from(lat >> l & 1 == 1)) - 0.5;
                }
            }
            v[i] = u8::from(s > 0.4);
        }
        v
    }

    fn idx(&self, v: &str) -> usize {
        self.names.iter().position(|n| n == v).unwrap()
    }

    /// The true `P(events ∧ given) / P(given)`.
    fn truth(&self, events: &[Event<'_>], given: &[Event<'_>]) -> f64 {
        let check = |unit: &(f64, u64, Vec<usize>, u64), evs: &[Event<'_>]| {
            evs.iter().all(|(v, x, val)| {
                let fixed: BTreeMap<usize, u8> = x.iter().map(|(k, l)| (self.idx(k), *l)).collect();
                self.solve(unit.1, &unit.2, unit.3, &fixed)[self.idx(v)] == *val
            })
        };
        let (mut num, mut den) = (0.0, 0.0);
        for u in self.units() {
            if check(&u, given) {
                den += u.0;
                if check(&u, events) {
                    num += u.0;
                }
            }
        }
        num / den
    }

    fn observational(&self) -> Dataset {
        let n = self.names.len();
        let mut p = vec![0.0; 1 << n];
        for (w, lat, e, f) in self.units() {
            let v = self.solve(lat, &e, f, &BTreeMap::new());
            p[v.iter()
                .enumerate()
                .map(|(i, b)| usize::from(*b) << i)
                .sum::<usize>()] += w;
        }
        let rows: Vec<Vec<String>> = (0..1usize << n)
            .map(|k| (0..n).map(|i| (k >> i & 1).to_string()).collect())
            .collect();
        Dataset::from_records(self.names.clone(), rows)
            .unwrap()
            .with_weights(p)
            .unwrap()
    }
}

/// LATE's ingredients in the IV graph: identified with Z -> X monotone and
/// equal to the truth; not identified without it.
#[test]
fn late_under_monotonicity() {
    let g = graph(&[("Z", "X"), ("X", "Y")], &[("X", "Y")]);
    let compliers: [Event<'_>; 2] = [("X", &[("Z", 0)], 0), ("X", &[("Z", 1)], 1)];
    for seed in 0..10u64 {
        let m = Monotone::new(&g, &[("Z", "X")], splitmix(seed));
        let p = m.observational();
        let cgma = Cgma::new(&g, &[("Z", "X", Sign::Positive)]).unwrap();
        for y in [0, 1] {
            let ev: [Event<'_>; 1] = [("Y", &[("X", 1)], y)];
            let est = m_id(&cgma, &ev, &compliers, &p).unwrap();
            assert!((est - m.truth(&ev, &compliers)).abs() < 1e-9, "seed {seed}");
        }
        let plain = Cgma::new(&g, &[]).unwrap();
        assert!(matches!(
            m_id(&plain, &compliers, &[], &p),
            Err(ParametricError::NotIdentifiable(_))
        ));
    }
}

/// The fairness graph of Fig. 19.33(d): Z, X -> M -> Y, Z -> Y, X -> Y.
fn fairness() -> Admg {
    graph(
        &[("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")],
        &[],
    )
}

/// Sec. 19.7.6.2: the total effect, and the (z, m)-specific effect, which
/// is identified once X -> M is monotone (Prop. 19.7.5). The book reports
/// TE = 0.175; the printed SCM and Table 19.1 give exactly 0.205
/// (½ (0.675 − 0.275) + ½ (0.66 − 0.65)), so the book's figure is
/// presumably its estimate from the 10 000 samples it draws.
#[test]
fn section_19_7_6_2_fair_machine_learning() {
    // M = 1{U = a} + X·1{U = c}, U ternary per value of Z; Y ~ Table 19.1.
    let pm = [[0.25, 0.5, 0.25], [0.5, 0.3, 0.2]];
    let py = [[[0.1, 0.8], [0.9, 0.6]], [[0.4, 0.9], [0.1, 0.8]]];
    let m_of = |x: u8, u: usize| u8::from(u == 0) + x * u8::from(u == 1);
    // Exact P(Z, X, M, Y) and the truths, enumerating Z, X, U and Y.
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    let mut joint = BTreeMap::new();
    for z in 0..2usize {
        for x in 0..2u8 {
            for (u, pu) in pm[z].iter().enumerate() {
                let w = 0.25 * pu;
                let m = m_of(x, u);
                for y in 0..2u8 {
                    let p1 = py[z][usize::from(x)][usize::from(m)];
                    let pw = w * if y == 1 { p1 } else { 1.0 - p1 };
                    *joint.entry([z as u8, x, m, y]).or_insert(0.0) += pw;
                }
            }
        }
    }
    for (k, w) in &joint {
        rows.push(k.iter().map(u8::to_string).collect::<Vec<_>>());
        weights.push(*w);
    }
    let p = Dataset::from_records(["Z", "X", "M", "Y"], rows)
        .unwrap()
        .with_weights(weights)
        .unwrap();
    let g = fairness();
    let cgma = Cgma::new(&g, &[("X", "M", Sign::Positive)]).unwrap();
    let ey = |x: u8| m_id(&cgma, &[("Y", &[("X", x)], 1)], &[], &p).unwrap();
    // The adjustment formula Σ_z P(z) Σ_m P(y | z, x, m) P(m | z, x).
    let adjust = |x: usize| -> f64 {
        (0..2)
            .map(|z| {
                let p1 = pm[z][0] + if x == 1 { pm[z][1] } else { 0.0 };
                0.5 * (py[z][x][1] * p1 + py[z][x][0] * (1.0 - p1))
            })
            .sum()
    };
    assert!((adjust(1) - adjust(0) - 0.205).abs() < 1e-12);
    assert!((ey(1) - ey(0) - 0.205).abs() < 1e-9);
    // (z, m)-TE by enumeration: E[Y_{x} | z, m].
    for z in 0..2usize {
        for m in 0..2u8 {
            let truth = |x: u8| {
                let (mut num, mut den) = (0.0, 0.0);
                for xf in 0..2u8 {
                    for u in 0..3 {
                        if m_of(xf, u) != m {
                            continue;
                        }
                        let w = 0.25 * pm[z][u];
                        den += w;
                        num += w * py[z][usize::from(x)][usize::from(m_of(x, u))];
                    }
                }
                num / den
            };
            let given: [Event<'_>; 2] = [("Z", &[], z as u8), ("M", &[], m)];
            let est = |x: u8| m_id(&cgma, &[("Y", &[("X", x)], 1)], &given, &p).unwrap();
            assert!(
                (est(1) - est(0) - (truth(1) - truth(0))).abs() < 1e-9,
                "z {z} m {m}"
            );
        }
    }
    // Without monotonicity of X -> M it is not identified.
    let plain = Cgma::new(&g, &[]).unwrap();
    assert!(
        m_id(
            &plain,
            &[("Y", &[("X", 1)], 1)],
            &[("Z", &[], 1), ("M", &[], 1)],
            &p
        )
        .is_err()
    );
}

/// PN in the fairness graph with X -> M, X -> Y, M -> Y monotone
/// (Prop. 19.7.5 (2)) on random monotone SCMs; and the global monotonicity
/// of Lemma 19.7.1.
#[test]
fn probability_of_necessity_under_monotonicity() {
    let g = fairness();
    let mono = [("X", "M"), ("X", "Y"), ("M", "Y")];
    let cgma = Cgma::new(
        &g,
        &[
            ("X", "M", Sign::Positive),
            ("X", "Y", Sign::Positive),
            ("M", "Y", Sign::Positive),
        ],
    )
    .unwrap();
    assert_eq!(
        cgma.globally_monotone("X", "Y").unwrap(),
        Some(Sign::Positive)
    );
    assert_eq!(
        Cgma::new(&g, &[("X", "Y", Sign::Positive)])
            .unwrap()
            .globally_monotone("X", "Y")
            .unwrap(),
        None
    );
    for seed in 0..6u64 {
        let m = Monotone::new(&g, &mono, splitmix(seed ^ 0xfe));
        let p = m.observational();
        let ev: [Event<'_>; 1] = [("Y", &[("X", 0)], 0)];
        let given: [Event<'_>; 2] = [("X", &[], 1), ("Y", &[], 1)];
        let est = m_id(&cgma, &ev, &given, &p).unwrap();
        assert!((est - m.truth(&ev, &given)).abs() < 1e-9, "seed {seed}");
    }
    // With M -> Y non-monotone it is not identified.
    let partial = Cgma::new(
        &g,
        &[("X", "M", Sign::Positive), ("X", "Y", Sign::Positive)],
    )
    .unwrap();
    let m = Monotone::new(&g, &[("X", "M"), ("X", "Y")], 3);
    let ev: [Event<'_>; 1] = [("Y", &[("X", 0)], 0)];
    assert!(
        m_id(
            &partial,
            &ev,
            &[("X", &[], 1), ("Y", &[], 1)],
            &m.observational()
        )
        .is_err()
    );
}

/// Random diagrams with every directed edge monotone and random
/// counterfactual queries: whenever M-ID answers, the answer is the truth.
#[test]
fn m_id_is_sound_on_random_monotone_scms() {
    const NAMES: [&str; 4] = ["A", "B", "C", "D"];
    let (mut answered, mut refused) = (0, 0);
    for seed in 0..24u64 {
        let mut r = splitmix(seed ^ 0x1907);
        let mut next = || {
            r = splitmix(r);
            r % 100
        };
        let mut g = Admg::new();
        for n in NAMES {
            g.add_node(n);
        }
        let mut directed = Vec::new();
        for (j, b) in NAMES.iter().enumerate() {
            for a in &NAMES[..j] {
                if next() < 50 {
                    g.directed(a, b);
                    directed.push((*a, *b));
                }
                if next() < 20 {
                    g.bidirected(a, b);
                }
            }
        }
        let m = Monotone::new(&g, &directed, splitmix(seed));
        let p = m.observational();
        let signs: Vec<(&str, &str, Sign)> = directed
            .iter()
            .map(|(a, b)| (*a, *b, Sign::Positive))
            .collect();
        let cgma = Cgma::new(&g, &signs).unwrap();
        // P(D_{A = a} = d | A = a', D = d') and P(D_{A = a} = d, C_{A = a'} = c).
        let (a, d, a2, d2) = (
            (next() % 2) as u8,
            (next() % 2) as u8,
            (next() % 2) as u8,
            (next() % 2) as u8,
        );
        let x: [(&str, u8); 1] = [("A", a)];
        let x2: [(&str, u8); 1] = [("A", a2)];
        let queries: Vec<(Vec<Event<'_>>, Vec<Event<'_>>)> = vec![
            (vec![("D", &x, d)], vec![("A", &[], a2), ("D", &[], d2)]),
            (vec![("D", &x, d), ("C", &x2, d2)], vec![]),
        ];
        for (ev, given) in queries {
            match m_id(&cgma, &ev, &given, &p) {
                Ok(est) => {
                    assert!(
                        (est - m.truth(&ev, &given)).abs() < 1e-8,
                        "seed {seed}: {ev:?} | {given:?}"
                    );
                    answered += 1;
                }
                Err(ParametricError::NotIdentifiable(_)) => refused += 1,
                Err(ParametricError::InvalidInput(_)) => {}
                Err(e) => panic!("seed {seed}: {e}"),
            }
        }
    }
    assert!(
        answered >= 15 && refused >= 3,
        "{answered} answered, {refused} refused"
    );
}
