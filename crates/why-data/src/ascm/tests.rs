//! Tests for chapter 14: the book's examples with exact numbers, and random
//! SCMs that check the bounds against the truth and against identification.

use super::*;
use crate::fairness::Dataset;
use crate::graph::cinfer::Input;
use crate::graph::cinfer::counterfactual::{CtfError, CtfQuery, ctf_infer};
use crate::graph::dseparation::Admg;
use crate::scm::{Dist, Model, Value, expr};

fn eq(name: &str, e: rssn::symbolic::core::Expr) -> (String, rssn::symbolic::core::Expr) {
    (name.to_string(), e)
}

fn bern(name: &str, p: f64) -> (String, Dist) {
    (name.to_string(), Dist::Bernoulli(p))
}

fn v(name: &str) -> rssn::symbolic::core::Expr {
    expr::var(name)
}

const T: Value = Value::Bool(true);
const F: Value = Value::Bool(false);

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn assert_bounds(b: Interval, lower: f64, upper: f64) {
    assert!(
        close(b.lower, lower) && close(b.upper, upper),
        "expected [{lower}, {upper}], got [{}, {}]",
        b.lower,
        b.upper
    );
}

/// `H`'s mechanism, the image mechanism and `P(U_H1 = 1)` vary across the
/// examples; the rest is Ex. 14.2.
fn faces(h: rssn::symbolic::core::Expr, image: &str, p_h1: f64) -> Ascm {
    let mut f = vec![
        eq("F", expr::xor(v("UF"), v("UY"))),
        eq("Y", v("UY")),
        eq("H", h),
    ];
    let mut u = vec![
        bern("UF", 0.4),
        bern("UY", 0.4),
        bern("UH1", p_h1),
        bern("UH2", 0.2),
    ];
    match image {
        // M† of Ex. 14.6 (Eq. 14.12).
        "dagger" => {
            f.push(eq("I1", expr::xor(expr::xor(v("Y"), v("H")), v("UI"))));
            f.push(eq("I2", expr::xor(v("Y"), v("UI"))));
            f.push(eq("I3", expr::xor(v("F"), v("H"))));
            f.push(eq("I4", v("UI")));
            u.push(bern("UI", 0.3));
        }
        // Ex. 14.4: the pixels show F, Y, H and whether the person smiles.
        "smile1" | "smile2" => {
            let s = if image == "smile1" {
                v("US")
            } else {
                expr::xor(v("US"), v("Y"))
            };
            f.push(eq("S", s));
            f.push(eq("I1", v("F")));
            f.push(eq("I2", v("Y")));
            f.push(eq("I3", v("H")));
            f.push(eq("I4", v("S")));
            u.push(bern("US", 0.5));
        }
        _ => unreachable!(),
    }
    let m = Model::new(f, u).unwrap();
    Ascm::new(m, &["F", "Y", "H"], &["I1", "I2", "I3", "I4"]).unwrap()
}

/// `H ← (¬Y ∧ U_H1) ⊕ (Y ∧ U_H2)` (Eq. 14.3).
fn h_star() -> rssn::symbolic::core::Expr {
    expr::xor(
        expr::and([expr::not(v("Y")), v("UH1")]),
        expr::and([v("Y"), v("UH2")]),
    )
}

fn faces_graph() -> Admg {
    // Fig. 14.3 at the generative level.
    let mut g = Admg::new();
    g.bidirected("F", "Y");
    g.directed("Y", "H");
    g
}

fn probability(d: &Dataset, row: &[&str]) -> f64 {
    let names = d.column_names().to_vec();
    (0..d.len())
        .filter(|&r| {
            names
                .iter()
                .zip(row)
                .all(|(n, x)| d.value(n, r) == Some(*x))
        })
        .map(|r| d.weights().unwrap()[r])
        .sum()
}

#[test]
fn table_14_1() {
    let p = faces(h_star(), "dagger", 0.4).observational().unwrap();
    let table = [
        (["0", "0", "0"], 0.216),
        (["0", "0", "1"], 0.144),
        (["0", "1", "0"], 0.128),
        (["0", "1", "1"], 0.032),
        (["1", "0", "0"], 0.144),
        (["1", "0", "1"], 0.096),
        (["1", "1", "0"], 0.192),
        (["1", "1", "1"], 0.048),
    ];
    for (row, want) in table {
        assert!(close(probability(&p, &row), want), "{row:?}");
    }
}

#[test]
fn examples_14_3_and_14_5_fit_the_data_but_disagree() {
    // P(F_{Y=0} = 0, H_{Y=0} = 1 | F = 0, Y = 1, H = 0): Eqs. 14.6, 14.7, 14.10.
    let query = |m: &mut Ascm| {
        m.counterfactual(
            &[("Y", F)],
            &[("F", F), ("H", T)],
            &[("F", F), ("Y", T), ("H", F)],
        )
        .unwrap()
    };
    let mut star = faces(h_star(), "dagger", 0.4);
    // M′ (Eq. 14.5): H reads U_Y instead of Y.
    let h_prime = expr::xor(
        expr::and([expr::not(v("UY")), v("UH1")]),
        expr::and([v("UY"), v("UH2")]),
    );
    let mut prime = faces(h_prime, "dagger", 0.4);
    // M(3) (Eq. 14.9), with P(U_H1 = 1) = 0.25.
    let h3 = expr::or([expr::and([expr::not(v("Y")), v("UH1")]), v("UH2")]);
    let mut three = faces(h3, "dagger", 0.25);

    let p = star.observational().unwrap();
    let vars = ["F", "Y", "H"];
    for m in [&prime, &three] {
        assert!(total_variation(&p, &m.observational().unwrap(), &vars).unwrap() < 1e-12);
    }
    assert!(close(query(&mut star), 0.4));
    assert!(close(query(&mut prime), 0.0));
    assert!(close(query(&mut three), 0.25));

    // Ex. 14.9: the optimal bounds are [0.25, 0.5]. M* lies inside, M(3)
    // attains the lower end, and M′, which breaks Fig. 14.3 (H does not
    // listen to Y), lies outside.
    let g = faces_graph();
    let q = CtfQuery::parse(&g, "P(f_{y'}, h'_{y'} | f, y, h)").unwrap();
    let values = [("F", "0"), ("Y", "1"), ("Y'", "0"), ("H", "0"), ("H'", "1")];
    assert_bounds(optimal_bounds(&g, &p, &q, &values).unwrap(), 0.25, 0.5);

    // With W = {F, Y} the bound collapses to 1 (Eq. 14.24).
    let q = CtfQuery::parse(&g, "P(f_{y'} | f, y)").unwrap();
    let b = optimal_bounds(&g, &p, &q, &[("F", "0"), ("Y", "1"), ("Y'", "0")]).unwrap();
    assert_bounds(b, 1.0, 1.0);
}

fn bits(s: &str) -> Vec<Value> {
    s.chars().map(|c| Value::Bool(c == '1')).collect()
}

#[test]
fn figure_14_5_and_the_feature_queries() {
    let mut m = faces(h_star(), "dagger", 0.4);
    let ctf = m.image_counterfactual(&[("Y", F)]).unwrap();
    let book = [
        ("0000", "0000", 0.1512),
        ("0001", "1101", 0.02304),
        ("0001", "0111", 0.01536),
        ("0010", "0010", 0.1008),
        ("0011", "1111", 0.03456),
        ("0011", "0101", 0.02304),
        ("0100", "0010", 0.02016),
        ("0100", "1000", 0.01344),
        ("0101", "0101", 0.0288),
        ("0110", "0000", 0.01344),
        ("0110", "1010", 0.00896),
        ("0111", "0111", 0.0432),
        ("1000", "1000", 0.0672),
        ("1001", "1111", 0.00864),
        ("1001", "0101", 0.00576),
        ("1010", "1010", 0.1008),
        ("1011", "1101", 0.00576),
        ("1011", "0111", 0.00384),
        ("1100", "0000", 0.05376),
        ("1100", "1010", 0.03584),
        ("1101", "1101", 0.0648),
        ("1110", "0010", 0.08064),
        ("1110", "1000", 0.05376),
        ("1111", "1111", 0.0432),
    ];
    assert_eq!(ctf.rows.len(), book.len());
    for (i, j, p) in book {
        assert!(close(ctf.probability(&bits(i), &bits(j)), p), "{i} -> {j}");
    }

    // Eqs. 14.14 and 14.16 for i = 0001, i′ = 1101.
    let (i, j) = (bits("0001"), bits("1101"));
    let pixel = |im: &[Value]| im[0].as_bool();
    assert!(close(ctf.feature_of(pixel, &i, &j), 0.08));
    let parity = |im: &[Value]| im[0].as_bool() ^ im[1].as_bool() ^ im[2].as_bool();
    assert!(close(ctf.feature_of(parity, &i, &j), 0.52));

    // That parity is h_F (Eq. 14.19), so Lemma 14.3.1 gives the same 0.52
    // at the generative level (Eq. 14.20).
    for im in ctf.rows.iter().flat_map(|(a, b, _)| [a, b]) {
        assert_eq!(
            m.label_of(im, &["F"]).unwrap().unwrap()[0].as_bool(),
            parity(im)
        );
    }
    let lemma = m
        .joint_counterfactual(&[("Y", F)], &[("F", F)], &[("F", F)])
        .unwrap();
    assert!(close(lemma, 0.52));

    // Lemma 14.3.1 for W = V and every pair of labels.
    let all = ["F", "Y", "H"];
    for a in 0..8 {
        for b in 0..8 {
            let lab =
                |k: usize| -> Vec<Value> { (0..3).map(|s| Value::Bool(k >> s & 1 == 1)).collect() };
            let (w, w2) = (lab(a), lab(b));
            let h = |im: &[Value]| m.label_of(im, &all).unwrap().unwrap();
            let lhs = ctf.feature(h, &w, &w2);
            let events: Vec<(&str, Value)> = all.iter().copied().zip(w2.iter().copied()).collect();
            let factual: Vec<(&str, Value)> = all.iter().copied().zip(w.iter().copied()).collect();
            let mut m2 = m.clone();
            let rhs = m2
                .joint_counterfactual(&[("Y", F)], &events, &factual)
                .unwrap();
            assert!(close(lhs, rhs), "{w:?} -> {w2:?}: {lhs} vs {rhs}");
        }
    }
}

#[test]
fn example_14_4_same_images_different_edits() {
    let mut one = faces(h_star(), "smile1", 0.4);
    let mut two = faces(h_star(), "smile2", 0.4);
    let cols = ["F", "Y", "H", "I1", "I2", "I3", "I4"];
    let t1 = one
        .model()
        .world(one.model().observational())
        .probability_table(Some(&cols))
        .unwrap();
    let t2 = two
        .model()
        .world(two.model().observational())
        .probability_table(Some(&cols))
        .unwrap();
    assert_eq!(t1.rows.len(), t2.rows.len());
    for ((a, p), (b, q)) in t1.rows.iter().zip(&t2.rows) {
        assert_eq!(a, b);
        assert!(close(*p, *q));
    }
    // A non-smiling young male, edited to a smiling old male with gray hair.
    let (i, j) = (bits("0100"), bits("0011"));
    let c1 = one.image_counterfactual(&[("Y", F)]).unwrap();
    let c2 = two.image_counterfactual(&[("Y", F)]).unwrap();
    assert!(close(c1.probability(&i, &j), 0.0));
    assert!(close(c2.probability(&i, &j), 0.4 * 0.4 * 0.8 * 0.4 * 0.5));
}

#[test]
fn a_non_invertible_image_is_rejected() {
    let m = Model::new(
        [eq("X", v("UX")), eq("I1", expr::and([v("X"), v("UI")]))],
        [bern("UX", 0.5), bern("UI", 0.5)],
    )
    .unwrap();
    assert_eq!(
        Ascm::new(m, &["X"], &["I1"]).unwrap_err(),
        AscmError::NotInvertible
    );
}

/// The colored MNIST models of Sec. 14.4, Eqs. 14.30–14.38, with the
/// pixels reduced to the factors and one unlabelled style bit. `C` draws its
/// Bernoulli from a uniform `U_C` on `0..100`.
fn mnist(frontdoor: bool) -> Ascm {
    let (p1, p2, p3) = if frontdoor {
        (0.9, 0.8, 0.7)
    } else {
        (0.8, 0.9, 0.75)
    };
    let (c, big) = if frontdoor {
        // C ← Bernoulli(0.95 − 0.1 D); B listens to U_D (confounding).
        (
            expr::lt(
                v("UC"),
                expr::sub(expr::int(95), expr::mul(expr::int(10), v("D"))),
            ),
            expr::lt(v("UD"), expr::int(4)),
        )
    } else {
        // C ← Bernoulli(0.05 + 0.1 U_D) (confounding); B listens to D.
        (
            expr::lt(
                v("UC"),
                expr::add(expr::int(5), expr::mul(expr::int(10), v("UD"))),
            ),
            expr::ge(v("D"), expr::int(5)),
        )
    };
    let b = expr::and([
        expr::or([expr::xor(big, v("U1")), expr::xor(v("C"), v("U2"))]),
        v("U3"),
    ]);
    let m = Model::new(
        [
            eq("D", v("UD")),
            eq("C", c),
            eq("B", b),
            eq("P1", v("D")),
            eq("P2", v("C")),
            eq("P3", v("B")),
            eq("P4", v("UI")),
        ],
        [
            ("UD".to_string(), Dist::Categorical(vec![0.1; 10])),
            ("UC".to_string(), Dist::Categorical(vec![0.01; 100])),
            bern("U1", p1),
            bern("U2", p2),
            bern("U3", p3),
            bern("UI", 0.5),
        ],
    )
    .unwrap();
    Ascm::new(m, &["D", "C", "B"], &["P1", "P2", "P3", "P4"]).unwrap()
}

fn int(i: i64) -> Value {
    Value::Int(i)
}

#[test]
fn mnist_backdoor_tasks() {
    let mut m = mnist(false);
    let p = m.observational().unwrap();
    let mut g = Admg::new();
    g.bidirected("D", "C");
    g.directed("D", "B");
    g.directed("C", "B");

    // Task 1: a red 3 with a bar, edited to a 6 (Fig. 14.8, Eqs. 14.33-34).
    let q = CtfQuery::parse(&g, "P(c_{d}, b_{d} | d', c, b)").unwrap();
    let vals = |b: &'static str| [("D", "6"), ("D'", "3"), ("C", "1"), ("B", b)];
    let upper = 0.75 * 0.28 / (0.75 * 0.82);
    let with_bar = optimal_bounds(
        &g,
        &p,
        &q,
        &[("D", "6"), ("D'", "3"), ("C", "1"), ("B", "1")],
    )
    .unwrap();
    assert_bounds(with_bar, 0.0, upper);
    assert!((with_bar.upper - 0.34).abs() < 0.005);
    // The bar disappears: P(C_{D=6} = 1, B_{D=6} = 0 | D = 3, C = 1, B = 1).
    let q0 = CtfQuery::parse(&g, "P(c_{d}, b'_{d} | d', c, b)").unwrap();
    let mut v0 = vals("1").to_vec();
    v0.push(("B'", "0"));
    assert_bounds(optimal_bounds(&g, &p, &q0, &v0).unwrap(), 1.0 - upper, 1.0);
    // The truth: 0.075 / 0.615.
    let truth = m
        .counterfactual(
            &[("D", int(6))],
            &[("C", T), ("B", T)],
            &[("D", int(3)), ("C", T), ("B", T)],
        )
        .unwrap();
    assert!(close(truth, 0.075 / 0.615));

    // Task 2: removing the bar from a green 1 changes nothing else (Eq. 14.35).
    let q = CtfQuery::parse(&g, "P(d_{b'}, c_{b'} | d, c, b)").unwrap();
    let b = optimal_bounds(
        &g,
        &p,
        &q,
        &[("D", "1"), ("C", "0"), ("B", "1"), ("B'", "0")],
    )
    .unwrap();
    assert_bounds(b, 1.0, 1.0);
}

#[test]
fn mnist_frontdoor_tasks() {
    let mut m = mnist(true);
    let p = m.observational().unwrap();
    let mut g = Admg::new();
    g.directed("D", "C");
    g.directed("C", "B");
    g.bidirected("D", "B");

    // Task 3: a green 7 with a bar, edited to a 2 (Fig. 14.11, Eqs. 14.39-41).
    let base = [("D", "2"), ("D'", "7"), ("C", "0"), ("B", "1")];
    let q = CtfQuery::parse(&g, "P(c_{d}, b_{d} | d', c, b)").unwrap();
    assert_bounds(optimal_bounds(&g, &p, &q, &base).unwrap(), 0.0, 1.0 / 3.0);
    let q = CtfQuery::parse(&g, "P(c_{d}, b'_{d} | d', c, b)").unwrap();
    let mut v = base.to_vec();
    v.push(("B'", "0"));
    assert_bounds(optimal_bounds(&g, &p, &q, &v).unwrap(), 0.0, 0.0);
    let q = CtfQuery::parse(&g, "P(c'_{d} | d', c, b)").unwrap();
    let mut v = base.to_vec();
    v.push(("C'", "1"));
    assert_bounds(optimal_bounds(&g, &p, &q, &v).unwrap(), 2.0 / 3.0, 1.0);
    // The truth sits on the upper end: C is comonotone in its noise.
    let truth = m
        .counterfactual(
            &[("D", int(2))],
            &[("C", F), ("B", T)],
            &[("D", int(7)), ("C", F), ("B", T)],
        )
        .unwrap();
    assert!(close(truth, 1.0 / 3.0));

    // Task 4: a red 4 without a bar, edited to green (Fig. 14.12,
    // Eqs. 14.42-43). B(0) = (U1 ∨ U2) ∧ U3 and B(1) = (U1 ∨ ¬U2) ∧ U3 at
    // D = 4.
    let (b0, b1) = (0.98 * 0.7, 0.92 * 0.7);
    let lower = (b0 + (1.0 - b1) - 1.0) / (1.0 - b1);
    let q = CtfQuery::parse(&g, "P(d_{c'}, b'_{c'} | d, c, b)").unwrap();
    let v = [("D", "4"), ("C", "1"), ("C'", "0"), ("B", "0"), ("B'", "1")];
    let with_bar = optimal_bounds(&g, &p, &q, &v).unwrap();
    assert_bounds(with_bar, lower, 1.0);
    assert!((with_bar.lower - 0.12).abs() < 0.005);
    let q = CtfQuery::parse(&g, "P(d_{c'}, b_{c'} | d, c, b)").unwrap();
    let v = [("D", "4"), ("C", "1"), ("C'", "0"), ("B", "0")];
    assert_bounds(optimal_bounds(&g, &p, &q, &v).unwrap(), 0.0, 1.0 - lower);
    let truth = m
        .counterfactual(
            &[("C", F)],
            &[("D", int(4)), ("B", T)],
            &[("D", int(4)), ("C", T), ("B", F)],
        )
        .unwrap();
    assert!(close(truth, 0.7 * 0.1 * 0.8 / (1.0 - b1)));
}

#[test]
fn ctf_consistency_of_the_examples() {
    // M(3) fits P(V) and attains the lower bound, so it is ctf-consistent;
    // M′ fits P(V) too but its value 0 is outside [0.25, 0.5].
    let g = faces_graph();
    let truth = faces(h_star(), "dagger", 0.4).observational().unwrap();
    let q = CtfQuery::parse(&g, "P(f_{y'}, h'_{y'} | f, y, h)").unwrap();
    let values = [("F", "0"), ("Y", "1"), ("Y'", "0"), ("H", "0"), ("H'", "1")];
    let ok = ctf_consistency(&g, &truth, &truth, &q, &values, 0.25, 1e-9).unwrap();
    assert!(ok.consistent && ok.fit < 1e-12);
    let bad = ctf_consistency(&g, &truth, &truth, &q, &values, 0.0, 1e-9).unwrap();
    assert!(!bad.consistent);
}

#[test]
fn errors_name_the_problem() {
    // Two free components: Y's and Z's response types both meet the query.
    let mut g = Admg::new();
    g.directed("X", "Y");
    g.directed("Y", "Z");
    let rows: Vec<[String; 3]> = (0..8)
        .map(|k| [k & 1, k >> 1 & 1, k >> 2 & 1].map(|b: i32| b.to_string()))
        .collect();
    let p = Dataset::from_records(["X", "Y", "Z"], rows)
        .unwrap()
        .with_weights(vec![0.05, 0.1, 0.15, 0.2, 0.1, 0.15, 0.1, 0.15])
        .unwrap();
    let q = CtfQuery::parse(&g, "P(y'_{x'}, z'_{y'} | x, y, z)").unwrap();
    let values = [
        ("X", "1"),
        ("X'", "0"),
        ("Y", "1"),
        ("Y'", "0"),
        ("Z", "1"),
        ("Z'", "0"),
    ];
    match optimal_bounds(&g, &p, &q, &values) {
        Err(AscmError::Nonlinear(cs)) => assert_eq!(cs, vec![vec!["Y"], vec!["Z"]]),
        other => panic!("expected Nonlinear, got {other:?}"),
    }
    // Conditioning on a counterfactual world.
    let q = CtfQuery::parse(&g, "P(y | z_{x})").unwrap();
    assert!(matches!(
        optimal_bounds(&g, &p, &q, &[("X", "1"), ("Y", "1"), ("Z", "1")]),
        Err(AscmError::CounterfactualEvidence(_))
    ));
}

// ---------------------------------------------------------------------------
// Random SCMs.

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// A random binary SCM over `V0 … V{n-1}` (topological order). Each
/// variable is a response function of its parents, the binary latents of
/// its bidirected edges and its own 4-valued noise: noise 0 and 1 force the
/// output (so every conditional is positive), 2 and 3 pick pseudo-random
/// functions.
struct Random {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    latents: Vec<(usize, usize, f64)>,
    noise: Vec<[f64; 4]>,
    seed: u64,
    graph: Admg,
}

impl Random {
    fn new(seed: u64, n: usize) -> Self {
        let mut r = seed;
        let mut next = || {
            r = splitmix(r);
            unit(r)
        };
        let names: Vec<String> = (0..n).map(|i| format!("V{i}")).collect();
        let mut graph = Admg::new();
        for name in &names {
            graph.add_node(name);
        }
        let mut parents = vec![Vec::new(); n];
        let mut latents = Vec::new();
        for j in 0..n {
            for i in 0..j {
                if parents[j].len() < 2 && next() < 0.45 {
                    parents[j].push(i);
                    graph.directed(&names[i], &names[j]);
                }
                if next() < 0.3 {
                    latents.push((i, j, 0.2 + 0.6 * next()));
                    graph.bidirected(&names[i], &names[j]);
                }
            }
        }
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
            latents,
            noise,
            seed,
            graph,
        }
    }

    fn f(&self, i: usize, v: &[u8], u: u64, e: usize) -> u8 {
        match e {
            0 => 0,
            1 => 1,
            _ => {
                let mut key = splitmix(self.seed ^ (i as u64) << 40 ^ (e as u64) << 32);
                for &p in &self.parents[i] {
                    key = splitmix(key ^ u64::from(v[p]) << 1 ^ p as u64);
                }
                for (l, (a, b, _)) in self.latents.iter().enumerate() {
                    if *a == i || *b == i {
                        key = splitmix(key ^ ((u >> l) & 1) << 1 ^ (l as u64) << 8);
                    }
                }
                u8::from(unit(key) < 0.5)
            }
        }
    }

    /// Every exogenous state: probability, latent bits, noises.
    fn units(&self) -> Vec<(f64, u64, Vec<usize>)> {
        let n = self.names.len();
        let mut out = Vec::new();
        for u in 0..(1u64 << self.latents.len()) {
            let pu: f64 = self
                .latents
                .iter()
                .enumerate()
                .map(|(l, (_, _, p))| if (u >> l) & 1 == 1 { *p } else { 1.0 - p })
                .product();
            for code in 0..4usize.pow(n as u32) {
                let e: Vec<usize> = (0..n).map(|i| code / 4usize.pow(i as u32) % 4).collect();
                let pe: f64 = e
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| self.noise[i][k])
                    .product();
                out.push((pu * pe, u, e));
            }
        }
        out
    }

    fn solve(&self, u: u64, e: &[usize], fixed: &[(usize, u8)]) -> Vec<u8> {
        let mut v = vec![0u8; self.names.len()];
        for i in 0..v.len() {
            v[i] = match fixed.iter().find(|(x, _)| *x == i) {
                Some((_, l)) => *l,
                None => self.f(i, &v, u, e[i]),
            };
        }
        v
    }

    fn distribution(&self) -> Dataset {
        let n = self.names.len();
        let mut p = vec![0.0; 1 << n];
        for (w, u, e) in self.units() {
            let v = self.solve(u, &e, &[]);
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

    /// `P(Y_{x} = y, plus | evidence)`, with `evidence` factual.
    fn truth(&self, x: (usize, u8), y: (usize, u8), evidence: &[(usize, u8)]) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (w, u, e) in self.units() {
            let v = self.solve(u, &e, &[]);
            if evidence.iter().all(|(i, l)| v[*i] == *l) {
                den += w;
                if self.solve(u, &e, &[x])[y.0] == y.1 {
                    num += w;
                }
            }
        }
        num / den
    }

    /// The response types of the largest c-component.
    fn largest_component(&self) -> usize {
        self.graph
            .c_components()
            .iter()
            .map(|c| {
                c.iter()
                    .map(|v| {
                        let i = self.names.iter().position(|n| n == v).unwrap();
                        1usize << (1 << self.parents[i].len())
                    })
                    .product::<usize>()
            })
            .max()
            .unwrap()
    }

    /// Whether every c-component is a bidirected clique.
    fn cliques_only(&self) -> bool {
        self.graph.c_components().iter().all(|c| {
            c.iter().enumerate().all(|(a, x)| {
                c.iter().skip(a + 1).all(|y| {
                    let (i, j) = (
                        self.names.iter().position(|n| n == x).unwrap(),
                        self.names.iter().position(|n| n == y).unwrap(),
                    );
                    self.latents
                        .iter()
                        .any(|(a, b, _)| (*a, *b) == (i.min(j), i.max(j)))
                })
            })
        })
    }
}

/// A query, its value names, and the factual evidence for the truth.
type Case<'a> = (String, Vec<(&'a str, String)>, Vec<(usize, u8)>);

/// Random graphs and queries: the true value always lies within the bounds,
/// and when the graph's c-components are cliques (so the bounds are tight)
/// an identifiable query has collapsed bounds.
#[test]
fn bounds_contain_the_truth_and_collapse_when_identified() {
    let (mut checked, mut collapsed, mut nonlinear) = (0, 0, 0);
    for seed in 0..80u64 {
        let m = Random::new(splitmix(seed ^ 0xa5c4), 4);
        if m.largest_component() > 1024 {
            continue;
        }
        let p = m.distribution();
        let r = splitmix(seed);
        let xi = (r % 3) as usize;
        let yi = xi + 1 + ((r >> 8) % (3 - xi as u64)) as usize;
        let (x, y) = (m.names[xi].to_lowercase(), m.names[yi].to_lowercase());
        let (xv, yv) = ((r >> 16) as u8 & 1, (r >> 17) as u8 & 1);
        let lv = |b: u8| b.to_string();
        let xu = m.names[xi].as_str();
        let yu = m.names[yi].as_str();
        let xp = format!("{xu}'");
        let yp = format!("{yu}'");
        // (query, evidence for the truth)
        let cases: Vec<Case<'_>> = vec![
            (
                format!("P({y}_{{{x}}})"),
                vec![(xu, lv(xv)), (yu, lv(yv))],
                vec![],
            ),
            (
                format!("P({y}_{{{x}}} | {x}')"),
                vec![(xu, lv(xv)), (xp.as_str(), lv(1 - xv)), (yu, lv(yv))],
                vec![(xi, 1 - xv)],
            ),
            (
                format!("P({y}_{{{x}}} | {x}', {y}')"),
                vec![
                    (xu, lv(xv)),
                    (xp.as_str(), lv(1 - xv)),
                    (yu, lv(yv)),
                    (yp.as_str(), lv(1 - yv)),
                ],
                vec![(xi, 1 - xv), (yi, 1 - yv)],
            ),
        ];
        for (text, values, evidence) in cases {
            let q = CtfQuery::parse(&m.graph, &text).unwrap();
            let values: Vec<(&str, &str)> = values.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let b = match optimal_bounds(&m.graph, &p, &q, &values) {
                Ok(b) => b,
                Err(AscmError::Nonlinear(_)) => {
                    nonlinear += 1;
                    continue;
                }
                Err(e) => panic!("seed {seed}, {text}: {e}"),
            };
            let truth = m.truth((xi, xv), (yi, yv), &evidence);
            assert!(
                b.lower - 1e-7 <= truth && truth <= b.upper + 1e-7,
                "seed {seed}, {text}: {truth} outside [{}, {}]",
                b.lower,
                b.upper
            );
            checked += 1;
            let identified = match ctf_infer(&m.graph, &q, &[Input::experimental(&[])]) {
                Ok(_) => true,
                Err(CtfError::NotIdentifiable { .. }) => false,
                Err(_) => continue,
            };
            if identified && m.cliques_only() {
                assert!(
                    b.upper - b.lower < 1e-7,
                    "seed {seed}, {text}: identified but [{}, {}]",
                    b.lower,
                    b.upper
                );
                collapsed += 1;
            }
        }
    }
    assert!(checked >= 150, "only {checked} queries checked");
    assert!(collapsed >= 40, "only {collapsed} identified queries");
    assert!(nonlinear < checked / 3, "{nonlinear} multilinear programs");
}
