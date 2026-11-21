//! Chapter 19: parametric identification, linear and monotonic.
//!
//! Walks through the chapter, asserting each result:
//!
//! 1. Linear SCMs: regression is biased under confounding (Exs. 19.1-19.2),
//!    and the single-door, back-door, conditional-instrument and
//!    instrumental-set criteria recover the true coefficients
//!    (Exs. 19.16-19.21, 19.28).
//! 2. Monotonicity: LATE in the IV graph (Ex. 19.34), and the
//!    sex-on-income example of Sec. 19.7.6.2, where the (z, m)-specific
//!    effect is identified once X -> M is monotone and differs from the
//!    naive (z, m)-TV.
//!
//! The book's companion Python has no chapter 19 code. Run with
//! `cargo run -p why-data --example chapter19_parametric`.

use std::collections::BTreeMap;

use why_data::fairness::Dataset;
use why_data::graph::dseparation::Admg;
use why_data::parametric::{
    Cgma, Event, LinearScm, Sign, back_door, conditional_instrument, generic_rank,
    instrumental_set, m_id, single_door,
};

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

fn part1() {
    println!("== 1. Linear SCMs ==");
    // Ex. 19.2: confounding biases the regression slope.
    let bow = graph(&[("X", "Y")], &[("X", "Y")]);
    let m = LinearScm::new(&bow, &[("X", "Y", 0.5)], &[], &[("X", "Y", 0.3)]).unwrap();
    let s = m.covariance();
    println!(
        "bow: λ_xy = 0.5 but regressing Y on X gives {:.2}; single door: {:?}",
        s.regression("Y", "X", &[]).unwrap(),
        single_door(&bow, "X", "Y").unwrap()
    );

    // Ex. 19.16: W is the single door.
    let g = graph(
        &[("W", "X"), ("W", "Y"), ("X", "Y")],
        &[("X", "T"), ("T", "Y")],
    );
    let m = LinearScm::new(
        &g,
        &[("W", "X", 0.8), ("W", "Y", -0.6), ("X", "Y", 0.4)],
        &[],
        &[("X", "T", 0.2), ("T", "Y", 0.3)],
    )
    .unwrap();
    let z = single_door(&g, "X", "Y").unwrap().unwrap();
    let beta = m.covariance().regression("Y", "X", &["W"]).unwrap();
    println!("Ex. 19.16: single door {z:?}, β_{{yx.w}} = {beta:.3} = λ_xy");
    assert!(close(beta, 0.4));

    // Ex. 19.17: the back door identifies the total effect.
    let g = graph(
        &[("S", "X"), ("S", "Y"), ("X", "W"), ("W", "Y"), ("X", "Y")],
        &[("S", "X"), ("W", "Y")],
    );
    let m = LinearScm::new(
        &g,
        &[
            ("S", "X", 0.7),
            ("S", "Y", 0.5),
            ("X", "W", 0.6),
            ("W", "Y", 0.9),
            ("X", "Y", 0.3),
        ],
        &[],
        &[("S", "X", 0.2), ("W", "Y", 0.25)],
    )
    .unwrap();
    let z = back_door(&g, "X", "Y").unwrap().unwrap();
    let beta = m.covariance().regression("Y", "X", &["S"]).unwrap();
    println!(
        "Ex. 19.17: back door {z:?}, β = {beta:.3} = δ_xy = λ_xw λ_wy + λ_xy = {:.3}",
        0.6 * 0.9 + 0.3
    );
    assert!(close(beta, 0.6 * 0.9 + 0.3));

    // Ex. 19.19: a conditional instrument.
    let g = graph(
        &[("W", "Z"), ("Z", "X"), ("X", "Y")],
        &[("X", "Y"), ("W", "Y")],
    );
    let m = LinearScm::new(
        &g,
        &[("W", "Z", 0.9), ("Z", "X", 0.7), ("X", "Y", -0.4)],
        &[],
        &[("X", "Y", 0.3), ("W", "Y", 0.2)],
    )
    .unwrap();
    let (z, w) = conditional_instrument(&g, "X", "Y").unwrap().unwrap();
    let s = m.covariance();
    let ws: Vec<&str> = w.iter().map(String::as_str).collect();
    let est = s.regression("Y", &z, &ws).unwrap() / s.regression("X", &z, &ws).unwrap();
    println!("Ex. 19.19: instrument {z} given {w:?}: λ_xy = {est:.3}");
    assert!(close(est, -0.4));

    // Exs. 19.20, 19.21, 19.28: instrumental sets and the flow-graph rank.
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
    let mut ex28 = ex21.clone();
    ex28.bidirected("Z2", "X2");
    for (name, g) in [
        ("Ex. 19.20", &ex20),
        ("Ex. 19.21", &ex21),
        ("Ex. 19.28", &ex28),
    ] {
        let rank = generic_rank(g, &["Z1", "Z2"], &["X1", "X2"]).unwrap();
        let (z, x) = instrumental_set(g, "Y").unwrap();
        println!("{name}: rank Σ_{{Z,X}} = {rank}; instrumental set {z:?} for {x:?}");
    }
}

fn part2() {
    println!("\n== 2. Monotonicity ==");
    // Sec. 19.7.6.2: Z age, X sex, M education, Y income.
    let pm = [[0.25, 0.5, 0.25], [0.5, 0.3, 0.2]];
    let py = [[[0.1, 0.8], [0.9, 0.6]], [[0.4, 0.9], [0.1, 0.8]]];
    let m_of = |x: u8, u: usize| u8::from(u == 0) + x * u8::from(u == 1);
    let mut joint: BTreeMap<[u8; 4], f64> = BTreeMap::new();
    for z in 0..2usize {
        for x in 0..2u8 {
            for (u, pu) in pm[z].iter().enumerate() {
                let m = m_of(x, u);
                let p1 = py[z][usize::from(x)][usize::from(m)];
                for y in 0..2u8 {
                    *joint.entry([z as u8, x, m, y]).or_insert(0.0) +=
                        0.25 * pu * if y == 1 { p1 } else { 1.0 - p1 };
                }
            }
        }
    }
    let rows: Vec<Vec<String>> = joint
        .keys()
        .map(|k| k.iter().map(u8::to_string).collect())
        .collect();
    let p = Dataset::from_records(["Z", "X", "M", "Y"], rows)
        .unwrap()
        .with_weights(joint.values().copied().collect())
        .unwrap();
    let g = graph(
        &[("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")],
        &[],
    );
    let cgma = Cgma::new(&g, &[("X", "M", Sign::Positive)]).unwrap();
    let te = m_id(&cgma, &[("Y", &[("X", 1)], 1)], &[], &p).unwrap()
        - m_id(&cgma, &[("Y", &[("X", 0)], 1)], &[], &p).unwrap();
    println!(
        "total effect of sex on income: {te:.3} (the book reports 0.175 from a sample; exactly 0.205)"
    );
    assert!(close(te, 0.205));
    for z in 0..2u8 {
        for m in 0..2u8 {
            let given: [Event<'_>; 2] = [("Z", &[], z), ("M", &[], m)];
            let te_zm = m_id(&cgma, &[("Y", &[("X", 1)], 1)], &given, &p).unwrap()
                - m_id(&cgma, &[("Y", &[("X", 0)], 1)], &given, &p).unwrap();
            let tv = py[usize::from(z)][1][usize::from(m)] - py[usize::from(z)][0][usize::from(m)];
            println!("  z = {z}, m = {m}: (z, m)-TE = {te_zm:+.3}, naive (z, m)-TV = {tv:+.3}");
        }
    }
    let plain = Cgma::new(&g, &[]).unwrap();
    let refused = m_id(
        &plain,
        &[("Y", &[("X", 1)], 1)],
        &[("Z", &[], 1), ("M", &[], 1)],
        &p,
    );
    println!("without X -> M monotone: {}", refused.unwrap_err());

    // LATE in the IV graph (Ex. 19.34): the compliers' share is the Wald
    // denominator.
    let iv = graph(&[("Z", "X"), ("X", "Y")], &[("X", "Y")]);
    let cg = Cgma::new(&iv, &[("Z", "X", Sign::Positive)]).unwrap();
    let rows = [
        ["0", "0", "0"],
        ["0", "1", "1"],
        ["1", "1", "1"],
        ["1", "0", "0"],
    ];
    let piv = Dataset::from_records(["Z", "X", "Y"], rows)
        .unwrap()
        .with_weights(vec![0.4, 0.1, 0.35, 0.15])
        .unwrap();
    let compliers = m_id(
        &cg,
        &[("X", &[("Z", 0)], 0), ("X", &[("Z", 1)], 1)],
        &[],
        &piv,
    )
    .unwrap();
    println!(
        "IV graph: share of compliers P(X_z0 = 0, X_z1 = 1) = {compliers:.2} = P(x1 | z1) − P(x1 | z0)"
    );
    assert!(close(compliers, 0.5));
}

fn main() {
    part1();
    part2();
}
