//! Chapter 17: multi-domain structural learning.
//!
//! Walks through the chapter, asserting each result:
//!
//! 1. Augmented selection diagrams (Exs. 17.12-17.15): which F-nodes and
//!    S-node edges each mix of observational and interventional data
//!    yields.
//! 2. MD-Markov invariances and equivalence (Exs. 17.16, 17.17, 17.20), and
//!    the duality of domains and interventions with observations alone
//!    (Thm 17.4.1).
//! 3. MD-PC (Ex. 17.23): the causal chain and what differs between two
//!    hospitals, learned from observations in one and trials in the other.
//! 4. Why domains are not interventions (Ex. 17.27): pooling two domains'
//!    observations (MDIX) makes I-FCI orient an edge backwards, while MD-FCI
//!    gets it right, both from exact tables.
//!
//! The figures give no edge lists; the diagrams are reconstructed from the
//! worked examples. The book's companion Python has no chapter 17 code.
//! Run with `cargo run -p why-data --example chapter17_multidomain`.

use why_data::discovery::{
    AugmentedOracle, DataTests, Invariance, Regime, SelectionOracle, augment_selection, i_fci,
    md_fci, md_markov_equivalent, md_pc,
};
use why_data::fairness::Dataset;
use why_data::graph::dseparation::Admg;
use why_data::scm::{Dist, Model, Value, WorldId, expr};

fn chain() -> Admg {
    let mut g = Admg::new();
    g.directed("X", "Y");
    g.directed("Y", "Z");
    g
}

fn part1() {
    println!("== 1. Augmented selection diagrams (Exs. 17.12-17.15) ==");
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["X"])];
    let cases: [(&str, Vec<Regime<'_>>); 4] = [
        ("⟨{}¹, {}²⟩", vec![(0, &[], true), (1, &[], true)]),
        ("⟨{}¹, {Y}²⟩", vec![(0, &[], true), (1, &["Y"], true)]),
        ("⟨{Z}¹, {Y}²⟩", vec![(0, &["Z"], true), (1, &["Y"], true)]),
        (
            "⟨{}¹, {Y}², {Z}²⟩",
            vec![(0, &[], true), (1, &["Y"], true), (1, &["Z"], true)],
        ),
    ];
    for (name, setup) in cases {
        let (_, f, sn) = augment_selection(&chain(), &s, &setup).unwrap();
        let fs: Vec<String> = f
            .iter()
            .map(|n| format!("{} -> {{{}}}", n.name, n.targets.join(", ")))
            .collect();
        println!(
            "{name:<20} F-nodes: [{}]; S12 -> {{{}}}",
            fs.join("; "),
            sn[0].targets.join(", ")
        );
    }
}

fn part2() {
    println!("\n== 2. Invariances and equivalence (Exs. 17.16-17.20, Thm 17.4.1) ==");
    let g = chain();
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["Y"])];
    let setup: [(usize, &[&str], bool); 3] = [(0, &[], true), (0, &["Z"], true), (1, &[], true)];
    let mut oracle = SelectionOracle::new(g.clone(), &s, &setup).unwrap();
    println!(
        "P¹(Y) = P¹_Z(Y): {}; P¹_Z(Y | X, Z) = P²(Y | X, Z): {}",
        oracle.invariant(0, 1, "Y", &[]).unwrap(),
        oracle.invariant(1, 2, "Y", &["X", "Z"]).unwrap()
    );
    let none: [(usize, usize, &[&str]); 0] = [];
    let eq = md_markov_equivalent(&g, &s, &setup, &g, &none, &setup).unwrap();
    println!("with and without S12 -> Y MD-Markov equivalent (Ex. 17.20): {eq}");
    assert!(!eq);
    // Thm 17.4.1: two domains' observations behave like ⟨∅, {Y}⟩.
    let mut md = SelectionOracle::new(g.clone(), &s, &[(0, &[], true), (1, &[], true)]).unwrap();
    let mut psi = AugmentedOracle::new(g, &[&[], &["Y"]]).unwrap();
    let mut agree = true;
    for (y, w) in [
        ("X", vec![]),
        ("Y", vec!["X"]),
        ("Z", vec!["Y"]),
        ("Z", vec![]),
    ] {
        agree &= md.invariant(0, 1, y, &w).unwrap() == psi.invariant(0, 1, y, &w).unwrap();
    }
    println!("observational domains agree with interventions on the S-node's children: {agree}");
    assert!(agree);
}

fn part3() {
    println!("\n== 3. MD-PC (Ex. 17.23) ==");
    let mut g = Admg::new();
    g.directed("X", "Z");
    g.directed("Z", "Y");
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["X"])];
    let setup: [(usize, &[&str], bool); 3] = [(0, &[], true), (1, &["Y"], true), (1, &["Z"], true)];
    let mut oracle = SelectionOracle::new(g, &s, &setup).unwrap();
    let pdag = md_pc(&["X", "Y", "Z"], &setup, &mut oracle).unwrap();
    println!("observations in hospital 1, trials on Y and Z in hospital 2:\n  {pdag}");
    assert!(pdag.edges().contains(&"S12 --> X".to_string()));
}

fn table(m: &Model, w: WorldId, vars: &[&str]) -> Dataset {
    let t = m.world(w).probability_table(Some(vars)).unwrap();
    let rows: Vec<Vec<String>> = t
        .rows
        .iter()
        .map(|(r, _)| r.iter().map(|x: &Value| x.as_int().to_string()).collect())
        .collect();
    let weights = t.rows.iter().map(|(_, p)| *p).collect();
    Dataset::from_records(vars.iter().copied(), rows)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

fn part4() {
    println!("\n== 4. Pooling domains is unsound (Ex. 17.27) ==");
    let v = expr::var;
    let model = |p_x: f64| {
        Model::new(
            [
                ("X".to_string(), v("UX")),
                (
                    "Y".to_string(),
                    expr::or([expr::and([v("X"), v("UY1")]), v("UY2")]),
                ),
            ],
            [
                ("UX".to_string(), Dist::Bernoulli(p_x)),
                ("UY1".to_string(), Dist::Bernoulli(0.7)),
                ("UY2".to_string(), Dist::Bernoulli(0.2)),
            ],
        )
        .unwrap()
    };
    let names = ["X", "Y"];
    // The lab (domain 1) and the hospital (domain 2) differ in X's mechanism.
    let lab = model(0.3);
    let mut trial = model(0.3);
    let soft = trial
        .intervene_with(&[(
            "Y",
            expr::or([expr::and([v("X"), v("UY1")]), expr::not(v("UY2"))]),
        )])
        .unwrap();
    let hospital = model(0.7);
    let p1 = table(&lab, lab.observational(), &names);
    let p1y = table(&trial, soft, &names);
    let p2 = table(&hospital, hospital.observational(), &names);
    let rows: Vec<Vec<String>> = [&p1, &p2]
        .iter()
        .flat_map(|d| {
            (0..d.len()).map(|r| {
                names
                    .iter()
                    .map(|n| d.value(n, r).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let weights: Vec<f64> = [&p1, &p2]
        .iter()
        .flat_map(|d| d.weights().unwrap().iter().map(|w| w / 2.0))
        .collect();
    let pooled = Dataset::from_records(names, rows)
        .unwrap()
        .with_weights(weights)
        .unwrap();
    let wrong = i_fci(
        &names,
        &[&[], &["Y"]],
        &mut DataTests::exact(&[&pooled, &p1y], 1e-9),
    )
    .unwrap();
    println!("I-FCI on the pooled observations and the lab trial: {wrong}");
    let setup: [(usize, &[&str], bool); 3] = [(0, &[], true), (0, &["Y"], true), (1, &[], true)];
    let right = md_fci(
        &names,
        &setup,
        &mut DataTests::exact(&[&p1, &p1y, &p2], 1e-9),
    )
    .unwrap();
    println!("MD-FCI with the domains kept apart:              {right}");
    assert!(wrong.edges().contains(&"X <-- Y".to_string()));
    assert!(right.edges().contains(&"X --> Y".to_string()));
}

fn main() {
    part1();
    part2();
    part3();
    part4();
}
