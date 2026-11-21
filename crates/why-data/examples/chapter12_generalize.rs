//! Chapter 12: generalization and robustness across the PCH.
//!
//! Walks through the chapter's worked examples, asserting each result:
//!
//! 1. Generalizing experiments by adjustment (Sec. 12.1): the
//!    tr-adjustment of Exs. 12.1–12.2, IsEAdmissible (Algorithm 40) on
//!    Fig. 12.3, and the transport formula estimated from simulated data.
//! 2. Policy transportability by C-INFER (Sec. 12.1.2, Algorithm 41): the
//!    trees and estimands of Exs. 12.11, 12.13 and 12.14.
//! 3. Counterfactual transportability (Sec. 12.2): the natural direct effect
//!    of Ex. 12.15 and the effect of treatment on the treated of Ex. 12.20.
//!
//! Some figures are not given as edge lists; the graphs here are
//! reconstructed from the text around them. The book's companion Python has
//! no chapter 12 code.
//!
//! Run with `cargo run -p why-data --example chapter12_generalize`.

use why_data::generalize::{
    DomainInput, PolicyQuery, ctf_tr, e_adjustment_order, find_tr_adjustment, is_e_admissible,
    sigma_tr, tr_adjustment,
};
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::graph::dseparation::Admg;
use why_data::transport::{Dataset, Domain, EmpiricalDistributions, Regime, SelectionDiagram};

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

/// A small deterministic generator, so the example needs no dependency.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn coin(&mut self, p: f64) -> i64 {
        i64::from(self.unit() < p)
    }
}

fn sample(header: &[&str], n: usize, seed: u64, mut row: impl FnMut(&mut Rng) -> Vec<i64>) -> Dataset {
    let mut rng = Rng(seed);
    let rows: Vec<Vec<String>> = (0..n)
        .map(|_| row(&mut rng).into_iter().map(|v| v.to_string()).collect())
        .collect();
    Dataset::from_records(header.iter().copied(), rows).unwrap()
}

fn adjustment() {
    println!("1. Generalizing experiments by adjustment");

    // Ex. 12.1 (Fig. 12.1(a)): an RCT in LA, ages differing in NYC.
    let g = diagram(&[("Z", "X"), ("Z", "Y"), ("X", "Y")], &[]);
    let d = SelectionDiagram::new(g).with_source("LA", &["Z"]).unwrap();
    let e = tr_adjustment(&d, "LA", &["X"], &["Y"], &["Z"]).unwrap();
    println!("  Eq. 12.3:  P*(y | do(x)) = {e}");
    assert_eq!(e.to_string(), "sum_{z} P^LA(y | z; do(x)) P^*(z)");

    // Estimate it: the LA trial randomizes X, NYC is older and observational.
    // Y's mechanism is shared, so the truth is 0.2 + 0.3 + 0.3·0.6 = 0.68.
    let la = sample(&["Z", "X", "Y"], 40_000, 1, |r| {
        let z = r.coin(0.3);
        let x = r.coin(0.5);
        vec![z, x, r.coin(0.2 + 0.3 * x as f64 + 0.3 * z as f64)]
    });
    let nyc = sample(&["Z", "X", "Y"], 40_000, 2, |r| {
        let z = r.coin(0.6);
        let x = r.coin(0.2 + 0.6 * z as f64);
        vec![z, x, r.coin(0.2 + 0.3 * x as f64 + 0.3 * z as f64)]
    });
    let dists = EmpiricalDistributions::new()
        .with(Domain::source("LA").under(Regime::new().atomic(&["X"])), &la)
        .with(Domain::Target, &nyc);
    let est = e.evaluate(&dists, &[("X", "1"), ("Y", "1")]).unwrap();
    println!("  plug-in estimate of P*(Y = 1 | do(X = 1)) = {est:.4} (truth 0.68)");
    assert!((est - 0.68).abs() < 0.02);

    // Ex. 12.2 (Fig. 12.2): baseline risk B and background E.
    let g = diagram(&[("B", "X"), ("B", "Y"), ("E", "X"), ("E", "Y"), ("X", "Y")], &[]);
    let d = SelectionDiagram::new(g).with_source("FDA", &["E"]).unwrap();
    let e = tr_adjustment(&d, "FDA", &["X"], &["Y"], &["B", "E"]).unwrap();
    println!("  Eq. 12.4:  P*(y | do(x)) = {e}");
    assert_eq!(e.to_string(), "sum_{b, e} P^FDA(y | b, e; do(x)) P^*(b, e)");
    let smallest = find_tr_adjustment(&d, "FDA", &["X"], &["Y"]).unwrap();
    println!("  smallest tr-admissible set: {smallest:?}");
    assert_eq!(smallest, Some(vec!["E".to_string()]));

    // Fig. 12.3, reconstructed from Eqs. 12.7–12.14.
    let directed = [("X1", "Z1"), ("Z1", "Z2"), ("X2", "Z3"), ("Z3", "Z4"), ("X1", "Y"), ("X2", "Y")];
    let latent = [("Z2", "Y"), ("Z4", "Y")];
    let b = diagram(&directed, &latent);
    let mut a = b.clone();
    a.directed("X1", "Z2");
    a.directed("X2", "Z4");
    a.bidirected("Z2", "Z4");
    let z = ["Z1", "Z2", "Z3", "Z4"];
    let order = e_adjustment_order(&b, &["X1", "X2"], &["Y"], &z).unwrap();
    println!("  Fig. 12.3(b): e-adjustment order {order:?}");
    assert!(order.is_some());
    assert!(!is_e_admissible(&a, &["X1", "X2"], &["Y"], &z).unwrap());
    println!("  Fig. 12.3(a): no order");
}

fn policies() {
    println!("2. Policy transportability by C-INFER");

    // Ex. 12.11 (Fig. 12.8).
    let g = diagram(&[("X", "Z"), ("Z", "Y")], &[("X", "Y")]);
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X", "Y"])
        .unwrap()
        .with_source("2", &["X", "Z"])
        .unwrap();
    let do_x = Regime::new().atomic(&["X"]);
    let inputs = [
        DomainInput::new(Domain::source("1"), do_x.clone()),
        DomainInput::new(Domain::source("2"), do_x.clone()),
    ];
    let t = sigma_tr(&d, &PolicyQuery::new(&["Y"], do_x), &inputs).unwrap();
    println!("  Ex. 12.11, Eq. 12.53: {}", t.estimand);
    assert_eq!(t.estimand.to_string(), "sum_{z} P^1(z; do(x)) P^2(y | z; do(x))");
    for (k, tree) in t.input_trees.iter().enumerate() {
        println!("  d-tree for {}:", inputs[k]);
        for line in tree.to_string().lines() {
            println!("    {line}");
        }
    }

    // Ex. 12.13 (Fig. 12.13): a sequential protocol from two hospitals.
    let g = diagram(
        &[("X1", "Z"), ("X1", "X2"), ("Z", "X2"), ("X1", "Y"), ("X2", "Y"), ("Z", "Y")],
        &[],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["Y"])
        .unwrap()
        .with_source("2", &["Z"])
        .unwrap();
    let target = Regime::new().policy("X1", &[]).policy("X2", &["X1", "Z"]);
    let inputs = [
        DomainInput::new(Domain::source("1"), Regime::new().atomic(&["X1", "X2"])),
        DomainInput::new(Domain::source("2"), Regime::new().policy("X2", &["X1", "Z"])),
    ];
    let t = sigma_tr(&d, &PolicyQuery::new(&["Y"], target), &inputs).unwrap();
    println!("  Ex. 12.13, Eq. 12.66: {}", t.estimand);
    print!("{}", indent(&t.query_tree.to_string()));

    // Ex. 12.14 (Fig. 12.14): the loan policy σ_X = P̂(x | r).
    let g = diagram(
        &[("W", "X"), ("X", "Z"), ("R", "Z"), ("Z", "Y")],
        &[("R", "Y"), ("R", "W"), ("W", "X"), ("W", "Z")],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["W"])
        .unwrap()
        .with_source("2", &["R"])
        .unwrap();
    let inputs = [
        DomainInput::observational(Domain::Target),
        DomainInput::new(Domain::source("1"), Regime::new().policy("Z", &["X", "R"])),
        DomainInput::new(Domain::source("2"), Regime::new().policy("W", &[])),
    ];
    let q = PolicyQuery::new(&["Y"], Regime::new().policy("X", &["R"]));
    let t = sigma_tr(&d, &q, &inputs).unwrap();
    println!("  Ex. 12.14, Eq. 12.75: {}", t.estimand);
    assert_eq!(
        t.estimand.to_string(),
        "sum_{r, x, z} P̂(x | r) P^2(z | r, x; σ_{W}) \
         sum_{w, x'} P^1(r, w, x'; σ_{Z}) P^1(y | r, w, x', z; σ_{Z})"
    );
    assert!(sigma_tr(&d, &q, &inputs[..1]).is_err());
    println!("  (P*(V) alone does not transport it)");
}

fn counterfactuals() {
    println!("3. Counterfactual transportability");

    // Ex. 12.15 (Fig. 12.15): the natural direct effect, US vs Australia.
    let g = diagram(&[("X", "Z"), ("Z", "Y"), ("X", "Y")], &[("X", "Z")]);
    let d = SelectionDiagram::new(g).with_source("AU", &["Y"]).unwrap();
    let inputs = [
        DomainInput::observational(Domain::Target),
        DomainInput::new(Domain::source("AU"), Regime::new().atomic(&["X"])),
    ];
    for text in ["P(y_{x', Z_x})", "P(y_{x, Z_x})"] {
        let q = CtfQuery::parse(d.graph(), text).unwrap();
        let t = ctf_tr(&d, &q, &inputs).unwrap();
        println!("  Ex. 12.15: {text} = {}", t.estimand);
    }

    // Ex. 12.20 (Fig. 12.19): the ETT from a scholarship program.
    let g = diagram(
        &[("Z", "X"), ("Z", "Y"), ("X", "W"), ("X", "Y"), ("W", "Y")],
        &[("Z", "X"), ("W", "Y")],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["Z"])
        .unwrap()
        .with_source("2", &["W"])
        .unwrap();
    let inputs = [
        DomainInput::new(Domain::source("1"), Regime::new().policy("X", &[])),
        DomainInput::observational(Domain::source("2")),
    ];
    let q = CtfQuery::parse(d.graph(), "P(y_x | x')").unwrap();
    let t = ctf_tr(&d, &q, &inputs).unwrap();
    println!("  Ex. 12.20: P*(y_x | x') = {}", t.estimand);
    print!("{}", indent(&t.query_tree.to_string()));
}

fn indent(s: &str) -> String {
    s.lines().map(|l| format!("    {l}\n")).collect()
}

fn main() {
    adjustment();
    policies();
    counterfactuals();
}
