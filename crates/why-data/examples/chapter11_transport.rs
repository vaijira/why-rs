//! Chapter 11: generalizability and transportability of statistical
//! quantities.
//!
//! Walks through the chapter's worked examples, asserting each result:
//!
//! 1. Direct transportability (Sec. 11.2.1): ListDTRDist on Ex. 11.5, and
//!    the `3^k` separators of Fig. 11.8.
//! 2. Transportable feature sets (Algorithm 33, Score-TR): the maximal sets
//!    of Exs. 11.11–11.12.
//! 3. Representations (Sec. 11.2.2): the determined, constrained and free
//!    features of Exs. 11.18 and 11.20.
//! 4. Partial observability (Sec. 11.3): the transport formulas of
//!    Exs. 11.31–11.34, evaluated as plug-in estimates from data.
//! 5. Data-driven invariance (Sec. 11.2.3) and domain adaptation
//!    (Sec. 11.4): Algorithms 34–37 on simulated data.
//!
//! Some figures are not given as edge lists; the graphs here are
//! reconstructed from the text around them. The book's companion Python has
//! no chapter 11 code.
//!
//! Run with `cargo run -p why-data --example chapter11_transport`.

use why_data::graph::dseparation::Admg;
use why_data::transport::{
    Dataset, Domain, EmpiricalDistributions, Estimand, Representation, SelectionDiagram, list_dtr_dist,
    multi_cause_agnostic_da, multi_cause_da, score_tr, source_invariance,
    statistical_transport, structure_agnostic_da, structure_informed_da,
    transportable_feature_sets,
};

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
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u64) -> i64 {
        (self.next() % n) as i64
    }
}

fn sample(header: &[&str], n: usize, seed: u64, mut row: impl FnMut(&mut Rng) -> Vec<i64>) -> Dataset {
    let mut rng = Rng(seed);
    let rows: Vec<Vec<String>> = (0..n)
        .map(|_| row(&mut rng).into_iter().map(|v| v.to_string()).collect())
        .collect();
    Dataset::from_records(header.iter().copied(), rows).unwrap()
}

fn direct_transportability() {
    println!("1. Direct transportability");

    // Ex. 11.5 (Fig. 11.15(b)): only P(Y | X, Z) and P(Y | X) transport.
    let g = diagram(&[("X", "Y"), ("Z", "Y"), ("Y", "R")], &[]);
    let d = SelectionDiagram::new(g).with_source("1", &["X", "R"]).unwrap();
    let found = list_dtr_dist(&d, "1", "Y", &["X", "Z", "R"]).unwrap();
    for x in &found {
        println!("  P*(Y | {}) = P(Y | {})", x.join(", "), x.join(", "));
    }
    assert_eq!(found.len(), 2);

    // Fig. 11.8 with k = 4: 3^4 separators.
    let mut g = Admg::new();
    let mut first = Vec::new();
    let mut all = Vec::new();
    for i in 1..=4 {
        let (a, b) = (format!("X{i}1"), format!("X{i}2"));
        g.directed(&a, &b);
        g.directed(&b, "Y");
        first.push(a.clone());
        all.push(a);
        all.push(b);
    }
    fn refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    let d = SelectionDiagram::new(g).with_source("1", &refs(&first)).unwrap();
    let n = list_dtr_dist(&d, "1", "Y", &refs(&all)).unwrap().len();
    println!("  Fig. 11.8, k = 4: {n} directly transportable sets (3^4 = 81)");
    assert_eq!(n, 81);
}

fn feature_sets() {
    println!("2. Transportable feature sets (Score-TR)");
    // Ex. 11.12 (Fig. 11.13): two overlaid copies of Fig. 11.12(a).
    let g = diagram(
        &[("X1", "Y"), ("Y", "X2"), ("X3", "X4"), ("X5", "Y"), ("Y", "X6"), ("X7", "X8")],
        &[("X3", "X2"), ("X7", "X6")],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X1", "X3", "X5", "X7"])
        .unwrap();
    let features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"];
    let sets = transportable_feature_sets(&d, &features, "Y").unwrap();
    // Each copy of Fig. 11.12(a) has five transportable sets, so the overlay
    // has 25. Fig. 11.14's caption reports 15 for the book's figure, whose
    // edges the text does not give; the four maximal sets match Ex. 11.12.
    println!(
        "  {} of 256 feature sets are transportable; the maximal ones:",
        sets.transportable.len()
    );
    for m in &sets.maximal {
        println!("    {{{}}}", m.join(", "));
    }
    assert_eq!(sets.maximal.len(), 4);
    // The full set is not: S → X3 ↔ X2 ← Y opens once X2 and X4 are given.
    assert!(score_tr(&d, &features, "Y").is_err());
}

fn representations() {
    println!("3. Representations");
    // Ex. 11.18: R = ⟨β·X1:3, X1 + … + X6⟩ with an injective integer β.
    let names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"];
    let phi = |x: &[i64]| (x[0] + 2 * x[1] + 4 * x[2], x[..6].iter().sum::<i64>());
    let rep = Representation::classify(&names, &vec![vec![0, 1]; 9], phi).unwrap();
    println!(
        "  Ex. 11.18: det = {:?}, cons = {:?}, free = {:?}",
        rep.determined(),
        rep.constrained(),
        rep.free()
    );
    assert_eq!(rep.free(), ["X7", "X8", "X9"]);

    // Ex. 11.20: R = ⟨X1X2X3/X4, X1X2/(X3X4), X2X3/(X1X4)⟩ over {1, 2, 3}.
    let reduce = |n: i64, d: i64| {
        let (mut a, mut b) = (n, d);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        (n / a, d / a)
    };
    let phi = |x: &[i64]| {
        (
            reduce(x[0] * x[1] * x[2], x[3]),
            reduce(x[0] * x[1], x[2] * x[3]),
            reduce(x[1] * x[2], x[0] * x[3]),
        )
    };
    let names = ["X1", "X2", "X3", "X4"];
    let rep = Representation::classify(&names, &vec![vec![1, 2, 3]; 4], phi).unwrap();
    println!(
        "  Ex. 11.20: det = {:?}, cons = {:?} (R† = X2 / X4)",
        rep.determined(),
        rep.constrained()
    );
    assert_eq!(rep.determined(), ["X1", "X3"]);

    let g = diagram(&[("X1", "Y"), ("X3", "Y"), ("Y", "X2")], &[("X4", "Y")]);
    let d = SelectionDiagram::new(g)
        .with_source("1", &["X4"])
        .unwrap()
        .with_source("2", &["X2"])
        .unwrap();
    let aug = d.augment("R+", &rep.constrained()).unwrap();
    let s = score_tr(&aug, &["X1", "X3", "R+"], "Y").unwrap();
    println!("  l_φ(r) = P*(y | x1, x3, r†) = {}", s.estimand);
}

fn partial_observability() {
    println!("4. Partial observability");
    // Ex. 11.31: X → Z → R → Y with R differing; the target measures Z, R.
    let g = diagram(&[("X", "Z"), ("Z", "R"), ("R", "Y")], &[]);
    let d = SelectionDiagram::new(g).with_source("1", &["R"]).unwrap();
    let e = statistical_transport(&d, &["X"], "Y", &["Z", "R"]).unwrap();
    println!("  Ex. 11.31: P*(y | x) = {e}");

    // Estimate it from data: binary chains, with R's mechanism shifted.
    let chain = |p_r: f64, seed: u64, n: usize| {
        sample(&["X", "Z", "R", "Y"], n, seed, |r| {
            let x = i64::from(r.unit() < 0.5);
            let z = x ^ i64::from(r.unit() < 0.2);
            let rr = z ^ i64::from(r.unit() < p_r);
            let y = rr ^ i64::from(r.unit() < 0.1);
            vec![x, z, rr, y]
        })
    };
    let source = chain(0.1, 1, 50_000);
    let target = chain(0.4, 2, 50_000);
    let target_w = Dataset::from_records(
        ["Z", "R"],
        (0..target.len()).map(|i| [target.value("Z", i).unwrap(), target.value("R", i).unwrap()]),
    )
    .unwrap();
    let dists = EmpiricalDistributions::new()
        .with(Domain::source("1"), &source)
        .with(Domain::Target, &target_w);
    let at = [("X", "1"), ("Y", "1")];
    let plug_in = e.evaluate(&dists, &at).unwrap();
    let naive = Estimand::p(Domain::source("1"), &["Y"], &["X"])
        .evaluate(&dists, &at)
        .unwrap();
    // The truth from the mechanisms: Y = X ⊕ e_Z ⊕ e_R ⊕ e_Y, so
    // P*(Y = 1 | X = 1) is the chance that the three flips cancel.
    let flip = |a: f64, b: f64| a * (1.0 - b) + b * (1.0 - a);
    let truth = 1.0 - flip(flip(0.2, 0.4), 0.1);
    println!("    P*(Y=1 | X=1): plug-in {plug_in:.4}, truth {truth:.4}, source's P(Y=1 | X=1) {naive:.4}");
    assert!((plug_in - truth).abs() < 0.02);
    assert!((naive - truth).abs() > 0.1);

    // Ex. 11.34: two sources, Δ¹ = {Z, A}, Δ² = {X2, Z, Y}; the target
    // measures Z and X2.
    let g = diagram(
        &[("X2", "Z"), ("Z", "R"), ("R", "X1"), ("X2", "X1"), ("X1", "Y"), ("A", "Y")],
        &[("X1", "A")],
    );
    let d = SelectionDiagram::new(g)
        .with_source("1", &["Z", "A"])
        .unwrap()
        .with_source("2", &["X2", "Z", "Y"])
        .unwrap();
    let e = statistical_transport(&d, &["X1", "X2"], "Y", &["Z", "X2"]).unwrap();
    println!("  Ex. 11.34: P*(y | x1, x2) = {e}");
    let e = statistical_transport(&d, &["X1", "X2"], "Y", &["R", "X2"]).unwrap();
    println!("  Ex. 11.34 with W = {{R, X2}}: {e}");
}

fn data_driven_and_adaptation() {
    println!("5. Data-driven invariance and domain adaptation");
    // Two sources: Y ← X1 ⊕ U shared, X2 ← Y ⊕ U2 with a shifted U2.
    let source = |p_x1: f64, p_u2: f64, seed: u64| {
        sample(&["X1", "X2", "Y"], 20_000, seed, |r| {
            let x1 = i64::from(r.unit() < p_x1);
            let y = x1 ^ i64::from(r.unit() < 0.2);
            vec![x1, y ^ i64::from(r.unit() < p_u2), y]
        })
    };
    let (s1, s2) = (source(0.3, 0.1, 3), source(0.6, 0.4, 4));
    for f in [&["X1"][..], &["X1", "X2"]] {
        let r = source_invariance(&[&s1, &s2], f, "Y").unwrap();
        println!(
            "  E[Y | {}]: max gap {:.3}, p = {:.3}",
            f.join(", "),
            r.max_gap,
            r.p_value
        );
    }
    assert!(source_invariance(&[&s1, &s2], &["X1"], "Y").unwrap().is_invariant(0.01));
    assert!(!source_invariance(&[&s1, &s2], &["X1", "X2"], "Y").unwrap().is_invariant(0.01));

    // Algorithms 34 and 35: one source shares Y's mechanism, one does not.
    let y_of = |x: i64, r: &mut Rng| i64::from(x >= 5) ^ i64::from(r.unit() > 0.9);
    let target = sample(&["X", "Y"], 120, 5, |r| {
        let x = r.below(10);
        vec![x, y_of(x, r)]
    });
    let same = sample(&["X", "Y"], 5000, 6, |r| {
        let x = r.below(10);
        vec![x, y_of(x, r)]
    });
    let other = sample(&["X", "Y"], 5000, 7, |r| vec![r.below(10), r.below(2)]);
    let (model, j) =
        structure_informed_da(&[&same, &other], &target, &[&[], &["Y"]], &["X"], "Y", 0.5).unwrap();
    println!(
        "  Alg. 34 pools sources {j:?}: P̂*(Y=1 | X=7) = {:.3}",
        model.probability(&["7"], "1")
    );
    let choice = structure_agnostic_da(&[&same, &other], &target, &["X"], "Y", 0.5, 1).unwrap();
    println!("  Alg. 35 selects sources {:?} (validation NLL {:.3})", choice.sources, choice.loss);
    assert_eq!(choice.sources, [0]);

    // Algorithms 36 and 37, Ex. 11.38: noisy subtraction with different
    // parents in source and target.
    let noise = |r: &mut Rng| if r.unit() < 0.91 { 0 } else { 1 + r.below(9) };
    let src = sample(&["X1", "X2", "X3", "Y"], 20_000, 8, |r| {
        let x = [r.below(10), r.below(10), r.below(10)];
        vec![x[0], x[1], x[2], (x[0] - x[1] + noise(r)).rem_euclid(10)]
    });
    let tgt = sample(&["X1", "X2", "X3", "Y"], 400, 9, |r| {
        let x = [r.below(10), r.below(10), r.below(10)];
        vec![x[0], x[1], x[2], (x[2] - x[1] + noise(r)).rem_euclid(10)]
    });
    let (model, _) =
        multi_cause_da(&[&src], &tgt, &[&[]], &[&["X1", "X2"]], &["X3", "X2"], "Y", 0.5).unwrap();
    println!("  Alg. 36: µ(y | x3 = 7, x2 = 2) predicts {}", model.predict(&["7", "2"]));
    assert_eq!(model.predict(&["7", "2"]), "5");
    let choice = multi_cause_agnostic_da(&[&src], &tgt, &["X1", "X2", "X3"], "Y", 0.5, 2).unwrap();
    println!(
        "  Alg. 37: {} candidates (Eq. 11.253), selects target {:?} with source {:?}",
        choice.candidates, choice.target_parents, choice.source_parents
    );
    assert_eq!(choice.candidates, 98);
}

fn main() {
    direct_transportability();
    feature_sets();
    representations();
    partial_observability();
    data_driven_and_adaptation();
}
