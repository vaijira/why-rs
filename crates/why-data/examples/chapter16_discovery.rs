//! Chapter 16: structural learning.
//!
//! Walks through the chapter, asserting each result:
//!
//! 1. MAGs (Alg. 53, Ex. 16.4) and PAGs: FCI learns Fig. 16.5(d) from the
//!    independences of Fig. 16.5(a) (Ex. 16.7), with B -> C visible and
//!    D -> C not (Lemma 16.1.4).
//! 2. Known interventions: I-FCI on Ex. 16.14 learns that X causes W and Y
//!    but not Z, and that X and Y are confounded. FCI cannot learn this.
//! 3. Unknown interventions: Ψ-FCI on Ex. 16.24, and Fig. 16.16's contrast
//!    between known and unknown targets (Ex. 16.15).
//! 4. Learning from samples: FCI with G-tests on simulated data.
//!
//! The figures give no edge lists; the diagrams are reconstructed from the
//! worked examples. The book's companion Python has no chapter 16 code.
//! Run with `cargo run -p why-data --example chapter16_discovery`.

use why_data::discovery::{
    AugmentedOracle, DataTests, GraphOracle, Invariance, fci, i_fci, i_markov_equivalent, mag,
    psi_fci, psi_markov_equivalent,
};
use why_data::fairness::Dataset;
use why_data::graph::dseparation::Admg;
use why_data::scm::{Dist, Model, expr};

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

fn part1() {
    println!("== 1. MAGs and PAGs (Exs. 16.4, 16.7) ==");
    let g = graph(&[("A", "B"), ("B", "C"), ("C", "D")], &[("B", "D")]);
    let m = mag(&g, &["A", "B", "C", "D"]).unwrap();
    println!("Ex. 16.4, MAG of A -> B -> C -> D with B <-> D: {m}");
    assert_eq!(
        m.edges(),
        ["A --> B", "A --> D", "B --> C", "B --> D", "C --> D"]
    );

    let g = graph(&[("A", "B"), ("D", "B"), ("B", "C")], &[("D", "C")]);
    let names = ["A", "B", "C", "D"];
    println!("Fig. 16.5(a): {}", mag(&g, &names).unwrap());
    let pag = fci(&names, &mut GraphOracle::new(g)).unwrap();
    println!(
        "FCI (Ex. 16.7): {pag}; B -> C visible: {}, D -> C visible: {}",
        pag.is_visible("B", "C").unwrap(),
        pag.is_visible("D", "C").unwrap()
    );
    assert_eq!(pag.edges(), ["A o-> B", "B --> C", "B <-o D", "C <-- D"]);
}

fn part2() {
    println!("\n== 2. Known interventions: I-FCI (Ex. 16.14) ==");
    let g = graph(&[("X", "Y"), ("Z", "Y"), ("X", "W")], &[("X", "Y")]);
    let names = ["W", "X", "Y", "Z"];
    let targets: [&[&str]; 2] = [&[], &["X"]];
    let mut oracle = AugmentedOracle::new(g.clone(), &targets).unwrap();
    println!(
        "invariances: P(Z) = P_X(Z) {}, P(W | X) = P_X(W | X) {}, P(Y | X) = P_X(Y | X) {}",
        oracle.invariant(0, 1, "Z", &[]).unwrap(),
        oracle.invariant(0, 1, "W", &["X"]).unwrap(),
        oracle.invariant(0, 1, "Y", &["X"]).unwrap()
    );
    let ipag = i_fci(&names, &targets, &mut oracle).unwrap();
    let pag = fci(&names, &mut GraphOracle::new(g)).unwrap();
    println!("I-FCI: {ipag}");
    println!("FCI from P(V) alone: {pag}");
    assert_eq!(
        ipag.edges(),
        ["F1 --> X", "F1 --> Y", "W <-- X", "X --> Y", "Y <-o Z"]
    );
}

fn part3() {
    println!("\n== 3. Unknown interventions: Ψ-FCI (Ex. 16.24, Fig. 16.16) ==");
    let g = graph(
        &[("Z", "X"), ("X", "Y"), ("Z", "Y"), ("Y", "W")],
        &[("X", "W")],
    );
    let mut oracle = AugmentedOracle::new(g, &[&[], &["X"]]).unwrap();
    let pag = psi_fci(&["W", "X", "Y", "Z"], &mut oracle).unwrap();
    println!("Ψ-FCI on Ex. 16.24 (targets hidden from the learner): {pag}");
    assert!(pag.edges().contains(&"Y <-- Z".to_string()));

    let g1 = graph(&[("X", "Y")], &[("X", "Y")]);
    let g2 = graph(&[("Y", "X")], &[]);
    let psi = psi_markov_equivalent(&g1, &[&[], &["X"]], &g2, &[&[], &["X", "Y"]]).unwrap();
    let known = i_markov_equivalent(&g1, &g2, &[&[], &["X"]]).unwrap();
    println!(
        "Ex. 16.15: X -> Y, X <-> Y under ⟨∅, X⟩ vs Y -> X under ⟨∅, XY⟩: Ψ-equivalent {psi}; with the targets known to be ⟨∅, X⟩, I-equivalent {known}"
    );
    assert!(psi && !known);
    let targets: [&[&str]; 2] = [&[], &["X"]];
    let mut oracle = AugmentedOracle::new(g1.clone(), &targets).unwrap();
    let names = ["X", "Y"];
    println!(
        "Fig. 16.16: Ψ-FCI {}; I-FCI {}; FCI {}",
        psi_fci(&names, &mut oracle).unwrap(),
        i_fci(&names, &targets, &mut oracle).unwrap(),
        fci(&names, &mut GraphOracle::new(g1)).unwrap()
    );
}

fn part4() {
    println!("\n== 4. Learning from samples ==");
    let v = expr::var;
    let m = Model::new(
        [
            ("A".to_string(), v("UA")),
            ("D".to_string(), expr::or([v("L"), v("UD")])),
            (
                "B".to_string(),
                expr::or([expr::and([v("A"), v("D")]), v("UB")]),
            ),
            (
                "C".to_string(),
                expr::or([expr::and([v("B"), v("UC1")]), expr::and([v("L"), v("UC2")])]),
            ),
        ],
        [
            ("UA".to_string(), Dist::Bernoulli(0.3)),
            ("L".to_string(), Dist::Bernoulli(0.4)),
            ("UD".to_string(), Dist::Bernoulli(0.2)),
            ("UB".to_string(), Dist::Bernoulli(0.15)),
            ("UC1".to_string(), Dist::Bernoulli(0.8)),
            ("UC2".to_string(), Dist::Bernoulli(0.5)),
        ],
    )
    .unwrap();
    let names = ["A", "B", "C", "D"];
    let all = m.world(m.observational()).world().variables().to_vec();
    let at: Vec<usize> = names
        .iter()
        .map(|n| all.iter().position(|a| a == n).unwrap())
        .collect();
    let mut state = 16u64;
    let mut uniform = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    for n in [500, 5_000, 20_000] {
        let rows: Vec<Vec<String>> = m
            .world(m.observational())
            .sample(n, &mut uniform)
            .unwrap()
            .into_iter()
            .map(|r| at.iter().map(|&i| r[i].as_int().to_string()).collect())
            .collect();
        let data = Dataset::from_records(names, rows).unwrap();
        let pag = fci(&names, &mut DataTests::g_test(&[&data], 0.01)).unwrap();
        println!("FCI with G-tests (α = 0.01) on {n:>6} samples of Fig. 16.5's model: {pag}");
        if n == 20_000 {
            assert_eq!(pag.edges(), ["A o-> B", "B --> C", "B <-o D", "C <-- D"]);
        }
    }
}

fn main() {
    part1();
    part2();
    part3();
    part4();
}
