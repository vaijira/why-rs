//! Chapter 20: the hierarchy of graphical models and counterfactual
//! realizability.
//!
//! 1. Layers L1 to L3 (Sec. 20.1): Exs. 20.6, 20.12-20.14 and 20.17. The ETT
//!    is in L2.25 and the NDE's unnested term in L2.5, and C-INFER
//!    identifies both from observations.
//! 2. Realizability (Sec. 20.2.2): Algorithm 68 on Exs. 20.19-20.20,
//!    Fig. 20.14 and Experiment 1.
//! 3. Decision-making (Secs. 20.2.3-20.2.4): a model with Experiment 3's
//!    averages, and MAB-OPT (Algorithm 70) against an L2 bandit.
//! 4. Other graphical models (Sec. 20.3): SWIGs (Algorithm 71, Ex. 20.30)
//!    and augmented DAGs (Algorithm 72, Exs. 20.42-20.43).
//!
//! The book's companion Python has no chapter 20 code. Run with
//! `cargo run -p why-data --example chapter20_hierarchy`.

use why_data::graph::cinfer::Input;
use why_data::graph::cinfer::counterfactual::{CtfQuery, ctf_infer};
use why_data::graph::dseparation::Admg;
use why_data::hierarchy::{
    Eci, Layer, MabScm, MabUnit, Term, ag_construct, ctf_realize, fisher_actions, layer,
    layer225_actions, mab_int, mab_opt, maximal_actions, swig,
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

fn show(q: &[Term]) -> String {
    let s: Vec<String> = q.iter().map(ToString::to_string).collect();
    format!("P({})", s.join(", "))
}

fn part1() {
    println!("== 1. Layers of PCH* ==");
    let fork = graph(&[("X", "Z"), ("X", "Y")], &[]);
    let x1 = [("X", 1)];
    let x0 = [("X", 0)];
    let cases: [(&[Term], Layer); 4] = [
        (
            &[Term::natural("X"), Term::new("Y", &x1), Term::new("Z", &x1)],
            Layer::L225,
        ),
        (
            &[Term::natural("X"), Term::new("Y", &x1), Term::new("Z", &x0)],
            Layer::L25,
        ),
        (&[Term::new("Y", &x1), Term::natural("Y")], Layer::L3),
        (&[Term::new("Y", &x1), Term::new("Z", &x1)], Layer::L2),
    ];
    for (q, want) in cases {
        let l = layer(&fork, q).unwrap();
        println!("fork Z <- X -> Y: {} is in {l:?}", show(q));
        assert_eq!(l, want);
    }

    // Ex. 20.17 / Fig. 20.10: the NDE's P(y_{x'z}, z_x) is in L2.5, and
    // C-INFER identifies it.
    let med = graph(&[("X", "Z"), ("Z", "Y"), ("X", "Y")], &[]);
    let nde = [
        Term::new("Y", &[("X", 0), ("Z", 1)]),
        Term::new("Z", &[("X", 1)]),
    ];
    let l = layer(&med, &nde).unwrap();
    assert_eq!(l, Layer::L25);
    let q = CtfQuery::parse(&med, "P(y_{x', z}, z_{x})").unwrap();
    let est = ctf_infer(&med, &q, &[Input::observational()])
        .unwrap()
        .estimand;
    println!(
        "mediation: {} is in {l:?}; P(y_{{x'z}}, z_x) = {est}",
        show(&nde)
    );

    // Ex. 20.14: the ETT in Fig. 20.8(a) is in L2.25.
    let g = graph(&[("Z", "X"), ("X", "Y")], &[]);
    let ett = [Term::new("Y", &x1), Term::natural("X")];
    let l = layer(&g, &ett).unwrap();
    assert_eq!(l, Layer::L225);
    let q = CtfQuery::parse(&g, "P(y_{x}, x')").unwrap();
    let est = ctf_infer(&g, &q, &[Input::observational()])
        .unwrap()
        .estimand;
    println!(
        "Z -> X -> Y: {} is in {l:?}; P(y_x, x') = {est}",
        show(&ett)
    );
}

fn part2() {
    println!("\n== 2. Realizability (Algorithm 68) ==");
    let xy = graph(&[("X", "Y")], &[]);
    let ett = [Term::new("Y", &[("X", 1)]), Term::natural("X")];
    let plan = ctf_realize(&xy, &ett, &maximal_actions(&xy).unwrap()).unwrap();
    let acts: Vec<String> = plan
        .interventions()
        .iter()
        .map(|(a, v)| format!("{a} : {v}"))
        .collect();
    println!("Ex. 20.19: {} with A†: {}", show(&ett), acts.join("; "));
    let err = ctf_realize(&xy, &ett, &fisher_actions(&xy).unwrap()).unwrap_err();
    println!("  with RAND(X) only: {err}");
    let ps = [
        Term::new("Y", &[("X", 1)]),
        Term::natural("X"),
        Term::natural("Y"),
    ];
    let err = ctf_realize(&xy, &ps, &maximal_actions(&xy).unwrap()).unwrap_err();
    println!("Ex. 20.20: {}: {err}", show(&ps));

    let q = [Term::new("Z", &[("X", 1)]), Term::new("W", &[("T", 1)])];
    let g1 = graph(&[("T", "A"), ("A", "Z"), ("A", "W"), ("X", "Z")], &[]);
    let g2 = graph(&[("T", "Z"), ("T", "W"), ("X", "Z")], &[]);
    let r1 = ctf_realize(&g1, &q, &maximal_actions(&g1).unwrap());
    let r2 = ctf_realize(&g2, &q, &maximal_actions(&g2).unwrap()).unwrap();
    let acts: Vec<String> = r2
        .interventions()
        .iter()
        .map(|(a, v)| format!("{a} : {v}"))
        .collect();
    println!("Fig. 20.14: {} in G1: {}", show(&q), r1.unwrap_err());
    println!("  in G2: {}", acts.join("; "));

    let e1 = graph(&[("X", "Y"), ("X", "Z")], &[("Y", "Z")]);
    let q = [Term::new("Y", &[("X", 1)]), Term::new("Z", &[("X", 0)])];
    assert!(ctf_realize(&e1, &q, &maximal_actions(&e1).unwrap()).is_ok());
    let err = ctf_realize(&e1, &q, &layer225_actions(&e1).unwrap()).unwrap_err();
    println!(
        "Experiment 1: {} needs per-child randomization; with one value per variable: {err}",
        show(&q)
    );
}

fn part3() {
    println!("\n== 3. Counterfactual decision-making ==");
    // A model with the averages of Table 20.5. Units with x' = 0 include a
    // 10% subgroup that D_{x''=1} singles out and that only x = 0 rewards.
    let u = |p, x, d: [u8; 2], y: [u8; 2]| MabUnit { p, z: 0, x, d, y };
    let scm = MabScm::new(vec![
        u(0.05, 0, [0, 1], [1, 0]),
        u(0.10, 0, [0, 0], [0, 1]),
        u(0.25, 0, [0, 0], [1, 1]),
        u(0.10, 0, [0, 0], [0, 0]),
        u(0.25, 1, [0, 0], [1, 1]),
        u(0.15, 1, [0, 0], [1, 0]),
        u(0.10, 1, [0, 0], [0, 1]),
    ])
    .unwrap();
    let s = scm.strategies();
    println!(
        "Experiment 3: obs {:.2}, int {:.2}, ETT {:.2}, opt {:.2} (Table 20.5: 0.65, 0.70, 0.75, 0.80)",
        s.obs, s.int, s.ett, s.opt
    );
    assert!((s.opt - 0.80).abs() < 1e-9 && (s.ett - 0.75).abs() < 1e-9);
    let (rounds, seeds) = (6000, 20u64);
    let mean = |f: &dyn Fn(u64) -> f64| (0..seeds).map(f).sum::<f64>() / seeds as f64;
    let opt = mean(&|seed| mab_opt(&scm, rounds, seed).mean(rounds / 2, rounds));
    let int = mean(&|seed| mab_int(&scm, rounds, seed).mean(rounds / 2, rounds));
    println!(
        "Thompson sampling, mean reward over rounds {}..{rounds} ({seeds} runs): MAB-OPT {opt:.3}, L2 {int:.3}",
        rounds / 2
    );
}

fn part4() {
    println!("\n== 4. SWIGs and augmented DAGs ==");
    let chain = graph(&[("X", "Z"), ("Z", "Y")], &[]);
    let s = swig(&chain, &[("X", 1), ("Z", 0)]).unwrap();
    let labels: Vec<String> = ["X", "Z", "Y"]
        .iter()
        .map(|v| s.label(v).unwrap().to_string())
        .collect();
    println!("Fig. 20.28(d): SWIG of X -> Z -> Y for do(x, z) has random nodes {labels:?}");
    let sz = swig(&chain, &[("Z", 1)]).unwrap();
    let q = [Term::natural("Z"), Term::new("Y", &[("Z", 1)])];
    println!(
        "Ex. 20.30: {} passes Prop. 20.3.4: {}",
        show(&q),
        sz.holds(&q, &chain)
    );

    let ecis = [
        Eci::new(&["U", "Z"], &["FX"], &[]),
        Eci::new(&["U"], &["Z"], &["FX"]),
        Eci::new(&["Y"], &["FX"], &["X", "U"]),
        Eci::new(&["Y"], &["Z"], &["X", "U", "FX"]),
    ];
    let g = ag_construct(&["Z", "X", "Y"], &["U"], &[("FX", "X")], &ecis).unwrap();
    println!("Ex. 20.42: AG-CONSTRUCT gives {:?}", g.edges());
    let ecis = [
        Eci::new(&["Z"], &["FX"], &[]),
        Eci::new(&["W"], &["X"], &["Z", "FX"]),
        Eci::new(&["Y"], &["Z"], &["X", "W", "FX"]),
        Eci::new(&["Y"], &["FX"], &["X", "W"]),
        Eci::new(&["W"], &["FX"], &["Z"]),
    ];
    let g = ag_construct(&["X", "W", "Y", "Z"], &[], &[("FX", "X")], &ecis).unwrap();
    println!("Ex. 20.43: {:?}; two augmented DAGs fit Z o-o W", g.edges());
}

fn main() {
    part1();
    part2();
    part3();
    part4();
}
