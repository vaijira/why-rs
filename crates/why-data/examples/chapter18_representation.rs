//! Chapter 18: causal representation learning.
//!
//! Runs CRID (Algorithm 65) on the chapter's examples and on the settings
//! of its experiments (Sec. 18.6), asserting what the book claims:
//!
//! 1. ΔV and ΔQ sets (Exs. 18.14-18.18).
//! 2. Ex. 18.38: observations in two domains, an intervention on V3 and a
//!    marginal one on V2, step by step.
//! 3. Sec. 18.6's chain, collider and non-Markovian settings.
//!
//! Learning the representations themselves (Sec. 18.6's neural proxies and
//! their MCC scores) is not reproduced; CRID is the chapter's algorithm and
//! it is graphical. The book's companion Python has no chapter 18 code. Run
//! with `cargo run -p why-data --example chapter18_representation`.

use why_data::graph::dseparation::Admg;
use why_data::representation::{crid, delta_q, delta_v};

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
    println!("== 1. Changed variables and ΔQ sets (Exs. 18.14-18.18) ==");
    let g = graph(&[("V2", "V1"), ("V2", "V3")], &[]);
    println!(
        "V1 <- V2 -> V3, σ(V3) vs observation: ΔV = {:?}, ΔQ = {:?}",
        delta_v(&g, &[], (0, &[]), (0, &["V3"])).unwrap(),
        delta_q(&g, &[], (0, &[]), (0, &["V3"])).unwrap()
    );
    let g = graph(&[], &[("V1", "V3")]);
    println!(
        "V1 <-> V3, σ(V3): ΔQ = {:?} (nothing is invariant)",
        delta_q(&g, &[], (0, &[]), (0, &["V3"])).unwrap()
    );
    let g = graph(&[("V1", "V2"), ("V2", "V3"), ("V3", "V4")], &[("V2", "V3")]);
    println!(
        "V1 -> V2 -> V3 -> V4, V2 <-> V3, σ(V2): ΔQ = {:?}",
        delta_q(&g, &[], (0, &[]), (0, &["V2"])).unwrap()
    );
}

fn part2() {
    println!("\n== 2. CRID on Ex. 18.38 ==");
    let g = graph(&[("V1", "V2"), ("V2", "V3")], &[("V1", "V2")]);
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["V3"])];
    let three = crid(&g, &s, &[(0, &[]), (1, &[]), (1, &["V3"])]).unwrap();
    println!("⟨{{}}¹, {{}}², {{V3}}²⟩:           {three}");
    let four = crid(&g, &s, &[(0, &[]), (1, &[]), (1, &["V3"]), (0, &["V2!"])]).unwrap();
    println!("plus a marginal σ on V2 in domain 1: {four}");
    assert_eq!(
        four.edges(),
        ["V1^ <- {V1}", "V2^ <- {V2}", "V3^ <- {V2, V3}"]
    );
}

fn part3() {
    println!("\n== 3. The settings of Sec. 18.6 ==");
    let chain = graph(&[("V1", "V2"), ("V2", "V3")], &[]);
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["V3"])];
    let a = crid(&chain, &s, &[(0, &[]), (1, &[]), (0, &["V3"])]).unwrap();
    println!("chain, ⟨{{}}¹, {{}}², {{V3}}¹⟩:            {a}");
    assert!(a.identifiable(&["V3"], &["V1"]).unwrap());
    let b = crid(
        &chain,
        &s,
        &[(0, &[]), (1, &[]), (0, &["V3"]), (0, &["V2!"])],
    )
    .unwrap();
    println!("chain, plus σ(∅) on V2:                {b}");
    assert!(b.identifiable(&["V2"], &["V1", "V3"]).unwrap());

    let collider = graph(&[("V1", "V2"), ("V3", "V2")], &[]);
    let mut s5: Vec<(usize, usize, &[&str])> = Vec::new();
    for i in 0..5 {
        for j in i + 1..5 {
            s5.push((i, j, &["V1", "V3"]));
        }
    }
    let obs: Vec<(usize, &[&str])> = (0..5).map(|d| (d, &[] as &[&str])).collect();
    let c = crid(&collider, &s5, &obs).unwrap();
    println!("collider, observations in 5 domains:   {c}");
    assert!(c.identifiable(&["V1", "V3"], &["V2"]).unwrap());

    let nm = graph(&[("V1", "V2"), ("V2", "V3"), ("V3", "V4")], &[("V2", "V3")]);
    let d = crid(&nm, &[], &[(0, &[]), (0, &["V3!"]), (0, &["V3!"])]).unwrap();
    println!("non-Markovian, two σ(∅) on V3:         {d}");
    assert!(d.identifiable(&["V3"], &["V1", "V2", "V4"]).unwrap());
}

fn main() {
    part1();
    part2();
    part3();
}
