//! Causal AI book, chapter 4 section 4.4 — C-INFER and causal factor trees.
//!
//! Runs [`why_data::graph::cinfer`] on the section's three canonical graphs
//! and on a surrogate experiment, printing the q-tree, each d-tree and the
//! estimand. The asserted estimands are the ones the section derives.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-data --example chapter4_cinfer
//! ```

use why_data::graph::cinfer::{CInferError, Input, Query, c_infer};
use why_data::graph::dseparation::Admg;

fn main() {
    front_door();
    back_door();
    not_identifiable();
    surrogate_experiment();
}

/// Prints every tree of an identification attempt and returns the estimand.
fn run(g: &Admg, q: &Query, inputs: &[Input]) -> Option<String> {
    match c_infer(g, q, inputs) {
        Ok(d) => {
            println!("  q-tree:\n{}", indent(&d.query_tree.to_string()));
            for (p, t) in inputs.iter().zip(&d.input_trees) {
                println!("  d-tree for {p}:\n{}", indent(&t.to_string()));
            }
            println!("  {q} = {}", d.estimand);
            Some(d.estimand.to_string())
        }
        Err(CInferError::NotIdentifiable {
            query_tree,
            input_trees,
            unmapped,
        }) => {
            println!("  q-tree:\n{}", indent(&query_tree.to_string()));
            for (p, t) in inputs.iter().zip(&input_trees) {
                println!("  d-tree for {p}:\n{}", indent(&t.to_string()));
            }
            let names: Vec<String> = unmapped.iter().map(ToString::to_string).collect();
            println!("  FAIL: no input yields {}", names.join(", "));
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|l| format!("    {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn front_door() {
    println!("\n=== Front door: X -> Z -> Y, X <-> Y ===");
    let mut g = Admg::new();
    g.directed("X", "Z");
    g.directed("Z", "Y");
    g.bidirected("X", "Y");
    let e = run(&g, &Query::new(&["X"], &["Y"]), &[Input::observational()]);
    assert_eq!(
        e.as_deref(),
        Some("sum_{z} P(z | x) sum_{x'} P(x') P(y | x', z)")
    );
}

fn back_door() {
    println!("\n=== Back door: Z -> X, Z -> Y, X -> Y ===");
    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("Z", "Y");
    g.directed("X", "Y");
    let e = run(&g, &Query::new(&["X"], &["Y"]), &[Input::observational()]);
    assert_eq!(e.as_deref(), Some("sum_{z} P(y | z, x) P(z)"));
}

fn not_identifiable() {
    println!("\n=== Not identifiable: X <-> Z, Z -> Y, X -> Y, X <-> Y ===");
    let mut g = Admg::new();
    g.directed("Z", "Y");
    g.directed("X", "Y");
    g.bidirected("X", "Z");
    g.bidirected("X", "Y");
    let e = run(&g, &Query::new(&["X"], &["Y"]), &[Input::observational()]);
    assert!(e.is_none());
    println!("  Q[Y] sits in the district {{X, Y, Z}} of an ancestral set: a hedge.");
}

fn surrogate_experiment() {
    println!("\n=== Surrogate experiment: Z -> X -> Y, X <-> Z <-> Y ===");
    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("X", "Y");
    g.bidirected("X", "Z");
    g.bidirected("Z", "Y");
    let q = Query::new(&["X"], &["Y"]);
    assert!(run(&g, &q, &[Input::observational()]).is_none());
    println!("\n  ...and again with an experiment on Z added:");
    let e = run(
        &g,
        &q,
        &[Input::observational(), Input::experimental(&["Z"])],
    );
    assert_eq!(e.as_deref(), Some("P(y | x, do(z))"));
}
