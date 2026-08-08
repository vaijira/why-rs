//! Causal AI book, chapter 4 part 1 — the back-door criterion.
//!
//! A migration of `ch4 (part 1).ipynb`, plus the nine-variable diagram from
//! `4-12.ipynb`, to [`why_data::graph::backdoor`].
//!
//! The notebook builds graphs from a `<NODES>/<EDGES>` text block and renders
//! them with graphviz; here they are built programmatically and the diagram is
//! printed as an edge list.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-data --example chapter4_part1
//! ```

use why_data::graph::backdoor::{self, Estimand};
use why_data::graph::dseparation::{Admg, EdgeKind};

/// Prints a diagram the way the notebook's `<EDGES>` block writes it.
fn print_graph(name: &str, g: &Admg) {
    use why_data::graph::EdgeRef;
    println!("  {name}:");
    let graph = g.graph();
    let mut edges: Vec<String> = graph
        .edge_references()
        .map(|e| {
            let arrow = match e.weight() {
                EdgeKind::Directed => "->",
                EdgeKind::Bidirected => "--",
            };
            format!("    {} {arrow} {}", graph[e.source()], graph[e.target()])
        })
        .collect();
    edges.sort();
    println!("{}", edges.join("\n"));
}

fn show_sets(sets: &[Vec<String>]) {
    if sets.is_empty() {
        println!("    (none — the effect is not identifiable by adjustment)");
        return;
    }
    for s in sets {
        if s.is_empty() {
            println!("    {{}}  (the empty set)");
        } else {
            println!("    {{{}}}", s.join(", "));
        }
    }
}

fn main() {
    definition_4_2_2();
    example_4_4();
    figure_4_12();
    not_migrated();
}

/// Definition 4.2.2 — the criterion itself, on the smallest graph that shows
/// each of its two conditions doing work.
fn definition_4_2_2() {
    println!("\n=== Definition 4.2.2 — the back-door criterion ===");

    // Condition 2: a confounder must be adjusted for.
    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("Z", "Y");
    g.directed("X", "Y");
    print_graph("confounded triangle", &g);
    println!(
        "    Z = {{}}   admissible? {}",
        backdoor::satisfies(&g, &["X"], &["Y"], &[]).unwrap()
    );
    println!(
        "    Z = {{Z}}  admissible? {}",
        backdoor::satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap()
    );
    assert!(!backdoor::satisfies(&g, &["X"], &["Y"], &[]).unwrap());
    assert!(backdoor::satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap());

    // Condition 1: a descendant of X is barred even when it would block a path.
    let mut g = Admg::new();
    g.directed("X", "M");
    g.directed("M", "Y");
    g.directed("Z", "X");
    g.directed("Z", "Y");
    print_graph("\n  mediator M on the causal path", &g);
    println!(
        "    Z = {{Z, M}} admissible? {}   <- M is a descendant of X",
        backdoor::satisfies(&g, &["X"], &["Y"], &["Z", "M"]).unwrap()
    );
    println!(
        "    Z = {{Z}}    admissible? {}",
        backdoor::satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap()
    );
    assert!(!backdoor::satisfies(&g, &["X"], &["Y"], &["Z", "M"]).unwrap());
    assert!(backdoor::satisfies(&g, &["X"], &["Y"], &["Z"]).unwrap());
}

/// Example 4.4 — SAT results. `Z1` is the school district, `Z2` the family's
/// monthly income, `X` the type of school, `Y` the SAT result. The district
/// administration wants `P(y | do(x), Z1 = z1)`, a `Z1`-specific effect.
fn example_4_4() {
    println!("\n=== Example 4.4 — SAT results, conditional back-door ===");

    let mut g = Admg::new();
    g.directed("Z1", "X");
    g.directed("Z2", "Z1");
    g.directed("Z2", "Y");
    g.directed("X", "Y");
    g.bidirected("Z1", "Z2");
    g.bidirected("X", "Z1");
    print_graph("G4", &g);

    // The query is specific to Z1, so every admissible set must contain it.
    println!("\n  admissible sets containing Z1:");
    let sets = backdoor::list_admissible_sets(&g, &["X"], &["Y"], &["Z1"]).unwrap();
    show_sets(&sets);
    assert_eq!(sets, vec![vec!["Z1".to_string(), "Z2".to_string()]]);

    println!(
        "\n  Z1 alone is not enough: {}",
        backdoor::satisfies(&g, &["X"], &["Y"], &["Z1"]).unwrap()
    );
    println!("  the back door X <-> Z1 <-> Z2 -> Y stays open until Z2 joins.");

    // Theorem 4.2.5, the W-specific adjustment, with W = {Z1} and Z = {Z2}.
    let estimand = Estimand::new(&g, &["X"], &["Y"], &["Z2"], &["Z1"])
        .unwrap()
        .expect("Z1 with Z2 is admissible");
    println!("\n  estimand:");
    println!("    {estimand}");
    println!("    latex: {}", estimand.to_latex());
    assert_eq!(
        estimand.to_string(),
        "P(y | do(x), z1) = sum_{z2} P(y | x, z2, z1) P(z2 | z1)"
    );
}

/// Notebook `4-12.ipynb` — the same question on a nine-variable diagram, where
/// the answer is no longer obvious by inspection.
fn figure_4_12() {
    println!("\n=== Figure 4.12 — listing every admissible set ===");

    let mut g = Admg::new();
    for (a, b) in [
        ("X", "Z1"),
        ("X", "Z4"),
        ("Z1", "Y"),
        ("Z2", "X"),
        ("Z2", "Z5"),
        ("Z3", "Y"),
        ("Z4", "Z3"),
        ("Z5", "Y"),
        ("Z6", "Z2"),
        ("Z6", "Z5"),
        ("Z7", "Z5"),
        ("Z7", "Y"),
    ] {
        g.directed(a, b);
    }
    g.bidirected("X", "Z6");
    g.bidirected("Z6", "Z7");
    print_graph("G (14 edges)", &g);

    let descendants = g.descendants_of(&["X"]).unwrap();
    println!("\n  De(X) = {{{}}}  — barred by condition 1", descendants.join(", "));

    let sets = backdoor::list_admissible_sets(&g, &["X"], &["Y"], &[]).unwrap();
    println!("\n  {} admissible sets, smallest first:", sets.len());
    show_sets(&sets);

    assert!(!sets.is_empty(), "the effect should be identifiable");
    for s in &sets {
        let refs: Vec<&str> = s.iter().map(String::as_str).collect();
        assert!(backdoor::satisfies(&g, &["X"], &["Y"], &refs).unwrap());
        assert!(!s.is_empty(), "X and Y are confounded through Z6");
    }

    // The adjustment formula for the smallest one.
    let smallest: Vec<&str> = sets[0].iter().map(String::as_str).collect();
    let estimand = Estimand::new(&g, &["X"], &["Y"], &smallest, &[])
        .unwrap()
        .expect("the listed set is admissible");
    println!("\n  estimand using the smallest set:");
    println!("    {estimand}");
}

/// What this example does not cover, stated rather than left as a silent gap.
fn not_migrated() {
    println!("\n=== Not migrated ===");
    println!(
        "  - parseGraph's <NODES>/<EDGES>/<TASK> text format: graphs here are\n  \
         \x20 built programmatically. That parser belongs in why-parser.\n  \
         - plot_causal_diagram: graphviz rendering has no equivalent yet."
    );
}
