//! Causal AI book, chapter 4 part 2 — the rules of do-calculus.
//!
//! A migration of `ch4 (part 2).ipynb` to [`why_data::graph::docalculus`] and
//! [`why_data::graph::identification`].
//!
//! The notebook tests each rule with `DoCalculusInspector` and plots the
//! mutilated graph it ran on; here the same graph comes back on the
//! [`Applicability`](why_data::graph::docalculus::Applicability) and is printed
//! as an edge list. Its closing sections call `DoCalculusEngine.compute`, which
//! is the Shpitser-Pearl ID algorithm; every estimand printed below was checked
//! against the output stored in the notebook.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-data --example chapter4_part2
//! ```

use why_data::graph::EdgeRef;
use why_data::graph::docalculus::{self, Applicability};
use why_data::graph::dseparation::{Admg, EdgeKind};
use why_data::graph::identification::{IdError, identify, identify_conditional};

fn edges(g: &Admg) -> String {
    let graph = g.graph();
    let mut out: Vec<String> = graph
        .edge_references()
        .map(|e| {
            let arrow = match e.weight() {
                EdgeKind::Directed => "->",
                EdgeKind::Bidirected => "--",
            };
            format!("{} {arrow} {}", graph[e.source()], graph[e.target()])
        })
        .collect();
    out.sort();
    if out.is_empty() {
        "(no edges)".to_string()
    } else {
        out.join(", ")
    }
}

/// Prints a rule test the way the notebook's `display_inspector_result` does,
/// plus the mutilated graph it plots alongside.
fn report(original: &Admg, r: &Applicability) {
    println!("  original:  {}", edges(original));
    println!("  mutilated: {}", edges(&r.graph));
    println!("{r}");
}

fn main() {
    rule_1();
    rule_2();
    rule_3();
    id_expressions();
    napkin();
}

/// Rule 1 — insertion and deletion of observations, on the notebook's `G1`.
fn rule_1() {
    println!("\n=== Rule 1 — insertion/deletion of observations ===");
    println!("  P(y | do(x), z, w) = P(y | do(x), w)  if (Y _||_ Z | X, W) in G_xbar\n");

    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("X", "Y");
    g.bidirected("X", "Y");

    let r = docalculus::rule_1(&g, &["Y"], &["Z"], &["X"], &[]).unwrap();
    report(&g, &r);
    assert!(r.holds);
    println!("\n  Cutting the arrowheads into X removes Z -> X and X <-> Y,");
    println!("  which leaves Z isolated: once we intervene on X, observing the");
    println!("  district Z tells us nothing more about Y.");

    // Without the intervention the same independence fails.
    let without = docalculus::rule_1(&g, &["Y"], &["Z"], &[], &[]).unwrap();
    assert!(!without.holds);
    println!(
        "\n  For contrast, with no do(x) the test is {} — Z -> X -> Y is open.",
        without.holds
    );
}

/// Rule 2 — action/observation exchange, on the notebook's `G2`, where it does
/// *not* apply, and then on a graph where it does.
fn rule_2() {
    println!("\n=== Rule 2 — action/observation exchange ===");
    println!("  P(y | do(x), do(z), w) = P(y | do(x), z, w)");
    println!("      if (Y _||_ Z | X, W) in G_xbar_zunderline\n");

    let mut g2 = Admg::new();
    g2.directed("Z", "X");
    g2.directed("Z", "Y");
    g2.directed("Y", "X");

    // The notebook's query: Y as outcome, X as the action, Z as covariate.
    let r = docalculus::rule_2(&g2, &["Y"], &["X"], &[], &["Z"]).unwrap();
    report(&g2, &r);
    assert!(!r.holds);
    println!("\n  X has no outgoing edges, so underlining it changes nothing,");
    println!("  and Y -> X leaves the two adjacent — no set can separate them.");

    println!("\n  The same rule in the back-door graph Z -> X -> Y, Z -> Y:\n");
    let mut g = Admg::new();
    g.directed("Z", "X");
    g.directed("X", "Y");
    g.directed("Z", "Y");
    let r = docalculus::rule_2(&g, &["Y"], &["X"], &[], &["Z"]).unwrap();
    report(&g, &r);
    assert!(r.holds);
    println!("\n  Underlining X deletes X -> Y; Z blocks the remaining");
    println!("  X <- Z -> Y. This is back-door adjustment, derived.");
}

/// Rule 3 — insertion and deletion of actions, on the notebook's `G3`.
fn rule_3() {
    println!("\n=== Rule 3 — insertion/deletion of actions ===");
    println!("  P(y | do(x), do(z), w) = P(y | do(x), w)");
    println!("      if (Y _||_ Z | X, W) in G_xbar_z(w)bar");
    println!("  where Z(W) is the part of Z that is not an ancestor of W.\n");

    let mut g3 = Admg::new();
    g3.directed("Z", "X");
    g3.directed("X", "Y");
    g3.bidirected("Z", "Y");
    g3.bidirected("X", "Y");

    let r = docalculus::rule_3(&g3, &["Y"], &["Z"], &["X"], &[]).unwrap();
    report(&g3, &r);
    assert!(r.holds);
    println!("\n  Cutting into X removes Z -> X and X <-> Y; cutting into Z then");
    println!("  removes Z <-> Y, leaving Z isolated. Z reached Y only through");
    println!("  Z -> X -> Y, and do(x) already severed that.");

    // The Z(W) subtlety: only non-ancestors of W get cut.
    println!("\n  The Z(W) restriction, on Z -> W with Z <-> Y:\n");
    let mut g = Admg::new();
    g.directed("Z", "W");
    g.bidirected("Z", "Y");

    let no_w = docalculus::rule_3(&g, &["Y"], &["Z"], &[], &[]).unwrap();
    println!("  W = {{}}:  {} -> holds = {}", no_w.graph_label, no_w.holds);
    let with_w = docalculus::rule_3(&g, &["Y"], &["Z"], &[], &["W"]).unwrap();
    println!("  W = {{W}}: {} -> holds = {}", with_w.graph_label, with_w.holds);
    assert!(no_w.holds && !with_w.holds);
    println!("\n  With W observed, Z is an ancestor of it, so Z(W) is empty and");
    println!("  nothing is cut — Z <-> Y survives and the action cannot be dropped.");
}

/// The notebook's `ID Expressions with Do-Calculus` section. Two queries on the
/// same diagram, one identifiable and one not.
fn id_expressions() {
    println!("\n=== ID expressions ===");
    println!("  The three rules are complete for identification: the ID algorithm");
    println!("  applies them mechanically and returns the estimand, or the hedge");
    println!("  that proves no estimand exists.\n");

    let mut g = Admg::new();
    g.directed("X", "Z1");
    g.directed("Z1", "Y");
    g.directed("Z2", "Y");
    g.bidirected("X", "Z2");
    g.bidirected("Z1", "Z2");
    println!("  G4: {}\n", edges(&g));

    // P(y | do(x), z2)
    print!("  P(y | do(x), z2) = ");
    match identify_conditional(&g, &["X"], &["Y"], &["Z2"]) {
        Ok(e) => println!("{e}"),
        Err(IdError::NotIdentifiable(h)) => println!("NOT IDENTIFIABLE\n    {h}"),
        Err(e) => println!("error: {e}"),
    }
    assert!(identify_conditional(&g, &["X"], &["Y"], &["Z2"]).is_err());

    // P(y | do(x, z2))
    let joint = identify(&g, &["X", "Z2"], &["Y"]).expect("identifiable");
    println!("  P(y | do(x, z2))  = {joint}");
    assert_eq!(joint.to_string(), "sum_{z1} P(y | z2, x, z1) P(z1 | x)");
    println!("\n  Intervening on Z2 as well makes the difference: the hedge above");
    println!("  is the c-component {{X, Z1, Z2}} closing over {{Z1, Z2}}, and do(z2)");
    println!("  cuts the bidirected edges that form it.");
}

/// The napkin graph, the notebook's closing example: identifiable, but by no
/// adjustment set at all.
fn napkin() {
    println!("\n=== The napkin graph ===");

    let mut g = Admg::new();
    g.directed("W", "Z");
    g.directed("Z", "X");
    g.directed("X", "Y");
    g.bidirected("W", "X");
    g.bidirected("W", "Y");
    println!("  {}\n", edges(&g));

    // No back-door set works here, which is what makes the example famous.
    let sets = why_data::graph::backdoor::list_admissible_sets(&g, &["X"], &["Y"], &[])
        .expect("valid query");
    println!("  back-door admissible sets: {}", if sets.is_empty() {
        "none".to_string()
    } else {
        format!("{sets:?}")
    });
    assert!(sets.is_empty());

    let e = identify(&g, &["X"], &["Y"]).expect("the napkin is identifiable");
    println!("  P(y | do(x)) = {e}");
    println!("  latex:         {}", e.to_latex());
    assert_eq!(
        e.to_string(),
        "[sum_{w} P(w) P(x, y | w, z)] / [sum_{w} P(w) P(x | w, z)]"
    );

    println!("\n  Identifiable without any admissible set: ID is strictly stronger");
    println!("  than the back-door criterion. Z is an instrument-like surrogate,");
    println!("  and the ratio is what the c-component factorisation leaves behind.");
}
