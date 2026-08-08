//! Causal AI book, chapter 2 part 2 — "Causal Diagram and Conditional
//! Independence".
//!
//! A migration of `chapter2_part2.ipynb`. Examples 2.10 to 2.13 use
//! [`why_data::scm`], and every probability printed is asserted against the
//! value the Python notebook produces. Examples 2.15 to 2.18 use
//! [`why_data::graph::dseparation`] in place of the notebook's
//! `listDSeparationPaths`, printing the same split into connected and separated
//! paths; the graphviz rendering has no equivalent here.
//!
//! The three graph structures — chain, fork, collider — are the point of the
//! chapter, so each of the first group also prints the causal diagram that
//! [`why_data::scm`] derives from the equations, per Definition 2.4.1.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-data --example chapter2_part2
//! ```

use why_data::graph::EdgeRef;
use why_data::graph::dseparation::Admg;
use why_data::scm::{Dist, Model, Value, WorldRef, expr};

const T: Value = Value::Bool(true);
const F: Value = Value::Bool(false);

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn show(label: &str, got: f64, expected: f64) {
    assert!(
        approx(got, expected),
        "{label}: got {got}, notebook says {expected}"
    );
    println!("  {label} = {got:.4}");
}

/// Definition 2.4.1 — the causal diagram of an SCM: a vertex per variable, and
/// an edge `V_j -> V_i` whenever `V_j` is an argument of `f_i`. `scm` builds
/// this from the free variables of each equation, so it comes for free.
fn print_diagram(w: WorldRef<'_>) {
    let g = w.world().graph();
    let mut edges: Vec<String> = g
        .edge_references()
        .map(|e| format!("{} -> {}", g[e.source()], g[e.target()]))
        .collect();
    edges.sort();
    println!("  causal diagram: {}", edges.join(", "));
}

/// Two independent events multiply; dependent ones do not. Reports which.
fn report_factorisation(joint: f64, a: f64, b: f64) {
    let product = a * b;
    if approx(joint, product) {
        println!("  joint {joint:.4} == {a:.4} * {b:.4} = {product:.4}  -> independent");
    } else {
        println!("  joint {joint:.4} != {a:.4} * {b:.4} = {product:.4}  -> dependent");
    }
}

fn main() {
    example_2_10();
    example_2_11();
    example_2_12();
    example_2_13();
    example_2_14();
    example_2_15();
    example_2_16();
    example_2_17();
    example_2_18();
}

/// Example 2.10 — causal chain `X -> Z -> Y`. X and Y are dependent.
fn example_2_10() {
    println!("\n=== Example 2.10 — causal chain (M1) ===");

    let bern = || Dist::Bernoulli(0.5);
    let m = Model::new(
        [
            ("x".to_string(), expr::var("ux")),
            (
                "z".to_string(),
                expr::and([expr::var("x"), expr::not(expr::var("uz"))]),
            ),
            ("y".to_string(), expr::and([expr::var("z"), expr::var("uy")])),
        ],
        [
            ("uz".to_string(), bern()),
            ("ux".to_string(), bern()),
            ("uy".to_string(), bern()),
        ],
    )
    .expect("M1 is well formed");

    println!("{m}");
    let w = m.world(m.observational());
    print_diagram(w);

    let joint = w.query(&[("x", T), ("y", T)], &[]).expect("query");
    let px = w.query(&[("x", T)], &[]).expect("query");
    let py = w.query(&[("y", T)], &[]).expect("query");
    show("P(x = 1, y = 1)", joint, 0.125_000_000_000_000_03);
    show("P(x = 1)", px, 0.500_000_000_000_000_1);
    show("P(y = 1)", py, 0.125_000_000_000_000_03);
    report_factorisation(joint, px, py);
}

/// Example 2.11 — common cause `X <- Z -> Y`. Conditioning on the fork's centre
/// makes X and Y independent.
fn example_2_11() {
    println!("\n=== Example 2.11 — common cause (M2) ===");

    let bern = || Dist::Bernoulli(0.5);
    let m = Model::new(
        [
            ("z".to_string(), expr::var("uz")),
            (
                "x".to_string(),
                expr::and([expr::var("z"), expr::not(expr::var("ux"))]),
            ),
            ("y".to_string(), expr::and([expr::var("z"), expr::var("uy")])),
        ],
        [
            ("uz".to_string(), bern()),
            ("ux".to_string(), bern()),
            ("uy".to_string(), bern()),
        ],
    )
    .expect("M2 is well formed");

    println!("{m}");
    let w = m.world(m.observational());
    print_diagram(w);

    let joint = w.query(&[("x", T), ("y", T)], &[("z", T)]).expect("query");
    let px = w.query(&[("x", T)], &[("z", T)]).expect("query");
    let py = w.query(&[("y", T)], &[("z", T)]).expect("query");
    show("P(x = 1, y = 1 | z = 1)", joint, 0.25);
    show("P(x = 1 | z = 1)", px, 0.5);
    show("P(y = 1 | z = 1)", py, 0.5);
    report_factorisation(joint, px, py);
}

/// Example 2.12 — common effect `X -> Z <- Y`. X and Y are independent until
/// the collider is conditioned on, which opens the path.
fn example_2_12() {
    println!("\n=== Example 2.12 — common effect / collider (M3) ===");

    let bern = || Dist::Bernoulli(0.5);
    let m = Model::new(
        [
            ("x".to_string(), expr::var("ux")),
            ("y".to_string(), expr::var("uy")),
            // Z := !Y & (!X | U_z)
            (
                "z".to_string(),
                expr::and([
                    expr::not(expr::var("y")),
                    expr::or([expr::not(expr::var("x")), expr::var("uz")]),
                ]),
            ),
        ],
        [
            ("uz".to_string(), bern()),
            ("ux".to_string(), bern()),
            ("uy".to_string(), bern()),
        ],
    )
    .expect("M3 is well formed");

    println!("{m}");
    let w = m.world(m.observational());
    print_diagram(w);

    println!("\nUnconditioned:");
    let joint = w.query(&[("x", T), ("y", T)], &[]).expect("query");
    let px = w.query(&[("x", T)], &[]).expect("query");
    let py = w.query(&[("y", T)], &[]).expect("query");
    show("P(x = 1, y = 1)", joint, 0.250_000_000_000_000_06);
    show("P(x = 1)", px, 0.500_000_000_000_000_1);
    show("P(y = 1)", py, 0.500_000_000_000_000_1);
    report_factorisation(joint, px, py);

    println!("\nConditioning on the collider z = 0:");
    let joint = w.query(&[("x", T), ("y", T)], &[("z", F)]).expect("query");
    let px = w.query(&[("x", T)], &[("z", F)]).expect("query");
    let py = w.query(&[("y", T)], &[("z", F)]).expect("query");
    show("P(x = 1, y = 1 | z = 0)", joint, 0.4);
    show("P(x = 1 | z = 0)", px, 0.600_000_000_000_000_1);
    show("P(y = 1 | z = 0)", py, 0.8);
    report_factorisation(joint, px, py);
}

/// Example 2.13 — collider extended with a descendant `W`. Conditioning on a
/// descendant of a collider opens the path too.
fn example_2_13() {
    println!("\n=== Example 2.13 — collider with a descendant (M4) ===");

    let bern = || Dist::Bernoulli(0.5);
    let m = Model::new(
        [
            ("x".to_string(), expr::var("ux")),
            ("y".to_string(), expr::var("uy")),
            (
                "z".to_string(),
                expr::and([
                    expr::not(expr::var("y")),
                    expr::or([expr::not(expr::var("x")), expr::var("uz")]),
                ]),
            ),
            // W := (U_w & !Z) | (!U_w & Z), i.e. Z xor U_w
            (
                "w".to_string(),
                expr::or([
                    expr::and([expr::var("uw"), expr::not(expr::var("z"))]),
                    expr::and([expr::not(expr::var("uw")), expr::var("z")]),
                ]),
            ),
        ],
        [
            ("uz".to_string(), bern()),
            ("ux".to_string(), bern()),
            ("uy".to_string(), bern()),
            ("uw".to_string(), Dist::Bernoulli(0.1)),
        ],
    )
    .expect("M4 is well formed");

    println!("{m}");
    let w = m.world(m.observational());
    print_diagram(w);

    println!("\nConditioning on w = 1, a descendant of the collider z:");
    let joint = w.query(&[("x", T), ("y", T)], &[("w", T)]).expect("query");
    let px = w.query(&[("x", T)], &[("w", T)]).expect("query");
    let py = w.query(&[("y", T)], &[("w", T)]).expect("query");
    show("P(x = 1, y = 1 | w = 1)", joint, 0.062_500_000_000_000_01);
    show("P(x = 1 | w = 1)", px, 0.375);
    show("P(y = 1 | w = 1)", py, 0.125_000_000_000_000_03);
    report_factorisation(joint, px, py);
}

/// The notebook's second "Example 2.13" — `Z := !(X xor Y)`, where knowing the
/// collider makes X fully determine Y.
fn example_2_14() {
    println!("\n=== Example 2.13 (second) — deterministic collider (M5) ===");

    let m = Model::new(
        [
            ("x".to_string(), expr::var("ux")),
            ("y".to_string(), expr::var("uy")),
            // Z := !((X & !Y) | (!X & Y))
            (
                "z".to_string(),
                expr::not(expr::or([
                    expr::and([expr::var("x"), expr::not(expr::var("y"))]),
                    expr::and([expr::not(expr::var("x")), expr::var("y")]),
                ])),
            ),
        ],
        [
            ("ux".to_string(), Dist::Bernoulli(0.5)),
            ("uy".to_string(), Dist::Bernoulli(0.5)),
        ],
    )
    .expect("M5 is well formed");

    println!("{m}");
    let w = m.world(m.observational());
    print_diagram(w);

    show(
        "P(y = 1 | x = 1, z = 1)",
        w.query(&[("y", T)], &[("x", T), ("z", T)]).expect("query"),
        1.0,
    );
    show(
        "P(y = 1 | z = 1)",
        w.query(&[("y", T)], &[("z", T)]).expect("query"),
        0.5,
    );
    println!("  knowing x on top of z pins y down completely");
}

/// The Markovian graph of figure 2.16.
fn fig_2_16() -> Admg {
    let mut g = Admg::new();
    g.directed("C", "S");
    g.directed("C", "R");
    g.directed("S", "W");
    g.directed("R", "W");
    g.directed("W", "L");
    g
}

/// Prints every path between `x` and `y`, split the way the notebook's
/// `listDSeparationPaths` splits them, then the verdict.
fn report_paths(g: &Admg, x: &str, y: &str, z: &[&str]) {
    let paths = g.paths(x, y, z).expect("valid query");
    let (open, blocked): (Vec<_>, Vec<_>) = paths.iter().partition(|p| p.is_open());

    println!("  adjusted: {{{}}}", z.join(", "));
    println!("  connected paths ({}):", open.len());
    for p in &open {
        println!("    {p}");
    }
    println!("  separated paths ({}):", blocked.len());
    for p in &blocked {
        let why = p.blocked_by.as_ref().expect("a blocked path has a blocker");
        println!("    {p}    [{why}]");
    }

    let separated = g.is_d_separated(&[x], &[y], z).expect("valid query");
    println!(
        "  ({x} _||_ {y} | Z) is {}",
        if separated { "TRUE — d-separated" } else { "FALSE — d-connected" }
    );
    // The two routines must never disagree: one enumerates paths, the other
    // never builds one.
    assert_eq!(separated, open.is_empty());
}

/// Example 2.15 — paths between S and W, nothing adjusted. They are adjacent,
/// so no set could separate them.
fn example_2_15() {
    println!("\n=== Example 2.15 — d-separation ===");
    let g = fig_2_16();
    report_paths(&g, "S", "W", &[]);
}

/// Example 2.16 — paths between R and L, adjusting for W. W is a non-collider
/// on both, so both close.
fn example_2_16() {
    println!("\n=== Example 2.16 — d-separation (2) ===");
    let g = fig_2_16();
    println!("Without adjustment:");
    report_paths(&g, "R", "L", &[]);
    println!("Adjusting for W:");
    report_paths(&g, "R", "L", &["W"]);
}

/// Example 2.17 — the non-Markovian variant, where S and R share an unobserved
/// common cause written `S <-> R`.
fn example_2_17() {
    println!("\n=== Example 2.17 — d-separation in non-Markovian models ===");
    let mut g = Admg::new();
    g.bidirected("S", "R");
    g.directed("S", "W");
    g.directed("R", "W");
    g.directed("W", "L");
    report_paths(&g, "S", "L", &[]);
}

/// Example 2.18 — figure 2.17, where *no* set d-separates X from Y. Rather than
/// assert the textbook's claim, this searches every subset of the remaining
/// variables and shows the search comes up empty.
fn example_2_18() {
    println!("\n=== Example 2.18 — no admissible set exists ===");
    let mut g = Admg::new();
    g.directed("X", "W");
    g.directed("T", "X");
    g.directed("T", "Z");
    g.directed("W", "Z");
    g.directed("Z", "Y");
    g.directed("R", "Y");
    g.directed("Y", "S");
    g.bidirected("W", "Y");

    report_paths(&g, "X", "Y", &[]);

    println!("\nThe textbook says Z = {{T}} blocks everything but X -> W -> Z -> Y:");
    report_paths(&g, "X", "Y", &["T"]);

    // Exhaustive search over every subset of V \ {X, Y}.
    let candidates = ["T", "W", "Z", "R", "S"];
    let mut found: Vec<String> = Vec::new();
    for mask in 0..(1u32 << candidates.len()) {
        let z: Vec<&str> = candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, n)| *n)
            .collect();
        if g.is_d_separated(&["X"], &["Y"], &z).expect("valid query") {
            found.push(format!("{{{}}}", z.join(", ")));
        }
    }
    println!(
        "\n  searched all {} subsets of {{{}}}: {} d-separate X from Y",
        1u32 << candidates.len(),
        candidates.join(", "),
        if found.is_empty() { "none".to_string() } else { found.join(", ") }
    );
    assert!(
        found.is_empty(),
        "no set should d-separate X from Y in figure 2.17"
    );
}
