//! The Causal AI book's notebook graphs, parsed from their original text.
//!
//! Every block below is copied verbatim from a notebook cell, so this shows the
//! whole path: the book's own file format in, an answer out.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-parser --example notebook_graphs
//! ```

use why_data::graph::{backdoor, identification};
use why_parser::fusion;

/// Example 2.15 / 2.16 — d-separation, with the task carried in the file.
const G1: &str = r"
<NODES>
C
S
R
W
L

<EDGES>
C -> S
C -> R
S -> W
R -> W
W -> L

<TASK>
treatment: R
outcome: L
adjusted: W
";

/// Example 2.17 — the non-Markovian variant, where `--` is a latent confounder.
const G3: &str = r"
<NODES>
S
R
W
L

<EDGES>
S -- R
S -> W
R -> W
W -> L

<TASK>
treatment: S
outcome: L
adjusted:
";

/// Example 4.4 — the SAT diagram, whose task names the covariate the query is
/// specific to.
const G4: &str = r"
<NODES>
X
Y
Z1
Z2

<EDGES>
Z1 -> X
Z2 -> Z1
Z2 -> Y
X -> Y
Z1 -- Z2
X -- Z1
";

/// The napkin, from the end of chapter 4.
const NAPKIN: &str = r"
<NODES>
W
Z
X
Y

<EDGES>
W -> Z
Z -> X
X -> Y
W -- X
W -- Y
";

fn main() {
    d_separation();
    non_markovian();
    back_door();
    napkin();
    errors();
}

fn d_separation() {
    println!("\n=== Example 2.16 — d-separation, task read from the file ===");
    let doc = fusion::parse(G1).expect("G1 parses");
    let (x, y, z) = doc.query().expect("the file has a <TASK>");
    println!("  treatment {x:?}, outcome {y:?}, adjusted {z:?}");

    let separated = doc.graph.is_d_separated(&x, &y, &z).expect("valid query");
    println!("  ({} _||_ {} | {}) is {separated}", x[0], y[0], z[0]);
    assert!(separated);

    for path in doc.graph.paths(x[0], y[0], &z).expect("valid query") {
        let status = match &path.blocked_by {
            Some(b) => format!("blocked: {b}"),
            None => "open".to_string(),
        };
        println!("    {path}   [{status}]");
    }
}

fn non_markovian() {
    println!("\n=== Example 2.17 — `--` is a latent confounder, not an undirected edge ===");
    let doc = fusion::parse(G3).expect("G3 parses");
    let (x, y, z) = doc.query().expect("the file has a <TASK>");

    // A bidirected edge carries no ancestry: R is not an ancestor of S.
    println!("  An(R) = {:?}", doc.graph.ancestors_of(&["R"]).unwrap());
    assert_eq!(doc.graph.ancestors_of(&["R"]).unwrap(), vec!["R"]);

    println!("  paths from {} to {}:", x[0], y[0]);
    for path in doc.graph.paths(x[0], y[0], &z).expect("valid query") {
        println!("    {path}");
    }
    assert!(!doc.graph.is_d_separated(&x, &y, &z).unwrap());
}

fn back_door() {
    println!("\n=== Example 4.4 — admissible sets from the parsed diagram ===");
    let doc = fusion::parse(G4).expect("G4 parses");
    println!("  variables: {:?}", doc.graph.variables());

    let sets = backdoor::list_admissible_sets(&doc.graph, &["X"], &["Y"], &["Z1"])
        .expect("valid query");
    println!("  admissible sets containing Z1: {sets:?}");
    assert_eq!(sets, vec![vec!["Z1".to_string(), "Z2".to_string()]]);

    let estimand = backdoor::Estimand::new(&doc.graph, &["X"], &["Y"], &["Z2"], &["Z1"])
        .expect("valid query")
        .expect("admissible");
    println!("  {estimand}");
}

fn napkin() {
    println!("\n=== The napkin — identified straight from the file ===");
    let doc = fusion::parse(NAPKIN).expect("the napkin parses");
    let e = identification::identify(&doc.graph, &["X"], &["Y"]).expect("identifiable");
    println!("  P(y | do(x)) = {e}");
    assert_eq!(
        e.to_string(),
        "[sum_{w} P(w) P(x, y | w, z)] / [sum_{w} P(w) P(x | w, z)]"
    );
}

fn errors() {
    println!("\n=== Errors carry the line they were found on ===");
    for bad in [
        "<NODES>\nX\n<TYPO>\n",
        "<NODES>\nX\nX\n<EDGES>\n",
        "<NODES>\nX\n<EDGES>\nX -> Q\n",
        "<NODES>\nX\nY\n<EDGES>\nX => Y\n",
    ] {
        let err = fusion::parse(bad).expect_err("should not parse");
        println!("  {err}");
    }
}
