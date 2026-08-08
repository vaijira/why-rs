//! The section-based causal-diagram format used by the Causal AI book.
//!
//! ```text
//! <NODES>
//! X
//! Y
//! Z
//!
//! <EDGES>
//! Z -> X
//! X -> Y
//! X -- Y
//!
//! <TASK>
//! treatment: X
//! outcome: Y
//! adjusted: Z
//! ```
//!
//! # Where this format comes from
//!
//! It is **not** a community interchange format — not DOT, not GraphML, and not
//! the dagitty syntax [`crate::dagitty`] handles. It is the input syntax of the
//! book's own editor tooling (`src/editor/` in the companion repository, reached
//! through `src/fusion.py`), and the Python there is a port of a TypeScript web
//! application: its source still carries commented-out TS, and its regexes are
//! written as JavaScript literals.
//!
//! The full format has far more sections than the notebooks use — `<TASK>`,
//! `<POPULATIONS>`, `<EXPERIMENTS>`, `<INTERVENTIONS>`, `<COVARIATES>`,
//! `<CONDITIONAL>`, `<OBSERVATIONS>`, `<EXTERNAL_DATA>`, `<QUERY>` and
//! `<TRANSFORMATION>` — covering transportability, selection bias and
//! sigma-calculus interventions. This parser implements `<NODES>`, `<EDGES>`
//! and `<TASK>`, which is everything the chapter 2 and 4 notebooks exercise;
//! any other recognised tag is rejected rather than silently ignored.
//!
//! # A gotcha worth knowing
//!
//! `--` means **bidirected** here — an unobserved common cause — not undirected.
//! In dagitty the same token means an undirected edge. Reading one format's
//! files with the other's assumptions silently changes the causal model.
//!
//! ```
//! use why_parser::fusion;
//!
//! let doc = fusion::parse("<NODES>\nX\nY\n\n<EDGES>\nX -> Y\n").unwrap();
//! assert_eq!(doc.graph.variables(), vec!["X", "Y"]);
//! ```

use std::fmt;

use why_data::graph::dseparation::Admg;
use why_data::types::Point;

/// What a variable is declared as.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum NodeKind {
    /// An ordinary observed variable.
    #[default]
    Basic,
    /// A latent variable, written `latent` in the file.
    ///
    /// Kept as declared: projecting a latent variable down to bidirected edges
    /// between its children is *not* done here.
    Latent,
    /// Any other declared type — selection bias, cluster, transportability and
    /// intervention nodes all land here.
    Other(String),
}

impl NodeKind {
    fn parse(word: &str) -> Self {
        match word {
            "basic" => Self::Basic,
            "latent" => Self::Latent,
            other => Self::Other(other.to_string()),
        }
    }
}

/// A declared variable, with whatever the file said about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// The variable's name, as edges and tasks refer to it.
    pub name: String,
    /// An optional display label, written in quotes.
    pub label: Option<String>,
    /// The declared type.
    pub kind: NodeKind,
    /// The layout position, if the file carried one.
    pub position: Option<Point<f64>>,
}

/// The `<TASK>` section: which variables play which role.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Task {
    /// `treatment:` — the intervention set `X`.
    pub treatment: Vec<String>,
    /// `outcome:` — the outcome set `Y`.
    pub outcome: Vec<String>,
    /// `adjusted:` — the conditioning set `Z`.
    pub adjusted: Vec<String>,
}

/// A parsed document.
#[derive(Clone, Debug)]
pub struct Document {
    /// Every declared variable, in declaration order.
    pub nodes: Vec<Node>,
    /// The diagram, ready for the algorithms in [`why_data::graph`].
    pub graph: Admg,
    /// The task, when the file had a `<TASK>` section.
    pub task: Option<Task>,
}

impl Document {
    /// The task's treatment, outcome and adjusted sets as borrowed slices,
    /// which is the shape the graph algorithms take.
    ///
    /// Returns `None` when the file declared no `<TASK>`.
    #[must_use]
    pub fn query(&self) -> Option<(Vec<&str>, Vec<&str>, Vec<&str>)> {
        fn refs(v: &[String]) -> Vec<&str> {
            v.iter().map(String::as_str).collect()
        }
        let task = self.task.as_ref()?;
        Some((
            refs(&task.treatment),
            refs(&task.outcome),
            refs(&task.adjusted),
        ))
    }
}

/// What went wrong, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// The 1-based line the problem was found on.
    pub line: usize,
    /// What the problem was.
    pub kind: ErrorKind,
}

/// The kinds of malformed input this parser rejects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A `<...>` line that is not a section this parser knows.
    UnknownTag(String),
    /// A required section is absent.
    MissingSection(&'static str),
    /// A line in `<NODES>` that is not a node declaration.
    MalformedNode,
    /// A line in `<EDGES>` that is not `from <op> to`.
    MalformedEdge,
    /// A line in `<TASK>` that is not `definition: names`.
    MalformedTask,
    /// The same variable was declared twice.
    DuplicateNode(String),
    /// An edge or task referred to a variable that was never declared.
    UnknownNode(String),
    /// An edge operator other than `->`, `--` or `-`.
    UnknownEdgeType(String),
    /// A task definition other than `treatment`, `outcome` or `adjusted`.
    UnknownTaskDefinition(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: ", self.line)?;
        match &self.kind {
            ErrorKind::UnknownTag(t) => write!(f, "unknown section tag {t}"),
            ErrorKind::MissingSection(s) => write!(f, "the {s} section is required"),
            ErrorKind::MalformedNode => f.write_str("expected a node declaration"),
            ErrorKind::MalformedEdge => f.write_str("expected `from <op> to`"),
            ErrorKind::MalformedTask => f.write_str("expected `definition: names`"),
            ErrorKind::DuplicateNode(n) => write!(f, "{n} is declared twice"),
            ErrorKind::UnknownNode(n) => write!(f, "{n} was never declared in <NODES>"),
            ErrorKind::UnknownEdgeType(t) => {
                write!(f, "{t} is not an edge type; expected ->, -- or -")
            }
            ErrorKind::UnknownTaskDefinition(d) => {
                write!(f, "{d} is not a task definition")
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// One section: its tag, and its lines paired with their 1-based numbers.
type Section<'a> = (String, Vec<(usize, &'a str)>);

/// Splits the input into sections.
fn sections(input: &str) -> Result<Vec<Section<'_>>, ParseError> {
    const KNOWN: [&str; 3] = ["<NODES>", "<EDGES>", "<TASK>"];
    let mut out: Vec<(String, Vec<(usize, &str)>)> = Vec::new();

    for (i, raw) in input.lines().enumerate() {
        let line = raw.trim();
        let number = i + 1;
        if line.is_empty() {
            continue;
        }
        if line.starts_with('<') {
            if !KNOWN.contains(&line) {
                return Err(ParseError {
                    line: number,
                    kind: ErrorKind::UnknownTag(line.to_string()),
                });
            }
            out.push((line.to_string(), Vec::new()));
            continue;
        }
        // Content before any tag is not in a section, so it is stray text.
        let Some((_, body)) = out.last_mut() else {
            return Err(ParseError {
                line: number,
                kind: ErrorKind::MissingSection("<NODES>"),
            });
        };
        body.push((number, line));
    }
    Ok(out)
}

/// Splits off the first whitespace-delimited token.
fn token(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

/// Whether the rest of a node line looks like `x,y` coordinates.
fn looks_like_coordinates(s: &str) -> bool {
    let head = token(s).0;
    head.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '+' || c == '.')
        && s.contains(',')
}

fn parse_node(line: &str, number: usize) -> Result<Node, ParseError> {
    let malformed = || ParseError {
        line: number,
        kind: ErrorKind::MalformedNode,
    };

    let (name, mut rest) = token(line);
    if name.is_empty() {
        return Err(malformed());
    }
    rest = rest.trim_start();

    let mut label = None;
    if let Some(tail) = rest.strip_prefix('"') {
        let end = tail.find('"').ok_or_else(malformed)?;
        label = Some(tail[..end].to_string());
        rest = tail[end + 1..].trim_start();
    }

    let mut kind = NodeKind::Basic;
    if !rest.is_empty() && !rest.starts_with('[') && !looks_like_coordinates(rest) {
        let (word, r) = token(rest);
        kind = NodeKind::parse(word);
        rest = r.trim_start();
    }

    // Options are recognised so they do not break the line, but nothing in this
    // crate consumes them yet.
    if rest.starts_with('[') {
        let end = rest.find(']').ok_or_else(malformed)?;
        rest = rest[end + 1..].trim_start();
    }

    let mut position = None;
    if !rest.is_empty() {
        let (x, y) = rest.split_once(',').ok_or_else(malformed)?;
        let x: f64 = x.trim().parse().map_err(|_| malformed())?;
        let y: f64 = y.trim().parse().map_err(|_| malformed())?;
        position = Some(Point::new(x, y));
    }

    Ok(Node {
        name: name.to_string(),
        label,
        kind,
        position,
    })
}

/// One edge: the endpoints and whether it is bidirected.
fn parse_edge(line: &str, number: usize) -> Result<(String, String, bool), ParseError> {
    let (from, rest) = token(line);
    let (op, rest) = token(rest);
    let (to, _) = token(rest);
    if from.is_empty() || op.is_empty() || to.is_empty() {
        return Err(ParseError {
            line: number,
            kind: ErrorKind::MalformedEdge,
        });
    }
    // `--` is bidirected in this format, not undirected. `-` is the format's
    // undirected edge, which has no place in an ADMG; treat it as bidirected so
    // the confounding it stands for is not silently dropped.
    let bidirected = match op {
        "->" => false,
        "--" | "-" => true,
        other => {
            return Err(ParseError {
                line: number,
                kind: ErrorKind::UnknownEdgeType(other.to_string()),
            });
        }
    };
    Ok((from.to_string(), to.to_string(), bidirected))
}

/// Parses a document in the book's `<NODES>/<EDGES>/<TASK>` format.
///
/// `<NODES>` and `<EDGES>` are required, `<TASK>` optional, and the sections may
/// appear in any order — though edges are resolved against the declared nodes,
/// so an edge naming an undeclared variable is an error wherever it sits.
///
/// # Errors
///
/// [`ParseError`], carrying the 1-based line number, for an unknown tag, a
/// malformed line, a duplicate declaration or a reference to an undeclared
/// variable.
pub fn parse(input: &str) -> Result<Document, ParseError> {
    let sections = sections(input)?;

    let find = |tag: &str| {
        sections
            .iter()
            .filter(|(t, _)| t == tag)
            .flat_map(|(_, body)| body.iter().copied())
            .collect::<Vec<(usize, &str)>>()
    };

    if !sections.iter().any(|(t, _)| t == "<NODES>") {
        return Err(ParseError {
            line: 1,
            kind: ErrorKind::MissingSection("<NODES>"),
        });
    }
    if !sections.iter().any(|(t, _)| t == "<EDGES>") {
        return Err(ParseError {
            line: 1,
            kind: ErrorKind::MissingSection("<EDGES>"),
        });
    }

    let mut nodes: Vec<Node> = Vec::new();
    let mut graph = Admg::new();
    for (number, line) in find("<NODES>") {
        let node = parse_node(line, number)?;
        if nodes.iter().any(|n| n.name == node.name) {
            return Err(ParseError {
                line: number,
                kind: ErrorKind::DuplicateNode(node.name),
            });
        }
        graph.add_node(&node.name);
        nodes.push(node);
    }

    let declared = |name: &str, number: usize| -> Result<(), ParseError> {
        if nodes.iter().any(|n| n.name == name) {
            Ok(())
        } else {
            Err(ParseError {
                line: number,
                kind: ErrorKind::UnknownNode(name.to_string()),
            })
        }
    };

    for (number, line) in find("<EDGES>") {
        let (from, to, bidirected) = parse_edge(line, number)?;
        declared(&from, number)?;
        declared(&to, number)?;
        if bidirected {
            graph.bidirected(&from, &to);
        } else {
            graph.directed(&from, &to);
        }
    }

    let task_lines = find("<TASK>");
    let task = if sections.iter().any(|(t, _)| t == "<TASK>") {
        let mut task = Task::default();
        for (number, line) in task_lines {
            let (definition, names) = line.split_once(':').ok_or(ParseError {
                line: number,
                kind: ErrorKind::MalformedTask,
            })?;
            // The book's parser does not trim these; trimming here means
            // `treatment: X, Y` behaves as it reads.
            let names: Vec<String> = names
                .split(',')
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(ToString::to_string)
                .collect();
            for name in &names {
                declared(name, number)?;
            }
            match definition.trim() {
                "treatment" => task.treatment = names,
                "outcome" => task.outcome = names,
                "adjusted" => task.adjusted = names,
                other => {
                    return Err(ParseError {
                        line: number,
                        kind: ErrorKind::UnknownTaskDefinition(other.to_string()),
                    });
                }
            }
        }
        Some(task)
    } else {
        None
    };

    Ok(Document { nodes, graph, task })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Example 2.15's graph, exactly as the notebook writes it.
    const G1: &str = "
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
treatment: S
outcome: W
adjusted:
";

    #[test]
    fn parses_the_d_separation_task() {
        let doc = parse(G1).unwrap();
        assert_eq!(doc.graph.variables(), vec!["C", "S", "R", "W", "L"]);
        let task = doc.task.expect("the file has a task");
        assert_eq!(task.treatment, vec!["S"]);
        assert_eq!(task.outcome, vec!["W"]);
        assert!(task.adjusted.is_empty(), "an empty list stays empty");
    }

    #[test]
    fn the_parsed_graph_answers_the_book_s_question() {
        // Example 2.16: adjusting for W d-separates R from L.
        let doc = parse(G1).unwrap();
        assert!(!doc.graph.is_d_separated(&["R"], &["L"], &[]).unwrap());
        assert!(doc.graph.is_d_separated(&["R"], &["L"], &["W"]).unwrap());
    }

    #[test]
    fn double_dash_is_bidirected_not_undirected() {
        // Example 2.17's non-Markovian graph.
        let doc = parse("<NODES>\nS\nR\nW\nL\n<EDGES>\nS -- R\nS -> W\nR -> W\nW -> L\n")
            .unwrap();
        // A bidirected edge carries no ancestry, which is what distinguishes it
        // from an undirected one.
        assert_eq!(doc.graph.ancestors_of(&["R"]).unwrap(), vec!["R"]);
        assert!(!doc.graph.is_d_separated(&["S"], &["L"], &[]).unwrap());
    }

    #[test]
    fn parses_the_napkin() {
        let doc = parse(
            "<NODES>\nW\nZ\nX\nY\n\n<EDGES>\nW -> Z\nZ -> X\nX -> Y\nW -- X\nW -- Y\n",
        )
        .unwrap();
        let e = why_data::graph::identification::identify(&doc.graph, &["X"], &["Y"]).unwrap();
        assert_eq!(
            e.to_string(),
            "[sum_{w} P(w) P(x, y | w, z)] / [sum_{w} P(w) P(x | w, z)]"
        );
    }

    #[test]
    fn parses_figure_4_12() {
        let doc = parse(
            "<NODES>\nX\nY\nZ1\nZ2\nZ3\nZ4\nZ5\nZ6\nZ7\n\n<EDGES>\n\
             X -> Z1\nX -> Z4\nZ1 -> Y\nZ2 -> X\nZ2 -> Z5\nZ3 -> Y\nZ4 -> Z3\n\
             Z5 -> Y\nZ6 -> Z2\nZ6 -> Z5\nZ7 -> Z5\nZ7 -> Y\nX -- Z6\nZ6 -- Z7\n",
        )
        .unwrap();
        assert_eq!(doc.nodes.len(), 9);
        assert_eq!(doc.graph.graph().edge_count(), 14);
        let sets =
            why_data::graph::backdoor::list_admissible_sets(&doc.graph, &["X"], &["Y"], &[])
                .unwrap();
        assert_eq!(sets.len(), 5);
    }

    #[test]
    fn optional_node_fields_are_parsed() {
        let doc = parse(
            "<NODES>\nX \"Treatment\" basic 1.5,-2\nY latent\nZ [style=dashed] 0,0\n\
             <EDGES>\nX -> Y\n",
        )
        .unwrap();
        assert_eq!(doc.nodes[0].label.as_deref(), Some("Treatment"));
        assert_eq!(doc.nodes[0].kind, NodeKind::Basic);
        assert_eq!(doc.nodes[0].position, Some(Point::new(1.5, -2.0)));
        assert_eq!(doc.nodes[1].kind, NodeKind::Latent);
        assert_eq!(doc.nodes[1].position, None);
        // Options are skipped, and the coordinates after them still parse.
        assert_eq!(doc.nodes[2].position, Some(Point::new(0.0, 0.0)));
    }

    #[test]
    fn a_task_is_optional() {
        let doc = parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n").unwrap();
        assert!(doc.task.is_none());
        assert!(doc.query().is_none());
    }

    #[test]
    fn query_borrows_the_three_sets() {
        let doc = parse(G1).unwrap();
        let (x, y, z) = doc.query().expect("the file has a task");
        assert_eq!(x, vec!["S"]);
        assert_eq!(y, vec!["W"]);
        assert!(z.is_empty());
        // And they plug straight into the graph algorithms.
        assert!(!doc.graph.is_d_separated(&x, &y, &z).unwrap());
    }

    #[test]
    fn errors_carry_line_numbers() {
        let err = parse("<NODES>\nX\n<WRONG>\n").unwrap_err();
        assert_eq!(err.line, 3);
        assert_eq!(err.kind, ErrorKind::UnknownTag("<WRONG>".to_string()));

        let err = parse("<NODES>\nX\nX\n<EDGES>\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::DuplicateNode("X".to_string()));
        assert_eq!(err.line, 3);

        let err = parse("<NODES>\nX\n<EDGES>\nX -> Q\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownNode("Q".to_string()));

        let err = parse("<NODES>\nX\nY\n<EDGES>\nX => Y\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownEdgeType("=>".to_string()));

        let err = parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n<TASK>\nnope: X\n").unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::UnknownTaskDefinition("nope".to_string())
        );

        // A missing <NODES> is reported as such, rather than as the undeclared
        // variable it happens to make X into.
        let err = parse("<EDGES>\nX -> Y\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::MissingSection("<NODES>"));

        let err = parse("<NODES>\nX\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::MissingSection("<EDGES>"));

        // Stray text before any tag.
        let err = parse("hello\n<NODES>\nX\n<EDGES>\n").unwrap_err();
        assert_eq!(err.line, 1);
    }

    #[test]
    fn a_task_referring_to_an_undeclared_variable_is_rejected() {
        let err =
            parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n<TASK>\ntreatment: Q\n").unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownNode("Q".to_string()));
    }

    #[test]
    fn task_lists_are_split_and_trimmed() {
        let doc =
            parse("<NODES>\nX\nY\nZ\n<EDGES>\nX -> Y\n<TASK>\nadjusted: X, Z\n").unwrap();
        let task = doc.task.unwrap();
        assert_eq!(task.adjusted, vec!["X", "Z"]);
    }
}
