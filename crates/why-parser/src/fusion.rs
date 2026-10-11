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
//! and `<TASK>`, which is everything the chapter 2 and 4 notebooks exercise,
//! plus the σ-calculus pair `<QUERY>` and `<INTERVENTIONS>`; any other tag is
//! rejected rather than silently ignored.
//!
//! # σ-calculus queries
//!
//! `<QUERY>` names a rule and the sets of
//! [`why_data::graph::sigmacalculus`]; `<INTERVENTIONS>` gives the policies,
//! grouped under `interventional:` (`σ_X`, the default), `treatment_current:`
//! (`σ_Z`) and `treatment_new:` (`σ_Z'`). Each policy is
//! `target type [inputs]`, the type one of `idle`, `atomic`, `conditional` or
//! `stochastic`, the inputs a comma list for the last two.
//!
//! ```
//! use why_parser::fusion;
//!
//! let doc = fusion::parse("
//! <NODES>
//! A
//! X
//! Y
//! Z
//!
//! <EDGES>
//! Z -> X
//! X -> Y
//! A -> X
//! A -> Y
//!
//! <QUERY>
//! rule: 1
//! outcome: Y
//! treatment: Z
//! interventional: X
//!
//! <INTERVENTIONS>
//! X conditional A
//! ").unwrap();
//! let r = doc.sigma_calculus().unwrap().unwrap();
//! assert!(r.holds);
//! assert_eq!(r.rewrite, "P(y | z; σ_x) = P(y | σ_x)");
//! ```
//!
//! Note that `treatment:` means different things in the two sections that use
//! it: `X`, the intervened set, in `<TASK>`, but `Z`, the variable a rule adds
//! or removes, in `<QUERY>`. The names are the book's.
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

use why_data::graph::dseparation::{Admg, DSepError};
use why_data::graph::sigmacalculus::{self, Applicability, SigmaError};
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

/// The `<QUERY>` section: a σ-calculus rule and the sets it is stated over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// `rule:` — 1, 2 or 3. The book defaults it to 1 when absent.
    pub rule: u8,
    /// `outcome:` — `Y`.
    pub outcome: Vec<String>,
    /// `treatment:` — `Z`, the variable the rule adds, removes or re-regimes.
    pub treatment: Vec<String>,
    /// `conditional:` — `W`.
    pub conditional: Vec<String>,
    /// `interventional:` — `X`, the variables under the policies `σ_X`.
    pub interventional: Vec<String>,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            rule: 1,
            outcome: Vec::new(),
            treatment: Vec::new(),
            conditional: Vec::new(),
            interventional: Vec::new(),
        }
    }
}

/// How a policy in `<INTERVENTIONS>` sets its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    /// `idle`: the natural mechanism.
    Idle,
    /// `atomic`: a constant.
    Atomic,
    /// `conditional`: a function of the inputs.
    Conditional,
    /// `stochastic`: a distribution given the inputs.
    Stochastic,
}

/// One policy line of `<INTERVENTIONS>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervention {
    /// The variable whose mechanism is replaced.
    pub target: String,
    /// What replaces it.
    pub kind: PolicyKind,
    /// The inputs of a conditional or stochastic policy; empty otherwise.
    pub inputs: Vec<String>,
}

impl Intervention {
    /// The policy as [`why_data::graph::sigmacalculus`] takes it.
    #[must_use]
    pub fn to_sigma(&self) -> sigmacalculus::Intervention<'_> {
        let inputs: Vec<&str> = self.inputs.iter().map(String::as_str).collect();
        match self.kind {
            PolicyKind::Idle => sigmacalculus::Intervention::idle(&self.target),
            PolicyKind::Atomic => sigmacalculus::Intervention::atomic(&self.target),
            PolicyKind::Conditional => {
                sigmacalculus::Intervention::conditional(&self.target, &inputs)
            }
            PolicyKind::Stochastic => sigmacalculus::Intervention::stochastic(&self.target, &inputs),
        }
    }
}

/// The `<INTERVENTIONS>` section, by group.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interventions {
    /// Under `interventional:`, or before any group header — `σ_X`.
    pub interventional: Vec<Intervention>,
    /// Under `treatment_current:` — `σ_Z`.
    pub treatment_current: Vec<Intervention>,
    /// Under `treatment_new:` — `σ_Z'`.
    pub treatment_new: Vec<Intervention>,
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
    /// The σ-calculus query, when the file had a `<QUERY>` section.
    pub sigma_query: Option<Query>,
    /// The σ-calculus policies, when the file had an `<INTERVENTIONS>` section.
    pub interventions: Option<Interventions>,
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

    /// Tests the `<QUERY>` rule against the graph, with the policies of
    /// `<INTERVENTIONS>` — what the book's `SigmaCalculusInspector.test` does.
    ///
    /// A variable listed under `interventional:` with no policy of its own is
    /// taken as idle, and likewise for `treatment:` in rules 2 and 3, so the
    /// regime is still written in the rewriting.
    ///
    /// Returns `None` when the file declared no `<QUERY>`.
    ///
    /// # Errors
    ///
    /// [`SigmaError::Graph`] with [`DSepError::Empty`] if the query has no
    /// outcome or no treatment — the book returns nothing in that case — and
    /// any error the rule itself reports.
    #[must_use]
    pub fn sigma_calculus(&self) -> Option<Result<Applicability, SigmaError>> {
        fn refs(v: &[String]) -> Vec<&str> {
            v.iter().map(String::as_str).collect()
        }
        fn convert(v: &[Intervention]) -> Vec<sigmacalculus::Intervention<'_>> {
            v.iter().map(Intervention::to_sigma).collect()
        }
        let q = self.sigma_query.as_ref()?;
        if q.outcome.is_empty() {
            return Some(Err(DSepError::Empty("outcome").into()));
        }
        if q.treatment.is_empty() {
            return Some(Err(DSepError::Empty("treatment").into()));
        }
        let none = Interventions::default();
        let intvs = self.interventions.as_ref().unwrap_or(&none);

        let mut sigma_x = convert(&intvs.interventional);
        for v in &q.interventional {
            if !sigma_x.iter().any(|i| i.target == v) {
                sigma_x.push(sigmacalculus::Intervention::idle(v));
            }
        }
        let sigma_z = convert(&intvs.treatment_current);
        let sigma_z_prime = convert(&intvs.treatment_new);
        let (y, z, w) = (refs(&q.outcome), refs(&q.treatment), refs(&q.conditional));
        Some(match q.rule {
            1 => sigmacalculus::rule_1(&self.graph, &y, &z, &w, &sigma_x),
            2 => sigmacalculus::rule_2(&self.graph, &y, &z, &w, &sigma_x, &sigma_z, &sigma_z_prime),
            _ => sigmacalculus::rule_3(&self.graph, &y, &z, &w, &sigma_x, &sigma_z, &sigma_z_prime),
        })
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
    /// A line in `<QUERY>` that is not `definition: value`.
    MalformedQuery,
    /// A query definition other than `rule`, `outcome`, `treatment`,
    /// `conditional` or `interventional`.
    UnknownQueryDefinition(String),
    /// A `rule:` that is not 1, 2 or 3.
    InvalidRule(String),
    /// A line in `<INTERVENTIONS>` that is not `target type [inputs]`, or an
    /// `idle` or `atomic` policy given inputs.
    MalformedIntervention,
    /// A policy type other than `idle`, `atomic`, `conditional` or
    /// `stochastic`.
    UnknownInterventionType(String),
    /// A group header other than `interventional:`, `treatment_current:` or
    /// `treatment_new:`.
    UnknownInterventionGroup(String),
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
            ErrorKind::MalformedQuery => f.write_str("expected `definition: value`"),
            ErrorKind::UnknownQueryDefinition(d) => write!(f, "{d} is not a query definition"),
            ErrorKind::InvalidRule(r) => write!(f, "{r} is not a rule; expected 1, 2 or 3"),
            ErrorKind::MalformedIntervention => {
                f.write_str("expected `target type [inputs]`, with inputs only for conditional or stochastic")
            }
            ErrorKind::UnknownInterventionType(t) => write!(
                f,
                "{t} is not a policy type; expected idle, atomic, conditional or stochastic"
            ),
            ErrorKind::UnknownInterventionGroup(g) => write!(
                f,
                "{g} is not a group; expected interventional, treatment_current or treatment_new"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

/// One section: its tag, and its lines paired with their 1-based numbers.
type Section<'a> = (String, Vec<(usize, &'a str)>);

/// Splits the input into sections.
fn sections(input: &str) -> Result<Vec<Section<'_>>, ParseError> {
    const KNOWN: [&str; 5] = ["<NODES>", "<EDGES>", "<TASK>", "<QUERY>", "<INTERVENTIONS>"];
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

/// Splits a comma list, trimming each name and dropping empty ones.
///
/// The book's parser does not trim; trimming here means `X, Y` behaves as it
/// reads.
fn names(list: &str) -> Vec<String> {
    list.split(',')
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(ToString::to_string)
        .collect()
}

/// One line of `<INTERVENTIONS>`: a group header, or a policy.
enum InterventionLine {
    Group(String),
    Policy(Intervention),
}

fn parse_intervention(line: &str, number: usize) -> Result<InterventionLine, ParseError> {
    let err = |kind| ParseError { line: number, kind };
    let (target, rest) = token(line);
    if rest.trim().is_empty()
        && let Some(group) = target.strip_suffix(':')
    {
        return Ok(InterventionLine::Group(group.to_string()));
    }
    let (kind, inputs) = token(rest);
    let kind = match kind {
        "idle" => PolicyKind::Idle,
        "atomic" => PolicyKind::Atomic,
        "conditional" => PolicyKind::Conditional,
        "stochastic" => PolicyKind::Stochastic,
        "" => return Err(err(ErrorKind::MalformedIntervention)),
        other => return Err(err(ErrorKind::UnknownInterventionType(other.to_string()))),
    };
    let inputs = names(inputs);
    // The book ignores inputs on these two; a policy that ignores its inputs
    // is more likely a typo for `conditional` than anything meant.
    if matches!(kind, PolicyKind::Idle | PolicyKind::Atomic) && !inputs.is_empty() {
        return Err(err(ErrorKind::MalformedIntervention));
    }
    Ok(InterventionLine::Policy(Intervention {
        target: target.to_string(),
        kind,
        inputs,
    }))
}

/// Parses a document in the book's `<NODES>/<EDGES>/<TASK>` format.
///
/// `<NODES>` and `<EDGES>` are required, `<TASK>`, `<QUERY>` and
/// `<INTERVENTIONS>` optional, and the sections may
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
            let (definition, list) = line.split_once(':').ok_or(ParseError {
                line: number,
                kind: ErrorKind::MalformedTask,
            })?;
            let names = names(list);
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

    let sigma_query = if sections.iter().any(|(t, _)| t == "<QUERY>") {
        let mut query = Query::default();
        for (number, line) in find("<QUERY>") {
            let (definition, value) = line.split_once(':').ok_or(ParseError {
                line: number,
                kind: ErrorKind::MalformedQuery,
            })?;
            let definition = definition.trim();
            if definition == "rule" {
                query.rule = match value.trim().parse::<u8>() {
                    Ok(r @ 1..=3) => r,
                    _ => {
                        return Err(ParseError {
                            line: number,
                            kind: ErrorKind::InvalidRule(value.trim().to_string()),
                        });
                    }
                };
                continue;
            }
            let names = names(value);
            for name in &names {
                declared(name, number)?;
            }
            match definition {
                "outcome" => query.outcome = names,
                "treatment" => query.treatment = names,
                "conditional" => query.conditional = names,
                "interventional" => query.interventional = names,
                other => {
                    return Err(ParseError {
                        line: number,
                        kind: ErrorKind::UnknownQueryDefinition(other.to_string()),
                    });
                }
            }
        }
        Some(query)
    } else {
        None
    };

    let interventions = if sections.iter().any(|(t, _)| t == "<INTERVENTIONS>") {
        let mut out = Interventions::default();
        let mut group = &mut out.interventional;
        for (number, line) in find("<INTERVENTIONS>") {
            match parse_intervention(line, number)? {
                InterventionLine::Group(g) => {
                    group = match g.as_str() {
                        "interventional" => &mut out.interventional,
                        "treatment_current" => &mut out.treatment_current,
                        "treatment_new" => &mut out.treatment_new,
                        _ => {
                            return Err(ParseError {
                                line: number,
                                kind: ErrorKind::UnknownInterventionGroup(g),
                            });
                        }
                    };
                }
                InterventionLine::Policy(p) => {
                    declared(&p.target, number)?;
                    for input in &p.inputs {
                        declared(input, number)?;
                    }
                    group.push(p);
                }
            }
        }
        Some(out)
    } else {
        None
    };

    Ok(Document {
        nodes,
        graph,
        task,
        sigma_query,
        interventions,
    })
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
    /// Builds a σ-calculus document from its graph, query and interventions.
    fn sigma_doc(nodes: &str, edges: &str, query: &str, intvs: &str) -> String {
        let nodes = nodes.replace(' ', "\n");
        let edges = edges.replace(", ", "\n");
        let query = query.replace("; ", "\n");
        let intvs = intvs.replace("; ", "\n");
        format!("<NODES>\n{nodes}\n<EDGES>\n{edges}\n<QUERY>\n{query}\n<INTERVENTIONS>\n{intvs}\n")
    }

    #[test]
    fn parses_query_and_interventions() {
        let doc = parse(&sigma_doc(
            "Y Z A",
            "Z -> Y, A -> Z, A -> Y",
            "rule: 2; outcome: Y; treatment: Z; conditional: A",
            "treatment_current:; Z atomic; treatment_new:; Z stochastic A, Y",
        ))
        .unwrap();
        let q = doc.sigma_query.as_ref().unwrap();
        assert_eq!(q.rule, 2);
        assert_eq!(q.outcome, vec!["Y"]);
        assert_eq!(q.treatment, vec!["Z"]);
        assert_eq!(q.conditional, vec!["A"]);
        assert!(q.interventional.is_empty());
        let i = doc.interventions.as_ref().unwrap();
        assert!(i.interventional.is_empty());
        assert_eq!(i.treatment_current[0].kind, PolicyKind::Atomic);
        assert_eq!(
            i.treatment_new[0],
            Intervention {
                target: "Z".into(),
                kind: PolicyKind::Stochastic,
                inputs: vec!["A".into(), "Y".into()],
            }
        );
        // The task section stays separate.
        assert!(doc.task.is_none());
    }

    #[test]
    fn sigma_calculus_agrees_with_the_book_where_the_book_is_right() {
        // Each file was run through the book's SigmaCalculusInspector. The
        // last field is its answer; where it differs, see the module docs of
        // why_data::graph::sigmacalculus.
        let cases = [
            // σ_X sets X from A alone, cutting Z -> X.
            ("X Y Z A", "Z -> X, X -> Y, A -> X, A -> Y", "rule: 1; outcome: Y; treatment: Z; interventional: X",
             "interventional:; X conditional A", true, true),
            // No policy: Z -> X -> Y stays open.
            ("X Y Z A", "Z -> X, X -> Y, A -> X, A -> Y", "rule: 1; outcome: Y; treatment: Z",
             "", false, false),
            // The new edge A -> X opens Z -> A -> X -> Y; the book never adds it.
            ("X Y Z A", "Z -> A, X -> Y", "rule: 1; outcome: Y; treatment: Z; interventional: X",
             "X conditional A", false, true),
            // σ_Z' keeps the back door Z <- A -> Y.
            ("Y Z A", "Z -> Y, A -> Z, A -> Y", "rule: 2; outcome: Y; treatment: Z",
             "treatment_current:; Z atomic; treatment_new:; Z conditional A", false, false),
            // ... which conditioning on A closes.
            ("Y Z A", "Z -> Y, A -> Z, A -> Y", "rule: 2; outcome: Y; treatment: Z; conditional: A",
             "treatment_current:; Z atomic; treatment_new:; Z conditional A", true, true),
            ("Y Z W", "Z -> W, Z -- Y", "rule: 3; outcome: Y; treatment: Z",
             "treatment_current:; Z atomic; treatment_new:; Z idle", true, true),
            // W keeps Z's arrowheads, so Z <-> Y survives the idle σ_Z'; the
            // book's do(z) cut leaks into that graph.
            ("Y Z W", "Z -> W, Z -- Y", "rule: 3; outcome: Y; treatment: Z; conditional: W",
             "treatment_current:; Z atomic; treatment_new:; Z idle", false, true),
            // Overlining Z cuts A -> Z; the book spares σ_Z's inputs.
            ("Y Z A", "A -> Y, A -> Z", "rule: 3; outcome: Y; treatment: Z",
             "treatment_current:; Z conditional A; treatment_new:; Z atomic", true, false),
        ];
        for (nodes, edges, query, intvs, holds, _book) in cases {
            let doc = parse(&sigma_doc(nodes, edges, query, intvs)).unwrap();
            let r = doc.sigma_calculus().unwrap().unwrap();
            assert_eq!(r.holds, holds, "{query} with {intvs}\n{r}");
        }
    }

    #[test]
    fn sigma_calculus_fills_in_idle_regimes() {
        // X is listed as interventional but given no policy: it is idle, so
        // the graph is untouched, but the regime is still written.
        let doc = parse(&sigma_doc(
            "X Y Z",
            "Z -> X, X -> Y",
            "rule: 1; outcome: Y; treatment: Z; interventional: X",
            "",
        ))
        .unwrap();
        let r = doc.sigma_calculus().unwrap().unwrap();
        assert!(!r.holds);
        assert_eq!(r.rewrite, "P(y | z; σ_x) = P(y | σ_x)");
    }

    #[test]
    fn sigma_calculus_needs_a_query_with_y_and_z() {
        let doc = parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n").unwrap();
        assert!(doc.sigma_calculus().is_none());
        let doc = parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n<QUERY>\noutcome: Y\n").unwrap();
        assert_eq!(
            doc.sigma_calculus().unwrap().unwrap_err(),
            SigmaError::Graph(DSepError::Empty("treatment"))
        );
        // The rule defaults to 1, as in the book.
        let doc = parse("<NODES>\nX\nY\n<EDGES>\nX -> Y\n<QUERY>\noutcome: Y\ntreatment: X\n")
            .unwrap();
        assert_eq!(doc.sigma_calculus().unwrap().unwrap().rule.number(), 1);
    }

    #[test]
    fn query_and_intervention_errors() {
        let base = "<NODES>\nX\nY\n<EDGES>\nX -> Y\n";
        let kind = |tail: &str| parse(&format!("{base}{tail}")).unwrap_err().kind;
        assert_eq!(kind("<QUERY>\nrule: 4\n"), ErrorKind::InvalidRule("4".into()));
        assert_eq!(kind("<QUERY>\nrule: one\n"), ErrorKind::InvalidRule("one".into()));
        assert_eq!(kind("<QUERY>\nadjusted: X\n"), ErrorKind::UnknownQueryDefinition("adjusted".into()));
        assert_eq!(kind("<QUERY>\noutcome Y\n"), ErrorKind::MalformedQuery);
        assert_eq!(kind("<QUERY>\noutcome: Q\n"), ErrorKind::UnknownNode("Q".into()));
        assert_eq!(kind("<INTERVENTIONS>\nX\n"), ErrorKind::MalformedIntervention);
        assert_eq!(kind("<INTERVENTIONS>\nX random\n"), ErrorKind::UnknownInterventionType("random".into()));
        assert_eq!(kind("<INTERVENTIONS>\nX atomic Y\n"), ErrorKind::MalformedIntervention);
        assert_eq!(kind("<INTERVENTIONS>\ncurrent:\n"), ErrorKind::UnknownInterventionGroup("current".into()));
        assert_eq!(kind("<INTERVENTIONS>\nQ atomic\n"), ErrorKind::UnknownNode("Q".into()));
        assert_eq!(kind("<INTERVENTIONS>\nX conditional Q\n"), ErrorKind::UnknownNode("Q".into()));
        let err = parse(&format!("{base}<QUERY>\noutcome: Y\nrule: 9\n")).unwrap_err();
        assert_eq!(err.line, 8);
    }
}
