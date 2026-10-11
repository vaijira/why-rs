//! C-INFER at layer 3: identifying counterfactual queries.
//!
//! The Causal AI book's chapter 5 carries C-INFER from interventional queries
//! `P(y | do(x))` to counterfactual ones such as the effect of treatment on the
//! treated `P(y_x | x')`, joint statements across worlds `P(y_x, z_{x'})`, and
//! nested ones such as the natural direct effect's `P(y_{x, M_{x'}})`. It is
//! the counterfactual identification algorithm of Correa, Lee and Bareinboim
//! (2021), laid out as factor trees.
//!
//! The query tree ([`gen_query_tree`]) takes the query apart:
//!
//! 1. **CUT** — the counterfactual unnesting theorem replaces each nested
//!    subscript by a summed value: `P(y_{X_z}) = sum_x P(y_x, x_z)`.
//! 2. **≡** — the query is rewritten into an equivalent canonical one: each
//!    subscript is cut down to the variables that can still reach its
//!    variable, `y_y = y` is dropped as certain, and on the conditioning
//!    event a conditioned `x_t` is pushed into the subscripts of its
//!    descendants by composition — `Y_t = Y_{t, x}` whenever `X_t = x`.
//! 3. **Σ** — the query is the marginal of its counterfactual ancestors
//!    `An(Y_*)`: each `W ∈ An(Y)` in `G` with `T` underlined, in the world of
//!    the `T` that can still reach it. Every ancestor not in the query gets a
//!    fresh value and is summed out.
//! 4. **Γ** — a conditional `P(y_* | x_*)` is a ratio of two such sums.
//!    Factors linked to `Y_*` by neither a summed value nor a shared
//!    c-component appear identically above and below the line and cancel.
//! 5. **δ** — the ancestral set factorises into ctf-factors, one per
//!    c-component of `G` restricted to its variables. Each counterfactual
//!    `W_t` is written `W[pa_w]`: its parents fixed to their values in world
//!    `t`.
//! 6. **Ξ** — a ctf-factor is **consistent** when no variable takes two values
//!    across its members and their parents. A consistent one collapses to the
//!    interventional c-factor `Q[C](c; pa_C)`. An inconsistent one, such as
//!    `P(y_x, x')` with `X <-> Y`, is not identifiable, and neither is the query.
//!
//! The c-factors at the leaves are then found in the d-trees of the inputs,
//! exactly as in [layer 2](super), and each is evaluated at its own values.
//!
//! ```
//! use why_data::graph::cinfer::Input;
//! use why_data::graph::cinfer::counterfactual::{CtfQuery, ctf_infer};
//! use why_data::graph::dseparation::Admg;
//!
//! // The front-door graph.
//! let mut g = Admg::new();
//! g.directed("X", "M");
//! g.directed("M", "Y");
//! g.bidirected("X", "Y");
//!
//! // The effect of treatment on the treated, identifiable here although
//! // X and Y are confounded.
//! let q = CtfQuery::parse(&g, "P(y_x | x')").unwrap();
//! let d = ctf_infer(&g, &q, &[Input::observational()]).unwrap();
//! assert_eq!(d.estimand.to_string(), "sum_{m} P(m | x) P(y | x', m)");
//!
//! // With an unobserved common cause it is not, in the plain bow arc.
//! let mut bow = Admg::new();
//! bow.directed("X", "Y");
//! bow.bidirected("X", "Y");
//! let q = CtfQuery::parse(&bow, "P(y_x | x')").unwrap();
//! assert!(ctf_infer(&bow, &q, &[Input::observational()]).is_err());
//! ```
//!
//! # Values
//!
//! Values are written as the textbooks write them: `y`, `y'`, `y''` for the
//! values of `Y`, distinguished by their primes. Different primes stand for
//! different values, which is what makes `P(y_y')` zero. Every estimand holds
//! for every value of each symbol, so `P(y_x | x')` stays correct even at
//! `x' = x`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use super::{CFactor, CTree, Input, Source, VarSet, c_components, evaluate, grow_input_tree, parents};
use crate::graph::dseparation::{Admg, DSepError};
use crate::graph::identification::Formula;

/// A value of `var`: `var` followed by `primes` primes, the name the
/// estimand uses for it.
fn value_name(var: &str, primes: usize) -> String {
    format!("{var}{}", "'".repeat(primes))
}

/// A value as written in a query: `x`, `x'`.
fn value_text(var: &str, primes: usize) -> String {
    format!("{}{}", var.to_lowercase(), "'".repeat(primes))
}

/// What a subscript sets a variable to.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Setting {
    /// A value, numbered by its primes: 0 for `x`, 1 for `x'`.
    Value(usize),
    /// Whatever value the variable takes in another world, as `X_z` does in
    /// `Y_{X_z}`.
    Nested(Box<Counterfactual>),
}

/// A counterfactual variable `V_t`: `V` in the world where the subscript `t`
/// was set by intervention.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Counterfactual {
    /// The variable.
    pub var: String,
    /// The intervention, keyed by the variable each entry sets.
    pub subscript: BTreeMap<String, Setting>,
}

impl Counterfactual {
    /// `var` in the actual world.
    #[must_use]
    pub fn new(var: &str) -> Self {
        Self {
            var: var.to_string(),
            subscript: BTreeMap::new(),
        }
    }

    /// The same counterfactual with `var` also set to its value with
    /// `primes` primes.
    #[must_use]
    pub fn set(mut self, var: &str, primes: usize) -> Self {
        self.subscript.insert(var.to_string(), Setting::Value(primes));
        self
    }

    /// The same counterfactual with `inner.var` also set to whatever value
    /// `inner` takes.
    #[must_use]
    pub fn nest(mut self, inner: Counterfactual) -> Self {
        self.subscript
            .insert(inner.var.clone(), Setting::Nested(Box::new(inner)));
        self
    }

    /// The event that this counterfactual takes its value with `primes`
    /// primes.
    #[must_use]
    pub fn is(self, primes: usize) -> Event {
        Event {
            ctf: self,
            value: primes,
        }
    }
}

fn write_subscript(f: &mut fmt::Formatter<'_>, sub: &BTreeMap<String, Setting>) -> fmt::Result {
    let items: Vec<String> = sub
        .iter()
        .map(|(k, s)| match s {
            Setting::Value(p) => value_text(k, *p),
            // Braced even when short, so `y_{M_{x'}, x}` reads unambiguously.
            Setting::Nested(c) => {
                let mut inner = String::new();
                for (k, s) in &c.subscript {
                    if !inner.is_empty() {
                        inner.push_str(", ");
                    }
                    match s {
                        Setting::Value(p) => inner.push_str(&value_text(k, *p)),
                        Setting::Nested(n) => inner.push_str(&n.to_string()),
                    }
                }
                if inner.is_empty() {
                    c.var.clone()
                } else {
                    format!("{}_{{{inner}}}", c.var)
                }
            }
        })
        .collect();
    match items.as_slice() {
        [] => Ok(()),
        [one] if matches!(sub.values().next(), Some(Setting::Value(_))) => write!(f, "_{one}"),
        _ => write!(f, "_{{{}}}", items.join(", ")),
    }
}

impl fmt::Display for Counterfactual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.var)?;
        write_subscript(f, &self.subscript)
    }
}

/// The event `V_t = v`, written `v_t`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Event {
    /// The counterfactual variable.
    pub ctf: Counterfactual,
    /// Its value, numbered by primes.
    pub value: usize,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&value_text(&self.ctf.var, self.value))?;
        write_subscript(f, &self.ctf.subscript)
    }
}

/// The query `P(events | given)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CtfQuery {
    /// The events whose probability is asked for.
    pub events: Vec<Event>,
    /// What is conditioned on; empty for a joint.
    pub given: Vec<Event>,
}

impl CtfQuery {
    /// `P(events)`.
    #[must_use]
    pub fn new(events: Vec<Event>) -> Self {
        Self {
            events,
            given: Vec::new(),
        }
    }

    /// The same query, conditioned on `given`.
    #[must_use]
    pub fn given(mut self, given: Vec<Event>) -> Self {
        self.given = given;
        self
    }

    /// Reads a query in textbook notation, such as `P(y_x | x')`,
    /// `y_{x, z'}, z` or `P(y_{x, M_{x'}})`.
    ///
    /// An event is a value — the variable's name with any primes — with an
    /// optional subscript after `_`. A subscript is one value, or a braced
    /// list of values and nested counterfactuals; a nested counterfactual is
    /// a name followed by its own subscript, as in `M_{x'}`. Names match the
    /// graph's variables ignoring case, so `y` is a value of `Y`. Since `_`
    /// introduces a subscript, a variable whose name contains one cannot be
    /// written this way; build the query with [`Counterfactual`] instead.
    ///
    /// The `P(` and `)` around the query are optional.
    ///
    /// # Errors
    ///
    /// [`ParseCtfError`], with the byte offset of the problem.
    pub fn parse(g: &Admg, input: &str) -> Result<Self, ParseCtfError> {
        Parser { g, s: input, at: 0 }.query()
    }
}

impl fmt::Display for CtfQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |es: &[Event]| es.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        if self.given.is_empty() {
            write!(f, "P({})", list(&self.events))
        } else {
            write!(f, "P({} | {})", list(&self.events), list(&self.given))
        }
    }
}

/// Why [`CtfQuery::parse`] rejected its input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseCtfError {
    /// The byte offset of the problem.
    pub at: usize,
    /// What was wrong there.
    pub message: String,
}

impl fmt::Display for ParseCtfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for ParseCtfError {}

struct Parser<'a> {
    g: &'a Admg,
    s: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn err<T>(&self, message: impl Into<String>) -> Result<T, ParseCtfError> {
        Err(ParseCtfError {
            at: self.at,
            message: message.into(),
        })
    }

    fn skip_ws(&mut self) {
        let rest = &self.s[self.at..];
        self.at += rest.len() - rest.trim_start().len();
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.s[self.at..].chars().next()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.at += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn query(mut self) -> Result<CtfQuery, ParseCtfError> {
        let wrapped = self.peek() == Some('P') && self.s[self.at + 1..].trim_start().starts_with('(');
        if wrapped {
            self.at += 1;
            self.eat('(');
        }
        let events = self.events()?;
        let given = if self.eat('|') { self.events()? } else { Vec::new() };
        if wrapped && !self.eat(')') {
            return self.err("expected `)`");
        }
        if self.peek().is_some() {
            return self.err("unexpected text after the query");
        }
        Ok(CtfQuery { events, given })
    }

    fn events(&mut self) -> Result<Vec<Event>, ParseCtfError> {
        let mut out = vec![self.event()?];
        while self.eat(',') {
            out.push(self.event()?);
        }
        Ok(out)
    }

    /// A name and its primes, resolved to a graph variable.
    fn name(&mut self) -> Result<(String, usize), ParseCtfError> {
        self.skip_ws();
        let start = self.at;
        let rest = &self.s[start..];
        let len = rest
            .find(|c: char| !c.is_ascii_alphanumeric())
            .unwrap_or(rest.len());
        if len == 0 || !rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
            return self.err("expected a variable");
        }
        let word = &rest[..len];
        self.at += len;
        let mut primes = 0;
        while self.s[self.at..].starts_with('\'') {
            primes += 1;
            self.at += 1;
        }
        let matches: Vec<&str> = self
            .g
            .variables()
            .into_iter()
            .filter(|v| v.eq_ignore_ascii_case(word))
            .collect();
        match matches.as_slice() {
            [v] => Ok(((*v).to_string(), primes)),
            [] => Err(ParseCtfError {
                at: start,
                message: format!("{word} is not a variable of the graph"),
            }),
            _ => Err(ParseCtfError {
                at: start,
                message: format!("{word} matches more than one variable when case is ignored"),
            }),
        }
    }

    fn event(&mut self) -> Result<Event, ParseCtfError> {
        let (var, value) = self.name()?;
        let subscript = self.subscript()?;
        Ok(Event {
            ctf: Counterfactual { var, subscript },
            value,
        })
    }

    fn subscript(&mut self) -> Result<BTreeMap<String, Setting>, ParseCtfError> {
        let mut sub = BTreeMap::new();
        if !self.s[self.at..].starts_with('_') {
            return Ok(sub);
        }
        self.at += 1;
        if self.eat('{') {
            loop {
                let at = self.at;
                let (var, primes) = self.name()?;
                let setting = if self.s[self.at..].starts_with('_') {
                    if primes > 0 {
                        self.at = at;
                        return self.err("a nested counterfactual takes no primes");
                    }
                    Setting::Nested(Box::new(Counterfactual {
                        subscript: self.subscript()?,
                        var: var.clone(),
                    }))
                } else {
                    Setting::Value(primes)
                };
                if sub.insert(var.clone(), setting).is_some() {
                    self.at = at;
                    return self.err(format!("{var} is set twice in one subscript"));
                }
                if !self.eat(',') {
                    break;
                }
            }
            if !self.eat('}') {
                return self.err("expected `}`");
            }
        } else {
            let (var, primes) = self.name()?;
            if self.s[self.at..].starts_with('_') {
                return self.err("a nested subscript needs braces, as in y_{X_z}");
            }
            sub.insert(var, Setting::Value(primes));
        }
        Ok(sub)
    }
}

/// A flat counterfactual: every subscript entry is a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Ctf {
    var: String,
    sub: BTreeMap<String, usize>,
}

/// A flat event, as the engine manipulates it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Ev {
    ctf: Ctf,
    value: usize,
}

impl Ev {
    fn public(&self) -> Event {
        Event {
            ctf: Counterfactual {
                var: self.ctf.var.clone(),
                subscript: self
                    .ctf
                    .sub
                    .iter()
                    .map(|(k, v)| (k.clone(), Setting::Value(*v)))
                    .collect(),
            },
            value: self.value,
        }
    }
}

/// A value symbol: a variable and its primes.
type Sym = (String, usize);

/// What a node of the counterfactual q-tree stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtfTerm {
    /// A counterfactual query: the one asked at the root, or its rewritten
    /// form.
    Query(CtfQuery),
    /// A joint over counterfactual events — the ancestral set, or one
    /// ctf-factor written `W[pa_w]`.
    Joint(Vec<Event>),
    /// An interventional c-factor evaluated at the values of its scope and
    /// of its parents outside it, as the Ξ-operator leaves it.
    Factor {
        /// The c-factor `Q[C | do(pa_C)]`.
        factor: CFactor,
        /// The value of each variable of the scope and of `do`, by primes.
        at: BTreeMap<String, usize>,
    },
}

impl fmt::Display for CtfTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query(q) => q.fmt(f),
            Self::Joint(es) => {
                let es: Vec<String> = es.iter().map(ToString::to_string).collect();
                write!(f, "P({})", es.join(", "))
            }
            Self::Factor { factor, at } => {
                let vals = |s: &VarSet| -> String {
                    s.iter()
                        .map(|v| value_text(v, at[v]))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                if factor.doing.is_empty() {
                    write!(f, "{factor}({})", vals(&factor.scope))
                } else {
                    write!(f, "{factor}({}; {})", vals(&factor.scope), vals(&factor.doing))
                }
            }
        }
    }
}

/// An operator on an edge of the counterfactual q-tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtfOperator {
    /// CUT: the nested subscripts were replaced by these summed values.
    Unnest {
        /// The values introduced, each summed over.
        summed: Vec<(String, usize)>,
    },
    /// The query was rewritten into an equivalent canonical form.
    Rewrite,
    /// Σ, ancestral expansion: the parent is the child with these values
    /// summed out.
    Sigma {
        /// The values summed out.
        summed: Vec<(String, usize)>,
    },
    /// Γ, unconditioning: the parent is the child summed over these values,
    /// divided by the same summed also over the queried events.
    Gamma {
        /// The values summed out of the numerator.
        summed: Vec<(String, usize)>,
    },
    /// δ, ctf-factorisation by c-components.
    Delta,
    /// Ξ, collapsing a consistent ctf-factor to an interventional c-factor.
    Collapse,
}

fn write_syms(syms: &[(String, usize)]) -> String {
    syms.iter()
        .map(|(v, k)| value_text(v, *k))
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for CtfOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unnest { summed } => write!(f, "CUT Σ_{{{}}}", write_syms(summed)),
            Self::Rewrite => f.write_str("≡"),
            Self::Sigma { summed } => write!(f, "Σ_{{{}}}", write_syms(summed)),
            Self::Gamma { summed } => write!(f, "Γ Σ_{{{}}}", write_syms(summed)),
            Self::Delta => f.write_str("δ"),
            Self::Collapse => f.write_str("Ξ"),
        }
    }
}

/// A counterfactual q-tree.
pub type CtfTree = CTree<CtfTerm, CtfOperator>;

/// What a counterfactual query evaluates to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtfEstimand {
    /// The query contradicts itself, as `P(y_y')` or `P(y_x, y'_x)` do.
    Zero,
    /// Every queried event is certain given the rest, as in `P(y_y)`.
    One,
    /// An expression in the inputs.
    Formula(Formula),
}

impl fmt::Display for CtfEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => f.write_str("0"),
            Self::One => f.write_str("1"),
            Self::Formula(e) => e.fmt(f),
        }
    }
}

/// A successful identification, with the trees that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CtfDerivation {
    /// The q-tree.
    pub query_tree: CtfTree,
    /// One d-tree per input, in the order the inputs were given.
    pub input_trees: Vec<CTree>,
    /// The source of every Ξ leaf, keyed by the leaf's index in the q-tree.
    pub mapping: BTreeMap<usize, Source>,
    /// The estimand.
    pub estimand: CtfEstimand,
}

/// Why a counterfactual query could not be identified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtfError {
    /// A ctf-factor gives one variable two values, so the Ξ-operator does not
    /// apply, and by Lemma 5.5.1 the query is not identifiable from any
    /// combination of observational and experimental data.
    Inconsistent {
        /// The q-tree, down to the ctf-factors.
        query_tree: CtfTree,
        /// The inconsistent ctf-factor, written `W[pa_w]`.
        factor: Vec<Event>,
    },
    /// Every ctf-factor collapsed, but some c-factor appears in no d-tree.
    NotIdentifiable {
        /// The q-tree.
        query_tree: CtfTree,
        /// The d-trees.
        input_trees: Vec<CTree>,
        /// The c-factors no input could produce.
        unmapped: Vec<CFactor>,
    },
    /// The conditioning event contradicts itself, so the conditional is
    /// undefined.
    ImpossibleCondition,
    /// A nested counterfactual sits under another variable's key.
    MisplacedNesting(String),
    /// The query or an input names a variable not in the graph, or the graph
    /// is cyclic.
    Graph(DSepError),
}

impl From<DSepError> for CtfError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl fmt::Display for CtfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inconsistent { factor, .. } => {
                let es: Vec<String> = factor.iter().map(ToString::to_string).collect();
                write!(f, "not identifiable: the ctf-factor P({}) is inconsistent", es.join(", "))
            }
            Self::NotIdentifiable { unmapped, .. } => {
                let names: Vec<String> = unmapped.iter().map(ToString::to_string).collect();
                write!(f, "not identifiable: no input yields {}", names.join(", "))
            }
            Self::ImpossibleCondition => f.write_str("the conditioning event is impossible"),
            Self::MisplacedNesting(v) => {
                write!(f, "the subscript entry for {v} nests a different variable")
            }
            Self::Graph(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for CtfError {}

/// Which side of the conditioning bar an event is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Query,
    Given,
}

/// What normalising the events found.
enum Normal {
    Events(Vec<(Side, Ev)>),
    /// The queried events contradict each other or the conditioning event.
    Zero,
    /// The conditioning events contradict each other.
    Impossible,
}

/// The working state of one identification.
struct Engine<'g> {
    g: &'g Admg,
    /// Every value symbol in use, per variable, so fresh ones never collide.
    used: BTreeMap<String, BTreeSet<usize>>,
    /// The symbols summed over: introduced by CUT, or given to an ancestor.
    summed: BTreeSet<Sym>,
}

impl Engine<'_> {
    fn fresh(&mut self, var: &str) -> usize {
        let taken = self.used.entry(var.to_string()).or_default();
        let k = (0..).find(|k| !taken.contains(k)).expect("unbounded");
        taken.insert(k);
        k
    }

    fn note(&mut self, c: &Counterfactual, value: Option<usize>) {
        if let Some(v) = value {
            self.used.entry(c.var.clone()).or_default().insert(v);
        }
        for (k, s) in &c.subscript {
            match s {
                Setting::Value(p) => {
                    self.used.entry(k.clone()).or_default().insert(*p);
                }
                Setting::Nested(inner) => self.note(inner, None),
            }
        }
    }

    fn validate(&self, c: &Counterfactual) -> Result<(), CtfError> {
        if self.g.node(&c.var).is_none() {
            return Err(DSepError::Unknown(c.var.clone()).into());
        }
        for (k, s) in &c.subscript {
            if self.g.node(k).is_none() {
                return Err(DSepError::Unknown(k.clone()).into());
            }
            if let Setting::Nested(inner) = s {
                if inner.var != *k {
                    return Err(CtfError::MisplacedNesting(k.clone()));
                }
                self.validate(inner)?;
            }
        }
        Ok(())
    }

    /// Flattens `c`, replacing each nested subscript by a fresh summed value
    /// and adding the nested counterfactual, at that value, to `extra`.
    fn flatten(
        &mut self,
        c: &Counterfactual,
        extra: &mut Vec<Ev>,
        memo: &mut HashMap<Counterfactual, usize>,
    ) -> Ctf {
        let mut sub = BTreeMap::new();
        for (k, s) in &c.subscript {
            let v = match s {
                Setting::Value(p) => *p,
                Setting::Nested(inner) => {
                    if let Some(&v) = memo.get(&**inner) {
                        v
                    } else {
                        let flat = self.flatten(inner, extra, memo);
                        let v = self.fresh(&inner.var);
                        self.summed.insert((inner.var.clone(), v));
                        memo.insert((**inner).clone(), v);
                        extra.push(Ev { ctf: flat, value: v });
                        v
                    }
                }
            };
            sub.insert(k.clone(), v);
        }
        Ctf {
            var: c.var.clone(),
            sub,
        }
    }

    /// `c` with its subscript cut down to the variables that still reach its
    /// variable once the whole subscript is intervened on — the exclusion
    /// restrictions.
    fn minimal(&self, mut c: Ctf) -> Result<Ctf, DSepError> {
        if c.sub.is_empty() {
            return Ok(c);
        }
        let keys: Vec<&str> = c.sub.keys().map(String::as_str).collect();
        let reach: BTreeSet<String> = self
            .g
            .cut_edges_into(&keys)?
            .ancestors_of(&[&c.var])?
            .into_iter()
            .map(ToString::to_string)
            .collect();
        c.sub.retain(|k, _| reach.contains(k));
        Ok(c)
    }

    /// `An(c)`: each `W` that reaches `c.var` without passing through the
    /// subscript, in the world of the part of the subscript that reaches `W`.
    fn ancestors(&self, c: &Ctf) -> Result<Vec<Ctf>, DSepError> {
        let keys: Vec<&str> = c.sub.keys().map(String::as_str).collect();
        let an = self.g.cut_edges_out_of(&keys)?;
        an.ancestors_of(&[&c.var])?
            .into_iter()
            .map(|w| {
                self.minimal(Ctf {
                    var: w.to_string(),
                    sub: c.sub.clone(),
                })
            })
            .collect()
    }

    fn is_summed(&self, var: &str, value: usize) -> bool {
        self.summed.contains(&(var.to_string(), value))
    }

    /// Replaces the summed symbol `(var, from)` by `to` everywhere: summing
    /// over a value that an event pins down picks out that one term.
    fn unify(&mut self, evs: &mut [(Side, Ev)], var: &str, from: usize, to: usize) {
        self.summed.remove(&(var.to_string(), from));
        for (_, e) in evs.iter_mut() {
            if e.ctf.var == var && e.value == from {
                e.value = to;
            }
            if let Some(v) = e.ctf.sub.get_mut(var)
                && *v == from
            {
                *v = to;
            }
        }
    }

    /// Two values of `var` meet. Returns whether they could be unified, which
    /// changed the events; `None` if both are fixed and differ.
    fn reconcile(
        &mut self,
        evs: &mut [(Side, Ev)],
        var: &str,
        a: usize,
        b: usize,
    ) -> Option<()> {
        if self.is_summed(var, a) {
            self.unify(evs, var, a, b);
            Some(())
        } else if self.is_summed(var, b) {
            self.unify(evs, var, b, a);
            Some(())
        } else {
            None
        }
    }

    /// One pass of the rewriting; `Ok(None)` once nothing changes.
    #[allow(clippy::too_many_lines)]
    fn rewrite_once(&mut self, evs: &mut Vec<(Side, Ev)>) -> Result<Option<Normal>, DSepError> {
        for (_, e) in evs.iter_mut() {
            e.ctf = self.minimal(e.ctf.clone())?;
        }

        // Effectiveness: Y_{y0} = y0.
        for i in 0..evs.len() {
            let e = &evs[i].1;
            if let Some(&v0) = e.ctf.sub.get(&e.ctf.var) {
                if v0 == e.value {
                    evs.remove(i);
                    return Ok(Some(Normal::Events(Vec::new())));
                }
                let (var, value) = (e.ctf.var.clone(), e.value);
                if self.reconcile(evs, &var, v0, value).is_some() {
                    return Ok(Some(Normal::Events(Vec::new())));
                }
                return Ok(Some(if evs[i].0 == Side::Given {
                    Normal::Impossible
                } else {
                    Normal::Zero
                }));
            }
        }

        // The same counterfactual twice.
        for i in 0..evs.len() {
            for j in i + 1..evs.len() {
                if evs[i].1.ctf != evs[j].1.ctf {
                    continue;
                }
                let (si, sj) = (evs[i].0, evs[j].0);
                if evs[i].1.value == evs[j].1.value {
                    // A queried event already given is certain.
                    let drop = if si == Side::Query { i } else { j };
                    evs.remove(drop);
                    return Ok(Some(Normal::Events(Vec::new())));
                }
                let (var, a, b) = (evs[i].1.ctf.var.clone(), evs[i].1.value, evs[j].1.value);
                if self.reconcile(evs, &var, a, b).is_some() {
                    return Ok(Some(Normal::Events(Vec::new())));
                }
                return Ok(Some(if si == Side::Given && sj == Side::Given {
                    Normal::Impossible
                } else {
                    Normal::Zero
                }));
            }
        }

        // Composition: on the event X_t = x, every descendant Y_t of X_t is
        // Y_{t, x}.
        for i in 0..evs.len() {
            let an = self.ancestors(&evs[i].1.ctf)?;
            for j in 0..evs.len() {
                if i == j || evs[j].0 != Side::Given {
                    continue;
                }
                let x = &evs[j].1;
                if x.ctf.var != evs[i].1.ctf.var
                    && !evs[i].1.ctf.sub.contains_key(&x.ctf.var)
                    && an.contains(&x.ctf)
                {
                    let (var, value) = (x.ctf.var.clone(), x.value);
                    evs[i].1.ctf.sub.insert(var, value);
                    return Ok(Some(Normal::Events(Vec::new())));
                }
            }
        }

        // A nested counterfactual whose value nothing refers to any more sums
        // out: sum_x P(A, X_z = x) = P(A).
        for i in 0..evs.len() {
            let (var, value) = (evs[i].1.ctf.var.clone(), evs[i].1.value);
            if !self.is_summed(&var, value) {
                continue;
            }
            let referenced = evs.iter().enumerate().any(|(j, (_, e))| {
                (j != i && e.ctf.var == var && e.value == value) || e.ctf.sub.get(&var) == Some(&value)
            });
            if !referenced {
                self.summed.remove(&(var, value));
                evs.remove(i);
                return Ok(Some(Normal::Events(Vec::new())));
            }
        }
        Ok(None)
    }

    fn normalize(&mut self, mut evs: Vec<(Side, Ev)>) -> Result<Normal, DSepError> {
        loop {
            match self.rewrite_once(&mut evs)? {
                None => return Ok(Normal::Events(evs)),
                Some(Normal::Events(_)) => {}
                Some(other) => return Ok(other),
            }
        }
    }
}

/// One ctf-factor of the ancestral set.
struct CtfFactor {
    /// Indices into the ancestral set.
    members: Vec<usize>,
    /// The members written `W[pa_w]`.
    events: Vec<Event>,
    /// The value of every variable it mentions, or `None` if one has two.
    at: Option<BTreeMap<String, usize>>,
    /// Every value symbol it mentions.
    symbols: BTreeSet<Sym>,
    factor: CFactor,
}

/// The result of [`gen_query_tree`]: the tree, and what composing needs.
struct Plan {
    tree: CtfTree,
    /// The Ξ leaves: q-tree index, target c-factor and its values.
    leaves: Vec<(usize, CFactor, BTreeMap<String, usize>)>,
    /// Summed in the numerator, or in the whole estimand.
    summed: Vec<Sym>,
    /// For a conditional: the queried values, summed in the denominator, and
    /// the original symbol each private one stands for.
    conditional: Option<(Vec<Sym>, BTreeMap<String, String>)>,
}

enum Built {
    Plan(Plan),
    Constant(CtfTree, CtfEstimand),
}

/// `GENQUERYTREE` (Algorithm 42): takes the query apart into the
/// interventional c-factors it needs, as the module documentation describes.
///
/// # Errors
///
/// [`CtfError::Inconsistent`] when some ctf-factor cannot be collapsed;
/// [`CtfError::ImpossibleCondition`], [`CtfError::MisplacedNesting`] and
/// [`CtfError::Graph`] for malformed queries.
pub fn gen_query_tree(g: &Admg, query: &CtfQuery) -> Result<CtfTree, CtfError> {
    Ok(match build(g, query)? {
        Built::Plan(p) => p.tree,
        Built::Constant(t, _) => t,
    })
}

#[allow(clippy::too_many_lines)]
fn build(g: &Admg, query: &CtfQuery) -> Result<Built, CtfError> {
    if query.events.is_empty() {
        return Err(DSepError::Empty("the queried events").into());
    }
    let mut engine = Engine {
        g,
        used: BTreeMap::new(),
        summed: BTreeSet::new(),
    };
    for e in query.events.iter().chain(&query.given) {
        engine.validate(&e.ctf)?;
        engine.note(&e.ctf, Some(e.value));
    }

    // CUT.
    let mut memo = HashMap::new();
    let mut flat: Vec<(Side, Ev)> = Vec::new();
    for (side, list) in [(Side::Query, &query.events), (Side::Given, &query.given)] {
        for e in list {
            let mut extra = Vec::new();
            let ctf = engine.flatten(&e.ctf, &mut extra, &mut memo);
            flat.push((side, Ev { ctf, value: e.value }));
            flat.extend(extra.into_iter().map(|x| (side, x)));
        }
    }
    let cut: Vec<Sym> = engine.summed.iter().cloned().collect();

    let mut tree = CtfTree::new(CtfTerm::Query(query.clone()));
    let normal = engine.normalize(flat)?;
    let evs = match normal {
        Normal::Impossible => return Err(CtfError::ImpossibleCondition),
        Normal::Zero => return Ok(Built::Constant(tree, CtfEstimand::Zero)),
        Normal::Events(evs) => evs,
    };
    let rewritten = CtfQuery {
        events: evs.iter().filter(|(s, _)| *s == Side::Query).map(|(_, e)| e.public()).collect(),
        given: evs.iter().filter(|(s, _)| *s == Side::Given).map(|(_, e)| e.public()).collect(),
    };
    let mut cur = tree.root;
    if rewritten != *query {
        let op = if cut.is_empty() {
            CtfOperator::Rewrite
        } else {
            CtfOperator::Unnest { summed: cut }
        };
        cur = tree.add_child(cur, CtfTerm::Query(rewritten), op);
    }
    if evs.iter().all(|(s, _)| *s == Side::Given) {
        return Ok(Built::Constant(tree, CtfEstimand::One));
    }
    let conditional = evs.iter().any(|(s, _)| *s == Side::Given);
    let mut evs = evs;

    // In a conditional, give each queried value that is also written
    // elsewhere a private symbol, so that the denominator can sum it alone.
    let mut alias = BTreeMap::new();
    let mut queried: Vec<Sym> = Vec::new();
    if conditional {
        for i in 0..evs.len() {
            let (side, e) = &evs[i];
            if *side != Side::Query || engine.is_summed(&e.ctf.var, e.value) {
                continue;
            }
            let (var, value) = (e.ctf.var.clone(), e.value);
            let shared = evs.iter().enumerate().any(|(j, (_, o))| {
                (j != i && o.ctf.var == var && o.value == value) || o.ctf.sub.get(&var) == Some(&value)
            });
            let private = if shared {
                let k = engine.fresh(&var);
                alias.insert(value_name(&var, k), value_name(&var, value));
                evs[i].1.value = k;
                k
            } else {
                value
            };
            queried.push((var, private));
        }
    }

    // The counterfactual ancestors, each with a value.
    let mut members: Vec<(Side, Ev)> = evs;
    let mut index: HashMap<Ctf, usize> = members
        .iter()
        .enumerate()
        .map(|(i, (_, e))| (e.ctf.clone(), i))
        .collect();
    let mut next = 0;
    while next < members.len() {
        for a in engine.ancestors(&members[next].1.ctf)? {
            if !index.contains_key(&a) {
                let v = engine.fresh(&a.var);
                engine.summed.insert((a.var.clone(), v));
                index.insert(a.clone(), members.len());
                members.push((Side::Given, Ev { ctf: a, value: v }));
            }
        }
        next += 1;
    }

    // Each member as W[pa_w].
    let pa_form: Vec<BTreeMap<String, usize>> = members
        .iter()
        .map(|(_, m)| {
            parents(g, &VarSet::from([m.ctf.var.clone()]))
                .into_iter()
                .map(|p| {
                    let v = match m.ctf.sub.get(&p) {
                        Some(&v) => v,
                        None => {
                            let pc = engine.minimal(Ctf {
                                var: p.clone(),
                                sub: m.ctf.sub.clone(),
                            })?;
                            members[index[&pc]].1.value
                        }
                    };
                    Ok((p, v))
                })
                .collect::<Result<_, DSepError>>()
        })
        .collect::<Result<_, _>>()?;

    let vars: VarSet = members.iter().map(|(_, m)| m.ctf.var.clone()).collect();
    let refs: Vec<&str> = vars.iter().map(String::as_str).collect();
    let mut factors: Vec<CtfFactor> = c_components(&g.induced_subgraph(&refs)?)
        .into_iter()
        .map(|comp| {
            let ms: Vec<usize> = (0..members.len())
                .filter(|&i| comp.contains(&members[i].1.ctf.var))
                .collect();
            let mut at: Option<BTreeMap<String, usize>> = Some(BTreeMap::new());
            let mut symbols = BTreeSet::new();
            let mut assign = |var: &str, v: usize, at: &mut Option<BTreeMap<String, usize>>| {
                symbols.insert((var.to_string(), v));
                if let Some(map) = at
                    && *map.entry(var.to_string()).or_insert(v) != v
                {
                    *at = None;
                }
            };
            for &i in &ms {
                let m = &members[i].1;
                assign(&m.ctf.var, m.value, &mut at);
                for (p, &v) in &pa_form[i] {
                    assign(p, v, &mut at);
                }
            }
            let doing: VarSet = ms
                .iter()
                .flat_map(|&i| pa_form[i].keys().cloned())
                .filter(|p| !comp.contains(p))
                .collect();
            let events = ms
                .iter()
                .map(|&i| {
                    Ev {
                        ctf: Ctf {
                            var: members[i].1.ctf.var.clone(),
                            sub: pa_form[i].clone(),
                        },
                        value: members[i].1.value,
                    }
                    .public()
                })
                .collect();
            CtfFactor {
                members: ms,
                events,
                at,
                symbols,
                factor: CFactor { scope: comp, doing },
            }
        })
        .collect();

    // Γ: keep the factors linked to the queried events.
    if conditional {
        let link: BTreeSet<Sym> = engine.summed.iter().cloned().chain(queried.iter().cloned()).collect();
        let mut keep: Vec<bool> = factors
            .iter()
            .map(|f| f.members.iter().any(|&i| members[i].0 == Side::Query))
            .collect();
        loop {
            let reached: BTreeSet<&Sym> = factors
                .iter()
                .zip(&keep)
                .filter(|(_, k)| **k)
                .flat_map(|(f, _)| f.symbols.intersection(&link))
                .collect();
            let mut grew = false;
            for (f, k) in factors.iter().zip(keep.iter_mut()) {
                if !*k && f.symbols.iter().any(|s| reached.contains(s)) {
                    *k = true;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        let mut k = keep.into_iter();
        factors.retain(|_| k.next().expect("one flag per factor"));
    }

    let kept: BTreeSet<usize> = factors.iter().flat_map(|f| f.members.iter().copied()).collect();
    let summed: Vec<Sym> = kept
        .iter()
        .map(|&i| (members[i].1.ctf.var.clone(), members[i].1.value))
        .filter(|s| engine.summed.contains(s))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let joint = CtfTerm::Joint(kept.iter().map(|&i| members[i].1.public()).collect());
    let op = if conditional {
        CtfOperator::Gamma { summed: summed.clone() }
    } else {
        CtfOperator::Sigma { summed: summed.clone() }
    };
    let top = tree.add_child(cur, joint, op);

    let mut leaves = Vec::new();
    let mut inconsistent = None;
    for f in &factors {
        let parent = if factors.len() > 1 {
            tree.add_child(top, CtfTerm::Joint(f.events.clone()), CtfOperator::Delta)
        } else {
            top
        };
        match &f.at {
            Some(at) => {
                let leaf = tree.add_child(
                    parent,
                    CtfTerm::Factor {
                        factor: f.factor.clone(),
                        at: at.clone(),
                    },
                    CtfOperator::Collapse,
                );
                leaves.push((leaf, f.factor.clone(), at.clone()));
            }
            None => {
                inconsistent.get_or_insert_with(|| f.events.clone());
            }
        }
    }
    if let Some(factor) = inconsistent {
        return Err(CtfError::Inconsistent {
            query_tree: tree,
            factor,
        });
    }
    Ok(Built::Plan(Plan {
        tree,
        leaves,
        summed,
        conditional: conditional.then_some((queried, alias)),
    }))
}

/// `C-INFER` at layer 3 (Algorithm 43): identifies `query` from `inputs` and
/// the diagram `g`.
///
/// Builds the q-tree, a d-tree per input towards its Ξ leaves, maps each leaf
/// to the first input that reaches it, and composes the estimand: each leaf's
/// c-factor is evaluated along its d-tree path, at the leaf's values, and the
/// q-tree's Σ or Γ finishes the job.
///
/// # Errors
///
/// [`CtfError::Inconsistent`] when a ctf-factor cannot be collapsed, which
/// no data can rescue; [`CtfError::NotIdentifiable`] when the inputs given
/// cannot produce some c-factor; the other variants for malformed input.
pub fn ctf_infer(g: &Admg, query: &CtfQuery, inputs: &[Input]) -> Result<CtfDerivation, CtfError> {
    for input in inputs {
        for z in &input.doing {
            if g.node(z).is_none() {
                return Err(DSepError::Unknown(z.clone()).into());
            }
        }
    }
    let plan = match build(g, query)? {
        Built::Constant(query_tree, estimand) => {
            return Ok(CtfDerivation {
                query_tree,
                input_trees: Vec::new(),
                mapping: BTreeMap::new(),
                estimand,
            });
        }
        Built::Plan(p) => p,
    };

    let targets: Vec<CFactor> = plan.leaves.iter().map(|(_, c, _)| c.clone()).collect();
    let input_trees = inputs
        .iter()
        .map(|p| grow_input_tree(g, p, &targets))
        .collect::<Result<Vec<_>, _>>()?;
    let mut mapping = BTreeMap::new();
    let mut unmapped = Vec::new();
    for (leaf, c, _) in &plan.leaves {
        match input_trees
            .iter()
            .enumerate()
            .find_map(|(k, t)| t.find(c).map(|n| (k, n)))
        {
            Some((input, node)) => {
                mapping.insert(*leaf, Source::Input { input, node });
            }
            None => unmapped.push(c.clone()),
        }
    }
    if !unmapped.is_empty() {
        return Err(CtfError::NotIdentifiable {
            query_tree: plan.tree,
            input_trees,
            unmapped,
        });
    }

    let order = g.topological_order()?;
    // Parents before children, so the chain rule can merge runs of terms.
    let mut leaves: Vec<&(usize, CFactor, BTreeMap<String, usize>)> = plan.leaves.iter().collect();
    leaves.sort_by_key(|(_, c, _)| order.iter().position(|v| c.scope.contains(v)));
    let mut factors = Vec::new();
    for (leaf, _, at) in leaves {
        let Some(&Source::Input { input, node }) = mapping.get(leaf) else {
            unreachable!("every leaf was mapped")
        };
        let mut f = evaluate(g, &input_trees[input], node, &inputs[input], &order)?;
        let names: BTreeMap<String, String> =
            at.iter().map(|(v, &k)| (v.clone(), value_name(v, k))).collect();
        f.substitute_free(&names);
        factors.push(f);
    }
    factors.sort_by_key(|f| !matches!(f, Formula::P { .. }));
    let product = match Formula::product(factors) {
        p @ Formula::Product(_) => p,
        other => Formula::Product(vec![other]),
    };
    let names = |syms: &[Sym]| -> Vec<String> { syms.iter().map(|(v, k)| value_name(v, *k)).collect() };

    let formula = match &plan.conditional {
        None => Formula::sum(names(&plan.summed), product),
        Some((queried, alias)) => {
            let mut num = Formula::sum(names(&plan.summed), product.clone());
            num.substitute_free(alias);
            // The queried values first: once they cancel, a summed value may
            // be left heading its own term, and cancel too.
            let mut over = names(queried);
            over.extend(names(&plan.summed));
            let den = Formula::sum(over, product).simplify();
            cancel_common(num.simplify(), den)
        }
    };
    let mut formula = formula.simplify();
    // The query's own values are free too, even those that dropped out: a
    // sum reusing `x'` would read as the `x'` of the query.
    let mut free = formula.free_vars();
    free.extend(query_values(query));
    formula.avoid_capture(&free);
    Ok(CtfDerivation {
        query_tree: plan.tree,
        input_trees,
        mapping,
        estimand: CtfEstimand::Formula(formula),
    })
}

/// The name of every value `query` writes, nested ones included.
fn query_values(query: &CtfQuery) -> BTreeSet<String> {
    fn walk(c: &Counterfactual, out: &mut BTreeSet<String>) {
        for (k, s) in &c.subscript {
            match s {
                Setting::Value(p) => {
                    out.insert(value_name(k, *p));
                }
                Setting::Nested(inner) => walk(inner, out),
            }
        }
    }
    let mut out = BTreeSet::new();
    for e in query.events.iter().chain(&query.given) {
        out.insert(value_name(&e.ctf.var, e.value));
        walk(&e.ctf, &mut out);
    }
    out
}

/// `num / den`, with each term of `den` that is also a term of `num` struck
/// from both — in `num` only where no sum binds anything it mentions.
///
/// The factors that Γ keeps for their link to the conditioning event, such as
/// `P(x')` in the effect of treatment on the treated, often come out this way.
fn cancel_common(num: Formula, den: Formula) -> Formula {
    let terms = |f: &Formula| -> Option<Vec<Formula>> {
        match f {
            Formula::P { .. } => Some(vec![f.clone()]),
            Formula::Product(fs) if fs.iter().all(|t| matches!(t, Formula::P { .. })) => {
                Some(fs.clone())
            }
            _ => None,
        }
    };
    let (Some(mut den_terms), false) = (terms(&den), matches!(&den, Formula::Product(f) if f.is_empty()))
    else {
        return if matches!(&den, Formula::Product(f) if f.is_empty()) {
            num
        } else {
            Formula::Ratio {
                num: Box::new(num),
                den: Box::new(den),
            }
        };
    };
    let (over, mut num_terms) = match num {
        Formula::Sum { over, body } => match *body {
            Formula::Product(fs) => (over, fs),
            other => (over, vec![other]),
        },
        Formula::Product(fs) => (Vec::new(), fs),
        other => (Vec::new(), vec![other]),
    };
    den_terms.retain(|d| {
        let free = d.free_vars();
        let pos = num_terms
            .iter()
            .position(|t| t == d && over.iter().all(|v| !free.contains(v)));
        match pos {
            Some(i) => {
                num_terms.remove(i);
                false
            }
            None => true,
        }
    });
    let num = Formula::sum(over, Formula::product(num_terms));
    if den_terms.is_empty() {
        num
    } else {
        Formula::Ratio {
            num: Box::new(num),
            den: Box::new(Formula::product(den_terms)),
        }
    }
}

#[cfg(test)]
mod tests;
