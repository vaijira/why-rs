//! Symbolic structural causal models over discrete variables.
//!
//! A [`Model`] is a set of structural equations `F` over endogenous variables
//! `V`, driven by independent exogenous variables `U` with known discrete
//! distributions. Equations are [`rssn`] symbolic expressions, so they can be
//! inspected, substituted into and rendered, not just evaluated.
//!
//! Inference is exact and by enumeration: every assignment to `U` is evaluated
//! and the resulting joint over `V` is accumulated. That is tractable only
//! because the exogenous support is small, which is the regime this module
//! targets — the textbook models used to reason about d-separation,
//! interventions and counterfactuals.
//!
//! Interventions produce *new worlds*. [`Model::intervene`] rewrites the
//! equations with `X` clamped and renames every endogenous variable to
//! `{V}_{X=x}`, registering those names in the model. Because all worlds share
//! the same exogenous draw during enumeration, a single query can mix variables
//! from different worlds, which is what makes counterfactual quantities such as
//! `P(Y_{x=1} = 1, Y_{x=0} = 0)` computable.
//!
//! ```
//! use why_data::scm::{expr, Dist, Model, Value};
//!
//! // x := ux,  z := x & !uz,  y := z & uy
//! let mut m = Model::new(
//!     [
//!         ("x".to_string(), expr::var("ux")),
//!         ("z".to_string(), expr::and([expr::var("x"), expr::not(expr::var("uz"))])),
//!         ("y".to_string(), expr::and([expr::var("z"), expr::var("uy")])),
//!     ],
//!     [
//!         ("ux".to_string(), Dist::Bernoulli(0.5)),
//!         ("uz".to_string(), Dist::Bernoulli(0.5)),
//!         ("uy".to_string(), Dist::Bernoulli(0.5)),
//!     ],
//! )
//! .unwrap();
//!
//! let t = Value::Bool(true);
//! let obs = m.observational();
//! assert!((m.world(obs).query(&[("y", t)], &[]).unwrap() - 0.125).abs() < 1e-9);
//!
//! // Intervening on x severs it from ux; y is unaffected by the mutilation.
//! let w = m.intervene(&[("x", t)]).unwrap();
//! assert!((m.world(w).query(&[("y", t)], &[]).unwrap() - 0.25).abs() < 1e-9);
//! ```
//!
//! # Relationship to the Python `SymbolicSCM`
//!
//! This module is a port of the `scm.py` accompanying the Causal AI book, whose
//! models the tests below reproduce. The surface maps across directly:
//!
//! | `scm.py` | here |
//! | --- | --- |
//! | `SymbolicSCM(f, pu)` | [`Model::new`] |
//! | `m.do({x: 1})` | [`Model::intervene`] |
//! | `m.query(x, given)` | [`WorldRef::query`] |
//! | `m.get_probability_table()` | [`WorldRef::probability_table`] |
//! | `m.sample(n)` | [`WorldRef::sample`] |
//! | `_repr_mimebundle_` | `impl Display for Model` |
//!
//! Four things are deliberately *not* a literal translation. Each is a place
//! where following the Python would have been wrong or unidiomatic, so they are
//! worth knowing before changing anything here.
//!
//! ## Evaluation order is derived, not assumed
//!
//! `scm.py` evaluates equations in `dict` insertion order, so a model whose
//! equations are not written topologically silently computes a wrong joint —
//! nothing detects it. Here the causal diagram is built first, from the free
//! variables of each equation, and [`toposort`] supplies the evaluation order.
//! A mis-ordered model therefore evaluates correctly, and a genuinely recursive
//! one is rejected with [`ScmError::Cyclic`]. Interventions carry the mutilated
//! graph, which [`World::graph`] exposes for use with the graph algorithms in
//! [`crate::graph`].
//!
//! ## Worlds are owned by the model, not linked in a cycle
//!
//! Python keeps a `_counterfactuals` map on every SCM in which each SCM refers
//! to itself and its children. That is a reference cycle; expressed as
//! `Arc<RwLock<..>>` it would compile and then leak. Instead [`Model`] owns a
//! `Vec<World>` plus an `owner` map from every renamed variable to the world
//! defining it, and worlds are addressed by [`WorldId`]. Cross-world queries
//! still work — every world is evaluated under the same exogenous draw — but
//! ownership stays acyclic and no interior mutability is needed.
//!
//! ## Failure is a `Result`, not an `assert`
//!
//! The eleven `assert`s of the original became [`ScmError`], so malformed
//! models and impossible queries are recoverable rather than a panic.
//!
//! ## Substitution and evaluation are implemented here, not delegated
//!
//! The most important one. [`rssn`] supplies the expression AST, but its
//! algorithms are arithmetic-only and fail *silently* on logic:
//!
//! - `rssn::symbolic::calculus::substitute` walks arithmetic nodes and leaves
//!   logical ones untouched. `substitute(x & !uz, "x", true)` returns its input
//!   unchanged, with no error. Using it for the [`Model::intervene`] renaming
//!   would make every intervention quietly reproduce the un-intervened model.
//! - `rssn::symbolic::real_roots::eval_expr` returns `0.0` for any logic node.
//! - `rssn::output::latex::to_latex` renders `(x && !(uz))` rather than LaTeX,
//!   and its module is feature-gated behind `plotters`.
//!
//! So `substitute`, `eval` and `render` are local to this module, over the
//! [`Expr`] subset that [`expr`] can build. Extend those when the subset grows;
//! do not reach for the `rssn` equivalents. What `rssn` *is* used for is the
//! AST itself — including `And`/`Or`/`Not`/`Xor`, the comparisons, and `Apply`
//! for uninterpreted mechanisms such as chapter 4's `se := f_se(u_se)`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use petgraph::algo::toposort;
use rssn::symbolic::core::Expr;

use crate::graph::{DiGraph, NodeIndex};

/// Builders for the [`Expr`] subset this module understands.
///
/// [`rssn`] expressions are built out of [`Arc`]s, which makes writing a model
/// by hand noisy. These wrappers keep structural equations readable.
pub mod expr {
    use std::sync::Arc;

    use rssn::symbolic::core::Expr;

    /// A variable reference.
    #[must_use]
    pub fn var(name: &str) -> Expr {
        Expr::Variable(name.to_string())
    }

    /// A boolean literal.
    #[must_use]
    pub fn boolean(b: bool) -> Expr {
        Expr::Boolean(b)
    }

    /// An integer literal.
    #[must_use]
    pub fn int(i: i64) -> Expr {
        Expr::Constant(i as f64)
    }

    /// Conjunction of any number of operands.
    #[must_use]
    pub fn and(items: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::And(items.into_iter().collect())
    }

    /// Disjunction of any number of operands.
    #[must_use]
    pub fn or(items: impl IntoIterator<Item = Expr>) -> Expr {
        Expr::Or(items.into_iter().collect())
    }

    /// Negation.
    #[must_use]
    pub fn not(e: Expr) -> Expr {
        Expr::Not(Arc::new(e))
    }

    /// Exclusive or.
    #[must_use]
    pub fn xor(a: Expr, b: Expr) -> Expr {
        Expr::Xor(Arc::new(a), Arc::new(b))
    }

    /// Addition.
    #[must_use]
    pub fn add(a: Expr, b: Expr) -> Expr {
        Expr::Add(Arc::new(a), Arc::new(b))
    }

    /// Subtraction.
    #[must_use]
    pub fn sub(a: Expr, b: Expr) -> Expr {
        Expr::Sub(Arc::new(a), Arc::new(b))
    }

    /// Multiplication.
    #[must_use]
    pub fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Mul(Arc::new(a), Arc::new(b))
    }

    /// Arithmetic negation.
    #[must_use]
    pub fn neg(a: Expr) -> Expr {
        Expr::Neg(Arc::new(a))
    }

    /// Equality comparison.
    #[must_use]
    pub fn eq(a: Expr, b: Expr) -> Expr {
        Expr::Eq(Arc::new(a), Arc::new(b))
    }

    /// Strictly-less-than comparison.
    #[must_use]
    pub fn lt(a: Expr, b: Expr) -> Expr {
        Expr::Lt(Arc::new(a), Arc::new(b))
    }

    /// Less-or-equal comparison.
    #[must_use]
    pub fn le(a: Expr, b: Expr) -> Expr {
        Expr::Le(Arc::new(a), Arc::new(b))
    }

    /// Strictly-greater-than comparison.
    #[must_use]
    pub fn gt(a: Expr, b: Expr) -> Expr {
        Expr::Gt(Arc::new(a), Arc::new(b))
    }

    /// Greater-or-equal comparison.
    #[must_use]
    pub fn ge(a: Expr, b: Expr) -> Expr {
        Expr::Ge(Arc::new(a), Arc::new(b))
    }

    /// An uninterpreted function applied to arguments, as in `f_x(u_x)`.
    ///
    /// The head is a function symbol, not a variable, so it contributes no
    /// parent to the causal diagram. Such equations describe a model's shape
    /// without committing to a mechanism, and cannot be evaluated.
    #[must_use]
    pub fn apply(f: &str, args: impl IntoIterator<Item = Expr>) -> Expr {
        let mut args: Vec<Expr> = args.into_iter().collect();
        let arg = if args.len() == 1 {
            args.remove(0)
        } else {
            Expr::Tuple(args)
        };
        Expr::Apply(Arc::new(var(f)), Arc::new(arg))
    }
}

/// A value taken by a variable of the model.
///
/// Boolean and integer values interoperate: a boolean used arithmetically
/// counts as `0`/`1`, and an integer used logically is false exactly when it
/// is `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// A boolean value.
    Bool(bool),
    /// An integer value.
    Int(i64),
}

impl Value {
    /// The value as an integer, mapping `false`/`true` to `0`/`1`.
    #[must_use]
    pub fn as_int(self) -> i64 {
        match self {
            Self::Bool(b) => i64::from(b),
            Self::Int(i) => i,
        }
    }

    /// The value as a boolean, mapping any non-zero integer to `true`.
    #[must_use]
    pub fn as_bool(self) -> bool {
        match self {
            Self::Bool(b) => b,
            Self::Int(i) => i != 0,
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", i32::from(*b)),
            Self::Int(i) => write!(f, "{i}"),
        }
    }
}

/// The distribution of an exogenous variable.
#[derive(Clone, Debug, PartialEq)]
pub enum Dist {
    /// A Bernoulli variable, given the probability of `true`.
    Bernoulli(f64),
    /// A categorical variable over `0..n`, given the probability of each value.
    Categorical(Vec<f64>),
}

impl Dist {
    /// The values with non-zero probability.
    ///
    /// A Bernoulli variable always supports both booleans, so that a degenerate
    /// probability still yields a well-formed two-valued domain.
    #[must_use]
    pub fn support(&self) -> Vec<Value> {
        match self {
            Self::Bernoulli(_) => vec![Value::Bool(false), Value::Bool(true)],
            Self::Categorical(p) => p
                .iter()
                .enumerate()
                .filter(|(_, p)| **p != 0.0)
                .map(|(i, _)| Value::Int(i as i64))
                .collect(),
        }
    }

    /// The probability of a value, or `0.0` outside the support.
    #[must_use]
    pub fn probability(&self, v: Value) -> f64 {
        match self {
            Self::Bernoulli(p) => {
                if v.as_bool() {
                    *p
                } else {
                    1.0 - *p
                }
            }
            Self::Categorical(p) => {
                let i = v.as_int();
                if i < 0 {
                    0.0
                } else {
                    p.get(i as usize).copied().unwrap_or(0.0)
                }
            }
        }
    }

    fn validate(&self, name: &str) -> Result<(), ScmError> {
        let ps: &[f64] = match self {
            Self::Bernoulli(p) => std::slice::from_ref(p),
            Self::Categorical(p) => {
                if p.is_empty() {
                    return Err(ScmError::EmptyDistribution(name.to_string()));
                }
                p
            }
        };
        if ps.iter().any(|p| !(0.0..=1.0).contains(p)) {
            return Err(ScmError::ProbabilityOutOfRange(name.to_string()));
        }
        if let Self::Categorical(p) = self {
            let total: f64 = p.iter().sum();
            if (total - 1.0).abs() > 1e-9 {
                return Err(ScmError::NotNormalized {
                    variable: name.to_string(),
                    total,
                });
            }
        }
        Ok(())
    }
}

/// Anything that can go wrong building or querying a [`Model`].
#[derive(Clone, Debug, PartialEq)]
pub enum ScmError {
    /// A categorical distribution has no values.
    EmptyDistribution(String),
    /// A probability lies outside `[0, 1]`.
    ProbabilityOutOfRange(String),
    /// A categorical distribution does not sum to one.
    NotNormalized {
        /// The offending exogenous variable.
        variable: String,
        /// The sum that was found.
        total: f64,
    },
    /// The same variable was declared twice.
    Duplicate(String),
    /// An equation refers to a variable that is neither endogenous nor exogenous.
    Unknown(String),
    /// The structural equations are mutually recursive.
    Cyclic(Vec<String>),
    /// A variable expected to be endogenous is not.
    NotEndogenous(String),
    /// A query mentions the same variable on both sides of the conditioning bar.
    Overlap(String),
    /// An intervention would redefine variables of an existing world.
    WorldCollision(String),
    /// An expression cannot be evaluated to a discrete value.
    Unevaluable(String),
    /// Conditioning on an event of probability zero.
    ZeroProbability,
}

impl fmt::Display for ScmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution(v) => write!(f, "distribution of {v} is empty"),
            Self::ProbabilityOutOfRange(v) => {
                write!(f, "distribution of {v} has a probability outside [0, 1]")
            }
            Self::NotNormalized { variable, total } => {
                write!(f, "distribution of {variable} sums to {total}, not 1")
            }
            Self::Duplicate(v) => write!(f, "{v} is declared twice"),
            Self::Unknown(v) => write!(f, "{v} is neither endogenous nor exogenous"),
            Self::Cyclic(vs) => write!(f, "structural equations are cyclic: {}", vs.join(", ")),
            Self::NotEndogenous(v) => write!(f, "{v} is not an endogenous variable"),
            Self::Overlap(v) => write!(f, "{v} appears on both sides of the query"),
            Self::WorldCollision(v) => write!(f, "{v} is already defined by another world"),
            Self::Unevaluable(e) => write!(f, "cannot evaluate {e} to a discrete value"),
            Self::ZeroProbability => write!(f, "conditioning on an event of probability zero"),
        }
    }
}

impl std::error::Error for ScmError {}

/// Identifies one world of a [`Model`]: the observational one, or one produced
/// by an intervention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldId(usize);

/// One set of structural equations together with its causal diagram.
///
/// A world is immutable. The observational world holds the equations as
/// written; each intervened world holds the mutilated, renamed copy produced by
/// [`Model::intervene`].
#[derive(Clone, Debug)]
pub struct World {
    /// Endogenous variables in declaration order.
    v: Vec<String>,
    /// Equations in evaluation (topological) order.
    f: Vec<(String, Expr)>,
    /// Maps a name as written by the caller to its name in this world.
    syn: HashMap<String, String>,
    graph: DiGraph<String, ()>,
    node: HashMap<String, NodeIndex>,
}

impl World {
    /// The endogenous variables, in declaration order.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.v
    }

    /// The structural equations, in topological order.
    #[must_use]
    pub fn equations(&self) -> &[(String, Expr)] {
        &self.f
    }

    /// The causal diagram induced by the equations.
    ///
    /// Nodes are every endogenous and exogenous variable; there is an edge from
    /// each variable occurring free in an equation to the variable it defines.
    /// In an intervened world the edges into the intervened variables are gone,
    /// so this is the mutilated graph.
    #[must_use]
    pub fn graph(&self) -> &DiGraph<String, ()> {
        &self.graph
    }

    /// The node index of a variable in [`World::graph`].
    #[must_use]
    pub fn node(&self, name: &str) -> Option<NodeIndex> {
        self.node.get(name).copied()
    }

    /// Resolves a caller-facing name to the name it has in this world.
    fn resolve(&self, name: &str) -> String {
        self.syn.get(name).cloned().unwrap_or_else(|| name.to_string())
    }

    /// Evaluates every equation under an exogenous assignment, extending `env`.
    fn evaluate(&self, env: &mut HashMap<String, Value>) -> Result<(), ScmError> {
        for (k, e) in &self.f {
            let value = eval(e, env)?;
            env.insert(k.clone(), value);
        }
        Ok(())
    }
}

impl fmt::Display for World {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Declaration order, not the topological order used for evaluation, so
        // the model reads back the way it was written.
        for k in &self.v {
            if let Some((_, e)) = self.f.iter().find(|(name, _)| name == k) {
                writeln!(f, "  {k} := {}", render(e))?;
            }
        }
        Ok(())
    }
}

/// A structural causal model and every world derived from it.
///
/// The model owns the exogenous distributions, which every world shares, plus
/// the observational world and one world per distinct intervention.
#[derive(Clone, Debug)]
pub struct Model {
    worlds: Vec<World>,
    /// Exogenous variables in declaration order.
    u: Vec<String>,
    pu: HashMap<String, Dist>,
    /// Maps every endogenous name in any world to the world defining it.
    owner: HashMap<String, usize>,
    /// Caches interventions by their (sorted) assignment.
    cache: HashMap<Vec<(String, String)>, usize>,
    precision: usize,
}

impl Model {
    /// Builds a model from structural equations and exogenous distributions.
    ///
    /// The equations may be given in any order: the evaluation order is derived
    /// from the induced causal diagram, so a model that is not written
    /// topologically still evaluates correctly, and one that is genuinely
    /// recursive is rejected with [`ScmError::Cyclic`].
    ///
    /// # Errors
    ///
    /// Returns [`ScmError`] if a variable is declared twice, an equation refers
    /// to an undeclared variable, a distribution is malformed, or the equations
    /// are cyclic.
    pub fn new(
        f: impl IntoIterator<Item = (String, Expr)>,
        pu: impl IntoIterator<Item = (String, Dist)>,
    ) -> Result<Self, ScmError> {
        let f: Vec<(String, Expr)> = f.into_iter().collect();
        let pu: Vec<(String, Dist)> = pu.into_iter().collect();

        let mut u = Vec::with_capacity(pu.len());
        let mut dists = HashMap::with_capacity(pu.len());
        for (name, dist) in pu {
            dist.validate(&name)?;
            if dists.insert(name.clone(), dist).is_some() {
                return Err(ScmError::Duplicate(name));
            }
            u.push(name);
        }

        let mut seen = HashSet::with_capacity(f.len());
        for (name, _) in &f {
            if dists.contains_key(name) {
                return Err(ScmError::Duplicate(name.clone()));
            }
            if !seen.insert(name.clone()) {
                return Err(ScmError::Duplicate(name.clone()));
            }
        }

        let world = build_world(f, HashMap::new(), &seen, &dists)?;
        let owner = world.v.iter().map(|v| (v.clone(), 0)).collect();

        Ok(Self {
            worlds: vec![world],
            u,
            pu: dists,
            owner,
            cache: HashMap::new(),
            precision: 4,
        })
    }

    /// Sets the number of significant digits used when displaying probabilities.
    #[must_use]
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    /// The observational world, in which nothing has been intervened on.
    #[must_use]
    pub fn observational(&self) -> WorldId {
        WorldId(0)
    }

    /// Borrows a world for querying.
    ///
    /// # Panics
    ///
    /// Panics if `id` did not come from this model.
    #[must_use]
    pub fn world(&self, id: WorldId) -> WorldRef<'_> {
        assert!(id.0 < self.worlds.len(), "world id belongs to another model");
        WorldRef { model: self, id }
    }

    /// The exogenous variables, in declaration order.
    #[must_use]
    pub fn exogenous(&self) -> &[String] {
        &self.u
    }

    /// The distribution of an exogenous variable.
    #[must_use]
    pub fn distribution(&self, name: &str) -> Option<&Dist> {
        self.pu.get(name)
    }

    /// Applies `do(x)`, returning the world it creates.
    ///
    /// The new world clamps each variable in `x` to its value, drops the edges
    /// into it, and renames every endogenous variable `v` to `{v}_{X=x}` so
    /// that variables of different worlds never collide. Those names stay
    /// queryable from any world of this model, which is what allows a single
    /// query to span worlds. Repeating an intervention returns the world that
    /// already exists rather than building a second copy.
    ///
    /// # Errors
    ///
    /// Returns [`ScmError::NotEndogenous`] if `x` names a variable that is not
    /// endogenous in the observational world, and [`ScmError::WorldCollision`]
    /// if the generated names are already taken.
    pub fn intervene(&mut self, x: &[(&str, Value)]) -> Result<WorldId, ScmError> {
        let replacements: Vec<(&str, Expr)> = x
            .iter()
            .map(|(k, v)| {
                let e = match v {
                    Value::Bool(b) => Expr::Boolean(*b),
                    Value::Int(i) => expr::int(*i),
                };
                (*k, e)
            })
            .collect();
        self.intervene_with(&replacements)
    }

    /// Replaces the equation of each named variable outright, returning the
    /// world it creates.
    ///
    /// The general form of [`Model::intervene`]: where that clamps a variable to
    /// a constant, this substitutes an arbitrary expression, so the variable can
    /// be made to depend on different parents rather than none at all. That is a
    /// *soft* or *conditional* intervention — a policy, in the language of the
    /// book's `causality.py`, whose `SCM.intervene` likewise takes functions
    /// rather than values.
    ///
    /// Renaming, world registration and caching work exactly as for a hard
    /// intervention; the subscript records the new equation instead of a value,
    /// so `do(x := z & uz)` yields `{y}_{x:=z & uz}`.
    ///
    /// # Errors
    ///
    /// As [`Model::intervene`], plus [`ScmError::Cyclic`] if the replacement
    /// introduces a cycle.
    pub fn intervene_with(&mut self, x: &[(&str, Expr)]) -> Result<WorldId, ScmError> {
        let mut key: Vec<(String, String)> = x
            .iter()
            .map(|(k, e)| ((*k).to_string(), render(e)))
            .collect();
        key.sort();
        key.dedup();
        if let Some(id) = self.cache.get(&key) {
            return Ok(WorldId(*id));
        }

        let base = &self.worlds[0];
        for (k, _) in &key {
            if !base.v.contains(k) {
                return Err(ScmError::NotEndogenous(k.clone()));
            }
        }

        // `x = 1` for a constant, `x := expr` for anything else, so a hard
        // intervention keeps the notation the book uses.
        let subscript = key
            .iter()
            .map(|(k, rendered)| {
                if rendered.chars().all(|c| c.is_ascii_digit() || c == '-')
                    || rendered == "true"
                    || rendered == "false"
                {
                    let v = match rendered.as_str() {
                        "true" => "1".to_string(),
                        "false" => "0".to_string(),
                        other => other.to_string(),
                    };
                    format!("{k}={v}")
                } else {
                    format!("{k}:={rendered}")
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        let clamped: HashMap<&str, &Expr> = x.iter().map(|(k, e)| (*k, e)).collect();
        let rename: HashMap<String, Expr> = base
            .v
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    Expr::Variable(format!("{{{k}}}_{{{subscript}}}")),
                )
            })
            .collect();

        let mut f = Vec::with_capacity(base.f.len());
        for k in &base.v {
            let e = base
                .f
                .iter()
                .find(|(name, _)| name == k)
                .map(|(_, e)| e)
                .expect("every endogenous variable has an equation");
            // A replacement is rewritten too: a soft intervention may name
            // other endogenous variables, and those are this world's copies.
            let e = match clamped.get(k.as_str()) {
                Some(replacement) => substitute(replacement, &rename),
                None => substitute(e, &rename),
            };
            let name = match &rename[k] {
                Expr::Variable(n) => n.clone(),
                _ => unreachable!("rename maps to variables"),
            };
            f.push((name, e));
        }

        for (name, _) in &f {
            if self.owner.contains_key(name) {
                return Err(ScmError::WorldCollision(name.clone()));
            }
        }

        let mut syn = base.syn.clone();
        for (k, e) in &rename {
            if let Expr::Variable(n) = e {
                syn.insert(k.clone(), n.clone());
            }
        }

        let endogenous: HashSet<String> = f.iter().map(|(k, _)| k.clone()).collect();
        let world = build_world(f, syn, &endogenous, &self.pu)?;

        let id = self.worlds.len();
        for v in &world.v {
            self.owner.insert(v.clone(), id);
        }
        self.worlds.push(world);
        self.cache.insert(key, id);
        Ok(WorldId(id))
    }

    /// Every assignment to `U` paired with its probability.
    fn exogenous_assignments(&self) -> Vec<(HashMap<String, Value>, f64)> {
        let supports: Vec<(String, Vec<Value>)> = self
            .u
            .iter()
            .map(|k| (k.clone(), self.pu[k].support()))
            .collect();

        let mut out = Vec::new();
        let mut counters = vec![0usize; supports.len()];
        if supports.iter().any(|(_, s)| s.is_empty()) {
            return out;
        }
        loop {
            let mut env = HashMap::with_capacity(supports.len());
            // Accumulate in log space: a wide exogenous space multiplies many
            // small probabilities together.
            let mut log_p = 0.0;
            for (i, (name, support)) in supports.iter().enumerate() {
                let value = support[counters[i]];
                log_p += self.pu[name].probability(value).ln();
                env.insert(name.clone(), value);
            }
            out.push((env, log_p.exp()));

            let mut i = supports.len();
            loop {
                if i == 0 {
                    return out;
                }
                i -= 1;
                counters[i] += 1;
                if counters[i] < supports[i].1.len() {
                    break;
                }
                counters[i] = 0;
            }
        }
    }

    fn probability_table(&self, columns: &[String]) -> Result<ProbabilityTable, ScmError> {
        // Only the worlds actually mentioned need evaluating.
        let mut needed: Vec<usize> = Vec::new();
        for c in columns {
            if self.pu.contains_key(c) {
                continue;
            }
            let world = *self.owner.get(c).ok_or_else(|| ScmError::Unknown(c.clone()))?;
            if !needed.contains(&world) {
                needed.push(world);
            }
        }

        let mut rows: BTreeMap<Vec<Value>, f64> = BTreeMap::new();
        for (mut env, p) in self.exogenous_assignments() {
            for w in &needed {
                self.worlds[*w].evaluate(&mut env)?;
            }
            let key = columns
                .iter()
                .map(|c| env.get(c).copied().ok_or_else(|| ScmError::Unknown(c.clone())))
                .collect::<Result<Vec<_>, _>>()?;
            *rows.entry(key).or_insert(0.0) += p;
        }

        Ok(ProbabilityTable {
            columns: columns.to_vec(),
            rows: rows.into_iter().collect(),
            precision: self.precision,
        })
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base = &self.worlds[0];
        writeln!(f, "V = {{{}}}", base.v.join(", "))?;
        writeln!(f, "U = {{{}}}", self.u.join(", "))?;
        writeln!(f, "F = {{")?;
        write!(f, "{base}")?;
        writeln!(f, "}}")?;
        writeln!(f, "P(U) = {{")?;
        for name in &self.u {
            match &self.pu[name] {
                Dist::Bernoulli(p) => {
                    writeln!(f, "  {name} ~ Bern({:.*})", self.precision, p)?;
                }
                Dist::Categorical(ps) => {
                    let ps = ps
                        .iter()
                        .map(|p| format!("{:.*}", self.precision, p))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "  {name} ~ Categorical([{ps}])")?;
                }
            }
        }
        write!(f, "}}")
    }
}

/// A world of a [`Model`], borrowed for querying.
#[derive(Clone, Copy, Debug)]
pub struct WorldRef<'a> {
    model: &'a Model,
    id: WorldId,
}

impl<'a> WorldRef<'a> {
    /// The world's identifier.
    #[must_use]
    pub fn id(self) -> WorldId {
        self.id
    }

    /// The world itself.
    #[must_use]
    pub fn world(self) -> &'a World {
        &self.model.worlds[self.id.0]
    }

    /// The joint distribution over `symbols`, marginalising everything else.
    ///
    /// Passing `None` tabulates this world's endogenous variables. Names are
    /// resolved through the world's renaming, so `"y"` means `{y}_{x=1}` inside
    /// the world produced by `do(x = 1)`; a fully-qualified name from another
    /// world is also accepted, and mixing the two yields a counterfactual
    /// joint.
    ///
    /// # Errors
    ///
    /// Returns [`ScmError::Unknown`] for a name no world defines, and
    /// [`ScmError::Unevaluable`] if an equation cannot be reduced to a discrete
    /// value.
    pub fn probability_table(
        self,
        symbols: Option<&[&str]>,
    ) -> Result<ProbabilityTable, ScmError> {
        let world = self.world();
        let columns: Vec<String> = match symbols {
            Some(s) => s.iter().map(|k| world.resolve(k)).collect(),
            None => world.v.clone(),
        };
        self.model.probability_table(&columns)
    }

    /// The conditional distribution `P(symbols | given)`, as a table.
    ///
    /// Every row of the joint over `symbols ∪ given` is divided by the total
    /// mass of its `given` group, so the probabilities sum to one within each
    /// setting of `given` rather than across the whole table. This is
    /// `causality.py`'s `get_distribution(conditioned_on = ...)`.
    ///
    /// Columns are `given` first, then `symbols`, so the groups read
    /// contiguously. Rows whose conditioning event has probability zero are
    /// dropped rather than yielding `NaN`.
    ///
    /// # Errors
    ///
    /// As [`WorldRef::probability_table`], plus [`ScmError::Overlap`] if a
    /// variable appears on both sides.
    pub fn conditional_table(
        self,
        symbols: Option<&[&str]>,
        given: &[&str],
    ) -> Result<ProbabilityTable, ScmError> {
        let world = self.world();
        let targets: Vec<String> = match symbols {
            Some(s) => s.iter().map(|k| world.resolve(k)).collect(),
            None => world.v.clone(),
        };
        let given: Vec<String> = given.iter().map(|k| world.resolve(k)).collect();

        for g in &given {
            if targets.contains(g) {
                return Err(ScmError::Overlap(g.clone()));
            }
        }

        let mut columns = given.clone();
        columns.extend(targets.iter().cloned());
        let joint = self.model.probability_table(&columns)?;

        // Total mass per setting of the conditioning variables.
        let width = given.len();
        let mut totals: BTreeMap<Vec<Value>, f64> = BTreeMap::new();
        for (values, p) in &joint.rows {
            *totals.entry(values[..width].to_vec()).or_insert(0.0) += p;
        }

        let rows = joint
            .rows
            .into_iter()
            .filter_map(|(values, p)| {
                let total = totals[&values[..width]];
                (total > 0.0).then(|| (values, p / total))
            })
            .collect();

        Ok(ProbabilityTable {
            columns,
            rows,
            precision: joint.precision,
        })
    }

    /// `P(x)`, or `P(x | given)` when `given` is non-empty.
    ///
    /// # Errors
    ///
    /// Returns [`ScmError::Overlap`] if a variable appears in both `x` and
    /// `given`, [`ScmError::ZeroProbability`] if `given` cannot occur, and
    /// otherwise the errors of [`WorldRef::probability_table`].
    pub fn query(self, x: &[(&str, Value)], given: &[(&str, Value)]) -> Result<f64, ScmError> {
        let world = self.world();
        let x: Vec<(String, Value)> = x.iter().map(|(k, v)| (world.resolve(k), *v)).collect();
        let given: Vec<(String, Value)> =
            given.iter().map(|(k, v)| (world.resolve(k), *v)).collect();

        for (k, _) in &x {
            if given.iter().any(|(g, _)| g == k) {
                return Err(ScmError::Overlap(k.clone()));
            }
        }

        if given.is_empty() {
            let columns: Vec<String> = x.iter().map(|(k, _)| k.clone()).collect();
            let table = self.model.probability_table(&columns)?;
            let wanted: Vec<Value> = x.iter().map(|(_, v)| *v).collect();
            return Ok(table
                .rows
                .iter()
                .filter(|(values, _)| *values == wanted)
                .map(|(_, p)| p)
                .sum());
        }

        let mut joint = x;
        joint.extend(given.iter().cloned());
        let numerator = self.raw_query(&joint)?;
        let denominator = self.raw_query(&given)?;
        if denominator == 0.0 {
            return Err(ScmError::ZeroProbability);
        }
        Ok(numerator / denominator)
    }

    /// Draws `n` rows from the joint over this world's endogenous variables.
    ///
    /// `uniform` supplies independent draws from `[0, 1)`; keeping the source
    /// of randomness in the caller's hands avoids pinning this crate to an RNG
    /// and makes tests reproducible.
    ///
    /// # Errors
    ///
    /// The errors of [`WorldRef::probability_table`].
    pub fn sample(
        self,
        n: usize,
        mut uniform: impl FnMut() -> f64,
    ) -> Result<Vec<Vec<Value>>, ScmError> {
        let table = self.probability_table(None)?;
        let total: f64 = table.rows.iter().map(|(_, p)| p).sum();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut target = uniform() * total;
            let mut chosen = table.rows.len() - 1;
            for (i, (_, p)) in table.rows.iter().enumerate() {
                target -= p;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            out.push(table.rows[chosen].0.clone());
        }
        Ok(out)
    }

    fn raw_query(self, assignment: &[(String, Value)]) -> Result<f64, ScmError> {
        let columns: Vec<String> = assignment.iter().map(|(k, _)| k.clone()).collect();
        let table = self.model.probability_table(&columns)?;
        let wanted: Vec<Value> = assignment.iter().map(|(_, v)| *v).collect();
        Ok(table
            .rows
            .iter()
            .filter(|(values, _)| *values == wanted)
            .map(|(_, p)| p)
            .sum())
    }
}

/// A joint distribution: one probability per assignment of its columns.
#[derive(Clone, Debug)]
pub struct ProbabilityTable {
    /// The tabulated variables, in column order.
    pub columns: Vec<String>,
    /// Assignments and their probabilities, ordered by assignment.
    pub rows: Vec<(Vec<Value>, f64)>,
    precision: usize,
}

impl fmt::Display for ProbabilityTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} | probability", self.columns.join(" "))?;
        for (values, p) in &self.rows {
            let values = values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(f, "{values} | {:.*}", self.precision, p)?;
        }
        Ok(())
    }
}

/// Assembles a world: derives the causal diagram, checks it is acyclic and
/// orders the equations topologically.
fn build_world(
    f: Vec<(String, Expr)>,
    syn: HashMap<String, String>,
    endogenous: &HashSet<String>,
    pu: &HashMap<String, Dist>,
) -> Result<World, ScmError> {
    let mut graph = DiGraph::<String, ()>::new();
    let mut node = HashMap::new();
    for (name, _) in &f {
        node.insert(name.clone(), graph.add_node(name.clone()));
    }
    for name in pu.keys() {
        node.entry(name.clone())
            .or_insert_with(|| graph.add_node(name.clone()));
    }

    for (name, e) in &f {
        let mut parents = HashSet::new();
        free_vars(e, &mut parents);
        let mut parents: Vec<String> = parents.into_iter().collect();
        parents.sort();
        for parent in parents {
            if !endogenous.contains(&parent) && !pu.contains_key(&parent) {
                return Err(ScmError::Unknown(parent));
            }
            graph.add_edge(node[&parent], node[name], ());
        }
    }

    let order = toposort(&graph, None).map_err(|cycle| {
        ScmError::Cyclic(vec![graph[cycle.node_id()].clone()])
    })?;

    let mut ordered = Vec::with_capacity(f.len());
    for idx in order {
        let name = &graph[idx];
        if let Some((k, e)) = f.iter().find(|(k, _)| k == name) {
            ordered.push((k.clone(), e.clone()));
        }
    }

    Ok(World {
        v: f.into_iter().map(|(k, _)| k).collect(),
        f: ordered,
        syn,
        graph,
        node,
    })
}

/// Collects the free variables of an expression.
///
/// The head of an [`Expr::Apply`] is a function symbol rather than a variable,
/// so it is deliberately not collected: `f_x(u_x)` depends on `u_x` alone.
fn free_vars(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Variable(name) => {
            out.insert(name.clone());
        }
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::Power(a, b)
        | Expr::Xor(a, b)
        | Expr::Eq(a, b)
        | Expr::Lt(a, b)
        | Expr::Le(a, b)
        | Expr::Gt(a, b)
        | Expr::Ge(a, b) => {
            free_vars(a, out);
            free_vars(b, out);
        }
        Expr::Neg(a) | Expr::Not(a) => free_vars(a, out),
        Expr::And(items) | Expr::Or(items) | Expr::AddList(items) | Expr::MulList(items)
        | Expr::Tuple(items) => {
            for item in items {
                free_vars(item, out);
            }
        }
        Expr::Apply(_, arg) => free_vars(arg, out),
        _ => {}
    }
}

/// Replaces free variables by expressions, structurally.
///
/// [`rssn`]'s own `substitute` walks arithmetic nodes only and silently leaves
/// logical ones untouched, which would quietly produce wrong models here.
fn substitute(e: &Expr, map: &HashMap<String, Expr>) -> Expr {
    macro_rules! binary {
        ($ctor:path, $a:expr, $b:expr) => {
            $ctor(
                Arc::new(substitute($a, map)),
                Arc::new(substitute($b, map)),
            )
        };
    }
    macro_rules! unary {
        ($ctor:path, $a:expr) => {
            $ctor(Arc::new(substitute($a, map)))
        };
    }
    macro_rules! nary {
        ($ctor:path, $items:expr) => {
            $ctor($items.iter().map(|i| substitute(i, map)).collect())
        };
    }

    match e {
        Expr::Variable(name) => map.get(name).cloned().unwrap_or_else(|| e.clone()),
        Expr::Add(a, b) => binary!(Expr::Add, a, b),
        Expr::Sub(a, b) => binary!(Expr::Sub, a, b),
        Expr::Mul(a, b) => binary!(Expr::Mul, a, b),
        Expr::Div(a, b) => binary!(Expr::Div, a, b),
        Expr::Power(a, b) => binary!(Expr::Power, a, b),
        Expr::Xor(a, b) => binary!(Expr::Xor, a, b),
        Expr::Eq(a, b) => binary!(Expr::Eq, a, b),
        Expr::Lt(a, b) => binary!(Expr::Lt, a, b),
        Expr::Le(a, b) => binary!(Expr::Le, a, b),
        Expr::Gt(a, b) => binary!(Expr::Gt, a, b),
        Expr::Ge(a, b) => binary!(Expr::Ge, a, b),
        Expr::Neg(a) => unary!(Expr::Neg, a),
        Expr::Not(a) => unary!(Expr::Not, a),
        Expr::And(items) => nary!(Expr::And, items),
        Expr::Or(items) => nary!(Expr::Or, items),
        Expr::AddList(items) => nary!(Expr::AddList, items),
        Expr::MulList(items) => nary!(Expr::MulList, items),
        Expr::Tuple(items) => nary!(Expr::Tuple, items),
        // The head stays a function symbol; only the arguments are rewritten.
        Expr::Apply(f, arg) => Expr::Apply(f.clone(), Arc::new(substitute(arg, map))),
        _ => e.clone(),
    }
}

/// Evaluates an expression to a discrete value under a total assignment.
fn eval(e: &Expr, env: &HashMap<String, Value>) -> Result<Value, ScmError> {
    let int = |e: &Expr| -> Result<i64, ScmError> { Ok(eval(e, env)?.as_int()) };
    let boolean = |e: &Expr| -> Result<bool, ScmError> { Ok(eval(e, env)?.as_bool()) };

    Ok(match e {
        Expr::Boolean(b) => Value::Bool(*b),
        Expr::Constant(c) => {
            if c.fract() != 0.0 {
                return Err(ScmError::Unevaluable(render(e)));
            }
            Value::Int(*c as i64)
        }
        Expr::Variable(name) => *env
            .get(name)
            .ok_or_else(|| ScmError::Unknown(name.clone()))?,
        Expr::Add(a, b) => Value::Int(int(a)? + int(b)?),
        Expr::Sub(a, b) => Value::Int(int(a)? - int(b)?),
        Expr::Mul(a, b) => Value::Int(int(a)? * int(b)?),
        Expr::Div(a, b) => {
            let (a, b) = (int(a)?, int(b)?);
            if b == 0 || a % b != 0 {
                return Err(ScmError::Unevaluable(render(e)));
            }
            Value::Int(a / b)
        }
        Expr::Power(a, b) => {
            let exponent = int(b)?;
            let exponent =
                u32::try_from(exponent).map_err(|_| ScmError::Unevaluable(render(e)))?;
            Value::Int(int(a)?.pow(exponent))
        }
        Expr::Neg(a) => Value::Int(-int(a)?),
        Expr::AddList(items) => {
            let mut total = 0;
            for item in items {
                total += int(item)?;
            }
            Value::Int(total)
        }
        Expr::MulList(items) => {
            let mut total = 1;
            for item in items {
                total *= int(item)?;
            }
            Value::Int(total)
        }
        Expr::Not(a) => Value::Bool(!boolean(a)?),
        Expr::And(items) => {
            let mut acc = true;
            for item in items {
                acc &= boolean(item)?;
            }
            Value::Bool(acc)
        }
        Expr::Or(items) => {
            let mut acc = false;
            for item in items {
                acc |= boolean(item)?;
            }
            Value::Bool(acc)
        }
        Expr::Xor(a, b) => Value::Bool(boolean(a)? ^ boolean(b)?),
        Expr::Eq(a, b) => Value::Bool(eval(a, env)? == eval(b, env)?),
        Expr::Lt(a, b) => Value::Bool(int(a)? < int(b)?),
        Expr::Le(a, b) => Value::Bool(int(a)? <= int(b)?),
        Expr::Gt(a, b) => Value::Bool(int(a)? > int(b)?),
        Expr::Ge(a, b) => Value::Bool(int(a)? >= int(b)?),
        // Notably `Expr::Apply`: an uninterpreted mechanism has no value.
        _ => return Err(ScmError::Unevaluable(render(e))),
    })
}

/// Renders an expression, parenthesising by precedence.
fn render(e: &Expr) -> String {
    let mut out = String::new();
    write_expr(e, 0, &mut out);
    out
}

fn write_expr(e: &Expr, parent: u8, out: &mut String) {
    fn binary(a: &Expr, op: &str, b: &Expr, prec: u8, parent: u8, out: &mut String) {
        let paren = prec < parent;
        if paren {
            out.push('(');
        }
        write_expr(a, prec, out);
        out.push_str(op);
        write_expr(b, prec + 1, out);
        if paren {
            out.push(')');
        }
    }

    fn nary(items: &[Expr], op: &str, prec: u8, parent: u8, out: &mut String) {
        let paren = prec < parent;
        if paren {
            out.push('(');
        }
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(op);
            }
            write_expr(item, prec, out);
        }
        if paren {
            out.push(')');
        }
    }

    match e {
        Expr::Variable(name) => out.push_str(name),
        Expr::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Expr::Constant(c) => {
            if c.fract() == 0.0 {
                out.push_str(&format!("{}", *c as i64));
            } else {
                out.push_str(&format!("{c}"));
            }
        }
        Expr::Or(items) => nary(items, " | ", 1, parent, out),
        Expr::Xor(a, b) => binary(a, " ^ ", b, 2, parent, out),
        Expr::And(items) => nary(items, " & ", 3, parent, out),
        Expr::Not(a) => {
            out.push('!');
            write_expr(a, 9, out);
        }
        Expr::Eq(a, b) => binary(a, " == ", b, 4, parent, out),
        Expr::Lt(a, b) => binary(a, " < ", b, 4, parent, out),
        Expr::Le(a, b) => binary(a, " <= ", b, 4, parent, out),
        Expr::Gt(a, b) => binary(a, " > ", b, 4, parent, out),
        Expr::Ge(a, b) => binary(a, " >= ", b, 4, parent, out),
        Expr::Add(a, b) => binary(a, " + ", b, 5, parent, out),
        Expr::Sub(a, b) => binary(a, " - ", b, 5, parent, out),
        Expr::AddList(items) => nary(items, " + ", 5, parent, out),
        Expr::Mul(a, b) => binary(a, " * ", b, 6, parent, out),
        Expr::Div(a, b) => binary(a, " / ", b, 6, parent, out),
        Expr::MulList(items) => nary(items, " * ", 6, parent, out),
        Expr::Neg(a) => {
            out.push('-');
            write_expr(a, 9, out);
        }
        Expr::Power(a, b) => binary(a, "**", b, 7, parent, out),
        Expr::Apply(f, arg) => {
            write_expr(f, 9, out);
            out.push('(');
            match &**arg {
                Expr::Tuple(items) => nary(items, ", ", 0, 0, out),
                other => write_expr(other, 0, out),
            }
            out.push(')');
        }
        Expr::Tuple(items) => {
            out.push('(');
            nary(items, ", ", 0, 0, out);
            out.push(')');
        }
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: Value = Value::Bool(true);
    const F: Value = Value::Bool(false);

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// The chain `x -> z -> y` from chapter 2 of the Causal AI book.
    fn chain() -> Model {
        Model::new(
            [
                ("x".to_string(), expr::var("ux")),
                (
                    "z".to_string(),
                    expr::and([expr::var("x"), expr::not(expr::var("uz"))]),
                ),
                (
                    "y".to_string(),
                    expr::and([expr::var("z"), expr::var("uy")]),
                ),
            ],
            [
                ("ux".to_string(), Dist::Bernoulli(0.5)),
                ("uz".to_string(), Dist::Bernoulli(0.5)),
                ("uy".to_string(), Dist::Bernoulli(0.5)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn marginals_of_a_chain() {
        let m = chain();
        let w = m.world(m.observational());
        assert!(close(w.query(&[("x", T)], &[]).unwrap(), 0.5));
        assert!(close(w.query(&[("z", T)], &[]).unwrap(), 0.25));
        assert!(close(w.query(&[("y", T)], &[]).unwrap(), 0.125));
    }

    #[test]
    fn conditioning_on_the_mediator_separates_the_chain() {
        let m = chain();
        let w = m.world(m.observational());
        // y is independent of x given z.
        let a = w.query(&[("y", T)], &[("z", T), ("x", T)]).unwrap();
        let b = w.query(&[("y", T)], &[("z", T)]).unwrap();
        assert!(close(a, b));
    }

    #[test]
    fn probability_table_is_a_distribution() {
        let m = chain();
        let table = m
            .world(m.observational())
            .probability_table(None)
            .unwrap();
        assert_eq!(table.columns, vec!["x", "z", "y"]);
        let total: f64 = table.rows.iter().map(|(_, p)| p).sum();
        assert!(close(total, 1.0));
    }

    #[test]
    fn intervention_severs_the_incoming_edges() {
        let m = chain();
        let base = m.world(m.observational()).world();
        let x = base.node("x").unwrap();
        assert_eq!(base.graph().neighbors_directed(x, crate::graph::Incoming).count(), 1);

        let mut m = m;
        let w = m.intervene(&[("x", T)]).unwrap();
        let world = m.world(w).world();
        let x = world.node("{x}_{x=1}").unwrap();
        assert_eq!(
            world.graph().neighbors_directed(x, crate::graph::Incoming).count(),
            0
        );
        // P(y | do(x = 1)) = P(!uz & uy) = 0.25
        assert!(close(m.world(w).query(&[("y", T)], &[]).unwrap(), 0.25));
    }

    #[test]
    fn intervening_twice_reuses_the_world() {
        let mut m = chain();
        let a = m.intervene(&[("x", T)]).unwrap();
        let b = m.intervene(&[("x", T)]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn counterfactuals_span_worlds() {
        let mut m = chain();
        let one = m.intervene(&[("x", T)]).unwrap();
        let zero = m.intervene(&[("x", F)]).unwrap();

        assert!(close(m.world(one).query(&[("y", T)], &[]).unwrap(), 0.25));
        assert!(close(m.world(zero).query(&[("y", T)], &[]).unwrap(), 0.0));

        // P(Y_{x=1} = 1, Y_{x=0} = 0): the probability of being helped by x.
        let joint = m
            .world(one)
            .query(&[("{y}_{x=1}", T), ("{y}_{x=0}", F)], &[])
            .unwrap();
        assert!(close(joint, 0.25));
    }

    #[test]
    fn categorical_arithmetic_model() {
        // Two fair dice: x = u1 + u2, y = u1 - u2.
        let faces = vec![0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
        let m = Model::new(
            [
                ("x".to_string(), expr::add(expr::var("u1"), expr::var("u2"))),
                ("y".to_string(), expr::sub(expr::var("u1"), expr::var("u2"))),
            ],
            [
                ("u1".to_string(), Dist::Categorical(faces.clone())),
                ("u2".to_string(), Dist::Categorical(faces)),
            ],
        )
        .unwrap();

        let w = m.world(m.observational());
        assert!(close(w.query(&[("x", Value::Int(7))], &[]).unwrap(), 6.0 / 36.0));
        assert!(close(w.query(&[("y", Value::Int(0))], &[]).unwrap(), 6.0 / 36.0));
        assert!(close(
            w.query(&[("x", Value::Int(7))], &[("y", Value::Int(0))])
                .unwrap(),
            0.0
        ));
    }

    #[test]
    fn equations_need_not_be_written_in_topological_order() {
        let m = Model::new(
            [
                (
                    "y".to_string(),
                    expr::and([expr::var("x"), expr::var("uy")]),
                ),
                ("x".to_string(), expr::var("ux")),
            ],
            [
                ("ux".to_string(), Dist::Bernoulli(0.5)),
                ("uy".to_string(), Dist::Bernoulli(0.5)),
            ],
        )
        .unwrap();
        assert!(close(
            m.world(m.observational()).query(&[("y", T)], &[]).unwrap(),
            0.25
        ));
    }

    #[test]
    fn cyclic_equations_are_rejected() {
        let err = Model::new(
            [
                ("x".to_string(), expr::and([expr::var("y"), expr::var("ux")])),
                ("y".to_string(), expr::var("x")),
            ],
            [("ux".to_string(), Dist::Bernoulli(0.5))],
        )
        .unwrap_err();
        assert!(matches!(err, ScmError::Cyclic(_)));
    }

    #[test]
    fn undeclared_variables_are_rejected() {
        let err = Model::new(
            [("x".to_string(), expr::var("nope"))],
            [("ux".to_string(), Dist::Bernoulli(0.5))],
        )
        .unwrap_err();
        assert_eq!(err, ScmError::Unknown("nope".to_string()));
    }

    #[test]
    fn unnormalised_distributions_are_rejected() {
        let err = Model::new(
            [("x".to_string(), expr::var("ux"))],
            [("ux".to_string(), Dist::Categorical(vec![0.5, 0.2]))],
        )
        .unwrap_err();
        assert!(matches!(err, ScmError::NotNormalized { .. }));
    }

    #[test]
    fn overlapping_queries_are_rejected() {
        let m = chain();
        let err = m
            .world(m.observational())
            .query(&[("y", T)], &[("y", T)])
            .unwrap_err();
        assert_eq!(err, ScmError::Overlap("y".to_string()));
    }

    #[test]
    fn uninterpreted_mechanisms_describe_but_do_not_evaluate() {
        // The chapter 4 model states its shape only: se := f_se(u_se).
        let m = Model::new(
            [
                ("se".to_string(), expr::apply("f_se", [expr::var("u_se")])),
                (
                    "sp".to_string(),
                    expr::apply("f_sp", [expr::var("se"), expr::var("u_sp")]),
                ),
            ],
            [
                ("u_se".to_string(), Dist::Bernoulli(0.5)),
                ("u_sp".to_string(), Dist::Bernoulli(0.5)),
            ],
        )
        .unwrap();

        // The causal diagram is still recovered: se -> sp, and f_se is not a parent.
        let world = m.world(m.observational()).world();
        let se = world.node("se").unwrap();
        let sp = world.node("sp").unwrap();
        assert!(world.graph().find_edge(se, sp).is_some());
        assert!(world.node("f_se").is_none());

        let err = m
            .world(m.observational())
            .query(&[("se", T)], &[])
            .unwrap_err();
        assert!(matches!(err, ScmError::Unevaluable(_)));
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let m = chain();
        let mut draws = [0.05_f64, 0.95, 0.5].into_iter().cycle();
        let rows = m
            .world(m.observational())
            .sample(3, || draws.next().unwrap())
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.len() == 3));
    }

    #[test]
    fn a_soft_intervention_rewires_rather_than_clamps() {
        // x := ux, z := x & !uz, y := z & uy.
        let mut m = chain();

        // Policy: make x follow uz instead of ux. x is no longer a function of
        // ux, so P(x = 1) is now governed by uz.
        let w = m.intervene_with(&[("x", expr::var("uz"))]).unwrap();
        assert_eq!(m.world(w).world().variables()[0], "{x}_{x:=uz}");
        assert!(close(m.world(w).query(&[("x", T)], &[]).unwrap(), 0.5));
        // z := x & !uz becomes uz & !uz, which is never true.
        assert!(close(m.world(w).query(&[("z", T)], &[]).unwrap(), 0.0));
        assert!(close(m.world(w).query(&[("y", T)], &[]).unwrap(), 0.0));

        // A hard intervention is the special case, and keeps the `x=1` notation.
        let hard = m.intervene(&[("x", T)]).unwrap();
        assert_eq!(m.world(hard).world().variables()[0], "{x}_{x=1}");
    }

    #[test]
    fn soft_and_hard_interventions_share_the_cache() {
        let mut m = chain();
        let a = m.intervene(&[("x", T)]).unwrap();
        // The same clamp spelled as an expression must reuse the world.
        let b = m.intervene_with(&[("x", expr::boolean(true))]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn conditional_tables_normalise_within_each_group() {
        let m = chain();
        let w = m.world(m.observational());
        let table = w.conditional_table(Some(&["y"]), &["z"]).unwrap();

        assert_eq!(table.columns, vec!["z", "y"]);
        // Each setting of z sums to one.
        let mut totals: std::collections::BTreeMap<Value, f64> =
            std::collections::BTreeMap::new();
        for (values, p) in &table.rows {
            *totals.entry(values[0]).or_insert(0.0) += p;
        }
        assert_eq!(totals.len(), 2);
        for (_, total) in totals {
            assert!(close(total, 1.0));
        }

        // And they agree with query().
        for (values, p) in &table.rows {
            let got = w.query(&[("y", values[1])], &[("z", values[0])]).unwrap();
            assert!(close(*p, got), "{values:?}: table {p} vs query {got}");
        }
    }

    #[test]
    fn conditioning_on_a_target_is_rejected() {
        let m = chain();
        let err = m
            .world(m.observational())
            .conditional_table(Some(&["y"]), &["y"])
            .unwrap_err();
        assert_eq!(err, ScmError::Overlap("y".to_string()));
    }

    #[test]
    fn model_renders_as_text() {
        let text = chain().to_string();
        assert!(text.contains("z := x & !uz"), "{text}");
        assert!(text.contains("ux ~ Bern(0.5000)"), "{text}");
    }
}
