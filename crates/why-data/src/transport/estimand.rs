//! Estimands whose factors come from different domains, and their plug-in
//! evaluation.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

use super::{Dataset, Domain, TransportError};
use crate::graph::identification::Formula;

/// A transport formula: an expression in the distributions of the source and
/// target domains.
///
/// It prints the way the book writes transport formulas, with the domain as
/// a superscript: `sum_{z} P^1(y | x, z) P^*(z | x)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Estimand {
    /// `P^domain(vars | given)`.
    P {
        /// The domain whose distribution this is.
        domain: Domain,
        /// The variables whose probability this is.
        vars: Vec<String>,
        /// What it is conditioned on; empty for a marginal.
        given: Vec<String>,
    },
    /// A product of factors; the empty product is 1.
    Product(Vec<Estimand>),
    /// A sum marginalising `over` out of `body`.
    Sum {
        /// The variables summed out.
        over: Vec<String>,
        /// What is being summed.
        body: Box<Estimand>,
    },
    /// A quotient.
    Ratio {
        /// The numerator.
        num: Box<Estimand>,
        /// The denominator.
        den: Box<Estimand>,
    },
}

impl Estimand {
    /// `P^domain(vars | given)`.
    pub fn p(domain: Domain, vars: &[&str], given: &[&str]) -> Self {
        Self::P {
            domain,
            vars: vars.iter().map(|s| (*s).to_string()).collect(),
            given: given.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// The constant 1.
    #[must_use]
    pub fn one() -> Self {
        Self::Product(Vec::new())
    }

    fn is_one(&self) -> bool {
        matches!(self, Self::Product(f) if f.is_empty())
    }

    fn factors(self) -> Vec<Self> {
        match self {
            Self::Product(f) => f,
            other => vec![other],
        }
    }

    /// A product, flattened, without unit factors, and reduced to its single
    /// factor when it has one.
    #[must_use]
    pub fn product(factors: Vec<Self>) -> Self {
        let mut flat = Vec::new();
        for f in factors {
            match f {
                Self::Product(inner) => flat.extend(inner),
                other => flat.push(other),
            }
        }
        if flat.len() == 1 {
            flat.remove(0)
        } else {
            Self::Product(flat)
        }
    }

    /// `sum_over body`, simplified: nested sums merge, factors that mention
    /// none of `over` move out of the sum, and a factor `P(A | B)` with
    /// `A ⊆ over` that nothing else mentions sums to one and disappears.
    #[must_use]
    pub fn sum(over: Vec<String>, body: Self) -> Self {
        let (mut over, body) = match body {
            Self::Sum { over: inner, body } => {
                let mut merged = over;
                merged.extend(inner);
                (merged, *body)
            }
            other => (over, other),
        };
        over.sort();
        over.dedup();
        if matches!(body, Self::Ratio { .. } | Self::Sum { .. }) {
            return if over.is_empty() {
                body
            } else {
                Self::Sum {
                    over,
                    body: Box::new(body),
                }
            };
        }
        let mut inner = body.factors();
        let mut outside = Vec::new();
        inner.retain(|f| {
            if over.iter().any(|v| f.mentions(v)) {
                true
            } else {
                outside.push(f.clone());
                false
            }
        });
        // Drop factors that sum to one, until none does.
        loop {
            let found = (0..inner.len()).find(|&i| {
                let Self::P { vars, given, .. } = &inner[i] else {
                    return false;
                };
                vars.iter().all(|v| over.contains(v))
                    && vars.iter().all(|v| !given.contains(v))
                    && vars.iter().all(|v| {
                        inner
                            .iter()
                            .enumerate()
                            .all(|(j, g)| j == i || !g.mentions(v))
                    })
            });
            let Some(i) = found else { break };
            if let Self::P { vars, .. } = inner.remove(i) {
                over.retain(|v| !vars.contains(v));
            }
        }
        // Summation indices nothing mentions any more are dropped too: the
        // factors left do not depend on them. (Only reachable when a variable
        // was summed that never appeared.)
        over.retain(|v| inner.iter().any(|f| f.mentions(v)));
        if !over.is_empty() {
            outside.push(Self::Sum {
                over,
                body: Box::new(Self::product(inner)),
            });
        } else {
            outside.extend(inner);
        }
        Self::product(outside)
    }

    /// `num / den`, cancelling the top-level factors they share.
    #[must_use]
    pub fn ratio(num: Self, den: Self) -> Self {
        let mut num = num.factors();
        let mut den = den.factors();
        let mut i = 0;
        while i < den.len() {
            if let Some(j) = num.iter().position(|f| f.same_as(&den[i])) {
                num.remove(j);
                den.remove(i);
            } else {
                i += 1;
            }
        }
        let num = Self::product(num);
        let den = Self::product(den);
        if den.is_one() {
            num
        } else {
            Self::Ratio {
                num: Box::new(num),
                den: Box::new(den),
            }
        }
    }

    /// Equality up to the order of a factor's conditioning set.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::P {
                    domain: d1,
                    vars: v1,
                    given: g1,
                },
                Self::P {
                    domain: d2,
                    vars: v2,
                    given: g2,
                },
            ) => {
                let set = |v: &[String]| v.iter().cloned().collect::<BTreeSet<_>>();
                d1 == d2 && set(v1) == set(v2) && set(g1) == set(g2)
            }
            _ => self == other,
        }
    }

    /// Converts an observational [`Formula`] into one read from `domain`.
    pub(crate) fn from_formula(f: &Formula, domain: &Domain) -> Self {
        match f {
            Formula::P { vars, given, .. } => Self::P {
                domain: domain.clone(),
                vars: vars.clone(),
                given: given.clone(),
            },
            Formula::Product(fs) => {
                Self::product(fs.iter().map(|f| Self::from_formula(f, domain)).collect())
            }
            Formula::Sum { over, body } => Self::sum(over.clone(), Self::from_formula(body, domain)),
            Formula::Ratio { num, den } => {
                Self::ratio(Self::from_formula(num, domain), Self::from_formula(den, domain))
            }
        }
    }

    /// Sorts every conditioning set by `rank` (of the unprimed name), so
    /// factors print in the diagram's order whichever algorithm built them.
    pub(crate) fn sort_given(&mut self, rank: &HashMap<String, usize>) {
        match self {
            Self::P { given, .. } => {
                given.sort_by_key(|g| rank.get(base(g)).copied().unwrap_or(usize::MAX));
            }
            Self::Product(fs) => fs.iter_mut().for_each(|f| f.sort_given(rank)),
            Self::Sum { body, .. } => body.sort_given(rank),
            Self::Ratio { num, den } => {
                num.sort_given(rank);
                den.sort_given(rank);
            }
        }
    }

    fn mentions(&self, v: &str) -> bool {
        match self {
            Self::P { vars, given, .. } => vars.iter().chain(given).any(|s| s == v),
            Self::Product(fs) => fs.iter().any(|f| f.mentions(v)),
            Self::Sum { over, body } => over.iter().any(|s| s == v) || body.mentions(v),
            Self::Ratio { num, den } => num.mentions(v) || den.mentions(v),
        }
    }

    /// The variables no enclosing sum binds: the ones the estimand is a
    /// function of.
    #[must_use]
    pub fn free_vars(&self) -> BTreeSet<String> {
        fn walk(e: &Estimand, bound: &mut Vec<String>, out: &mut BTreeSet<String>) {
            match e {
                Estimand::P { vars, given, .. } => out.extend(
                    vars.iter()
                        .chain(given)
                        .filter(|v| !bound.contains(v))
                        .cloned(),
                ),
                Estimand::Product(fs) => fs.iter().for_each(|f| walk(f, bound, out)),
                Estimand::Ratio { num, den } => {
                    walk(num, bound, out);
                    walk(den, bound, out);
                }
                Estimand::Sum { over, body } => {
                    let depth = bound.len();
                    bound.extend(over.iter().cloned());
                    walk(body, bound, out);
                    bound.truncate(depth);
                }
            }
        }
        let mut out = BTreeSet::new();
        walk(self, &mut Vec::new(), &mut out);
        out
    }

    /// Every domain a factor is read from.
    #[must_use]
    pub fn domains(&self) -> BTreeSet<Domain> {
        let mut out = BTreeSet::new();
        self.visit(&mut |d, _| {
            out.insert(d.clone());
        });
        out
    }

    /// The variables the target's factors mention: what must be measured in
    /// `π*`.
    #[must_use]
    pub fn target_variables(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.visit(&mut |d, vars| {
            if *d == Domain::Target {
                out.extend(vars.into_iter().cloned());
            }
        });
        out
    }

    fn visit(&self, f: &mut impl FnMut(&Domain, Vec<&String>)) {
        match self {
            Self::P {
                domain,
                vars,
                given,
            } => f(domain, vars.iter().chain(given).collect()),
            Self::Product(fs) => fs.iter().for_each(|e| e.visit(f)),
            Self::Sum { body, .. } => body.visit(f),
            Self::Ratio { num, den } => {
                num.visit(f);
                den.visit(f);
            }
        }
    }

    /// Evaluates the estimand at `at`, an assignment of every free variable.
    ///
    /// A summation index with trailing primes (`x'`) ranges over the levels
    /// of the unprimed variable and reads its distribution.
    ///
    /// A product with a factor that is exactly zero is zero even if another
    /// factor is undefined: a conditional on a null event only matters when
    /// it is weighted by something positive.
    ///
    /// # Errors
    ///
    /// [`TransportError::InvalidInput`] if a free variable is unassigned,
    /// [`TransportError::Positivity`] for a conditional on a null event, and
    /// whatever `dists` reports for a missing domain or variable.
    pub fn evaluate(
        &self,
        dists: &impl Distributions,
        at: &[(&str, &str)],
    ) -> Result<f64, TransportError> {
        let mut env: HashMap<String, String> = at
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        for v in self.free_vars() {
            if !env.contains_key(&v) {
                return Err(TransportError::InvalidInput(format!(
                    "no value given for `{v}`"
                )));
            }
        }
        self.eval(dists, &mut env)
    }

    fn eval(
        &self,
        dists: &impl Distributions,
        env: &mut HashMap<String, String>,
    ) -> Result<f64, TransportError> {
        match self {
            Self::P {
                domain,
                vars,
                given,
            } => {
                fn bind<'a>(
                    names: &'a [String],
                    env: &'a HashMap<String, String>,
                ) -> Vec<(&'a str, &'a str)> {
                    names
                        .iter()
                        .map(|n| (base(n), env[n].as_str()))
                        .collect()
                }
                dists.conditional(domain, &bind(vars, env), &bind(given, env))
            }
            Self::Product(fs) => {
                let mut total = 1.0;
                let mut failure = None;
                for f in fs {
                    match f.eval(dists, env) {
                        Ok(0.0) => return Ok(0.0),
                        Ok(v) => total *= v,
                        Err(e) => failure = failure.or(Some(e)),
                    }
                }
                failure.map_or(Ok(total), Err)
            }
            Self::Ratio { num, den } => {
                let d = den.eval(dists, env)?;
                if d <= 0.0 {
                    return Err(TransportError::Positivity(format!(
                        "the denominator {den} is zero"
                    )));
                }
                Ok(num.eval(dists, env)? / d)
            }
            Self::Sum { over, body } => {
                let mut levels = Vec::with_capacity(over.len());
                for v in over {
                    let l = dists.levels(base(v)).ok_or_else(|| {
                        TransportError::MissingData(format!("no levels for `{v}`"))
                    })?;
                    if l.is_empty() {
                        return Ok(0.0);
                    }
                    levels.push(l);
                }
                let saved: Vec<Option<String>> = over.iter().map(|v| env.get(v).cloned()).collect();
                let mut idx = vec![0usize; over.len()];
                let mut total = 0.0;
                let mut result = Ok(());
                'outer: loop {
                    for (k, v) in over.iter().enumerate() {
                        env.insert(v.clone(), levels[k][idx[k]].clone());
                    }
                    match body.eval(dists, env) {
                        Ok(x) => total += x,
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                    for k in 0..idx.len() {
                        idx[k] += 1;
                        if idx[k] < levels[k].len() {
                            continue 'outer;
                        }
                        idx[k] = 0;
                    }
                    break;
                }
                for (v, old) in over.iter().zip(saved) {
                    match old {
                        Some(o) => env.insert(v.clone(), o),
                        None => env.remove(v),
                    };
                }
                result.map(|()| total)
            }
        }
    }

    fn write(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lower = |vs: &[String]| {
            vs.iter()
                .map(|v| v.to_lowercase())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            Self::P {
                domain,
                vars,
                given,
            } => {
                // An atomic intervention's value is shown in `do(·)`, after
                // the bar, rather than among the conditioning variables.
                let (head, shown, regime) = match domain {
                    Domain::Policy => ("P̂".to_string(), given.clone(), None),
                    Domain::Under(d, r) => {
                        let fixed = r.atomic_variables();
                        let (set, shown): (Vec<String>, Vec<String>) =
                            given.iter().cloned().partition(|g| fixed.contains(base(g)));
                        let done: Vec<String> = fixed
                            .iter()
                            .map(|v| {
                                set.iter()
                                    .find(|g| base(g) == v)
                                    .unwrap_or(v)
                                    .to_lowercase()
                            })
                            .collect();
                        let soft: Vec<&str> = r
                            .iter()
                            .filter(|(v, _)| !fixed.contains(*v))
                            .map(|(v, _)| v)
                            .collect();
                        let mut parts = Vec::new();
                        if !done.is_empty() {
                            parts.push(format!("do({})", done.join(", ")));
                        }
                        if !soft.is_empty() {
                            parts.push(format!("σ_{{{}}}", soft.join(", ")));
                        }
                        (format!("P^{d}"), shown, Some(parts.join(", ")))
                    }
                    d => (format!("P^{d}"), given.clone(), None),
                };
                write!(f, "{head}({}", lower(vars))?;
                if !shown.is_empty() {
                    write!(f, " | {}", lower(&shown))?;
                }
                if let Some(r) = regime {
                    write!(f, "; {r}")?;
                }
                f.write_str(")")
            }
            Self::Product(fs) if fs.is_empty() => f.write_str("1"),
            Self::Product(fs) => {
                for (i, factor) in fs.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    if matches!(factor, Self::Sum { .. }) && i + 1 < fs.len() {
                        f.write_str("[")?;
                        factor.write(f)?;
                        f.write_str("]")?;
                    } else {
                        factor.write(f)?;
                    }
                }
                Ok(())
            }
            Self::Sum { over, body } => {
                write!(f, "sum_{{{}}} ", lower(over))?;
                match &**body {
                    Self::Ratio { .. } | Self::Sum { .. } => {
                        f.write_str("[")?;
                        body.write(f)?;
                        f.write_str("]")
                    }
                    other => other.write(f),
                }
            }
            Self::Ratio { num, den } => {
                f.write_str("[")?;
                num.write(f)?;
                f.write_str("] / [")?;
                den.write(f)?;
                f.write_str("]")
            }
        }
    }
}

/// The variable a name stands for: a summation index renamed to avoid
/// capture carries trailing primes (`x'`), as in the front-door formula.
fn base(name: &str) -> &str {
    name.trim_end_matches('\'')
}

impl fmt::Display for Estimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f)
    }
}

/// The distributions an [`Estimand`] is evaluated against.
pub trait Distributions {
    /// The values `var` ranges over, used to enumerate sums.
    fn levels(&self, var: &str) -> Option<Vec<String>>;

    /// `P^domain(event)`, the marginal probability of a joint assignment.
    ///
    /// # Errors
    ///
    /// [`TransportError::MissingData`] when the domain has no distribution or
    /// its distribution does not cover a variable of the event.
    fn probability(&self, domain: &Domain, event: &[(&str, &str)]) -> Result<f64, TransportError>;

    /// `P^domain(event | given)`. The default divides two calls to
    /// [`Distributions::probability`]; override it for a domain known only
    /// as a conditional, such as [`Domain::Policy`].
    ///
    /// # Errors
    ///
    /// [`TransportError::Positivity`] when `given` has probability zero, and
    /// whatever [`Distributions::probability`] reports.
    fn conditional(
        &self,
        domain: &Domain,
        event: &[(&str, &str)],
        given: &[(&str, &str)],
    ) -> Result<f64, TransportError> {
        let mut joint = event.to_vec();
        joint.extend(given.iter().copied());
        let num = self.probability(domain, &joint)?;
        if given.is_empty() {
            return Ok(num);
        }
        let den = self.probability(domain, given)?;
        if den <= 0.0 {
            return Err(TransportError::Positivity(format!(
                "P^{domain}({}) = 0",
                given
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(num / den)
    }
}

/// A domain and a sorted joint assignment.
type EventKey = (Domain, Vec<(String, String)>);

/// Empirical distributions, one [`Dataset`] per domain: evaluating an
/// estimand against them is the plug-in estimator.
///
/// The target's table may hold only the measured variables `W`; any factor
/// that needs more reports [`TransportError::MissingData`]. Row weights are
/// honoured.
///
/// ```
/// use why_data::transport::{Dataset, Domain, EmpiricalDistributions, Estimand};
///
/// let src = Dataset::from_csv("x,y\n0,0\n0,1\n1,1\n1,1\n").unwrap();
/// let dists = EmpiricalDistributions::new().with(Domain::source("1"), &src);
/// let e = Estimand::p(Domain::source("1"), &["y"], &["x"]);
/// assert_eq!(e.evaluate(&dists, &[("x", "0"), ("y", "1")]).unwrap(), 0.5);
/// ```
#[derive(Debug, Default)]
pub struct EmpiricalDistributions<'a> {
    data: Vec<(Domain, &'a Dataset)>,
    cache: RefCell<HashMap<EventKey, f64>>,
}

impl<'a> EmpiricalDistributions<'a> {
    /// No domains yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the sample for `domain`.
    #[must_use]
    pub fn with(mut self, domain: Domain, data: &'a Dataset) -> Self {
        self.data.retain(|(d, _)| *d != domain);
        self.data.push((domain, data));
        self
    }
}

impl Distributions for EmpiricalDistributions<'_> {
    fn levels(&self, var: &str) -> Option<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for (_, d) in &self.data {
            if let Some(levels) = d.levels(var) {
                for l in levels {
                    if !out.contains(l) {
                        out.push(l.clone());
                    }
                }
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn probability(&self, domain: &Domain, event: &[(&str, &str)]) -> Result<f64, TransportError> {
        let mut key: Vec<(String, String)> = event
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        key.sort();
        let cache_key = (domain.clone(), key);
        if let Some(p) = self.cache.borrow().get(&cache_key) {
            return Ok(*p);
        }
        let data = self
            .data
            .iter()
            .find(|(d, _)| d == domain)
            .map(|(_, d)| *d)
            .ok_or_else(|| TransportError::MissingData(format!("no data for domain {domain}")))?;
        let mut cols = Vec::with_capacity(event.len());
        for (var, value) in event {
            let col = data.column(var).map_err(|_| {
                TransportError::MissingData(format!("domain {domain} does not measure `{var}`"))
            })?;
            match col.levels.iter().position(|l| l == value) {
                Some(code) => cols.push((col, code as u32)),
                None => {
                    self.cache.borrow_mut().insert(cache_key, 0.0);
                    return Ok(0.0);
                }
            }
        }
        let mut hit = 0.0;
        let mut total = 0.0;
        for row in 0..data.len() {
            let w = data.weight(row);
            total += w;
            if cols.iter().all(|(c, code)| c.codes[row] == *code) {
                hit += w;
            }
        }
        if total <= 0.0 {
            return Err(TransportError::MissingData(format!(
                "domain {domain} has no rows"
            )));
        }
        let p = hit / total;
        self.cache.borrow_mut().insert(cache_key, p);
        Ok(p)
    }
}
