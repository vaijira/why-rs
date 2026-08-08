//! The Shpitser-Pearl ID algorithm.
//!
//! [`identify`] decides whether an interventional distribution `P(y | do(x))`
//! is computable from observational data plus the causal diagram, and when it
//! is, returns the estimand as a [`Formula`]. When it is not, it returns the
//! [`Hedge`] that witnesses the failure.
//!
//! This is what chapter 4's `DoCalculusEngine.compute` does, and it subsumes
//! the named criteria: given a back-door admissible set, ID derives the
//! adjustment formula; given the front-door graph, it derives the front-door
//! formula; on the bow arc it reports a hedge.
//!
//! ```
//! use why_data::graph::dseparation::Admg;
//! use why_data::graph::identification::identify;
//!
//! // Confounded triangle: adjusting for Z identifies the effect.
//! let mut g = Admg::new();
//! g.directed("Z", "X");
//! g.directed("Z", "Y");
//! g.directed("X", "Y");
//!
//! let estimand = identify(&g, &["X"], &["Y"]).unwrap();
//! assert_eq!(estimand.to_string(), "sum_{z} P(y | z, x) P(z)");
//! ```

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use super::dseparation::{Admg, DSepError};

/// An estimand: an expression in observational quantities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Formula {
    /// `P(vars | given)`, a conditional of the observational distribution.
    P {
        /// The variables whose probability this is.
        vars: Vec<String>,
        /// What it is conditioned on; empty for a plain marginal.
        given: Vec<String>,
    },
    /// A product of factors.
    Product(Vec<Formula>),
    /// A sum marginalising `over` out of `body`.
    Sum {
        /// The variables summed out.
        over: Vec<String>,
        /// What is being summed.
        body: Box<Formula>,
    },
    /// A quotient, which conditioning introduces when the numerator is not
    /// already a product of conditionals.
    Ratio {
        /// The numerator.
        num: Box<Formula>,
        /// The denominator.
        den: Box<Formula>,
    },
}

impl Formula {
    /// A product, flattened and reduced to its single factor when it has one.
    fn product(mut factors: Vec<Self>) -> Self {
        let mut flat = Vec::new();
        for f in factors.drain(..) {
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

    /// A sum, dropping the sum entirely when nothing is summed over, and
    /// discarding factors that marginalise to one.
    ///
    /// Nested sums are merged first: `sum_a sum_b` is `sum_{a,b}`, and flattening
    /// them lets the cancellation below see the whole product at once.
    fn sum(over: Vec<String>, body: Self) -> Self {
        let (over, body) = match body {
            Self::Sum {
                over: inner,
                body: inner_body,
            } => {
                let mut merged = over;
                merged.extend(inner);
                (merged, *inner_body)
            }
            other => (over, other),
        };
        let (over, body) = Self::cancel(over, body);
        if over.is_empty() {
            body
        } else {
            Self::Sum {
                over,
                body: Box::new(body),
            }
        }
    }

    /// Drops `sum_v P(v | ...)` factors, which are one.
    ///
    /// Summing a conditional over its own variable gives one, so the factor and
    /// the summation index both disappear — provided the variable appears
    /// nowhere else in the product. This is what turns the napkin denominator
    /// from `sum_y sum_w P(w) P(x|w,z) P(y|w,z,x)` into `sum_w P(w) P(x|w,z)`.
    fn cancel(over: Vec<String>, body: Self) -> (Vec<String>, Self) {
        let Self::Product(factors) = body else {
            return (over, body);
        };
        let mut factors = factors;
        let mut remaining = Vec::new();

        for v in over {
            let mentions: Vec<usize> = factors
                .iter()
                .enumerate()
                .filter(|(_, f)| f.mentions(&v))
                .map(|(i, _)| i)
                .collect();
            let only_as_its_own_head = mentions.len() == 1
                && matches!(&factors[mentions[0]], Self::P { vars, .. } if vars == std::slice::from_ref(&v));
            if only_as_its_own_head {
                factors.remove(mentions[0]);
            } else {
                remaining.push(v);
            }
        }
        (remaining, Self::product(factors))
    }

    /// Collapses the estimand into the form the textbooks print.
    ///
    /// Three sound rewritings, applied bottom-up:
    ///
    /// - a run of chain-rule factors becomes one joint —
    ///   `P(a | c) P(b | a, c)` is `P(a, b | c)`;
    /// - summing a joint over some of its own variables drops them —
    ///   `sum_a P(a, b)` is `P(b)`;
    /// - a joint over one of its own marginals is a conditional —
    ///   `P(a, b | c) / P(b | c)` is `P(a | b, c)`.
    #[must_use]
    pub fn simplify(self) -> Self {
        match self {
            Self::P { .. } => self,
            Self::Product(factors) => {
                let simplified: Vec<Self> = factors.into_iter().map(Self::simplify).collect();
                Self::product(collapse_runs(simplified))
            }
            Self::Sum { over, body } => {
                let collapsed = Self::sum(over, body.simplify());
                let Self::Sum { over, body } = &collapsed else {
                    return collapsed;
                };
                // sum_a P(a, b | c) = P(b | c)
                if let Self::P { vars, given } = &**body
                    && over
                        .iter()
                        .all(|v| vars.contains(v) && !given.contains(v))
                {
                    let kept: Vec<String> =
                        vars.iter().filter(|v| !over.contains(v)).cloned().collect();
                    if !kept.is_empty() {
                        return Self::P {
                            vars: kept,
                            given: given.clone(),
                        };
                    }
                }
                collapsed
            }
            Self::Ratio { num, den } => {
                let (num, den) = (num.simplify(), den.simplify());
                // P(a, b | c) / P(b | c) = P(a | b, c)
                if let (Self::P { vars: a, given: gn }, Self::P { vars: b, given: gd }) =
                    (&num, &den)
                    && gn == gd
                    && b.iter().all(|v| a.contains(v))
                {
                    let kept: Vec<String> =
                        a.iter().filter(|v| !b.contains(v)).cloned().collect();
                    if !kept.is_empty() {
                        let mut given = gn.clone();
                        given.extend(b.iter().cloned());
                        return Self::P { vars: kept, given };
                    }
                }
                Self::Ratio {
                    num: Box::new(num),
                    den: Box::new(den),
                }
            }
        }
    }

    /// Whether `v` occurs anywhere in this formula.
    fn mentions(&self, v: &str) -> bool {
        match self {
            Self::P { vars, given } => {
                vars.iter().any(|s| s == v) || given.iter().any(|s| s == v)
            }
            Self::Product(fs) => fs.iter().any(|f| f.mentions(v)),
            Self::Sum { over, body } => over.iter().any(|s| s == v) || body.mentions(v),
            Self::Ratio { num, den } => num.mentions(v) || den.mentions(v),
        }
    }

    /// Renames every occurrence of `from` to `to`.
    fn rename(&mut self, from: &str, to: &str) {
        let swap = |s: &mut String| {
            if s == from {
                *s = to.to_string();
            }
        };
        match self {
            Self::P { vars, given } => {
                vars.iter_mut().for_each(swap);
                given.iter_mut().for_each(swap);
            }
            Self::Product(fs) => fs.iter_mut().for_each(|f| f.rename(from, to)),
            Self::Sum { over, body } => {
                over.iter_mut().for_each(swap);
                body.rename(from, to);
            }
            Self::Ratio { num, den } => {
                num.rename(from, to);
                den.rename(from, to);
            }
        }
    }

    /// Primes any summation index that collides with a free variable.
    ///
    /// The front-door estimand sums over `x` while `x` is also the treatment
    /// being asked about; textbooks write the bound one `x'`, and so does this.
    fn avoid_capture(&mut self, free: &BTreeSet<String>) {
        match self {
            Self::P { .. } => {}
            Self::Product(fs) => fs.iter_mut().for_each(|f| f.avoid_capture(free)),
            Self::Ratio { num, den } => {
                num.avoid_capture(free);
                den.avoid_capture(free);
            }
            Self::Sum { over, body } => {
                let collisions: Vec<String> =
                    over.iter().filter(|v| free.contains(*v)).cloned().collect();
                for v in collisions {
                    let primed = format!("{v}'");
                    for slot in over.iter_mut() {
                        if *slot == v {
                            *slot = primed.clone();
                        }
                    }
                    body.rename(&v, &primed);
                }
                body.avoid_capture(free);
            }
        }
    }

    fn write(&self, f: &mut fmt::Formatter<'_>, latex: bool) -> fmt::Result {
        let lower = |vs: &[String]| {
            vs.iter()
                .map(|v| v.to_lowercase())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            Self::P { vars, given } => {
                let bar = if latex { " \\mid " } else { " | " };
                if given.is_empty() {
                    write!(f, "P({})", lower(vars))
                } else {
                    write!(f, "P({}{bar}{})", lower(vars), lower(given))
                }
            }
            Self::Product(factors) => {
                for (i, factor) in factors.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    factor.write(f, latex)?;
                }
                Ok(())
            }
            Self::Sum { over, body } => {
                if latex {
                    write!(f, "\\sum_{{{}}} ", lower(over))?;
                } else {
                    write!(f, "sum_{{{}}} ", lower(over))?;
                }
                // A bare product under a sum needs no brackets; anything that
                // could swallow following factors does.
                match &**body {
                    Self::Ratio { .. } | Self::Sum { .. } => {
                        f.write_str("[")?;
                        body.write(f, latex)?;
                        f.write_str("]")
                    }
                    other => other.write(f, latex),
                }
            }
            Self::Ratio { num, den } => {
                if latex {
                    f.write_str("\\frac{")?;
                    num.write(f, true)?;
                    f.write_str("}{")?;
                    den.write(f, true)?;
                    f.write_str("}")
                } else {
                    f.write_str("[")?;
                    num.write(f, false)?;
                    f.write_str("] / [")?;
                    den.write(f, false)?;
                    f.write_str("]")
                }
            }
        }
    }

    /// The estimand as LaTeX.
    #[must_use]
    pub fn to_latex(&self) -> String {
        struct Latex<'a>(&'a Formula);
        impl fmt::Display for Latex<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.write(f, true)
            }
        }
        Latex(self).to_string()
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write(f, false)
    }
}


/// Merges maximal runs of chain-rule factors into joint distributions.
///
/// `P(a | c) P(b | a, c)` is the chain rule for `P(a, b | c)`, so a run whose
/// conditioning sets grow by exactly the variables already accumulated collapses
/// into one term. Runs are maximal and greedy from the left, which is the order
/// the ID recursion emits factors in.
fn collapse_runs(factors: Vec<Formula>) -> Vec<Formula> {
    let head = |f: &Formula| match f {
        Formula::P { vars, given } if vars.len() == 1 => Some((vars[0].clone(), given.clone())),
        _ => None,
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < factors.len() {
        let Some((var, given)) = head(&factors[i]) else {
            out.push(factors[i].clone());
            i += 1;
            continue;
        };
        let common: BTreeSet<String> = given.iter().cloned().collect();
        let mut accumulated = vec![var];
        let mut j = i + 1;
        while j < factors.len() {
            let Some((next, next_given)) = head(&factors[j]) else {
                break;
            };
            let expected: BTreeSet<String> = common
                .iter()
                .cloned()
                .chain(accumulated.iter().cloned())
                .collect();
            if next_given.iter().cloned().collect::<BTreeSet<String>>() != expected {
                break;
            }
            accumulated.push(next);
            j += 1;
        }
        if accumulated.len() > 1 {
            out.push(Formula::P {
                vars: accumulated,
                given,
            });
            i = j;
        } else {
            out.push(factors[i].clone());
            i += 1;
        }
    }
    out
}

/// The obstruction that makes an effect non-identifiable.
///
/// A hedge is a pair of c-forests witnessing that two models agreeing on all
/// observational data disagree on the effect. `f` is the vertex set of the
/// larger, `f_prime` of the smaller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hedge {
    /// The larger c-forest.
    pub f: Vec<String>,
    /// The subset that forms the smaller one.
    pub f_prime: Vec<String>,
}

impl fmt::Display for Hedge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "hedge: {{{}}} over {{{}}}",
            self.f.join(", "),
            self.f_prime.join(", ")
        )
    }
}

/// Why identification failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdError {
    /// The effect is not identifiable, witnessed by a hedge.
    NotIdentifiable(Hedge),
    /// The query or graph is malformed.
    Graph(DSepError),
}

impl From<DSepError> for IdError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotIdentifiable(h) => write!(f, "not identifiable ({h})"),
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for IdError {}

/// The distribution the recursion carries.
///
/// `formula` is how it prints; `terms` holds the chain-rule factors when the
/// distribution is still a plain product of conditionals, which lets steps 6
/// and 7 emit `P(v | ...)` instead of a quotient. Once a marginal has to be
/// taken that cannot be expressed by dropping factors, `terms` becomes `None`
/// and conditioning falls back to a ratio.
#[derive(Clone, Debug)]
struct Dist {
    domain: BTreeSet<String>,
    terms: Option<Vec<(String, Vec<String>)>>,
    formula: Formula,
}

impl Dist {
    /// The observational distribution over the whole graph, factorised by the
    /// chain rule along a topological order.
    fn observational(order: &[String]) -> Self {
        let terms: Vec<(String, Vec<String>)> = order
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), order[..i].to_vec()))
            .collect();
        Self {
            domain: order.iter().cloned().collect(),
            formula: Self::product_of(&terms),
            terms: Some(terms),
        }
    }

    fn product_of(terms: &[(String, Vec<String>)]) -> Formula {
        Formula::product(
            terms
                .iter()
                .map(|(v, given)| Formula::P {
                    vars: vec![v.clone()],
                    given: given.clone(),
                })
                .collect(),
        )
    }

    /// The marginal over `keep`.
    ///
    /// When the factors are a chain-rule product and `keep` is closed under the
    /// order's predecessors, the marginal is just the surviving factors — the
    /// trailing ones sum to one. Otherwise it has to stay a sum.
    fn marginal(&self, keep: &BTreeSet<String>) -> Self {
        if let Some(terms) = &self.terms {
            let closed = terms
                .iter()
                .filter(|(v, _)| keep.contains(v))
                .all(|(_, given)| {
                    given
                        .iter()
                        .all(|p| !self.domain.contains(p) || keep.contains(p))
                });
            if closed {
                let kept: Vec<(String, Vec<String>)> = terms
                    .iter()
                    .filter(|(v, _)| keep.contains(v))
                    .cloned()
                    .collect();
                return Self {
                    domain: keep.clone(),
                    formula: Self::product_of(&kept),
                    terms: Some(kept),
                };
            }
        }
        let over: Vec<String> = self.domain.difference(keep).cloned().collect();
        Self {
            domain: keep.clone(),
            terms: None,
            formula: Formula::sum(over, self.formula.clone()),
        }
    }

    /// `P(var | given)` under this distribution.
    fn conditional(&self, var: &str, given: &[String]) -> Formula {
        if let Some(terms) = &self.terms
            && let Some((v, g)) = terms.iter().find(|(v, _)| v == var)
        {
            return Formula::P {
                vars: vec![v.clone()],
                given: g.clone(),
            };
        }
        let given_set: BTreeSet<String> = given.iter().cloned().collect();
        let mut num_keep = given_set.clone();
        num_keep.insert(var.to_string());
        Formula::Ratio {
            num: Box::new(self.marginal(&num_keep).formula),
            den: Box::new(self.marginal(&given_set).formula),
        }
    }

    /// The distribution restricted to `keep`, as a product of the conditionals
    /// of each surviving variable — steps 6 and 7.
    fn restrict(&self, keep: &BTreeSet<String>, order: &[String]) -> Self {
        let mut terms = Vec::new();
        let mut factors = Vec::new();
        let mut plain = true;
        for (i, v) in order.iter().enumerate() {
            if !keep.contains(v) {
                continue;
            }
            let given = order[..i].to_vec();
            let factor = self.conditional(v, &given);
            if let Formula::P { vars, given } = &factor {
                terms.push((vars[0].clone(), given.clone()));
            } else {
                plain = false;
            }
            factors.push(factor);
        }
        Self {
            domain: keep.clone(),
            terms: plain.then_some(terms),
            formula: Formula::product(factors),
        }
    }
}

fn sorted(set: &BTreeSet<String>) -> Vec<String> {
    set.iter().cloned().collect()
}

fn as_refs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

/// Identifies `P(y | do(x))` from the causal diagram, or reports why it cannot
/// be.
///
/// Implements ID (Shpitser & Pearl, 2006). The seven steps, in order:
///
/// 1. Nothing intervened on — marginalise.
/// 2. Drop the variables that are not ancestors of `Y`.
/// 3. Fold into `X` the variables that intervening on cannot matter for.
/// 4. Split on the c-components of `G \ X` and recurse on each.
/// 5. A single c-component that is the whole of `C(G)` is a hedge — fail.
/// 6. A single c-component that is already one of `C(G)` — read the answer off
///    the chain rule.
/// 7. Otherwise descend into the c-component of `G` containing it.
///
/// # Errors
///
/// [`IdError::NotIdentifiable`] with the witnessing [`Hedge`] when no estimand
/// exists, and [`IdError::Graph`] for an unknown variable, a cyclic diagram, or
/// overlapping `x` and `y`.
pub fn identify(g: &Admg, x: &[&str], y: &[&str]) -> Result<Formula, IdError> {
    let known: HashSet<&str> = g.variables().into_iter().collect();
    for v in x.iter().chain(y) {
        if !known.contains(v) {
            return Err(IdError::Graph(DSepError::Unknown((*v).to_string())));
        }
    }
    if let Some(v) = x.iter().find(|v| y.contains(v)) {
        return Err(IdError::Graph(DSepError::Overlap((*v).to_string())));
    }

    let order = g.topological_order()?;
    let dist = Dist::observational(&order);
    let xs: BTreeSet<String> = x.iter().map(ToString::to_string).collect();
    let ys: BTreeSet<String> = y.iter().map(ToString::to_string).collect();

    let mut formula = id(&ys, &xs, &dist, g, &order)?.simplify();
    // The query's own variables stay free, so a sum must not capture them.
    let free: BTreeSet<String> = xs.union(&ys).cloned().collect();
    formula.avoid_capture(&free);
    Ok(formula)
}

/// Identifies the conditional effect `P(y | do(x), z)`.
///
/// IDC (Shpitser & Pearl, 2006). Each variable of `z` that rule 2 licenses
/// treating as an intervention is moved into `x`; what remains is identified
/// jointly with `y` by [`identify`] and then normalised:
///
/// ```text
/// P(y | do(x), z) = ID(y ∪ z, x) / sum_y ID(y ∪ z, x)
/// ```
///
/// # Errors
///
/// As [`identify`]. A `z` that cannot be absorbed and whose joint with `y` is
/// not identifiable yields [`IdError::NotIdentifiable`].
pub fn identify_conditional(
    g: &Admg,
    x: &[&str],
    y: &[&str],
    z: &[&str],
) -> Result<Formula, IdError> {
    let mut xs: Vec<String> = x.iter().map(ToString::to_string).collect();
    let mut zs: Vec<String> = z.iter().map(ToString::to_string).collect();

    // Rule 2 turns an observation into an action whenever the exchange is
    // licensed, and every variable so moved makes the remaining problem easier.
    'outer: loop {
        for (i, candidate) in zs.iter().enumerate() {
            let rest: Vec<String> = zs
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| v.clone())
                .collect();
            let mutilated = g
                .cut_edges_into(&as_refs(&xs))?
                .cut_edges_out_of(&[candidate.as_str()])?;
            let mut given: Vec<String> = xs.clone();
            given.extend(rest.iter().cloned());
            if mutilated.is_d_separated(y, &[candidate.as_str()], &as_refs(&given))? {
                xs.push(candidate.clone());
                zs = rest;
                continue 'outer;
            }
        }
        break;
    }

    if zs.is_empty() {
        return identify(g, &as_refs(&xs), y);
    }

    let mut joint: Vec<String> = y.iter().map(ToString::to_string).collect();
    joint.extend(zs.iter().cloned());
    let numerator = identify(g, &as_refs(&xs), &as_refs(&joint))?;
    let denominator = Formula::sum(
        y.iter().map(ToString::to_string).collect(),
        numerator.clone(),
    );
    Ok(Formula::Ratio {
        num: Box::new(numerator),
        den: Box::new(denominator),
    }
    .simplify())
}

fn id(
    y: &BTreeSet<String>,
    x: &BTreeSet<String>,
    p: &Dist,
    g: &Admg,
    order: &[String],
) -> Result<Formula, IdError> {
    let v: BTreeSet<String> = g.variables().into_iter().map(ToString::to_string).collect();

    // 1. Nothing left to intervene on: marginalise what we have.
    if x.is_empty() {
        let over: Vec<String> = v.difference(y).cloned().collect();
        return Ok(Formula::sum(over, p.formula.clone()));
    }

    // 2. Only the ancestors of Y can matter.
    let an_y: BTreeSet<String> = g
        .ancestors_of(&as_refs(&sorted(y)))?
        .into_iter()
        .map(ToString::to_string)
        .collect();
    if v.difference(&an_y).next().is_some() {
        let sub = g.induced_subgraph(&as_refs(&sorted(&an_y)))?;
        let sub_order: Vec<String> =
            order.iter().filter(|v| an_y.contains(*v)).cloned().collect();
        let x_an: BTreeSet<String> = x.intersection(&an_y).cloned().collect();
        return id(y, &x_an, &p.marginal(&an_y), &sub, &sub_order);
    }

    // 3. Variables that intervening on X already makes irrelevant to Y can be
    //    intervened on too, for free.
    let cut = g.cut_edges_into(&as_refs(&sorted(x)))?;
    let an_y_cut: BTreeSet<String> = cut
        .ancestors_of(&as_refs(&sorted(y)))?
        .into_iter()
        .map(ToString::to_string)
        .collect();
    let w: BTreeSet<String> = v
        .difference(x)
        .filter(|v| !an_y_cut.contains(*v))
        .cloned()
        .collect();
    if !w.is_empty() {
        let x_w: BTreeSet<String> = x.union(&w).cloned().collect();
        return id(y, &x_w, p, g, order);
    }

    // 4. Several c-components in G \ X: the effect factorises over them.
    let rest: BTreeSet<String> = v.difference(x).cloned().collect();
    let g_minus_x = g.induced_subgraph(&as_refs(&sorted(&rest)))?;
    let components = g_minus_x.c_components();
    if components.len() > 1 {
        let mut factors = Vec::with_capacity(components.len());
        for s in &components {
            let s_set: BTreeSet<String> = s.iter().cloned().collect();
            let complement: BTreeSet<String> = v.difference(&s_set).cloned().collect();
            factors.push(id(&s_set, &complement, p, g, order)?);
        }
        let yx: BTreeSet<String> = y.union(x).cloned().collect();
        let over: Vec<String> = v.difference(&yx).cloned().collect();
        return Ok(Formula::sum(over, Formula::product(factors)));
    }

    let s: BTreeSet<String> = components
        .first()
        .map(|c| c.iter().cloned().collect())
        .unwrap_or_default();
    let g_components = g.c_components();

    // 5. The whole graph is one c-component: a hedge, and no estimand exists.
    if g_components.len() == 1 {
        return Err(IdError::NotIdentifiable(Hedge {
            f: sorted(&v),
            f_prime: sorted(&s),
        }));
    }

    // 6. The component is already one of G's: read it off the chain rule.
    if g_components.iter().any(|c| {
        let set: BTreeSet<String> = c.iter().cloned().collect();
        set == s
    }) {
        let restricted = p.restrict(&s, order);
        let over: Vec<String> = s.difference(y).cloned().collect();
        return Ok(Formula::sum(over, restricted.formula));
    }

    // 7. Descend into the c-component of G that contains it.
    let s_prime: BTreeSet<String> = g_components
        .iter()
        .map(|c| c.iter().cloned().collect::<BTreeSet<String>>())
        .find(|c| s.is_subset(c))
        .expect("every c-component of G \\ X lies inside one of G");
    let sub = g.induced_subgraph(&as_refs(&sorted(&s_prime)))?;
    let sub_order: Vec<String> = order
        .iter()
        .filter(|v| s_prime.contains(*v))
        .cloned()
        .collect();
    let x_s: BTreeSet<String> = x.intersection(&s_prime).cloned().collect();
    id(y, &x_s, &p.restrict(&s_prime, order), &sub, &sub_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backdoor;

    /// Z -> X -> Y with Z -> Y: the textbook confounder.
    fn back_door() -> Admg {
        let mut g = Admg::new();
        g.directed("Z", "X");
        g.directed("Z", "Y");
        g.directed("X", "Y");
        g
    }

    /// X -> M -> Y with X <-> Y: the front-door graph.
    fn front_door() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "M");
        g.directed("M", "Y");
        g.bidirected("X", "Y");
        g
    }

    /// X -> Y with X <-> Y: the bow arc, the smallest non-identifiable effect.
    fn bow_arc() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "Y");
        g.bidirected("X", "Y");
        g
    }

    /// W -> Z -> X -> Y with W <-> X and W <-> Y: the napkin.
    fn napkin() -> Admg {
        let mut g = Admg::new();
        g.directed("W", "Z");
        g.directed("Z", "X");
        g.directed("X", "Y");
        g.bidirected("W", "X");
        g.bidirected("W", "Y");
        g
    }

    #[test]
    fn derives_the_back_door_adjustment() {
        let g = back_door();
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        assert_eq!(e.to_string(), "sum_{z} P(y | z, x) P(z)");
    }

    #[test]
    fn derives_the_front_door_formula() {
        let g = front_door();
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        // The front-door formula. The inner sum's index is primed, so the bound
        // x cannot be confused with the treatment being asked about; factors
        // come out in topological order, so P(x') precedes P(y | x', m).
        assert_eq!(
            e.to_string(),
            "sum_{m} P(m | x) sum_{x'} P(x') P(y | x', m)"
        );
    }

    #[test]
    fn the_bow_arc_is_not_identifiable() {
        let g = bow_arc();
        let err = identify(&g, &["X"], &["Y"]).unwrap_err();
        match err {
            IdError::NotIdentifiable(h) => {
                assert_eq!(h.f, vec!["X".to_string(), "Y".to_string()]);
                assert_eq!(h.f_prime, vec!["Y".to_string()]);
            }
            other => panic!("expected a hedge, got {other}"),
        }
    }

    #[test]
    fn the_napkin_is_identifiable() {
        let g = napkin();
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        // The book prints
        //   \frac{\sum_{W} P(X,Y | W,Z) P(W)}{\sum_{W} P(X | W,Z) P(W)}
        // and this is the same expression, factors in topological order.
        assert_eq!(
            e.to_string(),
            "[sum_{w} P(w) P(x, y | w, z)] / [sum_{w} P(w) P(x | w, z)]"
        );
    }

    #[test]
    fn no_confounding_needs_no_adjustment() {
        let mut g = Admg::new();
        g.directed("X", "Y");
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        assert_eq!(e.to_string(), "P(y | x)");
    }

    #[test]
    fn an_unconfounded_mediator_needs_no_adjustment() {
        // X -> M -> Y with nothing confounded. The raw estimand is
        // sum_m P(m | x) P(y | x, m), which is the chain rule for P(m, y | x);
        // summing m out leaves P(y | x).
        let mut g = Admg::new();
        g.directed("X", "M");
        g.directed("M", "Y");
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        assert_eq!(e.to_string(), "P(y | x)");
    }

    /// Whenever the back-door module finds an admissible set, ID must succeed —
    /// the two are independent routes to the same conclusion.
    #[test]
    fn id_succeeds_wherever_a_back_door_set_exists() {
        let graphs = [back_door(), front_door(), napkin(), bow_arc(), {
            let mut g = Admg::new();
            g.directed("Z1", "X");
            g.directed("Z2", "Z1");
            g.directed("Z2", "Y");
            g.directed("X", "Y");
            g.bidirected("Z1", "Z2");
            g.bidirected("X", "Z1");
            g
        }];
        for g in &graphs {
            let sets = backdoor::list_admissible_sets(g, &["X"], &["Y"], &[]).unwrap();
            let identified = identify(g, &["X"], &["Y"]).is_ok();
            if !sets.is_empty() {
                assert!(
                    identified,
                    "back door found {sets:?} but ID failed on {:?}",
                    g.variables()
                );
            }
        }
    }

    /// The napkin has no admissible set yet is identifiable — ID is strictly
    /// stronger than the back-door criterion.
    #[test]
    fn id_is_stronger_than_the_back_door_criterion() {
        let g = napkin();
        assert!(
            backdoor::list_admissible_sets(&g, &["X"], &["Y"], &[])
                .unwrap()
                .is_empty(),
            "the napkin has no back-door admissible set"
        );
        assert!(identify(&g, &["X"], &["Y"]).is_ok());
    }

    #[test]
    fn the_effect_on_a_non_descendant_is_the_marginal() {
        // Y is not affected by X at all.
        let mut g = Admg::new();
        g.directed("X", "M");
        g.directed("Z", "Y");
        let e = identify(&g, &["X"], &["Y"]).unwrap();
        // X cannot reach Y at all, so the effect is just the marginal.
        assert_eq!(e.to_string(), "P(y)");
    }

    /// The notebook's `G4` for the ID section.
    fn ch4_g4() -> Admg {
        let mut g = Admg::new();
        g.directed("X", "Z1");
        g.directed("Z1", "Y");
        g.directed("Z2", "Y");
        g.bidirected("X", "Z2");
        g.bidirected("Z1", "Z2");
        g
    }

    #[test]
    fn ch4_g4_joint_intervention() {
        // P(y | do(x, z2)) — the notebook's second query, which prints
        // \sum_{Z1} P(Y | X,Z1,Z2) P(Z1 | X).
        let e = identify(&ch4_g4(), &["X", "Z2"], &["Y"]).unwrap();
        assert_eq!(e.to_string(), "sum_{z1} P(y | z2, x, z1) P(z1 | x)");
    }

    #[test]
    fn ch4_g4_conditional_query_is_not_identifiable() {
        // P(y | do(x), z2) — the notebook's first query, which reports
        // "P(Y | do(X), Z2) is not identifiable."
        let err = identify_conditional(&ch4_g4(), &["X"], &["Y"], &["Z2"]).unwrap_err();
        match err {
            IdError::NotIdentifiable(h) => {
                assert_eq!(h.f, vec!["X", "Z1", "Z2"]);
                assert_eq!(h.f_prime, vec!["Z1", "Z2"]);
            }
            other => panic!("expected a hedge, got {other}"),
        }
    }

    #[test]
    fn idc_absorbs_what_rule_two_licenses() {
        // In the back-door graph, conditioning on Z is the same as intervening
        // on it, so IDC should reduce to a plain ID with no ratio left over.
        let e = identify_conditional(&back_door(), &["X"], &["Y"], &["Z"]).unwrap();
        assert!(
            !matches!(e, Formula::Ratio { .. }),
            "Z should have been absorbed into the intervention, got {e}"
        );
    }

    #[test]
    fn queries_are_validated() {
        let g = back_door();
        assert_eq!(
            identify(&g, &["nope"], &["Y"]).unwrap_err(),
            IdError::Graph(DSepError::Unknown("nope".to_string()))
        );
        assert_eq!(
            identify(&g, &["X"], &["X"]).unwrap_err(),
            IdError::Graph(DSepError::Overlap("X".to_string()))
        );
    }

    #[test]
    fn latex_rendering() {
        let e = identify(&back_door(), &["X"], &["Y"]).unwrap();
        assert_eq!(e.to_latex(), "\\sum_{z} P(y \\mid z, x) P(z)");
        let e = identify(&napkin(), &["X"], &["Y"]).unwrap();
        assert!(e.to_latex().starts_with("\\frac{"), "{}", e.to_latex());
    }
}
