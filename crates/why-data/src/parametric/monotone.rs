//! Counterfactual identification under monotonicity (Sec. 19.7): causal
//! graphs with monotonicity assumptions, the monotonicity reduction lemma,
//! Algorithm 67 (M-REDUCE) and M-ID.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use super::ParametricError;
use super::linear::edges;
use crate::fairness::Dataset;
use crate::graph::cinfer::Input;
use crate::graph::cinfer::counterfactual::{CtfEstimand, CtfQuery, ctf_infer};
use crate::graph::dseparation::Admg;
use crate::transport::{Domain, EmpiricalDistributions, Estimand};

/// The direction of a monotonic edge (Def. 19.7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    /// The child never decreases when the parent increases.
    Positive,
    /// The child never increases when the parent increases.
    Negative,
}

/// A causal graph with monotonicity assumptions (Def. 19.7.2) over binary
/// variables (levels 0 and 1).
#[derive(Clone, Debug)]
pub struct Cgma {
    g: Admg,
    signs: BTreeMap<(String, String), Sign>,
    order: Vec<String>,
    parents: BTreeMap<String, Vec<String>>,
}

/// One term `W_{pa} = w` of a ctf-factor: a variable, a full assignment of
/// its parents, and its value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CtfTerm {
    /// `W`.
    pub var: String,
    /// The value of every parent of `W`.
    pub parents: BTreeMap<String, u8>,
    /// `w`.
    pub value: u8,
}

impl Cgma {
    /// The CGMA of `g` with the monotonic edges `monotone`.
    ///
    /// # Errors
    ///
    /// [`ParametricError::NotAnEdge`] for a monotonicity assumption on a
    /// pair that is not a directed edge, and graph errors (a cycle).
    pub fn new(g: &Admg, monotone: &[(&str, &str, Sign)]) -> Result<Self, ParametricError> {
        let (d, _) = edges(g);
        let mut signs = BTreeMap::new();
        for (x, y, s) in monotone {
            if !d.iter().any(|(a, b)| a == x && b == y) {
                return Err(ParametricError::NotAnEdge(format!("{x} -> {y}")));
            }
            signs.insert(((*x).to_string(), (*y).to_string()), *s);
        }
        let order = g.topological_order()?;
        let mut parents: BTreeMap<String, Vec<String>> =
            order.iter().map(|v| (v.clone(), Vec::new())).collect();
        for (a, b) in d {
            parents.get_mut(&b).expect("a node").push(a);
        }
        for p in parents.values_mut() {
            p.sort();
        }
        Ok(Self {
            g: g.clone(),
            signs,
            order,
            parents,
        })
    }

    /// The graph.
    #[must_use]
    pub fn graph(&self) -> &Admg {
        &self.g
    }

    /// Whether `x` is globally monotone in `z` (Lemma 19.7.1): every
    /// directed path from `z` to `x` uses monotonic edges only, with the
    /// same product of signs, which is returned. `None` when some path has a
    /// non-monotonic edge or the products disagree; `Some(Positive)` when no
    /// path exists.
    ///
    /// # Errors
    ///
    /// [`ParametricError::UnknownVariable`] for an unknown name.
    pub fn globally_monotone(&self, z: &str, x: &str) -> Result<Option<Sign>, ParametricError> {
        for v in [z, x] {
            if !self.parents.contains_key(v) {
                return Err(ParametricError::UnknownVariable(v.to_string()));
            }
        }
        let mut products: BTreeSet<bool> = BTreeSet::new();
        let mut broken = false;
        let children: BTreeMap<&str, Vec<&str>> = self
            .order
            .iter()
            .map(|v| {
                let ch = self
                    .parents
                    .iter()
                    .filter(|(_, ps)| ps.contains(v))
                    .map(|(c, _)| c.as_str())
                    .collect();
                (v.as_str(), ch)
            })
            .collect();
        fn walk<'a>(
            cgma: &Cgma,
            children: &BTreeMap<&'a str, Vec<&'a str>>,
            at: &'a str,
            target: &str,
            positive: bool,
            products: &mut BTreeSet<bool>,
            broken: &mut bool,
        ) {
            if at == target {
                products.insert(positive);
                return;
            }
            for &c in &children[at] {
                match cgma.signs.get(&(at.to_string(), c.to_string())) {
                    Some(s) => walk(
                        cgma,
                        children,
                        c,
                        target,
                        positive == (*s == Sign::Positive),
                        products,
                        broken,
                    ),
                    None => {
                        // A non-monotonic edge on a path that reaches the target.
                        if c == target
                            || cgma
                                .g
                                .descendants_of(&[c])
                                .is_ok_and(|d| d.contains(&target))
                        {
                            *broken = true;
                        }
                    }
                }
            }
        }
        if z != x {
            walk(self, &children, z, x, true, &mut products, &mut broken);
        }
        if broken || products.len() > 1 {
            return Ok(None);
        }
        Ok(Some(if products.contains(&false) {
            Sign::Negative
        } else {
            Sign::Positive
        }))
    }

    /// How two parent assignments of `w` compare for its value: `Less`
    /// when `W_a ≤ W_b` for every unit, by the monotonic parents and their
    /// signs. `None` when a non-monotonic parent differs or the monotonic
    /// parents disagree (Lemma 19.7.7).
    fn compare(
        &self,
        w: &str,
        a: &BTreeMap<String, u8>,
        b: &BTreeMap<String, u8>,
    ) -> Option<Ordering> {
        let mut out = Ordering::Equal;
        for p in &self.parents[w] {
            let (x, y) = (a[p], b[p]);
            if x == y {
                continue;
            }
            let dir = match self.signs.get(&(p.clone(), w.to_string()))? {
                Sign::Positive => x.cmp(&y),
                Sign::Negative => y.cmp(&x),
            };
            if out == Ordering::Equal {
                out = dir;
            } else if out != dir {
                return None;
            }
        }
        Some(out)
    }
}

/// A signed list of ctf-factors.
type Signed = Vec<(f64, Vec<CtfTerm>)>;

/// Algorithm 67 (M-REDUCE): rewrites a ctf-factor, under the CGMA's
/// monotonicity constraints, as a signed sum of ctf-factors in which each
/// variable appears at most once. It applies the impossibility check
/// (Def. 19.7.3), the simplification rule, and then the difference rule
/// (Lemma 19.7.2) in reverse topological order. A factor that is impossible
/// contributes nothing (an empty list).
///
/// ```
/// use std::collections::BTreeMap;
/// use why_data::graph::dseparation::Admg;
/// use why_data::parametric::{Cgma, CtfTerm, Sign, m_reduce};
///
/// // LATE's compliers in the IV graph: P(Y_{x1} = 1, X_{z0} = 0, X_{z1} = 1).
/// let mut g = Admg::new();
/// g.directed("Z", "X");
/// g.directed("X", "Y");
/// g.bidirected("X", "Y");
/// let cgma = Cgma::new(&g, &[("Z", "X", Sign::Positive)]).unwrap();
/// let t = |var: &str, pa: &[(&str, u8)], value: u8| CtfTerm {
///     var: var.into(),
///     parents: pa.iter().map(|(k, v)| ((*k).to_string(), *v)).collect::<BTreeMap<_, _>>(),
///     value,
/// };
/// let factor = [t("Y", &[("X", 1)], 1), t("X", &[("Z", 0)], 0), t("X", &[("Z", 1)], 1)];
/// // The difference rule keeps X = 1, the value Y's subscript uses.
/// let reduced = m_reduce(&cgma, &factor).unwrap();
/// assert_eq!(reduced.len(), 2);
/// assert_eq!(reduced[0].1, vec![t("X", &[("Z", 1)], 1), t("Y", &[("X", 1)], 1)]);
/// assert_eq!(reduced[1].0, -1.0);
/// ```
///
/// # Errors
///
/// [`ParametricError::NotIdentifiable`] when two terms of a variable differ
/// in a non-monotonic parent or are incomparable (Lemma 19.7.7), so the
/// factor is not identifiable (Thm 19.7.8); [`ParametricError::InvalidInput`]
/// for a term without its full parent assignment.
pub fn m_reduce(cgma: &Cgma, factor: &[CtfTerm]) -> Result<Signed, ParametricError> {
    for t in factor {
        let ps = cgma
            .parents
            .get(&t.var)
            .ok_or_else(|| ParametricError::UnknownVariable(t.var.clone()))?;
        if ps.len() != t.parents.len()
            || ps.iter().any(|p| !t.parents.contains_key(p))
            || t.value > 1
        {
            return Err(ParametricError::InvalidInput(format!(
                "{} needs a binary value and the values of all its parents",
                t.var
            )));
        }
    }
    let mut out = Vec::new();
    reduce(cgma, 1.0, factor.to_vec(), cgma.order.len(), &mut out)?;
    Ok(out)
}

/// Simplifies `factor` (impossibility and Rule 1), then applies Rule 2 to
/// the variables before position `upto` of the topological order, from the
/// last backwards.
fn reduce(
    cgma: &Cgma,
    sign: f64,
    mut factor: Vec<CtfTerm>,
    upto: usize,
    out: &mut Signed,
) -> Result<(), ParametricError> {
    factor.sort();
    factor.dedup();
    // Simplify each variable's terms.
    let mut kept: Vec<CtfTerm> = Vec::new();
    for var in &cgma.order {
        let terms: Vec<&CtfTerm> = factor.iter().filter(|t| &t.var == var).collect();
        if terms.is_empty() {
            continue;
        }
        // Every pair must be comparable (Lemma 19.7.7).
        for (i, a) in terms.iter().enumerate() {
            for b in &terms[i + 1..] {
                match cgma.compare(var, &a.parents, &b.parents) {
                    None => {
                        return Err(ParametricError::NotIdentifiable(format!(
                            "two terms of {var} are not comparable under the monotonicity constraints"
                        )));
                    }
                    // Def. 19.7.3: the same unit at ordered settings.
                    Some(Ordering::Equal) if a.value != b.value => return Ok(()),
                    Some(Ordering::Less) if a.value > b.value => return Ok(()),
                    Some(Ordering::Greater) if a.value < b.value => return Ok(()),
                    _ => {}
                }
            }
        }
        // Rule 1: of the zeros keep the highest setting, of the ones the
        // lowest.
        let pick = |value: u8, want: Ordering| {
            terms
                .iter()
                .filter(|t| t.value == value)
                .copied()
                .reduce(|a, b| {
                    if cgma.compare(var, &b.parents, &a.parents) == Some(want) {
                        b
                    } else {
                        a
                    }
                })
        };
        kept.extend(pick(0, Ordering::Greater).cloned());
        kept.extend(pick(1, Ordering::Less).cloned());
    }
    // Rule 2 on the last variable (before `upto`) that still has two terms.
    for pos in (0..upto).rev() {
        let var = &cgma.order[pos];
        let terms: Vec<CtfTerm> = kept.iter().filter(|t| &t.var == var).cloned().collect();
        if terms.len() < 2 {
            continue;
        }
        let (zero, one) = if terms[0].value == 0 {
            (terms[0].clone(), terms[1].clone())
        } else {
            (terms[1].clone(), terms[0].clone())
        };
        let rest: Vec<CtfTerm> = kept.iter().filter(|t| &t.var != var).cloned().collect();
        // Keep the value that a child's subscript uses (Alg. 67, line 16).
        let used = rest.iter().find_map(|t| t.parents.get(var).copied());
        let (first, second) = if used == Some(0) {
            // Eq. 19.169: P(W_t = 0) − P(W_{t′} = 0).
            let mut other = one.clone();
            other.value = 0;
            (zero, other)
        } else {
            // Eq. 19.168: P(W_{t′} = 1) − P(W_t = 1).
            let mut other = zero.clone();
            other.value = 1;
            (one, other)
        };
        let mut a = rest.clone();
        a.push(first);
        let mut b = rest;
        b.push(second);
        reduce(cgma, sign, a, pos, out)?;
        reduce(cgma, -sign, b, pos, out)?;
        return Ok(());
    }
    kept.sort();
    out.push((sign, kept));
    Ok(())
}

/// A counterfactual variable `V_x`: the variable and the intervention,
/// restricted to its ancestors.
type CtfVar = (String, BTreeMap<String, u8>);

impl Cgma {
    fn ancestors(&self, v: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut stack = vec![v.to_string()];
        while let Some(x) = stack.pop() {
            for p in &self.parents[&x] {
                if out.insert(p.clone()) {
                    stack.push(p.clone());
                }
            }
        }
        out
    }

    fn restrict(&self, v: &str, x: &BTreeMap<String, u8>) -> BTreeMap<String, u8> {
        let an = self.ancestors(v);
        x.iter()
            .filter(|(k, _)| an.contains(*k))
            .map(|(k, val)| (k.clone(), *val))
            .collect()
    }

    /// CTF-FACTOR: the ctf-factors whose sum is `P(events)`, each event a
    /// variable, an intervention and a value. The counterfactual parents
    /// of each event are summed over.
    fn factors(
        &self,
        events: &[(String, BTreeMap<String, u8>, u8)],
    ) -> Result<Vec<Vec<CtfTerm>>, ParametricError> {
        let mut fixed: BTreeMap<CtfVar, u8> = BTreeMap::new();
        let mut needed: BTreeSet<CtfVar> = BTreeSet::new();
        for (v, x, val) in events {
            if !self.parents.contains_key(v) {
                return Err(ParametricError::UnknownVariable(v.clone()));
            }
            if let Some(forced) = x.get(v) {
                if forced != val {
                    return Ok(Vec::new());
                }
                continue;
            }
            let key = (v.clone(), self.restrict(v, x));
            if fixed
                .insert(key.clone(), *val)
                .is_some_and(|old| old != *val)
            {
                return Ok(Vec::new());
            }
            needed.insert(key);
        }
        // Close under counterfactual parents.
        let mut stack: Vec<CtfVar> = needed.iter().cloned().collect();
        while let Some((v, x)) = stack.pop() {
            for p in &self.parents[&v] {
                if x.contains_key(p) {
                    continue;
                }
                let key = (p.clone(), self.restrict(p, &x));
                if needed.insert(key.clone()) {
                    stack.push(key);
                }
            }
        }
        let free: Vec<CtfVar> = needed
            .iter()
            .filter(|k| !fixed.contains_key(*k))
            .cloned()
            .collect();
        if free.len() > 16 {
            return Err(ParametricError::InvalidInput(
                "too many counterfactual parents to sum".into(),
            ));
        }
        let mut out = Vec::new();
        for mask in 0..(1u32 << free.len()) {
            let mut value = fixed.clone();
            for (i, k) in free.iter().enumerate() {
                value.insert(k.clone(), u8::from(mask >> i & 1 == 1));
            }
            let mut terms: BTreeMap<(String, BTreeMap<String, u8>), u8> = BTreeMap::new();
            let mut ok = true;
            for (v, x) in &needed {
                let parents: BTreeMap<String, u8> = self.parents[v]
                    .iter()
                    .map(|p| {
                        let val = x
                            .get(p)
                            .copied()
                            .unwrap_or_else(|| value[&(p.clone(), self.restrict(p, x))]);
                        (p.clone(), val)
                    })
                    .collect();
                let w = value[&(v.clone(), x.clone())];
                if terms
                    .insert((v.clone(), parents), w)
                    .is_some_and(|old| old != w)
                {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push(
                    terms
                        .into_iter()
                        .map(|((var, parents), value)| CtfTerm {
                            var,
                            parents,
                            value,
                        })
                        .collect(),
                );
            }
        }
        Ok(out)
    }

    /// Identifies a reduced ctf-factor with C-INFER and evaluates it on `p`.
    fn evaluate(&self, factor: &[CtfTerm], p: &Dataset) -> Result<f64, ParametricError> {
        let name = |v: &str, level: u8| {
            let base = v.to_lowercase();
            if level == 1 { base } else { format!("{base}'") }
        };
        let events: Vec<String> = factor
            .iter()
            .map(|t| {
                let head = name(&t.var, t.value);
                if t.parents.is_empty() {
                    head
                } else {
                    let sub: Vec<String> = t.parents.iter().map(|(k, v)| name(k, *v)).collect();
                    format!("{head}_{{{}}}", sub.join(", "))
                }
            })
            .collect();
        let text = format!("P({})", events.join(", "));
        let q = CtfQuery::parse(&self.g, &text)
            .map_err(|e| ParametricError::InvalidInput(format!("{text}: {e:?}")))?;
        let d = ctf_infer(&self.g, &q, &[Input::observational()])?;
        Ok(match d.estimand {
            CtfEstimand::Zero => 0.0,
            CtfEstimand::One => 1.0,
            CtfEstimand::Formula(f) => {
                let values: Vec<(String, String)> = self
                    .order
                    .iter()
                    .flat_map(|v| {
                        [
                            (v.clone(), "1".to_string()),
                            (format!("{v}'"), "0".to_string()),
                        ]
                    })
                    .collect();
                let at: Vec<(&str, &str)> = values
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let dists = EmpiricalDistributions::new().with(Domain::Target, p);
                Estimand::from_formula(&f, &Domain::Target).evaluate(&dists, &at)?
            }
        })
    }

    fn joint(
        &self,
        events: &[(String, BTreeMap<String, u8>, u8)],
        p: &Dataset,
    ) -> Result<f64, ParametricError> {
        let mut total = 0.0;
        for factor in self.factors(events)? {
            for (sign, reduced) in m_reduce(self, &factor)? {
                total += sign * self.evaluate(&reduced, p)?;
            }
        }
        Ok(total)
    }
}

/// A counterfactual event `Y_x = y` for [`m_id`]: the variable, the
/// intervention `x` as `(variable, level)` pairs (empty for a factual
/// event) and the level `y`.
pub type Event<'a> = (&'a str, &'a [(&'a str, u8)], u8);

/// M-ID: `P(events | given)` under the CGMA, from the observational
/// distribution `p` (exact table or sample; levels `"0"`/`"1"`).
///
/// Each event is `(variable, intervention, value)`; `given` holds natural
/// (factual) or counterfactual events too. Each joint is expanded into
/// ctf-factors over the counterfactual parents (CTF-FACTOR), each factor is
/// reduced by Algorithm 67, and each reduced factor is identified by C-INFER
/// (chapter 5) and evaluated on `p`.
///
/// ```
/// use why_data::fairness::Dataset;
/// use why_data::graph::dseparation::Admg;
/// use why_data::parametric::{Cgma, Sign, m_id};
///
/// // The IV graph with P(X = 1 | z) = 0.2 + 0.5 z for compliers; data:
/// // P(Z, X, Y) for a model where X = 1 iff z-compliant or always-taker.
/// let mut g = Admg::new();
/// g.directed("Z", "X");
/// g.directed("X", "Y");
/// g.bidirected("X", "Y");
/// let cgma = Cgma::new(&g, &[("Z", "X", Sign::Positive)]).unwrap();
/// let rows = [["0", "0", "0"], ["0", "1", "1"], ["1", "1", "1"], ["1", "0", "0"]];
/// let p = Dataset::from_records(["Z", "X", "Y"], rows)
///     .unwrap()
///     .with_weights(vec![0.4, 0.1, 0.35, 0.15])
///     .unwrap();
/// // The share of compliers, P(X_{z0} = 0, X_{z1} = 1) = P(x1 | z1) − P(x1 | z0).
/// let c = m_id(&cgma, &[("X", &[("Z", 0)], 0), ("X", &[("Z", 1)], 1)], &[], &p).unwrap();
/// assert!((c - (0.7 - 0.2)).abs() < 1e-9);
/// ```
///
/// # Errors
///
/// [`ParametricError::NotIdentifiable`] when Algorithm 67 fails or a
/// reduced factor is not identifiable, [`ParametricError::InvalidInput`]
/// when `given` has probability zero, and evaluation errors.
pub fn m_id(
    cgma: &Cgma,
    events: &[Event<'_>],
    given: &[Event<'_>],
    p: &Dataset,
) -> Result<f64, ParametricError> {
    let own = |e: &[Event<'_>]| -> Vec<(String, BTreeMap<String, u8>, u8)> {
        e.iter()
            .map(|(v, x, val)| {
                (
                    (*v).to_string(),
                    x.iter().map(|(k, l)| ((*k).to_string(), *l)).collect(),
                    *val,
                )
            })
            .collect()
    };
    let mut all = own(events);
    all.extend(own(given));
    let num = cgma.joint(&all, p)?;
    if given.is_empty() {
        return Ok(num);
    }
    let den = cgma.joint(&own(given), p)?;
    if den <= 0.0 {
        return Err(ParametricError::InvalidInput(
            "the conditioning event has probability zero".into(),
        ));
    }
    Ok(num / den)
}
