//! Causal explanations of events (the Causal AI book, chapter 6, section 6.2):
//! natural total effects, generalized direct effects and counterfactual
//! (L3) Shapley values, with SHAP (L1) and do-Shapley (L2) for comparison.
//!
//! A *why query* `Why(y | e; X)` (Def. 6.2.1) asks why the outcome `Y = y`
//! happened, given the observed evidence `E = e`, in terms of explanatory
//! variables `X ⊆ E \ {Y}`. Everything in the section reduces to one
//! counterfactual mean per set `S` of variables:
//!
//! ```text
//! m(S) = Σ_{u, u'} Y_{S(u')}(u) P(u | e) P(u') = Σ_s P(S = s) · E[Y_{S=s} | e]
//! ```
//!
//! `S` is set to the values of a fresh unit `u'` drawn from the population, so
//! `S(u') ~ P(S)` jointly, while `Y` keeps responding to a unit `u` that is
//! consistent with the evidence. Then:
//!
//! * `NTE(S, Y | e) = m(∅) − m(S)`, where `m(∅) = E[Y | e]` (Defs. 6.2.2,
//!   6.2.3);
//! * `GDE_S(X, Y | e) = NTE(S ∪ {X}) − NTE(S) = m(S) − m(S ∪ {X})`
//!   (Def. 6.2.5);
//! * `φ^L3_X` is the Shapley value of `X` in the game `v(S) = NTE(S)` over the
//!   explanatory set (Def. 6.2.6). Its values sum to `NTE(X, Y | e)`
//!   (Cor. 6.2.2).
//!
//! # A fully specified model is required
//!
//! `E[Y_{S=s} | e]` conditions a counterfactual on factual evidence that
//! usually includes `Y` itself. Such quantities, e.g. `P(y_x | x', y')`, are
//! not identifiable from observational or experimental data, even without
//! confounding (Thm 6.1.13 (iv)). The counterfactual C-INFER in
//! [`graph::cinfer::counterfactual`](crate::graph::cinfer::counterfactual)
//! rejects them as inconsistent ctf-factors. So this module works on a
//! [`Model`], as all the book's examples do, and computes every quantity
//! exactly by enumerating the exogenous variables. The cost is
//! `O(2^|X| · |D_X| · |D_U|)`, which fits textbook-sized models.
//!
//! ```
//! use why_data::explain::Explainer;
//! use why_data::scm::{expr, Dist, Model, Value};
//!
//! // Ex. 6.36-6.39, rain and sprinkler: Y := X1 ∨ X2.
//! let m = Model::new(
//!     [
//!         ("x1".to_string(), expr::var("u1")),
//!         ("x2".to_string(), expr::var("u2")),
//!         ("y".to_string(), expr::or([expr::var("x1"), expr::var("x2")])),
//!     ],
//!     [("u1".to_string(), Dist::Bernoulli(0.5)), ("u2".to_string(), Dist::Bernoulli(0.5))],
//! )
//! .unwrap();
//! let t = Value::Bool(true);
//! let mut why = Explainer::new(&m, "y", &[("x1", t), ("x2", t), ("y", t)], &["x1", "x2"]).unwrap();
//! assert_eq!(why.gde("x1", &[]).unwrap(), 0.0); // not a but-for cause
//! assert!((why.gde("x1", &["x2"]).unwrap() - 0.25).abs() < 1e-12);
//! let phi = why.l3_shapley().unwrap();
//! assert!((phi.values[0] - 0.125).abs() < 1e-12 && (phi.values[1] - 0.125).abs() < 1e-12);
//! ```

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use crate::scm::{Model, ScmError, Value};

/// Everything that can go wrong while explaining an event.
#[derive(Debug, Clone, PartialEq)]
pub enum ExplainError {
    /// An error from the underlying model.
    Scm(ScmError),
    /// The name is not an endogenous variable of the model.
    NotEndogenous(String),
    /// An explanatory variable has no observed value in the evidence.
    NotObserved(String),
    /// The outcome was listed among the explanatory variables or in a set `S`.
    OutcomeInSet(String),
    /// A variable was listed twice.
    Duplicate(String),
    /// `GDE_S(X, ·)` was asked with `X ∈ S`.
    AlreadyInSet(String),
    /// Exact Shapley values over more than 20 variables are not attempted.
    TooMany(usize),
}

impl From<ScmError> for ExplainError {
    fn from(e: ScmError) -> Self {
        Self::Scm(e)
    }
}

impl fmt::Display for ExplainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scm(e) => write!(f, "{e}"),
            Self::NotEndogenous(v) => write!(f, "`{v}` is not an endogenous variable"),
            Self::NotObserved(v) => write!(f, "explanatory variable `{v}` is not in the evidence"),
            Self::OutcomeInSet(v) => write!(f, "the outcome `{v}` cannot explain itself"),
            Self::Duplicate(v) => write!(f, "`{v}` is listed twice"),
            Self::AlreadyInSet(v) => write!(f, "`{v}` is already in the adjustment set"),
            Self::TooMany(n) => write!(f, "{n} explanatory variables; at most 20 are supported"),
        }
    }
}

impl std::error::Error for ExplainError {}

/// A variable attribution: one value per explanatory variable, in the order
/// given to [`Explainer::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct Attribution {
    /// The explanatory variables.
    pub variables: Vec<String>,
    /// Their attributions.
    pub values: Vec<f64>,
    /// What the attributions add up to: `v(X) − v(∅)` for the game played.
    /// For L3 that is `NTE(X, Y | e)`.
    pub total: f64,
}

impl Attribution {
    /// The attribution of one variable.
    pub fn get(&self, var: &str) -> Option<f64> {
        self.variables.iter().position(|v| v == var).map(|i| self.values[i])
    }
}

impl fmt::Display for Attribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (v, x) in self.variables.iter().zip(&self.values) {
            write!(f, "{v}: {x:+.4}  ")?;
        }
        write!(f, "(total {:+.4})", self.total)
    }
}

/// Answers a why query `Why(y | e; X)` on a fully specified model.
///
/// Results of `m(S)` are cached, so asking for NTEs, GDEs and Shapley values
/// of the same query reuses the work.
#[derive(Debug, Clone)]
pub struct Explainer {
    model: Model,
    outcome: String,
    outcome_index: usize,
    evidence: Vec<(String, Value)>,
    explanatory: Vec<String>,
    cache: HashMap<Vec<String>, f64>,
}

impl Explainer {
    /// Sets up `Why(outcome | evidence; explanatory)`.
    ///
    /// The evidence may include the outcome's observed value (the usual
    /// case) but need not. Every explanatory variable must be observed in the
    /// evidence, and the evidence must have positive probability.
    pub fn new(
        model: &Model,
        outcome: &str,
        evidence: &[(&str, Value)],
        explanatory: &[&str],
    ) -> Result<Self, ExplainError> {
        let base = model.world(model.observational()).world();
        let endogenous = |v: &str| {
            if base.variables().iter().any(|n| n == v) {
                Ok(())
            } else {
                Err(ExplainError::NotEndogenous(v.to_string()))
            }
        };
        endogenous(outcome)?;
        let mut seen = BTreeSet::new();
        for (v, _) in evidence {
            endogenous(v)?;
            if !seen.insert(*v) {
                return Err(ExplainError::Duplicate(v.to_string()));
            }
        }
        let mut seen = BTreeSet::new();
        for v in explanatory {
            endogenous(v)?;
            if *v == outcome {
                return Err(ExplainError::OutcomeInSet(v.to_string()));
            }
            if !evidence.iter().any(|(e, _)| e == v) {
                return Err(ExplainError::NotObserved(v.to_string()));
            }
            if !seen.insert(*v) {
                return Err(ExplainError::Duplicate(v.to_string()));
            }
        }
        let mut ex = Self {
            model: model.clone(),
            outcome: outcome.to_string(),
            outcome_index: base.variables().iter().position(|n| n == outcome).unwrap(),
            evidence: evidence.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            explanatory: explanatory.iter().map(|v| v.to_string()).collect(),
            cache: HashMap::new(),
        };
        // Fails with ZeroProbability now rather than on the first query.
        ex.m(&[])?;
        Ok(ex)
    }

    /// The explanatory variables.
    pub fn explanatory(&self) -> &[String] {
        &self.explanatory
    }

    /// `E[col | given]` in the observational world, where `col` may be a
    /// fully qualified variable of another world.
    fn conditional_mean(&self, col: &str, given: &[(String, Value)]) -> Result<f64, ExplainError> {
        let obs = self.model.world(self.model.observational());
        let mut cols: Vec<&str> = vec![col];
        cols.extend(given.iter().map(|(k, _)| k.as_str()));
        let table = obs.probability_table(Some(&cols))?;
        let (mut num, mut den) = (0.0, 0.0);
        for (values, p) in &table.rows {
            let matches = values[1..]
                .iter()
                .zip(given)
                .all(|(a, (_, b))| a.as_int() == b.as_int());
            if matches {
                num += p * values[0].as_int() as f64;
                den += p;
            }
        }
        if den <= 0.0 {
            return Err(ScmError::ZeroProbability.into());
        }
        Ok(num / den)
    }

    /// The name `Y` has in the world `do(assignment)`.
    fn outcome_in(&mut self, assignment: &[(&str, Value)]) -> Result<String, ExplainError> {
        if assignment.is_empty() {
            return Ok(self.outcome.clone());
        }
        let w = self.model.intervene(assignment)?;
        Ok(self.model.world(w).world().variables()[self.outcome_index].clone())
    }

    fn validate_set(&self, set: &[&str]) -> Result<Vec<String>, ExplainError> {
        let base = self.model.world(self.model.observational()).world();
        let mut out = BTreeSet::new();
        for v in set {
            if !base.variables().iter().any(|n| n == v) {
                return Err(ExplainError::NotEndogenous(v.to_string()));
            }
            if *v == self.outcome {
                return Err(ExplainError::OutcomeInSet(v.to_string()));
            }
            if !out.insert(v.to_string()) {
                return Err(ExplainError::Duplicate(v.to_string()));
            }
        }
        Ok(out.into_iter().collect())
    }

    /// `m(S) = Σ_s P(S = s) E[Y_{S=s} | e]`.
    fn m(&mut self, set: &[&str]) -> Result<f64, ExplainError> {
        let key = self.validate_set(set)?;
        if let Some(v) = self.cache.get(&key) {
            return Ok(*v);
        }
        let evidence = self.evidence.clone();
        let value = if key.is_empty() {
            self.conditional_mean(&self.outcome, &evidence)?
        } else {
            let names: Vec<&str> = key.iter().map(String::as_str).collect();
            let joint = self
                .model
                .world(self.model.observational())
                .probability_table(Some(&names))?;
            let mut total = 0.0;
            for (values, p) in &joint.rows {
                if *p == 0.0 {
                    continue;
                }
                let assignment: Vec<(&str, Value)> =
                    names.iter().copied().zip(values.iter().copied()).collect();
                let y = self.outcome_in(&assignment)?;
                total += p * self.conditional_mean(&y, &evidence)?;
            }
            total
        };
        self.cache.insert(key, value);
        Ok(value)
    }

    /// `E[Y | e]`, the explanandum's factual side.
    pub fn factual(&mut self) -> Result<f64, ExplainError> {
        self.m(&[])
    }

    /// `NTE(S, Y | e) = Σ [Y(u) − Y_{S(u')}(u)] P(u | e) P(u')`
    /// (Def. 6.2.3). `S` may be any set of endogenous variables other than
    /// `Y`; with `S = X` it is the saturated NTE of Def. 6.2.2.
    pub fn nte(&mut self, set: &[&str]) -> Result<f64, ExplainError> {
        Ok(self.m(&[])? - self.m(set)?)
    }

    /// `GDE_S(X, Y | e) = NTE(S ∪ {X}) − NTE(S)` (Def. 6.2.5): what letting
    /// `X` vary naturally adds once `S` already does.
    pub fn gde(&mut self, x: &str, given: &[&str]) -> Result<f64, ExplainError> {
        if given.contains(&x) {
            return Err(ExplainError::AlreadyInSet(x.to_string()));
        }
        let mut with: Vec<&str> = given.to_vec();
        with.push(x);
        Ok(self.m(given)? - self.m(&with)?)
    }

    /// Counterfactual Shapley values `φ^L3` (Def. 6.2.6): each explanatory
    /// variable's GDE averaged over all orderings. They are causally
    /// admissible, have explanatory power and respect normality (Thm 6.2.6),
    /// and they sum to `NTE(X, Y | e)`.
    pub fn l3_shapley(&mut self) -> Result<Attribution, ExplainError> {
        let base = self.m(&[])?;
        self.shapley(|ex, s| Ok(base - ex.m(s)?))
    }

    /// SHAP as the book defines it in Ex. 6.40/6.41 (layer 1): the Shapley
    /// value of the game `v(S) = E[Y | x_S]`, with `x_S` the observed values.
    /// Not causally admissible.
    pub fn shap_l1(&mut self) -> Result<Attribution, ExplainError> {
        self.shapley(|ex, s| {
            let given: Vec<(String, Value)> =
                s.iter().map(|v| (v.to_string(), ex.observed(v))).collect();
            let y = ex.outcome.clone();
            ex.conditional_mean(&y, &given)
        })
    }

    /// do-Shapley (layer 2, Janzing et al. 2020; Heskes et al. 2020): the
    /// Shapley value of `v(S) = E[Y | do(x_S)]`. Not causally admissible
    /// either (Ex. 6.41).
    pub fn do_shapley_l2(&mut self) -> Result<Attribution, ExplainError> {
        self.shapley(|ex, s| {
            let assignment: Vec<(&str, Value)> = s.iter().map(|v| (*v, ex.observed(v))).collect();
            let y = ex.outcome_in(&assignment)?;
            ex.conditional_mean(&y, &[])
        })
    }

    fn observed(&self, var: &str) -> Value {
        self.evidence.iter().find(|(k, _)| k == var).unwrap().1
    }

    /// Exact Shapley values of the game `game` over the explanatory set.
    fn shapley(
        &mut self,
        mut game: impl FnMut(&mut Self, &[&str]) -> Result<f64, ExplainError>,
    ) -> Result<Attribution, ExplainError> {
        let n = self.explanatory.len();
        if n > 20 {
            return Err(ExplainError::TooMany(n));
        }
        let vars = self.explanatory.clone();
        let mut value = vec![0.0; 1 << n];
        for (mask, slot) in value.iter_mut().enumerate() {
            let set: Vec<&str> = (0..n)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| vars[i].as_str())
                .collect();
            *slot = game(self, &set)?;
        }
        // weight[k] = k! (n − k − 1)! / n!
        let mut fact = vec![1.0f64; n + 1];
        for i in 1..=n {
            fact[i] = fact[i - 1] * i as f64;
        }
        let weight: Vec<f64> = (0..n.max(1))
            .map(|k| fact[k] * fact[n.saturating_sub(k + 1)] / fact[n])
            .collect();
        let values = (0..n)
            .map(|i| {
                (0..1usize << n)
                    .filter(|mask| mask >> i & 1 == 0)
                    .map(|mask| {
                        let k = mask.count_ones() as usize;
                        weight[k] * (value[mask | 1 << i] - value[mask])
                    })
                    .sum()
            })
            .collect();
        Ok(Attribution {
            variables: vars,
            values,
            total: value[(1 << n) - 1] - value[0],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scm::{Dist, expr};
    use rssn::symbolic::core::Expr;

    const EPS: f64 = 1e-12;
    const T: Value = Value::Bool(true);
    const F: Value = Value::Bool(false);

    fn model(f: Vec<(&str, Expr)>, u: Vec<(&str, Dist)>) -> Model {
        Model::new(
            f.into_iter().map(|(k, e)| (k.to_string(), e)),
            u.into_iter().map(|(k, d)| (k.to_string(), d)),
        )
        .unwrap()
    }

    fn rain_sprinkler() -> Model {
        model(
            vec![
                ("x1", expr::var("u1")),
                ("x2", expr::var("u2")),
                ("y", expr::or([expr::var("x1"), expr::var("x2")])),
            ],
            vec![("u1", Dist::Bernoulli(0.5)), ("u2", Dist::Bernoulli(0.5))],
        )
    }

    /// Ex. 6.38 and 6.39: every GDE and both L3 values.
    #[test]
    fn rain_and_sprinkler() {
        let mut ex =
            Explainer::new(&rain_sprinkler(), "y", &[("x1", T), ("x2", T), ("y", T)], &["x1", "x2"])
                .unwrap();
        assert_eq!(ex.factual().unwrap(), 1.0);
        assert!((ex.m(&["x1"]).unwrap() - 1.0).abs() < EPS);
        assert!((ex.m(&["x1", "x2"]).unwrap() - 0.75).abs() < EPS);
        assert!(ex.gde("x1", &[]).unwrap().abs() < EPS);
        assert!(ex.gde("x2", &[]).unwrap().abs() < EPS);
        assert!((ex.gde("x1", &["x2"]).unwrap() - 0.25).abs() < EPS);
        assert!((ex.gde("x2", &["x1"]).unwrap() - 0.25).abs() < EPS);
        let phi = ex.l3_shapley().unwrap();
        assert!((phi.get("x1").unwrap() - 0.125).abs() < EPS);
        assert!((phi.get("x2").unwrap() - 0.125).abs() < EPS);
        // Cor. 6.2.2: they decompose the saturated NTE.
        assert!((phi.total - ex.nte(&["x1", "x2"]).unwrap()).abs() < EPS);
        assert!((phi.total - 0.25).abs() < EPS);
    }

    /// Ex. 6.37, watering plants: neither single variables nor the full set
    /// change the outcome, but {X1, X2} does.
    #[test]
    fn watering_plants() {
        let m = model(
            vec![
                ("x1", expr::var("u")),
                ("x2", expr::var("u")),
                ("x3", expr::not(expr::var("u"))),
                ("y", expr::or([expr::var("x1"), expr::var("x2"), expr::var("x3")])),
            ],
            vec![("u", Dist::Bernoulli(0.5))],
        );
        let ev = [("x1", T), ("x2", T), ("x3", F), ("y", T)];
        let mut ex = Explainer::new(&m, "y", &ev, &["x1", "x2", "x3"]).unwrap();
        for v in ["x1", "x2", "x3"] {
            assert!(ex.nte(&[v]).unwrap().abs() < EPS);
        }
        assert!(ex.nte(&["x1", "x2", "x3"]).unwrap().abs() < EPS);
        assert!((ex.nte(&["x1", "x2"]).unwrap() - 0.5).abs() < EPS);
    }

    /// Ex. 6.40, shark attacks: SHAP credits shark attacks with ice-cream
    /// profits, L3 does not.
    #[test]
    fn shark_attacks() {
        let m = model(
            vec![("x1", expr::var("u")), ("x2", expr::var("u")), ("y", expr::var("x2"))],
            vec![("u", Dist::Bernoulli(0.5))],
        );
        let mut ex = Explainer::new(&m, "y", &[("x1", T), ("x2", T), ("y", T)], &["x1", "x2"]).unwrap();
        assert!((ex.shap_l1().unwrap().get("x1").unwrap() - 0.25).abs() < EPS);
        let l3 = ex.l3_shapley().unwrap();
        assert!(l3.get("x1").unwrap().abs() < EPS);
        assert!((l3.get("x2").unwrap() - 0.5).abs() < EPS);
    }

    /// Ex. 6.41, flu shot: SHAP and do-Shapley both say 0.81, L3 says 0.
    #[test]
    fn flu_shot() {
        // Response types 0..4: (Y0, Y1) = (0,0), (0,1), (1,0), (1,1).
        let t = |k: i64| expr::eq(expr::var("ut"), expr::int(k));
        let m = model(
            vec![
                ("x", expr::var("ux")),
                (
                    "y",
                    expr::or([
                        expr::and([expr::var("x"), expr::or([t(1), t(3)])]),
                        expr::and([expr::not(expr::var("x")), expr::or([t(2), t(3)])]),
                    ]),
                ),
            ],
            vec![
                ("ux", Dist::Bernoulli(0.1)),
                ("ut", Dist::Categorical(vec![0.09, 0.90, 0.0, 0.01])),
            ],
        );
        let mut ex = Explainer::new(&m, "y", &[("x", T), ("y", F)], &["x"]).unwrap();
        assert!((ex.shap_l1().unwrap().values[0] - 0.81).abs() < 1e-9);
        assert!((ex.do_shapley_l2().unwrap().values[0] - 0.81).abs() < 1e-9);
        assert!(ex.l3_shapley().unwrap().values[0].abs() < EPS);
    }

    /// Ex. 6.43, normality: the same fire is explained more by dryness in
    /// the rainforest, where dry ground is rare.
    #[test]
    fn lightning_normality() {
        let fire = |p_dry: f64| {
            let m = model(
                vec![
                    ("x1", expr::var("u1")),
                    ("x2", expr::var("u2")),
                    ("y", expr::and([expr::var("x1"), expr::var("x2")])),
                ],
                vec![("u1", Dist::Bernoulli(0.01)), ("u2", Dist::Bernoulli(p_dry))],
            );
            Explainer::new(&m, "y", &[("x1", T), ("x2", T), ("y", T)], &["x1", "x2"]).unwrap()
        };
        let (mut rain, mut desert) = (fire(0.05), fire(0.95));
        assert!((rain.gde("x2", &[]).unwrap() - 0.95).abs() < 1e-9);
        assert!((desert.gde("x2", &[]).unwrap() - 0.05).abs() < 1e-9);
        assert!((rain.gde("x1", &[]).unwrap() - 0.99).abs() < 1e-9);
        assert!((desert.gde("x1", &[]).unwrap() - 0.99).abs() < 1e-9);
        let (r, d) = (rain.l3_shapley().unwrap(), desert.l3_shapley().unwrap());
        assert!(r.get("x2").unwrap() > d.get("x2").unwrap());
    }

    /// Random binary SCMs, X1 -> X2 -> X3 -> Y plus X1 -> Y and a variable N
    /// that never reaches Y. Checks Cor. 6.2.2 (efficiency), Thm 6.2.1 for
    /// every ordering, and admissibility for N.
    #[test]
    fn random_models_decompose_and_are_admissible() {
        let mut state = 0x1234_5678_u64;
        let mut unit = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let v = expr::var;
        for _ in 0..20 {
            // Random boolean mechanisms from a small menu, with independent noise.
            fn pick(k: f64, a: Expr, b: Expr) -> Expr {
                match (k * 4.0) as u32 {
                    0 => expr::and([a, b]),
                    1 => expr::or([a, b]),
                    2 => expr::xor(a, b),
                    _ => expr::and([a, expr::not(b)]),
                }
            }
            let k: Vec<f64> = (0..5).map(|_| unit()).collect();
            let f = vec![
                ("x1", v("u1")),
                ("n", pick(k[0], v("x1"), v("un"))),
                ("x2", pick(k[1], v("x1"), v("u2"))),
                ("x3", pick(k[2], v("x2"), v("u3"))),
                ("y", pick(k[3], pick(k[4], v("x3"), v("x1")), v("uy"))),
            ];
            let u = ["u1", "un", "u2", "u3", "uy"]
                .into_iter()
                .map(|n| (n, Dist::Bernoulli(0.2 + 0.6 * unit())))
                .collect();
            let m = model(f, u);
            // Use the most likely factual world as evidence.
            let table = m.world(m.observational()).probability_table(None).unwrap();
            let (row, _) = table
                .rows
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let names = m.world(m.observational()).world().variables().to_vec();
            let ev: Vec<(&str, Value)> = names.iter().map(String::as_str).zip(row.iter().copied()).collect();
            let xs = ["x1", "n", "x2", "x3"];
            let mut ex = Explainer::new(&m, "y", &ev, &xs).unwrap();
            let phi = ex.l3_shapley().unwrap();
            let nte = ex.nte(&xs).unwrap();
            assert!((phi.values.iter().sum::<f64>() - nte).abs() < 1e-9);
            assert!(phi.get("n").unwrap().abs() < 1e-9, "{phi}");
            // Thm 6.2.1 along one ordering.
            let order = ["x3", "n", "x1", "x2"];
            let mut sum = 0.0;
            for i in 0..order.len() {
                sum += ex.gde(order[i], &order[..i]).unwrap();
            }
            assert!((sum - nte).abs() < 1e-9);
        }
    }

    #[test]
    fn query_errors() {
        let m = rain_sprinkler();
        let ev = [("x1", T), ("x2", T), ("y", T)];
        assert_eq!(
            Explainer::new(&m, "y", &ev, &["y"]).unwrap_err(),
            ExplainError::OutcomeInSet("y".into())
        );
        assert_eq!(
            Explainer::new(&m, "y", &[("x1", T)], &["x2"]).unwrap_err(),
            ExplainError::NotObserved("x2".into())
        );
        assert_eq!(
            Explainer::new(&m, "q", &ev, &[]).unwrap_err(),
            ExplainError::NotEndogenous("q".into())
        );
        // Impossible evidence: Y = 0 although X1 = 1.
        assert!(matches!(
            Explainer::new(&m, "y", &[("x1", T), ("y", F)], &["x1"]),
            Err(ExplainError::Scm(ScmError::ZeroProbability))
        ));
        let mut ex = Explainer::new(&m, "y", &ev, &["x1", "x2"]).unwrap();
        assert_eq!(ex.gde("x1", &["x1"]).unwrap_err(), ExplainError::AlreadyInSet("x1".into()));
    }
}
