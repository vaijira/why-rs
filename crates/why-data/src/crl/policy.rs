//! Policies: sequences of decision rules `π_i(x_i | s_i)`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Index;

use super::PolicySpace;

/// The values of a stage's input states `S_i`, as a decision rule sees them.
///
/// Values are strings, matching [`super::Dataset`]. In an SCM a boolean
/// renders as `0`/`1`.
#[derive(Debug, Clone, Copy)]
pub struct State<'a> {
    pairs: &'a [(&'a str, &'a str)],
}

impl<'a> State<'a> {
    /// Wraps `(variable, value)` pairs in `S_i` order.
    #[must_use]
    pub fn new(pairs: &'a [(&'a str, &'a str)]) -> Self {
        Self { pairs }
    }

    /// The value of a state variable.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&'a str> {
        self.pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| *v)
    }

    /// The values, in `S_i` order.
    pub fn values(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.pairs.iter().map(|(_, v)| *v)
    }

    /// The `(variable, value)` pairs, in `S_i` order.
    #[must_use]
    pub fn pairs(&self) -> &'a [(&'a str, &'a str)] {
        self.pairs
    }

    /// Whether the rule sees no state, as in a multi-armed bandit.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

impl Index<&str> for State<'_> {
    type Output = str;

    /// # Panics
    ///
    /// Panics if `name` is not one of the input states.
    fn index(&self, name: &str) -> &str {
        self.get(name)
            .unwrap_or_else(|| panic!("`{name}` is not an input state of this decision rule"))
    }
}

impl fmt::Display for State<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (k, v)) in self.pairs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{k}={v}")?;
        }
        Ok(())
    }
}

/// A policy `π = (π_1, …, π_H)`: one decision rule per action.
pub trait Policy {
    /// `π_i(value | state)` for the action variable `action`.
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64;
}

impl<P: Policy + ?Sized> Policy for &P {
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64 {
        (**self).probability(action, state, value)
    }
}

/// A deterministic policy from a closure `(action, state) ↦ value`.
///
/// ```
/// use why_data::crl::{Deterministic, Policy, State};
///
/// // Ex. 8.24: X1 ← S1, X2 ← ¬S1.
/// let pi = Deterministic(|x: &str, s: &State<'_>| match x {
///     "X1" => s["S1"].to_string(),
///     _ => if &s["S1"] == "1" { "0".into() } else { "1".into() },
/// });
/// let s = [("S1", "1")];
/// assert_eq!(pi.probability("X2", &State::new(&s), "0"), 1.0);
/// ```
pub struct Deterministic<F>(pub F);

impl<F> Policy for Deterministic<F>
where
    F: Fn(&str, &State<'_>) -> String,
{
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64 {
        f64::from(u8::from((self.0)(action, state) == value))
    }
}

impl<F> fmt::Debug for Deterministic<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Deterministic(..)")
    }
}

/// A stochastic policy from a closure `(action, state, value) ↦ probability`.
pub struct Stochastic<F>(pub F);

impl<F> Policy for Stochastic<F>
where
    F: Fn(&str, &State<'_>, &str) -> f64,
{
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64 {
        (self.0)(action, state, value)
    }
}

impl<F> fmt::Debug for Stochastic<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Stochastic(..)")
    }
}

/// `π_UNIF`: every action uniform over its domain, ignoring the state
/// (Eq. 8.82).
#[derive(Debug, Clone, Default)]
pub struct Uniform {
    domains: HashMap<String, Vec<String>>,
}

impl Uniform {
    /// A uniform policy over the given `(action, domain)` pairs.
    pub fn new<I, A, D, V>(domains: I) -> Self
    where
        I: IntoIterator<Item = (A, D)>,
        A: Into<String>,
        D: IntoIterator<Item = V>,
        V: Into<String>,
    {
        Self {
            domains: domains
                .into_iter()
                .map(|(a, d)| (a.into(), d.into_iter().map(Into::into).collect()))
                .collect(),
        }
    }

    /// The domain of an action.
    #[must_use]
    pub fn domain(&self, action: &str) -> Option<&[String]> {
        self.domains.get(action).map(Vec::as_slice)
    }
}

impl Policy for Uniform {
    fn probability(&self, action: &str, _: &State<'_>, value: &str) -> f64 {
        match self.domains.get(action) {
            Some(d) if d.iter().any(|v| v == value) => 1.0 / d.len() as f64,
            _ => 0.0,
        }
    }
}

/// One deterministic decision rule, as a lookup table from `S_i` values to an
/// action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabularRule {
    /// The action variable `X_i`.
    pub action: String,
    /// The input states `S_i`.
    pub states: Vec<String>,
    /// The domain of `X_i`. States missing from the table fall back to the
    /// uniform distribution over it.
    pub domain: Vec<String>,
    /// `S_i` values, in [`TabularRule::states`] order, to the chosen action.
    pub table: BTreeMap<Vec<String>, String>,
}

impl TabularRule {
    /// The action chosen at a state, if the table covers it.
    #[must_use]
    pub fn action_for(&self, state: &[&str]) -> Option<&str> {
        let key: Vec<String> = state.iter().map(|s| (*s).to_string()).collect();
        self.table.get(&key).map(String::as_str)
    }
}

/// A deterministic policy given as one lookup table per stage, as returned by
/// [`super::optimize`].
///
/// A state the table does not cover gets the uniform distribution over the
/// action's domain. That is what an agent committed to this policy does in
/// histories its data never showed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabularPolicy {
    rules: Vec<TabularRule>,
}

impl TabularPolicy {
    /// An empty table for every stage of `space`, over the given domains in
    /// stage order.
    ///
    /// # Panics
    ///
    /// Panics if `domains` does not have one entry per stage.
    #[must_use]
    pub fn new(space: &PolicySpace, domains: Vec<Vec<String>>) -> Self {
        assert_eq!(domains.len(), space.horizon(), "one domain per stage");
        Self {
            rules: space
                .stages()
                .iter()
                .zip(domains)
                .map(|(stage, domain)| TabularRule {
                    action: stage.action.clone(),
                    states: stage.states.clone(),
                    domain,
                    table: BTreeMap::new(),
                })
                .collect(),
        }
    }

    /// Sets `π_i(state) = action` for the stage whose action is `x`.
    ///
    /// # Panics
    ///
    /// Panics if no stage has action `x`.
    pub fn set(&mut self, x: &str, state: &[&str], action: &str) {
        let rule = self
            .rules
            .iter_mut()
            .find(|r| r.action == x)
            .unwrap_or_else(|| panic!("no stage with action `{x}`"));
        rule.table.insert(
            state.iter().map(|s| (*s).to_string()).collect(),
            action.to_string(),
        );
    }

    /// The rules, in stage order.
    #[must_use]
    pub fn rules(&self) -> &[TabularRule] {
        &self.rules
    }

    /// The rule for action `x`.
    #[must_use]
    pub fn rule(&self, x: &str) -> Option<&TabularRule> {
        self.rules.iter().find(|r| r.action == x)
    }
}

impl Policy for TabularPolicy {
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64 {
        let Some(rule) = self.rule(action) else {
            return 0.0;
        };
        let key: Vec<&str> = rule
            .states
            .iter()
            .map(|s| state.get(s).unwrap_or(""))
            .collect();
        match rule.action_for(&key) {
            Some(a) => f64::from(u8::from(a == value)),
            None if rule.domain.iter().any(|v| v == value) => 1.0 / rule.domain.len() as f64,
            None => 0.0,
        }
    }
}

impl fmt::Display for TabularPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, rule) in self.rules.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            if rule.table.is_empty() {
                write!(f, "{} ~ Unif({{{}}})", rule.action, rule.domain.join(", "))?;
                continue;
            }
            // A rule that ignores its state reads best as a constant.
            let mut values = rule.table.values();
            let first = values.next().expect("non-empty");
            if values.all(|v| v == first) {
                write!(f, "{} ← {first}", rule.action)?;
                continue;
            }
            let lines: Vec<String> = rule
                .table
                .iter()
                .map(|(k, v)| {
                    let s: Vec<String> = rule
                        .states
                        .iter()
                        .zip(k)
                        .map(|(n, x)| format!("{n}={x}"))
                        .collect();
                    format!("{}({}) = {v}", rule.action, s.join(", "))
                })
                .collect();
            write!(f, "{}", lines.join("\n"))?;
        }
        Ok(())
    }
}
