//! Generalization and robustness across the PCH: chapter 12, transporting
//! interventional (layer 2) and counterfactual (layer 3) quantities from the
//! domains and regimes where data were collected to a target domain `π*`.
//!
//! Chapter 11 ([`crate::transport`]) moved statistical quantities between
//! domains. Here the quantity is causal, and the data may themselves be
//! experimental. Everything builds on chapter 11's [`SelectionDiagram`],
//! whose `Δ^{k,*}` says which mechanisms may differ between source `k` and
//! the target, and on its [`Estimand`], whose terms name the domain they are
//! read from — now with the regime as well: `P^1(y | z; σ_Z)`, `P^2(z;
//! do(x))`, and the target policy's own `P̂(x | r)`.
//!
//! # Generalizing experiments by adjustment (Sec. 12.1)
//!
//! [`tr_adjustment`] is the transport formula of Thm 12.1.3,
//! `P*(y | do(x)) = Σ_z P(y | do(x), z) P*(z)`, valid exactly when `Z`
//! satisfies the tr-adjustment criterion ([`is_tr_admissible`],
//! Def. 12.1.3; [`is_tr_admissible_singleton`] is Def. 12.1.1). Its first
//! half is the e-adjustment criterion (Def. 12.1.2), decided by
//! [`is_e_admissible`] — Algorithm 40, IsEAdmissible — with
//! [`e_adjustment_order`] returning the order it found.
//! [`pre_treatment_subset`] is Thm 12.1.6, and [`find_tr_adjustment`]
//! searches for an admissible set.
//!
//! # Policy transportability by C-INFER (Sec. 12.1.2)
//!
//! [`sigma_tr`] decides the σ-TR task `⟨P*(y | w; σ_X), P, D^Δ⟩` (signature
//! 12.35): every input is a [`DomainInput`] `P^k(V; σ_Z)`, from the target or
//! a source, natural or under a [`Regime`] of atomic and soft interventions.
//! The q-tree ([`gen_query_tree`]) splits the query into c-factors of `π*`
//! under `σ_X`; each input's d-tree ([`gen_input_tree`], Algorithm 41) seeks
//! them with `Σ` and `δ` in `G_{σ_Z}`, and ends with the new operators:
//!
//! | Operator | Rewrites | Licensed when |
//! | --- | --- | --- |
//! | `π` (Lemma 12.1.10) | `Q^k[T; σ] -> Q*[T; σ]` | `T ∩ Δ^{k,*} = ∅` |
//! | `σ` (Lemma 12.1.11) | `Q[T; σ_Z] -> Q[T; σ_X]` | `T ∩ (X ∪ Z) = ∅` |
//!
//! Atomic interventions give the g-TR task (Thm 12.1.12) as a special case.
//!
//! # Counterfactual transportability (Sec. 12.2)
//!
//! [`ctf_tr`] decides ctf-TR (Def. 12.2.1): the query tree is chapter 5's
//! ([`crate::graph::cinfer::counterfactual`], Algorithm 42), whose Ξ leaves
//! are c-factors of the target, and the d-trees are Algorithm 41's.
//!
//! # Evaluating estimands
//!
//! [`Estimand::evaluate`] takes any [`crate::transport::Distributions`].
//! Register each input under [`DomainInput::estimand_domain`] — for data,
//! [`crate::transport::EmpiricalDistributions::with`] — and wrap them in
//! [`WithPolicy`] to supply the target policy's `P̂(x | pa*)`.
//!
//! # Departures from the text
//!
//! * **Inputs keep their intervened variables.** A d-tree's root is
//!   `Q^k[V; σ_Z]` over all of `V`, where the book writes
//!   `Q^k[V \ Z | do(z)]` for an atomic regime. The intervened variables are
//!   c-components of their own in `G_{σ_Z}`, so `δ` splits them off and the
//!   two searches reach the same factors. An atomic input is evaluated as a
//!   trial that randomizes `Z`, which is what an experiment at every level
//!   amounts to, so `do(z)` terms read `z` as a conditioning variable.
//! * **Conditional σ-TR queries are a plain ratio.** `P*(y | w; σ_X)` is
//!   `P*(y, w; σ_X) / P*(w; σ_X)` over `An(Y ∪ W)` in `G_{σ_X}`. That is
//!   sound, but unlike layer-2 C-INFER it does not drop the part of `W`
//!   that rule 2 would turn into interventions, so it can miss conditional
//!   queries that Thm 12.1.13 says are transportable. Unconditional queries,
//!   which all the book's examples are, follow the book's procedure.
//! * **Condition (12.16) is rule 3.** The text's `G_{X(Z_{≤i−1})}` is read as
//!   `G` with the edges into `X(Z_{≤i−1})` cut, the treatments that are not
//!   ancestors of `Z_{≤i−1}`, since that is what licenses
//!   `P(z_i | do(x), z_{<i}) = P(z_i | z_{<i})`. Def. 12.1.1's condition (i)
//!   names no graph and is read the same way.
//! * **Exercise 12.12's solution** drops `Z` from `P(y | x1, x2, z; σ*)`
//!   because "`Y ⊥ Z | X1, X2` in `G_{σ*}`". That fails in Fig. 12.5 (there
//!   is an edge `Z → Y`), and Eq. 12.67 keeps `z`.
//!
//! Not implemented: the exercises' informal tasks, and conditional σ-TR
//! completeness (see above).

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::graph::cinfer::counterfactual::Event;
use crate::graph::dseparation::DSepError;
use crate::transport::{Distributions, Domain, TransportError};

mod adjustment;
mod counterfactual;
mod policy;
#[cfg(test)]
mod tests;

pub use crate::transport::{Estimand, Intervention, Regime, SelectionDiagram};
pub use adjustment::{
    e_adjustment_order, find_tr_adjustment, is_e_admissible, is_tr_admissible,
    is_tr_admissible_singleton, pre_treatment_subset, tr_adjustment,
};
pub use counterfactual::{CtfTrDerivation, CtfTrEstimand, ctf_tr};
pub use policy::{
    DomainInput, PolicyQuery, TrDerivation, TrFactor, TrOperator, TrSource, TrTerm, TrTree,
    gen_input_tree, gen_query_tree, map_factors, sigma_tr,
};

/// Why a transport failed.
#[derive(Clone, Debug, PartialEq)]
pub enum TrError {
    /// Some target c-factor appears in no d-tree.
    NotTransportable {
        /// The d-trees, which show how far each input got.
        input_trees: Vec<TrTree>,
        /// The target c-factors no input could produce.
        unmapped: Vec<TrFactor>,
    },
    /// A ctf-factor gives one variable two values, so the Ξ-operator does not
    /// apply and no combination of data transports the query.
    Inconsistent {
        /// The inconsistent ctf-factor, written `W[pa_w]`.
        factor: Vec<Event>,
    },
    /// The conditioning event contradicts itself.
    ImpossibleCondition,
    /// The query, an input or the diagram is malformed.
    Invalid(TransportError),
}

impl From<TransportError> for TrError {
    fn from(e: TransportError) -> Self {
        Self::Invalid(e)
    }
}

impl From<DSepError> for TrError {
    fn from(e: DSepError) -> Self {
        Self::Invalid(TransportError::Graph(e))
    }
}

impl fmt::Display for TrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotTransportable { unmapped, .. } => {
                let names: Vec<String> = unmapped.iter().map(ToString::to_string).collect();
                write!(f, "not transportable: no input yields {}", names.join(", "))
            }
            Self::Inconsistent { factor } => {
                let es: Vec<String> = factor.iter().map(ToString::to_string).collect();
                write!(f, "not transportable: the ctf-factor P({}) is inconsistent", es.join(", "))
            }
            Self::ImpossibleCondition => f.write_str("the conditioning event is impossible"),
            Self::Invalid(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for TrError {}

/// The target policy `σ*_X`, as the conditional tables `P̂(x | pa*)` its
/// estimand terms need.
///
/// ```
/// use why_data::generalize::TargetPolicy;
///
/// let mut pi = TargetPolicy::new();
/// pi.set("X", &[("R", "0")], "1", 0.8);
/// pi.set("X", &[("R", "0")], "0", 0.2);
/// assert_eq!(pi.probability("X", "1", &[("R", "0")]), Some(0.8));
/// ```
#[derive(Clone, Debug, Default)]
pub struct TargetPolicy {
    tables: HashMap<String, PolicyRows>,
}

/// One variable's policy: parent assignment (sorted) → value → probability.
type PolicyRows = HashMap<Vec<(String, String)>, BTreeMap<String, f64>>;

impl TargetPolicy {
    /// No tables yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `P̂(var = value | parents)`.
    pub fn set(&mut self, var: &str, parents: &[(&str, &str)], value: &str, p: f64) {
        let mut key: Vec<(String, String)> = parents
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        key.sort();
        self.tables
            .entry(var.to_string())
            .or_default()
            .entry(key)
            .or_default()
            .insert(value.to_string(), p);
    }

    /// `P̂(var = value | parents)`, or `None` if it was never set. A value
    /// missing from a row that was set has probability zero.
    #[must_use]
    pub fn probability(&self, var: &str, value: &str, parents: &[(&str, &str)]) -> Option<f64> {
        let mut key: Vec<(String, String)> = parents
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        key.sort();
        let row = self.tables.get(var)?.get(&key)?;
        Some(row.get(value).copied().unwrap_or(0.0))
    }
}

/// Distributions with the target policy alongside: [`Domain::Policy`] terms
/// are read from `policy`, everything else from `dists`.
#[derive(Debug)]
pub struct WithPolicy<'a, D> {
    dists: &'a D,
    policy: &'a TargetPolicy,
}

impl<'a, D: Distributions> WithPolicy<'a, D> {
    /// `dists` and `policy` together.
    pub fn new(dists: &'a D, policy: &'a TargetPolicy) -> Self {
        Self { dists, policy }
    }
}

impl<D: Distributions> Distributions for WithPolicy<'_, D> {
    fn levels(&self, var: &str) -> Option<Vec<String>> {
        self.dists.levels(var)
    }

    fn probability(&self, domain: &Domain, event: &[(&str, &str)]) -> Result<f64, TransportError> {
        if *domain == Domain::Policy {
            return Err(TransportError::InvalidInput(
                "the target policy is known only as a conditional".into(),
            ));
        }
        self.dists.probability(domain, event)
    }

    fn conditional(
        &self,
        domain: &Domain,
        event: &[(&str, &str)],
        given: &[(&str, &str)],
    ) -> Result<f64, TransportError> {
        if *domain != Domain::Policy {
            return self.dists.conditional(domain, event, given);
        }
        let [(var, value)] = event else {
            return Err(TransportError::InvalidInput(
                "a policy term sets exactly one variable".into(),
            ));
        };
        self.policy
            .probability(var, value, given)
            .ok_or_else(|| TransportError::MissingData(format!("no policy row for {var} given {given:?}")))
    }
}
