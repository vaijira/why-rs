//! The hierarchy of graphical models: chapter 20, "Graphical Models and
//! the Causal Hierarchy".
//!
//! Between interventions (L2) and full counterfactuals (L3) sit two layers
//! of distributions that counterfactual randomization makes physically
//! accessible: L2.25, where a randomized variable passes one value to all
//! of its children, and L2.5, where each child can receive its own. This
//! module decides which layer a query lives in, whether an agent with a
//! given set of physical actions can sample from it, and how to act on
//! the extra information. It also implements the constructions of the
//! two graphical frameworks that Sec. 20.3 compares with causal diagrams.
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Term`] | a potential response `W_t` |
//! | [`layer`], [`Layer`] | Defs. 20.1.11-20.1.12, Lemma 20.1.4 and Thm 20.1.6: L1 ⊆ L2 ⊆ L2.25 ⊆ L2.5 ⊆ L3 |
//! | [`reachability_set`] | Def. 20.1.16, the counterfactual reachability set `CRS(W*)` |
//! | [`ancestor_set`] | Def. 20.2.5, `An(W*)` |
//! | [`ctf_realize`], [`Action`], [`Plan`] | Algorithms 68-69 (CTF-REALIZE, COMPATIBLE), Thm 20.2.5 and Cor. 20.2.6 |
//! | [`MabScm`], [`mab_opt`] | Sec. 20.2.3: the strategies of the MAB template, Thm 20.2.8, Cor. 20.2.9, and Algorithm 70 (MAB-OPT) with Thompson sampling |
//! | [`swig`], [`Swig`] | Algorithm 71 (SWIG-CONSTRUCT), Prop. 20.3.4 and the counterfactual adjustment criterion (Def. 20.3.9) |
//! | [`ag_construct`], [`Eci`] | Algorithm 72 (AG-CONSTRUCT) over a set of extended conditional independences |
//!
//! # Book notes
//!
//! * **COMPATIBLE is missing a conflict.** Algorithm 69 never checks that a
//!   variable which must reach a child naturally is not also under a
//!   Fisherian `RAND`. Without that check, `P(Y_x, Z)` in `Z ← X → Y` comes
//!   out realizable from `RAND(X)` alone, and the plan returns `Z_x` in
//!   place of `Z`. [`ctf_realize`] treats `RAND(V)` as one more channel
//!   that a natural requirement conflicts with.
//! * **Def. 20.1.16's merging rule** is read as follows. Let `I` be the
//!   terms whose reachability sets hold some entry for `R`, and `x∪` the
//!   union of their subscripts. The entries merge into `‖R_{x∪}‖` when `x∪`
//!   is consistent and `‖W_{x∪}‖ = ‖W_x‖` for every term in `I`. With this
//!   reading, Lemma 20.1.4's L2.25 test holds exactly when CTF-REALIZE
//!   succeeds under one-value-per-variable randomization, and whenever
//!   Prop. 20.3.4's SWIG test does. The tests check both on random graphs.
//! * Phase 2 of Algorithm 68 draws each randomization from a device and
//!   discards units whose draw differs from the tag. [`Plan::sample`]
//!   writes the tagged values directly, which yields the same distribution
//!   (Lemma 20.2.1) without rejection.
//! * **The ancestor set leaves out the intervened variables.** Def. 20.2.5
//!   ranges over `An(Y)` in `G_X̄`, which contains `X` itself. The book's
//!   examples (`An(Y_x, X) = {Y_x, X}`, Ex. 20.19) leave the intervened
//!   variables out, and so does [`ancestor_set`].
//!
//! Not implemented:
//! * The ctf-calculus restricted to L2.25 or L2.5 (Def. 20.1.15). The book
//!   gives no algorithm for it. [`layer`] decides membership, and
//!   [`crate::graph::cinfer`] identifies the query once it is in a layer.
//! * The SEM-RR and FFRCISTG constructions and the DT-model inference
//!   examples of Sec. 20.3. Those are arguments about the frameworks, not
//!   procedures.
//! * The simulations of Experiments 1-3. The book prints their results but
//!   not their models. The example builds a model with Experiment 3's
//!   numbers.

use std::collections::BTreeMap;
use std::fmt;

use crate::discovery::DiscoveryError;
use crate::graph::dseparation::DSepError;

mod augmented;
mod bandit;
mod layers;
mod realize;
#[cfg(test)]
mod tests;

pub use augmented::{Eci, ag_construct};
pub use bandit::{MabRun, MabScm, MabUnit, Strategies, mab_int, mab_opt};
pub use layers::{Layer, Swig, ancestor_set, layer, reachability_set, swig};
pub use realize::{
    Action, Plan, Tag, ctf_realize, fisher_actions, layer225_actions, maximal_actions,
};

/// A potential response `W_t = ?`: a variable and the intervention in its
/// subscript, as `(variable, value)` pairs. An empty subscript is the
/// natural (factual) variable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    /// The variable.
    pub var: String,
    /// The subscript.
    pub sub: BTreeMap<String, u8>,
}

impl Term {
    /// `var` under the intervention `sub`.
    #[must_use]
    pub fn new(var: &str, sub: &[(&str, u8)]) -> Self {
        Self {
            var: var.to_string(),
            sub: sub.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
        }
    }

    /// The natural variable `var`.
    #[must_use]
    pub fn natural(var: &str) -> Self {
        Self::new(var, &[])
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.var)?;
        if !self.sub.is_empty() {
            let s: Vec<String> = self.sub.iter().map(|(k, v)| format!("{k}={v}")).collect();
            write!(f, "_{{{}}}", s.join(","))?;
        }
        Ok(())
    }
}

/// What can go wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum HierarchyError {
    /// A graph operation failed, or the graph has a directed cycle.
    Graph(String),
    /// A name is not a variable of the graph.
    UnknownVariable(String),
    /// An input is malformed.
    InvalidInput(String),
    /// CTF-REALIZE fails: no procedure built from the actions samples the
    /// query.
    NotRealizable(String),
    /// The discovery engine failed.
    Discovery(DiscoveryError),
}

impl From<DSepError> for HierarchyError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e.to_string())
    }
}

impl From<DiscoveryError> for HierarchyError {
    fn from(e: DiscoveryError) -> Self {
        Self::Discovery(e)
    }
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) | Self::InvalidInput(e) => f.write_str(e),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::NotRealizable(s) => write!(f, "not realizable: {s}"),
            Self::Discovery(e) => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for HierarchyError {}
