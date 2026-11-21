//! Neural causal models: chapter 13, "Causal Generative Models".
//!
//! A neural causal model (NCM, Def. 13.1.7) is an SCM whose exogenous
//! variables are uniform on `[0, 1]` and whose mechanisms are feedforward
//! neural networks. NCMs are as expressive as SCMs (Thm 13.1.4), so on their
//! own they inherit the causal hierarchy theorem: fitting `P(V)` says almost
//! nothing about `L2` or `L3` (Corol. 13.1.5, Ex. 13.6). The causal diagram
//! is the inductive bias that bridges the layers. A *G-constrained* NCM
//! (Def. 13.1.11) gives each complete confounded component of `G` one shared
//! noise `Û_C` and each mechanism exactly its parents in `G`. Such an NCM
//! satisfies every equality constraint `G` imposes on all three layers
//! (Thms 13.1.6 and 13.2.3), and loses no expressiveness under them
//! (Thms 13.1.7 and 13.2.4). A query is then identifiable from `G` and the
//! data exactly when every G-NCM that fits the data agrees on it
//! (Thms 13.1.8 and 13.2.5), and when it is, any fitted G-NCM computes it by
//! mutilation (Corols. 13.1.9 and 13.2.6).
//!
//! This module has what does not need a tensor library:
//!
//! * [`c2_components`] (Def. 13.1.10) and [`NcmStructure`], the scopes of
//!   Def. 13.1.11's construction.
//! * [`expand`], which turns a counterfactual query such as `P(y_{x, W_{x'}} | x')`
//!   into the worlds an NCM has to simulate, nested subscripts included
//!   (Def. 13.3.1).
//! * [`StepNcm`], NCMs built from the book's binary-step units `f̂_AND`,
//!   `f̂_OR`, `f̂_NOT` and `f̂_≥z`. Their queries are computed exactly by
//!   splitting `[0, 1]^k` at the units' thresholds, which reproduces
//!   Examples 13.5, 13.6, 13.10, 13.15 and 13.16 to the last digit.
//!   [`StepNcm::sample`] is Algorithm 48 (counterfactual sampling by
//!   rejection).
//!
//! Training NCMs (Algorithms 44–47 and 49) needs automatic differentiation
//! and lives in the `why-ncm` crate, which builds on these types.

use std::fmt;

use crate::graph::cinfer::counterfactual::CtfError;
use crate::graph::dseparation::DSepError;

mod step;
mod structure;
#[cfg(test)]
mod tests;
mod worlds;

pub use step::{
    Activation, Input, StepNcm, Unit, and, at_least, below, linear, noise, not, or, parent,
};
pub use structure::{NcmStructure, c2_components};
pub use worlds::{Assignment, Expanded, Fix, World, expand};

/// What an NCM operation can get wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum NcmError {
    /// The graph is malformed or names an unknown variable.
    Graph(DSepError),
    /// The query is malformed.
    Query(CtfError),
    /// A query value, such as `X'`, has no level assigned.
    MissingValue(String),
    /// A variable was never given a mechanism or a domain.
    UnknownVariable(String),
    /// A value is not one of the variable's levels.
    OutOfDomain {
        /// The variable.
        var: String,
        /// The offending value.
        value: String,
    },
    /// A mechanism feeds a noise into a unit together with other inputs, so
    /// [`StepNcm`] cannot find the regions where the network is constant.
    NotAxisAligned(String),
    /// The mechanisms depend on one another in a cycle.
    Cyclic(String),
    /// The conditioning event has probability zero.
    ImpossibleCondition,
    /// Enumerating the query's worlds would take more than the given number
    /// of cells.
    TooLarge(usize),
    /// An input argument is out of range.
    InvalidInput(String),
}

impl From<DSepError> for NcmError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl From<CtfError> for NcmError {
    fn from(e: CtfError) -> Self {
        Self::Query(e)
    }
}

impl fmt::Display for NcmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => e.fmt(f),
            Self::Query(e) => write!(f, "{e:?}"),
            Self::MissingValue(v) => write!(f, "no level given for the value {v}"),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::OutOfDomain { var, value } => write!(f, "{value} is not a level of {var}"),
            Self::NotAxisAligned(v) => {
                write!(
                    f,
                    "the mechanism of {v} mixes a noise with other inputs in one unit"
                )
            }
            Self::Cyclic(v) => write!(f, "the mechanisms are cyclic through {v}"),
            Self::ImpossibleCondition => f.write_str("the conditioning event has probability zero"),
            Self::TooLarge(n) => write!(f, "the query needs more than {n} cells"),
            Self::InvalidInput(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for NcmError {}
