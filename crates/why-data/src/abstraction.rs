//! Causal abstractions: chapter 15, "Causal Abstractions & Generation".
//!
//! A high-level model `M_H` over coarse variables `V_H` (calories, a
//! completed treatment, a digit) abstracts a low-level model `M_L` over fine
//! variables `V_L` (grams of each nutrient, each dose, pixels) through a map
//! `τ: D_{V_L} → D_{V_H}`. The chapter's working class of maps is the
//! *constructive abstraction function* (Def. 15.2.2). An intervariable
//! clustering `C` groups low-level variables into one cluster per high-level
//! variable; variables left out are projected away. An intravariable
//! clustering `D` then partitions each cluster's joint values into the
//! high-level levels (Def. 15.2.1). Here a cluster's partition is a function
//! from its members' values to a level, which names the block a tuple falls
//! in.
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Abstraction`] | `τ` from clusters `C` and `D` (Defs. 15.2.1, 15.2.2); [`Abstraction::transform`] is `τ(P(V_L))`; [`Abstraction::cdag`] is the cluster causal diagram (Def. 15.3.1) |
//! | [`Term`], [`low_level`], [`HighLevel`] | a quantity `Q` (Eq. 15.19) and its counterpart `τ(Q)` (Eq. 15.20) |
//! | [`q_consistency`], [`layer_consistency`] | Q-τ and `L_i`-τ consistency (Def. 15.2.3) |
//! | [`abstract_invariance`] | the abstract invariance condition (Thm 15.2.2) |
//! | [`construct`], [`Constructed`] | Algorithm 51: `M_H` from `M_L` |
//! | [`conditional_invariance`] | the conditional AIC (Def. 15.2.5) in its local, C-DAG form, checkable on data |
//! | [`observational_partition`], [`interventional_partition`] | Eqs. 15.67, 15.68 and the causal coarsening theorem (Thm 15.2.5) |
//! | [`abstract_identify`], [`abstract_estimate`] | abstract identification (Def. 15.3.2) by the dual abstract ID theorem (Thm 15.3.2) |
//!
//! Algorithm 52 (NeuralAbstractID) trains NCMs and lives in the `why-ncm`
//! crate.
//!
//! # Readings
//!
//! * **A cluster's inputs.** Alg. 51 builds `f^H_i(p̃a, u) =
//!   τ_{C_i}(f^L_V(p̃a_V, u_V) : V ∈ C_i)` with `p̃a` any preimage of the
//!   high-level parents. Ex. 15.8 shows what that means (Eq. 15.60). Members of
//!   `C_i` are computed in order from one another, and a variable outside
//!   every cluster (`R` there) is computed from `u`, not fixed. So
//!   [`Constructed`] evaluates `M_H` as `M_L` under `do(` the other clusters
//!   `= p̃a)`, and [`abstract_invariance`] checks Eq. 15.50 the same way, for
//!   each pair of preimages of the same high-level parents. Under the
//!   literal reading, which also fixes projected-away variables, Ex. 15.5's
//!   own clustering would break the AIC through `R`.
//! * **C-DAGs over projected-away variables.** Def. 15.3.1 takes a
//!   partition of `V`. With variables left out (Ex. 15.9 drops `R`),
//!   [`Abstraction::cdag`] first takes the latent projection onto the
//!   clustered variables, which is how Ex. 15.9's `X ↔ Y` arises.
//! * **`L3`-τ consistency over two worlds.** [`layer_consistency`] checks
//!   every query over one world (layers 1 and 2) or two worlds (layer 3).
//!   Queries over more worlds are not enumerated.
//!
//! # Book notes
//!
//! * Alg. 51's output line writes `τ: D_{V_H} → D_{V_L}`, and Prop. 15.3.1
//!   writes the same. `τ` maps low to high.
//! * Eq. 15.47 writes `τ_{C1}` for `τ_{C2}`, and Eq. 15.12 writes `D` for
//!   `B` in `τ(R, D, C, F, P, D)`.
//! * Fig. 15.12(a)'s C-DAG, `R → X → Y` with `R ↔ Y`, follows from Eq. 15.24:
//!   `R` and `Y` share `U_RY`.
//! * Ex. 15.9's `P(Y_{X=1} = 1)` values, 0.95 and 0.29, are rounded. Exactly,
//!   they are `0.66 + 0.34 · P(U_Y1)` and `0.34 · P(U_Y1)` with
//!   `P(U_Y1) = 0.852941`, about `29/34`.

use std::fmt;

use crate::fairness::FairnessError;
use crate::graph::cinfer::counterfactual::CtfError;
use crate::graph::dseparation::DSepError;
use crate::scm::ScmError;
use crate::transport::TransportError;

mod clusters;
mod consistency;
mod construct;
mod identify;
mod invariance;
#[cfg(test)]
mod tests;

pub use clusters::Abstraction;
pub use consistency::{
    HighLevel, LayerCheck, Mismatch, Term, layer_consistency, low_level, q_consistency,
};
pub use construct::{AicViolation, Constructed, abstract_invariance, construct};
pub use identify::{abstract_estimate, abstract_identify};
pub use invariance::{conditional_invariance, interventional_partition, observational_partition};

/// What can go wrong with an abstraction.
#[derive(Clone, Debug, PartialEq)]
pub enum AbstractionError {
    /// A cluster has no members, or a cluster name is used twice.
    BadCluster(String),
    /// A low-level variable is in two clusters.
    Overlap(String),
    /// A name is not a cluster of the abstraction.
    UnknownCluster(String),
    /// A name is not a variable of the model, graph or data.
    UnknownVariable(String),
    /// An intervention or assignment covers part of a cluster: only unions
    /// of clusters have a high-level counterpart.
    Partial(String),
    /// A high-level level has no low-level preimage the model can take.
    NoPreimage {
        /// The cluster.
        cluster: String,
        /// The level.
        level: String,
    },
    /// The clustering is not admissible (Def. 15.2.1): some descendant of
    /// the named cluster outside it is also its ancestor.
    NotAdmissible(String),
    /// The model breaks the abstract invariance condition (Thm 15.2.2).
    Invariance(Box<AicViolation>),
    /// The structural model failed.
    Scm(ScmError),
    /// A graph operation failed.
    Graph(String),
    /// The query is not identifiable, or malformed.
    Identification(CtfError),
    /// Evaluating an estimand failed.
    Transport(TransportError),
    /// The data are malformed.
    Data(String),
    /// An enumeration needs more than the given number of entries.
    TooLarge(usize),
}

impl From<ScmError> for AbstractionError {
    fn from(e: ScmError) -> Self {
        Self::Scm(e)
    }
}

impl From<DSepError> for AbstractionError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e.to_string())
    }
}

impl From<CtfError> for AbstractionError {
    fn from(e: CtfError) -> Self {
        Self::Identification(e)
    }
}

impl From<TransportError> for AbstractionError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<FairnessError> for AbstractionError {
    fn from(e: FairnessError) -> Self {
        Self::Data(e.to_string())
    }
}

impl fmt::Display for AbstractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadCluster(c) => write!(f, "bad cluster {c}: empty or named twice"),
            Self::Overlap(v) => write!(f, "{v} is in two clusters"),
            Self::UnknownCluster(c) => write!(f, "unknown cluster {c}"),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::Partial(c) => write!(f, "only part of cluster {c} is assigned"),
            Self::NoPreimage { cluster, level } => {
                write!(f, "no low-level value of {cluster} maps to {level}")
            }
            Self::NotAdmissible(c) => {
                write!(f, "the clustering is not admissible around cluster {c}")
            }
            Self::Invariance(v) => write!(f, "the abstract invariance condition fails: {v}"),
            Self::Scm(e) => e.fmt(f),
            Self::Graph(e) => f.write_str(e),
            Self::Identification(e) => write!(f, "{e:?}"),
            Self::Transport(e) => write!(f, "{e:?}"),
            Self::Data(s) => write!(f, "bad data: {s}"),
            Self::TooLarge(n) => write!(f, "the enumeration needs more than {n} entries"),
        }
    }
}

impl std::error::Error for AbstractionError {}
