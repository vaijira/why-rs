//! Parametric identification: chapter 19, "Parametric Identification
//! (Linear & Monotonic)".
//!
//! Effects that no amount of data identifies non-parametrically can be
//! identified once the mechanisms have a known shape. This module covers
//! the chapter's two shapes.
//!
//! **Linear SCMs** (Secs. 19.2-19.5): `V = Λᵀ V + U`, `Cov(U) = Ω`, so the
//! data reduce to `Σ = (I − Λ)⁻ᵀ Ω (I − Λ)⁻¹`. Identification asks which
//! coefficients `λ_xy` (direct effects) or total effects `δ_xy` that pins
//! down.
//!
//! | Item | Book |
//! | --- | --- |
//! | [`LinearScm`] | Eqs. 19.3-19.22; [`LinearScm::trek_covariance`] is the trek rule (Def. 19.2.1, Eq. 19.38) |
//! | [`Covariance`] | the data: regression coefficients `β_{yx.z}` and instrumental-set systems |
//! | [`single_door`] | Thm 19.3.1 |
//! | [`back_door`] | Thm 19.3.2 |
//! | [`conditional_instrument`] | Defs. 19.4.1-19.4.2, Thms 19.4.1-19.4.2 |
//! | [`generic_rank`] | Thm 19.5.2: rank of `Σ_{S,T}` by vertex max-flow in the flow graph (Def. 19.5.1, Lemma 19.5.1) |
//! | [`instrumental_set`] | Def. 19.5.3 found by Algorithm 66 (maximal match-block) |
//!
//! **Monotonicity** (Sec. 19.7): binary variables whose mechanisms are
//! monotone in some parents. Counterfactuals that are not identifiable
//! without the constraint can become identifiable with it.
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Cgma`] | Defs. 19.7.1-19.7.2; [`Cgma::globally_monotone`] is Lemma 19.7.1 |
//! | [`CtfTerm`], [`m_reduce`] | Lemma 19.7.2 (MRL), Def. 19.7.3, Lemma 19.7.7, Algorithm 67 |
//! | [`m_id`] | M-ID: factorizes a counterfactual into ctf-factors, reduces them with Algorithm 67, and identifies each with C-INFER (chapter 5) |
//!
//! # Book notes
//!
//! * Thm 19.4.1 prints `δ_xy = σ_zx / σ_zy`. Eq. 19.100 and the text give
//!   `σ_zy / σ_zx`, which is correct.
//! * Sec. 19.7.6.2 reports a total effect of 0.175 for its SCM. The printed
//!   mechanisms and Table 19.1 give exactly 0.205; 0.175 is presumably the
//!   estimate from the 10 000 samples drawn there.
//! * Algorithm 67 is applied to ctf-factors whose terms carry their full
//!   parent assignments (the output of CTF-FACTOR). [`m_id`] produces them
//!   by expanding each counterfactual over the values of its parents.
//!
//! Not implemented: the graph decompositions and recursive total-effect
//! identification of Sec. 19.6 (the book gives no algorithm for the
//! decomposed covariance matrices), and the 401k data experiment.

use std::fmt;

use crate::graph::cinfer::counterfactual::CtfError;
use crate::graph::dseparation::DSepError;
use crate::transport::TransportError;

mod linear;
mod monotone;
#[cfg(test)]
mod tests;

pub use linear::{
    Covariance, LinearScm, back_door, conditional_instrument, generic_rank, instrumental_set,
    single_door,
};
pub use monotone::{Cgma, CtfTerm, Event, Sign, m_id, m_reduce};

/// What can go wrong.
#[derive(Clone, Debug, PartialEq)]
pub enum ParametricError {
    /// A graph operation failed.
    Graph(String),
    /// A name is not a variable.
    UnknownVariable(String),
    /// A coefficient or covariance is given for a pair with no such edge.
    NotAnEdge(String),
    /// A system of equations is degenerate.
    Singular,
    /// An input is malformed.
    InvalidInput(String),
    /// The counterfactual is not identifiable under the monotonicity
    /// constraints (Alg. 67 fails, Thm 19.7.8), or a reduced factor is not
    /// identifiable.
    NotIdentifiable(String),
    /// Evaluating an estimand failed.
    Transport(TransportError),
}

impl From<DSepError> for ParametricError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e.to_string())
    }
}

impl From<TransportError> for ParametricError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<CtfError> for ParametricError {
    fn from(e: CtfError) -> Self {
        Self::NotIdentifiable(format!("{e:?}"))
    }
}

impl fmt::Display for ParametricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => f.write_str(e),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::NotAnEdge(e) => write!(f, "{e} is not an edge of the graph"),
            Self::Singular => f.write_str("the system of equations is singular"),
            Self::InvalidInput(s) => f.write_str(s),
            Self::NotIdentifiable(s) => write!(f, "not identifiable: {s}"),
            Self::Transport(e) => write!(f, "{e:?}"),
        }
    }
}

impl std::error::Error for ParametricError {}
