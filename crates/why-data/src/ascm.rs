//! Image counterfactuals: chapter 14, "Image-based Representations &
//! Generation".
//!
//! An augmented SCM (ASCM, Def. 14.1.1) generates an image in two stages.
//! First a generative-level SCM produces labelled factors `V` (gender, age,
//! digit, colour, …). Then a mechanism `f_I(V, U_I)` mixes them with
//! unlabelled factors into the pixels `I`. `f_I` is invertible in `V`, so a
//! labeller `h` reads `V = h(I)` off any image. Editing an image "had `X`
//! been `x′`" is the image counterfactual distribution `P(I, I_{x′})`
//! (Eq. 14.4). It is almost never identifiable from `P(V, I)`
//! (Corol. 14.2.1), and not even with the causal diagram (Thm 14.2.2). So
//! the chapter relaxes the target twice:
//!
//! * a **care set** `W ⊆ V`: the feature counterfactual (F-ctf) query
//!   `λ_φ` sums `P(I, I_{x′})` over all image pairs that agree on the
//!   features `W = φ(I)` (Def. 14.3.1). For `φ = h_W` it equals the
//!   generative-level `P(w, w′_{x′})` (Lemma 14.3.1);
//! * **optimal bounds**: an estimator is *ctf-consistent* when it fits
//!   `P(V, I)` and its F-ctf query lies in the optimal bounds that `P(V)`
//!   and `G` imply (Def. 14.3.2). Any proxy that fits the data and respects
//!   `G` is one (Thm 14.3.2).
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Ascm`] | Def. 14.1.1 on an [`scm::Model`](crate::scm::Model) with discrete pixels, the labeller `h` (Eq. 14.1) |
//! | [`ImageCounterfactual`] | `P(I, I_{x′})` (Eq. 14.4, Fig. 14.5) and the F-ctf query `λ_φ` (Def. 14.3.1) |
//! | [`optimal_bounds`] | the optimal bounds of Def. 14.3.2 (2), from `P(V)` and `G` |
//! | [`ctf_consistency`] | Def. 14.3.2 for a proxy model's `P̂(V)` and F-ctf value |
//!
//! Learning a proxy from images (Algorithm 50, ANCM) needs a tensor library
//! and lives in the `why-ncm` crate.
//!
//! # Optimal bounds
//!
//! [`optimal_bounds`] solves the canonical model exactly. Every SCM over
//! discrete `V` compatible with `G` agrees on all three layers with one in
//! which each variable is a *response type*, a function from its parents'
//! configurations to its levels, and the response types of each c-component
//! follow a joint distribution `q_C` independent of the other components.
//! `P(V)` fixes Tian's c-factor `Q[C]` of each component, which is linear in
//! `q_C`. The query is a sum, over the joint assignments of its worlds, of
//! products of one cell probability per component.
//!
//! A component whose cells all have the same probability under every
//! feasible `q_C` contributes a constant. With at most one component left,
//! the query is linear in it and two linear programs give the exact bounds.
//! In every example of the chapter only one component is left. With two or
//! more the program is multilinear, and [`optimal_bounds`] returns
//! [`AscmError::Nonlinear`] rather than a looser answer.
//!
//! Two consequences of the construction:
//!
//! * The bounds are tight when every c-component is a C²-component, that is,
//!   when every pair in it shares a bidirected edge, as in all the book's
//!   graphs. Otherwise `q_C` may couple variables that `G` keeps
//!   independent, and the interval is a valid outer bound.
//! * `P(V)` should satisfy `G`'s constraints exactly: an exact
//!   distribution, not a sample. A violation inside a c-component makes the
//!   program infeasible ([`AscmError::Incompatible`]). One across components
//!   goes unnoticed, since each component is constrained by its own c-factor.
//!   Tian's factorization also needs the conditionals it divides by to be
//!   positive ([`AscmError::Positivity`]).
//!
//! # Book errata
//!
//! * Ex. 14.2 says `Y = 0` is young. The mechanism of `H`, the claim that
//!   older people go gray more often, `do(Y = 0)` as "make older" and
//!   Ex. 14.3's "young male (`Y = 1`)" all need `Y = 0` to be old.
//!   Ex. 14.3 also writes `do(Y = 1)` for `do(Y = 0)`.
//! * Ex. 14.4 says `M(1)` and `M(2)` induce different observational
//!   distributions and the same counterfactual. It is the reverse, which is
//!   the point of the example.
//! * Eqs. 14.33, 14.35 and 14.41 condition on the wrong factual image
//!   (`D = 6`, `B = 0` and `D = 6, C = 1` respectively). The text and the
//!   figures use `D = 3`, `B = 1` and `D = 7, C = 0`.
//! * Eqs. 14.31 and 14.37 write `U_D` in all four mechanisms. Their diagrams
//!   draw `D → B` (backdoor) and `D → C` (frontdoor), so those two
//!   mechanisms must read `D`, or intervening on the digit would change
//!   nothing. The other two read `U_D`, which is the drawn confounding.
//!   The precedence in `B`'s mechanism is also open. Only
//!   `((1[D ≥ 5] ⊕ U1) ∨ (C ⊕ U2)) ∧ U3` reproduces the printed bounds.

use std::fmt;

use crate::fairness::FairnessError;
use crate::graph::dseparation::DSepError;
use crate::ncm::NcmError;
use crate::scm::ScmError;

mod bounds;
mod consistency;
mod model;
#[cfg(test)]
mod tests;

pub use crate::crl::Interval;
pub use bounds::optimal_bounds;
pub use consistency::{CtfConsistency, ctf_consistency, total_variation};
pub use model::{Ascm, ImageCounterfactual};

/// What can go wrong with an ASCM or a bound.
#[derive(Clone, Debug, PartialEq)]
pub enum AscmError {
    /// The structural model failed to build or evaluate.
    Scm(ScmError),
    /// The graph or the query is malformed.
    Structure(NcmError),
    /// A name is not an endogenous variable of the model or the graph.
    UnknownVariable(String),
    /// A level is not one the data show for the variable.
    UnknownLevel {
        /// The variable.
        var: String,
        /// The level.
        level: String,
    },
    /// The data set has no column for a variable of the graph.
    MissingColumn(String),
    /// The data set is malformed, or empty.
    Data(String),
    /// Two images with different labels coincide, so `f_I` is not
    /// invertible in `V` (Def. 14.1.1 (3)).
    NotInvertible,
    /// A conditional that Tian's factorization divides by has probability
    /// zero.
    Positivity(String),
    /// The conditioning event has probability zero.
    ImpossibleCondition,
    /// The query conditions on a counterfactual world. Only the factual
    /// (natural) world may be conditioned on.
    CounterfactualEvidence(String),
    /// No SCM compatible with the graph induces the distribution: it breaks
    /// one of the graph's constraints.
    Incompatible,
    /// The query couples the free response types of several c-components,
    /// so its bounds are a multilinear program. The components are listed.
    Nonlinear(Vec<Vec<String>>),
    /// The canonical model needs more response types or assignments than the
    /// given limit.
    TooLarge(usize),
}

impl From<ScmError> for AscmError {
    fn from(e: ScmError) -> Self {
        Self::Scm(e)
    }
}

impl From<NcmError> for AscmError {
    fn from(e: NcmError) -> Self {
        match e {
            NcmError::ImpossibleCondition => Self::ImpossibleCondition,
            e => Self::Structure(e),
        }
    }
}

impl From<DSepError> for AscmError {
    fn from(e: DSepError) -> Self {
        Self::Structure(e.into())
    }
}

impl From<FairnessError> for AscmError {
    fn from(e: FairnessError) -> Self {
        Self::Data(e.to_string())
    }
}

impl fmt::Display for AscmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scm(e) => e.fmt(f),
            Self::Structure(e) => e.fmt(f),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::UnknownLevel { var, level } => {
                write!(f, "{level} is not a level of {var} in the data")
            }
            Self::MissingColumn(v) => write!(f, "the data have no column for {v}"),
            Self::Data(s) => write!(f, "bad data: {s}"),
            Self::NotInvertible => f.write_str("the image mechanism is not invertible in V"),
            Self::Positivity(s) => write!(f, "positivity fails: P({s}) = 0"),
            Self::ImpossibleCondition => f.write_str("the conditioning event has probability zero"),
            Self::CounterfactualEvidence(v) => {
                write!(f, "the query conditions on {v} in a counterfactual world")
            }
            Self::Incompatible => {
                f.write_str("no SCM compatible with the graph induces the distribution")
            }
            Self::Nonlinear(cs) => {
                let cs: Vec<String> = cs.iter().map(|c| format!("{{{}}}", c.join(", "))).collect();
                write!(
                    f,
                    "the bounds couple several c-components ({}), a multilinear program",
                    cs.join(", ")
                )
            }
            Self::TooLarge(n) => write!(f, "the canonical model needs more than {n} entries"),
        }
    }
}

impl std::error::Error for AscmError {}
