//! Structural learning: chapters 16, "Structural Learning", and 17,
//! "Multi-Domain Structural Learning".
//!
//! Learning a causal diagram from data can only reach an equivalence class:
//! every diagram that implies the same constraints fits the data equally
//! well. With latent variables the class of observational equivalence is
//! that of a maximal ancestral graph (MAG), and the learned object is a
//! partial ancestral graph (PAG). A PAG keeps the adjacencies and only the
//! edge marks every MAG in the class shares, with a circle where they
//! differ. Interventional data add invariances across distributions, and
//! F-nodes in an augmented diagram represent them (Defs. 16.2.5, 16.2.11).
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Pag`], [`Mark`] | partial mixed graphs (Def. 16.1.7); MAGs are PAGs without circles; [`Pag::is_visible`] is Lemma 16.1.4 |
//! | [`mag`] | Algorithms 53/55: the MAG of a causal diagram over its observed variables |
//! | [`markov_equivalent`] | Prop. 16.1.1 |
//! | [`fci`] | Algorithm 54 (FCI), rules R1-R4, R8-R10 (Fig. 16.6) |
//! | [`augment`] | augmented diagrams (Def. 16.2.5, or Def. 16.2.11 with one F-node per pair) |
//! | [`i_markov_equivalent`], [`psi_markov_equivalent`] | Thms 16.2.6 and 16.2.8 |
//! | [`i_fci`] | Algorithms 56-58 (I-FCI, known targets), with R11 and R12 |
//! | [`psi_fci`], [`psi_pc`] | Algorithm 59 (Ψ-FCI) and Algorithm 60 (Ψ-PC), unknown targets |
//! | [`augment_selection`], [`md_markov_equivalent`] | chapter 17: augmented selection diagrams (Def. 17.2.4), MD-Markov equivalence (Thm 17.2.5) |
//! | [`md_pc`], [`md_fci`] | Algorithms 61 and 64 (MD-PC, MD-FCI), with Algorithms 62-63; [`SelectionOracle`] answers by Prop. 17.2.4 |
//! | [`Independence`], [`Invariance`] | the constraint tests the learners ask: [`GraphOracle`] and [`AugmentedOracle`] answer from a diagram, [`DataTests`] from data |
//!
//! The learners take any [`Independence`] (FCI) or [`Invariance`] (I-FCI,
//! Ψ-FCI, Ψ-PC) implementation. An oracle built from the true diagram makes
//! the output exactly the equivalence class. [`DataTests`] runs G-tests on
//! samples, or compares exact tables within a tolerance.
//!
//! # Readings
//!
//! * **Targets carry mechanism indices.** A target is a list of entries
//!   `"X"` or `"X:2"`, the second meaning mechanism 2 for `X`. The symmetric
//!   difference (Def. 16.2.2) keeps a variable when exactly one target
//!   has it, or both do with different mechanisms (Ex. 16.9).
//! * **Phase I searches every subset**, as Alg. 54 prints it, smallest
//!   first, so separating sets are the first found in that order.
//! * **Ψ-PC's orientation rules are Meek's R1-R3 (Fig. 16.17).** Colliders
//!   are oriented with tails, and undirected edges are written `o-o`.
//!
//! Figures 16.5, 16.12, 16.14 and 16.16 give no edge lists. The tests use
//! diagrams reconstructed from the worked examples, and say so. For Fig.
//! 16.14(a), Ex. 16.24 needs `Z *-> X` (the R4 path `⟨F1, X, Z, Y⟩`) and `X`
//! an ancestor of `W` (`F1` adjacent to `W` through `⟨F1, X, U, W⟩`).
//! Together they force `Z o-> W` in the Ψ-PAG, an edge the narration does
//! not mention.
//!
//! Not implemented: the score-based and parametric learners the
//! bibliographical notes mention, and rules R5-R7, which only apply under
//! selection bias (which the book also sets aside).

use std::fmt;

use crate::graph::dseparation::DSepError;

mod fci;
mod interventional;
mod mag;
mod multidomain;
mod oracle;
mod pag;
#[cfg(test)]
mod tests;

pub use fci::fci;
pub use interventional::{
    FNode, augment, i_fci, i_markov_equivalent, psi_fci, psi_markov_equivalent, psi_pc,
};
pub use mag::{mag, markov_equivalent};
pub use multidomain::{
    Regime, SNode, SelectionOracle, augment_selection, md_fci, md_markov_equivalent, md_pc,
};
pub use oracle::{AugmentedOracle, DataTests, GraphOracle, Independence, Invariance};
pub use pag::{Mark, Pag};

/// What can go wrong when learning a structure.
#[derive(Clone, Debug, PartialEq)]
pub enum DiscoveryError {
    /// A graph operation failed.
    Graph(String),
    /// A name is not a variable of the graph or the data.
    UnknownVariable(String),
    /// A variable's name clashes with an F-node's (`F1`, `F2`, …).
    Reserved(String),
    /// The two graphs are not over the same variables.
    Mismatch,
    /// The data are malformed.
    Data(String),
    /// An input is out of range.
    InvalidInput(String),
}

impl From<DSepError> for DiscoveryError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e.to_string())
    }
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => f.write_str(e),
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::Reserved(v) => write!(f, "{v} is reserved for an F-node"),
            Self::Mismatch => f.write_str("the graphs are over different variables"),
            Self::Data(s) => write!(f, "bad data: {s}"),
            Self::InvalidInput(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for DiscoveryError {}
