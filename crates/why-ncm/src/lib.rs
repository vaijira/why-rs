//! Neural causal models trained with [`burn`]: chapter 13 of the Causal AI
//! book, "Causal Generative Models", Secs. 13.1.3, 13.2.3 and 13.3.
//!
//! [`why_data::ncm`] has the structure of a G-constrained NCM
//! (Def. 13.1.11), the worlds a counterfactual query needs, and NCMs built by
//! hand from step units. This crate learns NCMs from data:
//!
//! | Item | Book |
//! | --- | --- |
//! | [`Ncm`] | a G-NCM with one MLP per variable (Def. 13.1.11, Eq. 13.141) |
//! | [`Ncm::log_likelihood`], [`Ncm::estimate`] | Eqs. 13.142–13.147 (explicit) or Eq. 13.151 (implicit) |
//! | [`Ncm::sample`] | Algorithm 48 (counterfactual sampling by rejection) |
//! | [`fit`] | train an NCM on data alone (line 3 of Algorithm 45) |
//! | [`neural_id`] | Algorithm 44 (`L2` query from `P(V)`) and Algorithm 46 (NeuralID), training as Algorithm 47 ([`Kind::Explicit`]) or Algorithm 49 ([`Kind::Implicit`]) |
//! | [`symbolic_then_neural`] | Algorithm 45: decide with C-INFER, estimate with an NCM |
//! | [`Ncm::distribution`] | the fitted `P̂(V)` |
//! | [`ancm`], [`Ancm`] | Algorithm 50 (chapter 14): a G-NCM that generates labelled images, trained as a VAE; samples and edits of `P̂(I, I_{x′})` |
//!
//! # The two implementations
//!
//! Both kinds share one network per variable, `φ_V(pa_V, û_V)`. It reads
//! one-hot parents and `noise_dim` uniforms per C²-component of `V`, and
//! outputs logits over `V`'s levels plus one gate logit.
//!
//! * [`Kind::Explicit`] (MLE-NCM, Sec. 13.3.1) reads the logits as
//!   `P(V = v | pa_V, û_V)`. Probabilities come from averaging over draws of
//!   `Û` (Eq. 13.142) with the worlds enumerated (Eq. 13.145). Training
//!   maximizes the likelihood while pushing the query up or down
//!   (Eq. 13.148).
//! * [`Kind::Implicit`] (Sec. 13.3.2) samples `V = argmax(φ_V + G_V)` with
//!   the Gumbel noise of Eq. 13.141 and never computes a probability.
//!   Training matches samples to the data by maximum mean discrepancy and
//!   pushes the query with the log losses of Table 13.6. Gradients pass the
//!   argmax through the Gumbel-softmax (straight-through).
//!
//! **Use the explicit model to decide identifiability.** It reproduces the
//! book's results on every example here. The implicit model fits Markovian
//! models and decides their (always identifiable) effects. With bidirected
//! edges it fits `P(V)` only to a few points per cell, because MMD with
//! straight-through gradients barely teaches the networks to route
//! confounding through `Û_C`. On the front-door graph of Ex. 13.10 it then
//! returns about 0.40 where the truth is 0.469. The book also finds the
//! GAN-NCM less stable (Sec. 13.3.3).
//!
//! # Departures from the text
//!
//! * **Worlds share each variable's own noise, not just `Û_C`.** Eq. 13.141
//!   adds a Gumbel `G_{V=v}` per variable to sample from `σ(φ_V)`. That
//!   noise is exogenous, so it must be the same in every world. Eq. 13.146
//!   instead multiplies `σ_{v_ij}` across worlds as if each world drew a
//!   fresh one. That breaks the consistency axiom: with `X = x` observed,
//!   the product gives `P(Y_x ≠ Y | X = x) > 0`. The implicit model shares
//!   `G_V` across worlds. The explicit one takes `V` as the inverse CDF of
//!   `σ(φ_V)` at a shared `ε_V ~ Unif(0, 1)`. It reads the levels in reverse
//!   in the worlds where a shared `δ_V ~ Unif(0, 1)` falls below the gate
//!   `g_V = sigmoid(·)`. Within a world nothing changes, since
//!   `P(V = v | pa, û)` is still `σ_v`. Across worlds the joint is a sum,
//!   over which worlds reverse, of overlaps of CDF intervals. That sum is
//!   exact and differentiable.
//! * **Why the gate.** Without it, a binary variable is comonotone across
//!   worlds given `û`. A high probability of sufficiency then needs `φ_V` to
//!   be a near step function of `û`, which training does not find: on
//!   Ex. 13.16 the maximizing model stalled at PS ≈ 0.04 where 1 is
//!   attainable. With the gate the same comparison spans [0, 1]. The gate
//!   adds an independent per-variable noise, so the model stays a
//!   G-constrained SCM.
//! * **While training, an empty cross-world overlap passes a small
//!   gradient** (a straight-through leak), and the query's log loss is not
//!   clamped. Both would otherwise cut the gradient exactly when a model
//!   starts with the query at 0. Reported values are always exact.
//! * **The implicit divergence is MMD, not a GAN.** Eq. 13.154 allows any
//!   divergence `D_P` and names MMD. The loss is MMD itself (an unbiased
//!   estimate, Gaussian kernels), not its square. The square is flat at the
//!   optimum, so the query term would win there however small `λ` got.
//! * **The estimate is the midpoint** of the two models' values. Algorithm 44
//!   returns either one; when the query is identified they agree up to `τ`.
//! * **Training is full-batch over the distinct rows**, each weighted by its
//!   frequency. Algorithm 47 loops over data points; the gradient is the
//!   same in expectation, and an exact distribution is just a weighted
//!   table. The learning rate decays to a tenth over training.
//!
//! Not implemented: continuous variables, the GAN discriminator of the
//! GAN-NCM, and the empirical tuning of `τ` (Sec. 13.3.1.2).

use std::fmt;

use why_data::fairness::Dataset;
use why_data::ncm::NcmError as StructureError;

mod ancm;
mod data;
mod explicit;
mod identify;
mod implicit;
mod model;
#[cfg(test)]
mod tests;

pub use ancm::{Ancm, AncmConfig, Edit, ImageData, ancm};
pub use identify::{NeuralIdResult, Objective, fit, neural_id, symbolic_then_neural};
pub use model::Ncm;

/// What can go wrong when building, training or querying an NCM.
#[derive(Clone, Debug, PartialEq)]
pub enum NcmError {
    /// The graph, the query or the model is malformed (see
    /// [`why_data::ncm::NcmError`]).
    Structure(StructureError),
    /// A data set lacks a column for some variable of the graph.
    MissingColumn(String),
    /// The query or a data set uses a level the model was not built with.
    UnknownLevel {
        /// The variable.
        var: String,
        /// The level.
        level: String,
    },
    /// The query names a variable the graph does not have.
    UnknownVariable(String),
    /// No data were given, or a data set has no rows.
    NoData,
    /// The conditioning event has probability zero under the model.
    ImpossibleCondition,
    /// Enumerating the query's worlds needs more cells than
    /// [`NcmConfig::max_cells`].
    TooLarge(usize),
    /// Images that do not match their labels or one another.
    BadImages(String),
}

impl From<StructureError> for NcmError {
    fn from(e: StructureError) -> Self {
        match e {
            StructureError::ImpossibleCondition => Self::ImpossibleCondition,
            e => Self::Structure(e),
        }
    }
}

impl From<why_data::graph::dseparation::DSepError> for NcmError {
    fn from(e: why_data::graph::dseparation::DSepError) -> Self {
        Self::Structure(e.into())
    }
}

impl fmt::Display for NcmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structure(e) => e.fmt(f),
            Self::MissingColumn(v) => write!(f, "a data set has no column for {v}"),
            Self::UnknownLevel { var, level } => {
                write!(f, "{level} is not a level of {var} in the data")
            }
            Self::UnknownVariable(v) => write!(f, "unknown variable {v}"),
            Self::NoData => f.write_str("no data"),
            Self::ImpossibleCondition => f.write_str("the conditioning event has probability zero"),
            Self::TooLarge(n) => write!(f, "the query needs more than {n} cells"),
            Self::BadImages(s) => write!(f, "bad images: {s}"),
        }
    }
}

impl std::error::Error for NcmError {}

/// How an NCM's networks are read (Sec. 13.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The MLE-NCM: networks output probabilities (Sec. 13.3.1,
    /// Algorithm 47).
    Explicit,
    /// Networks output samples (Def. 13.1.11 as it stands; Sec. 13.3.2,
    /// Algorithm 49).
    Implicit,
}

/// Architecture and training settings.
#[derive(Clone, Debug)]
pub struct NcmConfig {
    /// Explicit or implicit.
    pub kind: Kind,
    /// Width of each hidden layer.
    pub hidden: usize,
    /// Number of hidden layers per network.
    pub layers: usize,
    /// Number of uniforms in each `Û_C`.
    pub noise_dim: usize,
    /// Training epochs `T`.
    pub epochs: usize,
    /// Adam's learning rate `η` at the first epoch. It decays geometrically
    /// to a tenth of this by the last.
    pub learning_rate: f64,
    /// `λ` at the first and at the last epoch. It decays geometrically
    /// between them (Sec. 13.3.3.1 uses 1 → 0.001).
    pub lambda: (f64, f64),
    /// Monte-Carlo draws of `Û` per step (`m`).
    pub samples: usize,
    /// Draws of `Û` for the final estimates.
    pub eval_samples: usize,
    /// The threshold `τ` of Eq. 13.149.
    pub tolerance: f64,
    /// Softmax temperature of the straight-through gradient
    /// ([`Kind::Implicit`] only).
    pub temperature: f64,
    /// The most cells an explicit query may enumerate.
    pub max_cells: usize,
    /// Seeds parameter initialization and every draw of `Û`.
    pub seed: u64,
}

impl Default for NcmConfig {
    fn default() -> Self {
        Self {
            kind: Kind::Explicit,
            hidden: 32,
            layers: 2,
            noise_dim: 2,
            epochs: 1500,
            learning_rate: 0.01,
            lambda: (1.0, 0.001),
            samples: 256,
            eval_samples: 8192,
            tolerance: 0.05,
            temperature: 1.0,
            max_cells: 1 << 16,
            seed: 0,
        }
    }
}

/// One available distribution `P(V_z)` (Def. 13.2.4): data from the natural
/// regime when `intervened` is empty, else from an experiment that set the
/// variables of `intervened`. The rows record the values they were set to,
/// and may mix several (a randomized trial).
#[derive(Clone, Debug)]
pub struct Experiment {
    /// The intervened variables `Z_k`.
    pub intervened: Vec<String>,
    /// Rows over every variable of the graph. Row weights, if any, are
    /// frequencies, so an exact distribution is a weighted table.
    pub data: Dataset,
}

impl Experiment {
    /// Observational data `P(V)`.
    #[must_use]
    pub fn observational(data: Dataset) -> Self {
        Self {
            intervened: Vec::new(),
            data,
        }
    }

    /// Data from an experiment on `z`, `P(V_z)`.
    #[must_use]
    pub fn interventional(z: &[&str], data: Dataset) -> Self {
        Self {
            intervened: z.iter().map(|s| (*s).to_string()).collect(),
            data,
        }
    }
}

/// SplitMix64, for reproducible draws of `Û` without a dependency.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
