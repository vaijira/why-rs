//! Causal fairness analysis on tabular data (chapter 6, Plečko & Bareinboim).
//!
//! Given a dataset and its projection onto the *Standard Fairness Model*
//! ([`Sfm`]: protected attribute `X`, confounders `Z`, mediators `W`, outcome
//! `Y`), this module estimates the `x`-specific decomposition of the total
//! variation
//!
//! ```text
//! TV_{x0,x1}(y) = x-DE_{x0,x1}(y | x0) − x-IE_{x1,x0}(y | x0) − x-SE_{x1,x0}(y)
//! ```
//!
//! with the non-parametric plug-in estimands the SFM identifies:
//!
//! ```text
//! x-DE = Σ_{z,w} [E(y | x1,z,w) − E(y | x0,z,w)] P(w | x0,z) P(z | x0)
//! x-IE = Σ_{z,w}  E(y | x1,z,w) [P(w | x0,z) − P(w | x1,z)] P(z | x0)
//! x-SE = Σ_z      E(y | x1,z)   [P(z | x0)   − P(z | x1)]
//! ```
//!
//! # Sign convention
//!
//! These are the paper's definitions, as counterfactual contrasts:
//!
//! * `x-DE = E[Y_{x1,W_{x0}} | x0] − E[Y_{x0} | x0]`
//! * `x-IE = E[Y_{x1,W_{x0}} | x0] − E[Y_{x1} | x0]`
//! * `x-SE = E[Y_{x1} | x0]        − E[Y_{x1} | x1]`
//!
//! Some summaries of the chapter print x-IE and x-SE with the brackets the
//! other way round. Those versions satisfy `TV = DE + IE + SE`, not the
//! identity they are quoted next to. With the signs used here the
//! decomposition is exact on any dataset where positivity holds, and the tests
//! check that.
//!
//! # Data
//!
//! `X`, `Z` and `W` are treated as categorical: `Z` and `W` may span several
//! columns and are summed over their joint levels. That keeps the estimator
//! fully non-parametric, but it needs every `(z, w)` stratum seen under `x0`
//! to also be seen under `x1`. If one is missing,
//! [`FairnessError::Positivity`] names it. `Y` must be numeric, so a binary
//! outcome coded `0`/`1` gives probabilities, and a real-valued prediction
//! `Ŷ` gives expectations.
//!
//! # The three fairness tasks (Sec. 6.1.4)
//!
//! * **Task 1, bias detection and quantification.** [`Cookbook`] is
//!   Algorithm 15. Besides the x-specific measures above, it uses the
//!   symmetric x-DE/x-IE ([`Measures::de_sym`], Def. 6.1.32) and z-specific
//!   follow-up tests ([`z_specific_measures`]). [`general_measures`] gives
//!   TE, NDE, NIE and Exp-SE.
//! * **Task 2, fair prediction.** [`FairPredictor`] is in-processing with
//!   causal constraints (Thm 6.1.32), solved exactly. [`CausalIf`] is
//!   Algorithm 16, sequential optimal transport.
//! * **Task 3, fair decision-making.** [`benefit_fairness`] is Algorithm 17
//!   and [`utilitarian_benefit_fairness`] is Algorithm 18, both driven by the
//!   benefit `Δ` from [`estimate_benefit`]. [`canonical_type_bounds`] is
//!   Prop. 6.1.40.

mod cookbook;
mod dataset;
mod decision;
mod estimate;
mod predict;
#[cfg(test)]
mod tests;
mod transport;

use std::fmt;

pub use cookbook::{
    BootstrapConfig, BusinessNecessity, Cookbook, FairnessReport, MeasureTest, SubgroupTest,
    Verdict,
};
pub use dataset::Dataset;
pub use decision::{
    BenefitPolicy, CanonicalBounds, benefit_fairness, canonical_type_bounds, estimate_benefit,
    utilitarian_benefit_fairness,
};
pub use estimate::{
    GeneralMeasures, Measures, Sfm, ZEffects, ZMeasures, estimate, general_measures,
    z_specific_measures,
};
pub use predict::{Constraints, FairPredictor};
pub use transport::CausalIf;

/// Everything that can go wrong while loading data or estimating measures.
#[derive(Debug, Clone, PartialEq)]
pub enum FairnessError {
    /// Malformed CSV input.
    Csv {
        /// 1-based line number.
        line: usize,
        /// What went wrong.
        message: String,
    },
    /// Two columns share a name.
    DuplicateColumn(String),
    /// A record has a different number of fields than the header.
    RaggedRow {
        /// 1-based record number, header excluded.
        row: usize,
        /// Number of columns in the header.
        expected: usize,
        /// Number of fields found.
        found: usize,
    },
    /// A column named in the SFM does not exist.
    UnknownColumn(String),
    /// A column was given more than one SFM role.
    OverlappingRoles(String),
    /// A requested level of `X` never occurs.
    UnknownLevel {
        /// The column searched.
        column: String,
        /// The missing level.
        level: String,
    },
    /// `x0` and `x1` are the same level.
    IdenticalContrast(String),
    /// The outcome column holds a value that is not a finite number.
    NonNumericOutcome {
        /// The outcome column.
        column: String,
        /// The offending value.
        value: String,
    },
    /// Weights are the wrong length, negative or not finite.
    InvalidWeights(String),
    /// No (positively weighted) rows have `X` at this level.
    EmptyGroup {
        /// The empty level.
        level: String,
    },
    /// A stratum with support under `x0` has none under `x1`, so the plug-in
    /// estimand is undefined.
    Positivity {
        /// The level of `X` without support (always `x1`).
        level: String,
        /// The stratum, as `Z` values or `Z; W` values.
        stratum: String,
    },
    /// The bootstrap could not produce enough valid replicates.
    Bootstrap(String),
    /// A predictor was asked about an `(x, z, w)` cell it was not fitted on.
    UnseenCell(String),
    /// An argument is out of range or inconsistent.
    InvalidInput(String),
}

impl fmt::Display for FairnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Csv { line, message } => write!(f, "CSV line {line}: {message}"),
            Self::DuplicateColumn(c) => write!(f, "duplicate column `{c}`"),
            Self::RaggedRow {
                row,
                expected,
                found,
            } => write!(f, "record {row} has {found} fields, expected {expected}"),
            Self::UnknownColumn(c) => write!(f, "no column named `{c}`"),
            Self::OverlappingRoles(c) => {
                write!(f, "column `{c}` is given more than one SFM role")
            }
            Self::UnknownLevel { column, level } => {
                write!(f, "level `{level}` never occurs in column `{column}`")
            }
            Self::IdenticalContrast(l) => write!(f, "x0 and x1 are both `{l}`"),
            Self::NonNumericOutcome { column, value } => {
                write!(f, "outcome column `{column}` holds non-numeric value `{value}`")
            }
            Self::InvalidWeights(m) => write!(f, "invalid weights: {m}"),
            Self::EmptyGroup { level } => write!(f, "no rows with X = `{level}`"),
            Self::Positivity { level, stratum } => write!(
                f,
                "positivity violated: no rows with X = `{level}` in stratum [{stratum}], \
                 which occurs under the other level"
            ),
            Self::Bootstrap(m) => write!(f, "bootstrap failed: {m}"),
            Self::UnseenCell(m) => write!(f, "cell not seen in training: {m}"),
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for FairnessError {}

/// SplitMix64: small, fast and dependency-free. The bootstrap only needs
/// reproducible uniform draws, not cryptographic quality.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
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

    /// Uniform in `0..n` (Lemire's multiply-shift; bias is below 2⁻⁶⁴·n).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
