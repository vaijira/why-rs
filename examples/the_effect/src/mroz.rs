//! The `Mroz` data set as a `burn` dataset, straight from the polars frame
//! [`causaldata`] hands back.
//!
//! Burn reads a [`DataFrame`] natively through
//! [`DataframeDataset`](burn::data::dataset::DataframeDataset), which
//! deserializes one row at a time into a struct whose field names are the
//! column names -- so the frame goes to burn as-is, with no intermediate copy
//! of the data and no second schema to keep in step. All that is left here is
//! the row filter and the two regression specifications.

use burn::data::dataset::DataframeDataset;
use causaldata::datasets::mroz;
use polars::prelude::*;
use serde::Deserialize;

/// Married woman record: one row of the data set.
///
/// The field names are the column names `mroz::load` returns; the types are
/// what the tensor code wants, and the row deserializer converts on the way in
/// (`inc` is `f64` in the frame, the counts are `i64`).
#[derive(Deserialize, Debug, Clone)]
pub struct MarriedWoman {
    /// labor-force participation; a factor with levels: no; yes
    pub lfp: bool,

    /// number of children 5 years old or younger.
    pub k5: u8,

    /// number of children 6 to 18 years old.
    pub k618: u8,

    /// age in years.
    pub age: u8,

    /// wife's college attendance; a factor with levels: no; yes.
    pub wc: bool,

    /// husband's college attendance; a factor with levels: no; yes.
    pub hc: bool,

    /// log expected wage rate; for women in the labor force, the actual wage rate;
    /// for women not in the labor force, an imputed value based on the regression of lwg on the other variables.
    pub lwg: f32,

    /// family income exclusive of wife's income.
    pub inc: f32,
}

/// The Mroz data frame has 753 rows and 8 columns. The observations, from the
/// Panel Study of Income Dynamics (PSID), are married women.
pub type MrozDataset = DataframeDataset<MarriedWoman>;

/// The rows both regressions are fitted on.
///
/// Keeps just working women, and drops the single observation with a
/// non-positive `inc`: the regressor is log(inc), which is undefined there.
/// `statsmodels` does the same implicitly by dropping the NaN.
pub fn load() -> Result<MrozDataset, Box<dyn std::error::Error>> {
    // `load` has already mapped the yes/no columns to booleans, so `lfp` is a
    // mask as it stands.
    let df = mroz::load()?;
    let keep = df.column("lfp")?.bool()? & &df.column("inc")?.f64()?.gt(0.0);

    Ok(MrozDataset::new(df.filter(&keep)?)?)
}

/// Which of the two regressions from the chapter to fit. The outcome is `lwg`
/// in both cases; these name the right-hand side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Spec {
    /// `lwg ~ linc`
    #[default]
    M1,
    /// `lwg ~ linc + wc + k5`
    M2,
}

impl Spec {
    /// Regressor names in column order, excluding the intercept.
    pub fn regressors(&self) -> &'static [&'static str] {
        match self {
            Spec::M1 => &["linc"],
            Spec::M2 => &["linc", "wc", "k5"],
        }
    }

    /// How many columns the design matrix has, intercept aside.
    pub fn num_features(&self) -> usize {
        self.regressors().len()
    }

    /// Regressor values for one observation, in `regressors()` order.
    ///
    /// `wc` is a yes/no factor, so it enters as a 0/1 dummy exactly like
    /// statsmodels' `wc[T.yes]`; `k5` is a count and enters as-is.
    pub fn row(&self, w: &MarriedWoman) -> Vec<f32> {
        let linc = w.inc.ln();

        match self {
            Spec::M1 => vec![linc],
            Spec::M2 => vec![linc, if w.wc { 1.0 } else { 0.0 }, w.k5 as f32],
        }
    }
}
