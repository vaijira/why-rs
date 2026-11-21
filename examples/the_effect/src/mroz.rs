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
    // mask as it stands. `causaldata` currently returns a Polars 0.54 frame
    // while `burn-dataset` 0.22 expects Polars 0.55, so filter the rows and
    // rebuild the columns in Polars 0.55.
    let df = mroz::load()?;
    let lfp: Vec<bool> = df.column("lfp")?.bool()?.iter().flatten().collect();
    let k5: Vec<i64> = df.column("k5")?.i64()?.into_no_null_iter().collect();
    let k618: Vec<i64> = df.column("k618")?.i64()?.into_no_null_iter().collect();
    let age: Vec<i64> = df.column("age")?.i64()?.into_no_null_iter().collect();
    let wc: Vec<bool> = df.column("wc")?.bool()?.iter().flatten().collect();
    let hc: Vec<bool> = df.column("hc")?.bool()?.iter().flatten().collect();
    let lwg: Vec<f64> = df.column("lwg")?.f64()?.into_no_null_iter().collect();
    let inc: Vec<f64> = df.column("inc")?.f64()?.into_no_null_iter().collect();

    let keep: Vec<usize> = (0..df.height())
        .filter(|&i| lfp[i] && inc[i] > 0.0)
        .collect();
    let pick_bool = |src: &[bool]| -> Vec<bool> { keep.iter().map(|&i| src[i]).collect() };
    let pick_i64 = |src: &[i64]| -> Vec<i64> { keep.iter().map(|&i| src[i]).collect() };
    let pick_f64 = |src: &[f64]| -> Vec<f64> { keep.iter().map(|&i| src[i]).collect() };

    let filtered = DataFrame::new(
        keep.len(),
        vec![
            Column::new("lfp".into(), pick_bool(&lfp)),
            Column::new("k5".into(), pick_i64(&k5)),
            Column::new("k618".into(), pick_i64(&k618)),
            Column::new("age".into(), pick_i64(&age)),
            Column::new("wc".into(), pick_bool(&wc)),
            Column::new("hc".into(), pick_bool(&hc)),
            Column::new("lwg".into(), pick_f64(&lwg)),
            Column::new("inc".into(), pick_f64(&inc)),
        ],
    )?;

    Ok(MrozDataset::new(filtered)?)
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
