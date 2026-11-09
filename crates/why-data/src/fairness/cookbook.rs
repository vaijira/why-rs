//! Algorithm 15, the Fairness Cookbook: bias detection and quantification.

use std::fmt;

use super::estimate::{Measures, Prepared, Sfm};
use super::{Dataset, FairnessError, SplitMix64};

/// Which pathways from `X` to `Y` are legally justified by business
/// necessity. The direct pathway never can be.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BusinessNecessity {
    /// The spurious pathway through the confounders `Z` is permitted.
    pub z: bool,
    /// The indirect pathway through the mediators `W` is permitted.
    pub w: bool,
}

impl BusinessNecessity {
    /// `∅`: every pathway must be fair.
    pub const NONE: Self = Self { z: false, w: false };
    /// `{Z}`.
    pub const Z: Self = Self { z: true, w: false };
    /// `{W}`.
    pub const W: Self = Self { z: false, w: true };
    /// `{Z, W}`.
    pub const ZW: Self = Self { z: true, w: true };
}

impl fmt::Display for BusinessNecessity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match (self.z, self.w) {
            (false, false) => "∅",
            (true, false) => "{Z}",
            (false, true) => "{W}",
            (true, true) => "{Z, W}",
        })
    }
}

/// Nonparametric bootstrap settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootstrapConfig {
    /// Number of resamples.
    pub replicates: usize,
    /// RNG seed. Runs with the same seed are reproducible.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            replicates: 500,
            seed: 0x5EED_F00D,
        }
    }
}

/// A point estimate with its bootstrap uncertainty.
///
/// The decision rule is the book's (Sec. 6.1.4.1): reject `H0: θ = 0` when the
/// percentile confidence interval excludes zero. The Wald p-value is reported
/// alongside for reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureTest {
    /// Plug-in estimate on the full data.
    pub estimate: f64,
    /// Bootstrap standard deviation.
    pub std_error: f64,
    /// Lower end of the percentile confidence interval.
    pub ci_low: f64,
    /// Upper end of the percentile confidence interval.
    pub ci_high: f64,
    /// Two-sided Wald p-value, `2 (1 − Φ(|estimate| / std_error))`.
    pub p_value: f64,
}

impl MeasureTest {
    fn from_samples(estimate: f64, samples: &mut [f64], alpha: f64) -> Self {
        let m = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / m;
        let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (m - 1.0);
        let std_error = var.sqrt();
        samples.sort_by(f64::total_cmp);
        let p_value = if std_error > 0.0 {
            erfc(estimate.abs() / std_error / std::f64::consts::SQRT_2)
        } else if estimate == 0.0 {
            1.0
        } else {
            0.0
        };
        Self {
            estimate,
            std_error,
            ci_low: quantile(samples, alpha / 2.0),
            ci_high: quantile(samples, 1.0 - alpha / 2.0),
            p_value,
        }
    }

    /// Whether the confidence interval excludes zero.
    pub fn rejects(&self) -> bool {
        self.ci_low > 0.0 || self.ci_high < 0.0
    }
}

/// A z-specific follow-up test in one confounder stratum.
#[derive(Debug, Clone, PartialEq)]
pub struct SubgroupTest {
    /// The stratum, as `name=value` pairs.
    pub stratum: String,
    /// `P(z)`.
    pub weight: f64,
    /// The test, with a Bonferroni-corrected interval across strata.
    pub test: MeasureTest,
}

/// Outcome of one cookbook check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// `H0` rejected in the whole population. Either the measure is non-zero,
    /// or, for a business-necessity pathway audited against a reference
    /// outcome, the prediction's effect differs from the reference's.
    Discrimination,
    /// `H0` not rejected overall, but rejected in at least one confounder
    /// stratum by the z-specific follow-up (Alg. 15, step 3/4 "additionally").
    SubgroupDiscrimination,
    /// `H0` not rejected, overall or in any stratum.
    NoEvidence,
    /// The pathway is in the business-necessity set and no reference outcome
    /// was given, so it was not tested.
    Permitted,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Discrimination => "DISCRIMINATION (H0 rejected)",
            Self::SubgroupDiscrimination => "DISCRIMINATION in a subgroup (z-specific H0 rejected)",
            Self::NoEvidence => "no evidence (H0 not rejected)",
            Self::Permitted => "permitted by business necessity (not tested)",
        })
    }
}

/// Algorithm 15. Estimates the symmetric x-DE and x-IE and the x-SE,
/// bootstraps them, and judges disparate treatment and disparate impact
/// against a business-necessity set.
///
/// The outcome audited is always `sfm.y`. To audit a predictor `Ŷ`, put the
/// prediction column in `sfm.y` and the true outcome in
/// [`Cookbook::with_reference`]. A pathway in the business-necessity set is
/// then not required to vanish. Instead, its effect on `Ŷ` is tested against
/// its effect on `Y` (Eqs. 6.310 and 6.312), using the same bootstrap
/// resamples so that the difference is paired.
///
/// When the population-level DE (or IE) test does not reject, the z-specific
/// effect is tested in every confounder stratum. Those intervals are
/// Bonferroni-corrected across strata; the three top-level tests are not
/// corrected against each other.
///
/// ```
/// use why_data::fairness::{BusinessNecessity, Cookbook, Dataset, Sfm, Verdict};
///
/// let mut csv = String::from("x,w,y\n");
/// for i in 0..400 {
///     // W depends on X and Y depends only on W: a purely indirect effect.
///     let x = i % 2;
///     let w = usize::from(i % 8 < 3 + 2 * x);
///     let y = usize::from(i % 3 < 1 + w);
///     csv.push_str(&format!("{x},{w},{y}\n"));
/// }
/// let data = Dataset::from_csv(&csv).unwrap();
/// let report = Cookbook::new(Sfm::new("x", Vec::<String>::new(), ["w"], "y"), "0", "1")
///     .with_business_necessity(BusinessNecessity::W)
///     .run(&data)
///     .unwrap();
/// assert_eq!(report.indirect, Verdict::Permitted);
/// ```
#[derive(Debug, Clone)]
pub struct Cookbook {
    /// SFM projection, with `sfm.y` the outcome audited.
    pub sfm: Sfm,
    /// Baseline level of `X`; the measures are specific to this group.
    pub x0: String,
    /// Comparison level of `X`.
    pub x1: String,
    /// Permitted pathways.
    pub business_necessity: BusinessNecessity,
    /// True-outcome column when `sfm.y` is a prediction.
    pub reference: Option<String>,
    /// Significance level for every interval.
    pub alpha: f64,
    /// Bootstrap settings.
    pub bootstrap: BootstrapConfig,
}

impl Cookbook {
    /// A cookbook with an empty business-necessity set, `alpha = 0.05` and the
    /// default bootstrap.
    pub fn new(sfm: Sfm, x0: impl Into<String>, x1: impl Into<String>) -> Self {
        Self {
            sfm,
            x0: x0.into(),
            x1: x1.into(),
            business_necessity: BusinessNecessity::NONE,
            reference: None,
            alpha: 0.05,
            bootstrap: BootstrapConfig::default(),
        }
    }

    /// Sets the business-necessity set.
    pub fn with_business_necessity(mut self, bn: BusinessNecessity) -> Self {
        self.business_necessity = bn;
        self
    }

    /// Names the true-outcome column, when `sfm.y` holds predictions.
    pub fn with_reference(mut self, column: impl Into<String>) -> Self {
        self.reference = Some(column.into());
        self
    }

    /// Sets the significance level.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the bootstrap configuration.
    pub fn with_bootstrap(mut self, bootstrap: BootstrapConfig) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Runs the audit.
    ///
    /// A bootstrap replicate that leaves a group empty, or breaks positivity in
    /// some stratum, is discarded and counted in
    /// [`FairnessReport::replicates_failed`]. Fewer than two usable
    /// replicates is an error. A stratum that breaks positivity only affects
    /// its own z-specific samples.
    pub fn run(&self, data: &Dataset) -> Result<FairnessReport, FairnessError> {
        let prep = Prepared::new(data, &self.sfm, &self.x0, &self.x1, &self.sfm.y)?;
        let ref_prep = self
            .reference
            .as_deref()
            .map(|r| Prepared::new(data, &self.sfm, &self.x0, &self.x1, r))
            .transpose()?;

        let t = prep.tables(None)?;
        let measures = prep.x_specific(&t)?;
        let z_point = prep.z_specific(&t);
        let reference = ref_prep.as_ref().map(|p| p.measures(None)).transpose()?;
        let subgroups = !self.sfm.z.is_empty();

        let n = prep.len();
        let nz = z_point.len();
        let mut rng = SplitMix64::new(self.bootstrap.seed);
        let mut counts = vec![0u32; n];
        let mut s: [Vec<f64>; 6] = Default::default();
        let mut s_zde = vec![Vec::new(); nz];
        let mut s_zie = vec![Vec::new(); nz];
        let mut failed = 0;

        for _ in 0..self.bootstrap.replicates {
            counts.fill(0);
            for _ in 0..n {
                counts[rng.below(n)] += 1;
            }
            let Ok(t) = prep.tables(Some(&counts)) else {
                failed += 1;
                continue;
            };
            let Ok(m) = prep.x_specific(&t) else {
                failed += 1;
                continue;
            };
            let r = match &ref_prep {
                Some(p) => match p.measures(Some(&counts)) {
                    Ok(r) => Some(r),
                    Err(_) => {
                        failed += 1;
                        continue;
                    }
                },
                None => None,
            };
            s[0].push(m.tv);
            s[1].push(m.de_sym);
            s[2].push(m.ie_sym);
            s[3].push(m.se);
            if let Some(r) = r {
                s[4].push(m.ie_sym - r.ie_sym);
                s[5].push(m.se - r.se);
            }
            if subgroups {
                for (z, zm) in prep.z_specific(&t).into_iter().enumerate() {
                    if let Some(e) = zm.effects {
                        s_zde[z].push(e.de_sym);
                        s_zie[z].push(e.ie_sym);
                    }
                }
            }
        }

        let used = s[0].len();
        if used < 2 {
            return Err(FairnessError::Bootstrap(format!(
                "only {used} of {} replicates were usable; strata are too thin for \
                 the plug-in estimator",
                self.bootstrap.replicates
            )));
        }

        let alpha = self.alpha;
        let [s_tv, s_de, s_ie, s_se, s_ie_x, s_se_x] = &mut s;
        let tv = MeasureTest::from_samples(measures.tv, s_tv, alpha);
        let de = MeasureTest::from_samples(measures.de_sym, s_de, alpha);
        let ie = MeasureTest::from_samples(measures.ie_sym, s_ie, alpha);
        let se = MeasureTest::from_samples(measures.se, s_se, alpha);
        let (ie_excess, se_excess) = match reference {
            Some(r) => (
                Some(MeasureTest::from_samples(measures.ie_sym - r.ie_sym, s_ie_x, alpha)),
                Some(MeasureTest::from_samples(measures.se - r.se, s_se_x, alpha)),
            ),
            None => (None, None),
        };

        let subgroup_tests = |samples: &mut [Vec<f64>], pick: fn(&super::ZEffects) -> f64| {
            let testable: Vec<usize> = (0..nz)
                .filter(|&z| z_point[z].effects.is_some() && samples[z].len() >= 2)
                .collect();
            let level = alpha / testable.len().max(1) as f64;
            testable
                .into_iter()
                .map(|z| SubgroupTest {
                    stratum: z_point[z].stratum.clone(),
                    weight: z_point[z].weight,
                    test: MeasureTest::from_samples(
                        pick(z_point[z].effects.as_ref().unwrap()),
                        &mut samples[z],
                        level,
                    ),
                })
                .collect::<Vec<_>>()
        };
        let subgroup_direct = subgroup_tests(&mut s_zde, |e| e.de_sym);
        let subgroup_indirect = subgroup_tests(&mut s_zie, |e| e.ie_sym);

        let judge = |overall: &MeasureTest, sub: &[SubgroupTest]| {
            if overall.rejects() {
                Verdict::Discrimination
            } else if sub.iter().any(|t| t.test.rejects()) {
                Verdict::SubgroupDiscrimination
            } else {
                Verdict::NoEvidence
            }
        };
        let excess = |t: &Option<MeasureTest>| match t {
            Some(t) => judge(t, &[]),
            None => Verdict::Permitted,
        };
        let bn = self.business_necessity;

        Ok(FairnessReport {
            x0: self.x0.clone(),
            x1: self.x1.clone(),
            disparate_treatment: judge(&de, &subgroup_direct),
            indirect: if bn.w {
                excess(&ie_excess)
            } else {
                judge(&ie, &subgroup_indirect)
            },
            spurious: if bn.z { excess(&se_excess) } else { judge(&se, &[]) },
            measures,
            reference,
            tv,
            de,
            ie,
            se,
            ie_excess,
            se_excess,
            subgroup_direct,
            subgroup_indirect,
            business_necessity: bn,
            alpha,
            replicates_used: used,
            replicates_failed: failed,
        })
    }
}

/// Result of [`Cookbook::run`]. Its `Display` impl prints a readable summary.
#[derive(Debug, Clone, PartialEq)]
pub struct FairnessReport {
    /// Baseline level of `X`.
    pub x0: String,
    /// Comparison level of `X`.
    pub x1: String,
    /// Point estimates for the audited outcome, including the asymmetric
    /// transitions.
    pub measures: Measures,
    /// Point estimates for the reference outcome, if one was given.
    pub reference: Option<Measures>,
    /// Total variation (not judged; shown for context).
    pub tv: MeasureTest,
    /// `x-DE^sym`.
    pub de: MeasureTest,
    /// `x-IE^sym`.
    pub ie: MeasureTest,
    /// `x-SE_{x1,x0}`.
    pub se: MeasureTest,
    /// `x-IE^sym(Ŷ) − x-IE^sym(Y)`, when a reference outcome was given.
    pub ie_excess: Option<MeasureTest>,
    /// `x-SE(Ŷ) − x-SE(Y)`, when a reference outcome was given.
    pub se_excess: Option<MeasureTest>,
    /// `z-DE^sym` per confounder stratum (empty when `Z = ∅`).
    pub subgroup_direct: Vec<SubgroupTest>,
    /// `z-IE^sym` per confounder stratum (empty when `Z = ∅`).
    pub subgroup_indirect: Vec<SubgroupTest>,
    /// Disparate treatment: x-DE^sym, then z-DE^sym.
    pub disparate_treatment: Verdict,
    /// Disparate impact through `W`.
    pub indirect: Verdict,
    /// Disparate impact through `Z`.
    pub spurious: Verdict,
    /// Business-necessity set used.
    pub business_necessity: BusinessNecessity,
    /// Significance level used.
    pub alpha: f64,
    /// Bootstrap replicates that produced estimates.
    pub replicates_used: usize,
    /// Bootstrap replicates discarded for empty groups or positivity.
    pub replicates_failed: usize,
}

impl fmt::Display for FairnessReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ci = (1.0 - self.alpha) * 100.0;
        writeln!(
            f,
            "Fairness Cookbook  x0 = {}, x1 = {}  (alpha = {}, {} replicates, {} discarded)",
            self.x0, self.x1, self.alpha, self.replicates_used, self.replicates_failed
        )?;
        let row = |f: &mut fmt::Formatter<'_>, name: &str, t: &MeasureTest| {
            writeln!(
                f,
                "  {name:<22} {:>+9.4}  se {:.4}  {ci:.0}% CI [{:+.4}, {:+.4}]{}",
                t.estimate,
                t.std_error,
                t.ci_low,
                t.ci_high,
                if t.rejects() { "  *" } else { "" }
            )
        };
        row(f, "TV", &self.tv)?;
        row(f, "x-DE^sym", &self.de)?;
        row(f, "x-IE^sym", &self.ie)?;
        row(f, "x-SE", &self.se)?;
        if let Some(t) = &self.ie_excess {
            row(f, "x-IE^sym(Ŷ − Y)", t)?;
        }
        if let Some(t) = &self.se_excess {
            row(f, "x-SE(Ŷ − Y)", t)?;
        }
        // Subgroup tests matter only where the population test found nothing.
        let follow_ups = [
            ("z-DE^sym", &self.subgroup_direct, &self.de),
            ("z-IE^sym", &self.subgroup_indirect, &self.ie),
        ];
        for (name, tests, overall) in follow_ups {
            if overall.rejects() {
                continue;
            }
            for t in tests.iter().filter(|t| t.test.rejects()) {
                row(f, &format!("{name} [{}]", t.stratum), &t.test)?;
            }
        }
        writeln!(f, "  BN-set {}", self.business_necessity)?;
        writeln!(f, "  Disparate treatment (DE):  {}", self.disparate_treatment)?;
        writeln!(f, "  Disparate impact (IE, W):  {}", self.indirect)?;
        write!(f, "  Disparate impact (SE, Z):  {}", self.spurious)
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let pos = p * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Complementary error function (Numerical Recipes `erfcc`; absolute error
/// below 1.2e-7, ample for a p-value).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * poly.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[cfg(test)]
mod erfc_tests {
    #[test]
    fn known_values() {
        assert!((super::erfc(0.0) - 1.0).abs() < 1e-6);
        // Φ(1.96) ≈ 0.975 ⇒ two-sided p ≈ 0.05.
        assert!((super::erfc(1.96 / std::f64::consts::SQRT_2) - 0.049_996).abs() < 1e-5);
        assert!((super::erfc(-1.0) - 1.842_700_79).abs() < 1e-6);
    }
}
