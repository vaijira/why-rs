//! Data-driven transportability (Sec. 11.2.3): causal mechanistic stability
//! and the source invariance property.

use std::collections::{BTreeMap, BTreeSet};

use super::{Dataset, TransportError, subsets};

/// Assumption 11.2.12, causal mechanistic stability: a mechanism invariant
/// across every pair of sources is invariant in the target too, that is,
/// `∪_k Δ^{*,k} ⊆ ∪_{i,j} Δ^{i,j}`.
///
/// `between_sources` lists the `Δ^{i,j}` and `with_target` the `Δ^{*,k}`.
/// The assumption is about the unobserved models, so this only checks a
/// hypothesised set of discrepancies against it.
#[must_use]
pub fn cms_holds(between_sources: &[&[&str]], with_target: &[&[&str]]) -> bool {
    let sources: BTreeSet<&str> = between_sources.iter().flat_map(|d| d.iter().copied()).collect();
    with_target
        .iter()
        .flat_map(|d| d.iter())
        .all(|v| sources.contains(v))
}

/// The empirical scores `E_{P^i}[Y | R = r]` of one value of a
/// representation.
#[derive(Clone, Debug)]
pub struct LevelScores {
    /// The value `r`.
    pub level: String,
    /// `E_{P^i}[Y | R = r]` per source, `None` where the source never shows
    /// `r`.
    pub means: Vec<Option<f64>>,
    /// The (weighted) number of rows with `R = r` per source.
    pub counts: Vec<f64>,
}

/// The source invariance property (Def. 11.2.11) measured on data.
#[derive(Clone, Debug)]
pub struct InvarianceReport {
    /// The scores per value of `R`.
    pub levels: Vec<LevelScores>,
    /// The largest difference between two sources' scores at any `r`.
    pub max_gap: f64,
    /// The homogeneity statistic `Σ_r Σ_i n_i (m_i − m̄_r)² / v_r`, with `v_r`
    /// the pooled variance of `Y` given `R = r`. For binary `Y` it is the sum
    /// of Pearson's χ² statistics of each level.
    pub statistic: f64,
    /// Its degrees of freedom, `Σ_r (K_r − 1)` over the `K_r` sources that
    /// show `r`.
    pub df: usize,
    /// The asymptotic p-value of the χ² test of equal scores.
    pub p_value: f64,
    /// Values of `R` some source never shows. The book requires coverage,
    /// `P^i(r) > 0` in every source, before any of this means anything.
    pub uncovered: Vec<String>,
}

impl InvarianceReport {
    /// Whether the data are consistent with source invariance at level
    /// `alpha`: full coverage and a p-value of at least `alpha`.
    #[must_use]
    pub fn is_invariant(&self, alpha: f64) -> bool {
        self.uncovered.is_empty() && self.p_value >= alpha
    }
}

/// Measures the source invariance property `E_{P^i}[Y | r] = E_{P^j}[Y | r]`
/// (Def. 11.2.11) for the representation that maps a row to the values of
/// `features`; `Y` must be numeric.
///
/// Under CMS ([`cms_holds`]) and r-faithfulness (Assumption 11.2.15), the
/// score of a representation is transportable to the target exactly when it
/// has this property (Thm 11.2.16). With an invariance property, the minimal
/// invariant feature set of Thm 11.2.9 is the first one
/// [`invariant_feature_sets`] returns.
///
/// # Errors
///
/// [`TransportError::InvalidInput`] for fewer than two sources, a missing
/// column, or a non-numeric outcome.
pub fn source_invariance(
    datasets: &[&Dataset],
    features: &[&str],
    y: &str,
) -> Result<InvarianceReport, TransportError> {
    source_invariance_by(datasets, y, |d, row| {
        features
            .iter()
            .map(|f| d.value(f, row).map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .map(|v| v.join(","))
    })
}

/// As [`source_invariance`], for an arbitrary representation `phi`, which
/// maps a row to the value of `R` (or `None` if the row lacks a column).
///
/// # Errors
///
/// As [`source_invariance`].
pub fn source_invariance_by(
    datasets: &[&Dataset],
    y: &str,
    phi: impl Fn(&Dataset, usize) -> Option<String>,
) -> Result<InvarianceReport, TransportError> {
    if datasets.len() < 2 {
        return Err(TransportError::InvalidInput(
            "source invariance needs at least two sources".into(),
        ));
    }
    let k = datasets.len();
    // Per level, per source: (Σw, Σw·y, Σw·y²).
    let mut acc: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();
    for (i, d) in datasets.iter().enumerate() {
        for row in 0..d.len() {
            let level = phi(d, row).ok_or_else(|| {
                TransportError::InvalidInput(format!("source {i} lacks a feature column"))
            })?;
            let raw = d
                .value(y, row)
                .ok_or_else(|| TransportError::InvalidInput(format!("source {i} lacks `{y}`")))?;
            let v: f64 = raw.trim().parse().map_err(|_| {
                TransportError::InvalidInput(format!("`{y}` value `{raw}` is not numeric"))
            })?;
            let w = d.weight(row);
            let e = &mut acc.entry(level).or_insert_with(|| vec![(0.0, 0.0, 0.0); k])[i];
            e.0 += w;
            e.1 += w * v;
            e.2 += w * v * v;
        }
    }
    let mut levels = Vec::new();
    let mut uncovered = Vec::new();
    let mut max_gap: f64 = 0.0;
    let mut statistic = 0.0;
    let mut df = 0;
    for (level, per) in acc {
        let means: Vec<Option<f64>> = per
            .iter()
            .map(|(n, s, _)| (*n > 0.0).then(|| s / n))
            .collect();
        if means.iter().any(Option::is_none) {
            uncovered.push(level.clone());
        }
        let present: Vec<f64> = means.iter().flatten().copied().collect();
        if present.len() >= 2 {
            let hi = present.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lo = present.iter().copied().fold(f64::INFINITY, f64::min);
            max_gap = max_gap.max(hi - lo);
            let n: f64 = per.iter().map(|p| p.0).sum();
            let mean = per.iter().map(|p| p.1).sum::<f64>() / n;
            let var = per.iter().map(|p| p.2).sum::<f64>() / n - mean * mean;
            if var > 1e-12 {
                statistic += per
                    .iter()
                    .filter(|p| p.0 > 0.0)
                    .map(|(ni, si, _)| ni * (si / ni - mean).powi(2))
                    .sum::<f64>()
                    / var;
            }
            df += present.len() - 1;
        }
        levels.push(LevelScores {
            level,
            means,
            counts: per.iter().map(|p| p.0).collect(),
        });
    }
    let p_value = if df == 0 {
        1.0
    } else {
        chi2_survival(statistic, df as f64)
    };
    Ok(InvarianceReport {
        levels,
        max_gap,
        statistic,
        df,
        p_value,
        uncovered,
    })
}

/// Every subset of `features` whose representation passes
/// [`InvarianceReport::is_invariant`] at level `alpha`, smallest first, with
/// its report.
///
/// # Errors
///
/// [`TransportError::TooLarge`] beyond 20 features, and whatever
/// [`source_invariance`] reports.
pub fn invariant_feature_sets(
    datasets: &[&Dataset],
    features: &[&str],
    y: &str,
    alpha: f64,
) -> Result<Vec<(Vec<String>, InvarianceReport)>, TransportError> {
    let mut out = Vec::new();
    for s in subsets(features)? {
        let report = source_invariance(datasets, &s, y)?;
        if report.is_invariant(alpha) {
            out.push((s.iter().map(|v| (*v).to_string()).collect(), report));
        }
    }
    Ok(out)
}

/// `P(χ²_df > x)`, the regularised upper incomplete gamma `Q(df/2, x/2)`.
pub(crate) fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos, g = 7, n = 9.
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularised upper incomplete gamma: a series below `a + 1`, a continued
/// fraction above (Numerical Recipes, §6.2).
fn gamma_q(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut del = sum;
        let mut ap = a;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}
