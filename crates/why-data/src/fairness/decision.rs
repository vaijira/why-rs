//! Task 3, fair decision-making in the outcome-control setting
//! (Sec. 6.1.4.4–6.1.4.6): benefit fairness (Algorithm 17), the utilitarian
//! gap control (Algorithm 18) and the canonical-type bounds (Prop. 6.1.40).
//!
//! Every policy is driven by the degree of benefit
//! `Δ(x, z, w) = E[Y_{d1} − Y_{d0} | x, z, w]` (Def. 6.1.38). It can be
//! estimated from historical decisions with [`estimate_benefit`], or supplied
//! from elsewhere (a causal model, a trial, or a *fair* outcome model for
//! Causal Benefit Fairness, Def. 6.1.42).
//!
//! # Harmful units
//!
//! Prop. 6.1.41 states Algorithm 17 is optimal for `max E[Y_d]` subject to
//! `P(d) ≤ b`. Treating a unit with `Δ < 0` lowers `E[Y_d]`, so these
//! functions never treat units with `Δ ≤ 0`, even when budget is left over.
//! Algorithm 18's budget equation (6.493) is likewise relaxed to `≤ b`. When
//! every `Δ` is positive this changes nothing.

use std::collections::HashMap;

use super::estimate::{Sfm, contrast_codes, named_columns, numeric_levels};
use super::{Dataset, FairnessError};

/// Plug-in estimate of `Δ(x, z, w) = E[Y | d1, x, z, w] − E[Y | d0, x, z, w]`
/// for every row of `data`.
///
/// This equals `E[Y_{d1} − Y_{d0} | x, z, w]` when the historical decision
/// had no unobserved confounding with `Y` given `(X, Z, W)`, i.e. the SFM
/// extended with `D` (Fig. 6.27). Every `(x, z, w)` cell must contain both
/// decisions. `Y` is `sfm.y` and must be numeric.
pub fn estimate_benefit(
    data: &Dataset,
    sfm: &Sfm,
    decision: &str,
    d0: &str,
    d1: &str,
) -> Result<Vec<f64>, FairnessError> {
    let dcol = data.column(decision)?;
    let (k0, k1) = contrast_codes(dcol, decision, d0, d1)?;
    let xcol = data.column(&sfm.x)?;
    let cols = named_columns(data, &sfm.z)?
        .into_iter()
        .chain(named_columns(data, &sfm.w)?)
        .collect::<Vec<_>>();
    let ycol = data.column(&sfm.y)?;
    let yvals = numeric_levels(ycol);

    let key = |i: usize| -> Vec<u32> {
        std::iter::once(xcol.codes[i])
            .chain(cols.iter().map(|(_, c)| c.codes[i]))
            .collect()
    };
    let mut sums: HashMap<Vec<u32>, [(f64, f64); 2]> = HashMap::new();
    for i in 0..data.len() {
        let dc = dcol.codes[i] as usize;
        let d = match dc {
            _ if dc == k0 => 0,
            _ if dc == k1 => 1,
            _ => continue,
        };
        let yc = ycol.codes[i] as usize;
        let y = yvals[yc].ok_or_else(|| FairnessError::NonNumericOutcome {
            column: sfm.y.clone(),
            value: ycol.levels[yc].clone(),
        })?;
        let w = data.weight(i);
        let e = &mut sums.entry(key(i)).or_insert([(0.0, 0.0); 2])[d];
        e.0 += w * y;
        e.1 += w;
    }

    (0..data.len())
        .map(|i| {
            let k = key(i);
            let describe = || {
                std::iter::once(format!("{}={}", sfm.x, xcol.levels[k[0] as usize]))
                    .chain(
                        cols.iter()
                            .zip(&k[1..])
                            .map(|((n, c), &v)| format!("{n}={}", c.levels[v as usize])),
                    )
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let s = sums.get(&k).copied().unwrap_or([(0.0, 0.0); 2]);
            for (d, level) in [(0, d0), (1, d1)] {
                if s[d].1 <= 0.0 {
                    return Err(FairnessError::Positivity {
                        level: level.to_string(),
                        stratum: describe(),
                    });
                }
            }
            Ok(s[1].0 / s[1].1 - s[0].0 / s[0].1)
        })
        .collect()
}

/// A decision policy that treats units by their degree of benefit.
#[derive(Debug, Clone, PartialEq)]
pub struct BenefitPolicy {
    /// `P(d = 1)` for every row of the input data.
    pub treat_probability: Vec<f64>,
    /// `δ(x0), δ(x1)`: units with `Δ` above the threshold are treated, units
    /// exactly at it with the boundary probability. Algorithm 17 uses one
    /// threshold for everyone, so the two are equal unless a group
    /// exhausted its positive-benefit units.
    pub thresholds: [f64; 2],
    /// Probability of treating a unit with `Δ` exactly at its group's
    /// threshold.
    pub boundary_probability: [f64; 2],
    /// `P(d | x0), P(d | x1)`.
    pub rates: [f64; 2],
    /// `P(d)` over all rows.
    pub rate: f64,
    /// `E[Δ · d]`: expected gain over treating no one.
    pub expected_benefit: f64,
}

impl BenefitPolicy {
    /// `P(d | x1) − P(d | x0)`, the allocation disparity (Def. 6.1.41 (i)).
    /// Decompose it by adding [`BenefitPolicy::column`] to the data and
    /// running [`estimate`](super::estimate) on it.
    pub fn disparity(&self) -> f64 {
        self.rates[1] - self.rates[0]
    }

    /// The treatment probabilities as strings, ready for
    /// [`Dataset::with_column`].
    pub fn column(&self) -> Vec<String> {
        self.treat_probability.iter().map(|p| format!("{p}")).collect()
    }
}

/// Distinct positive benefit levels of a group, best first, with their
/// weight as a fraction of the group.
struct Ladder {
    levels: Vec<(f64, f64)>,
}

impl Ladder {
    fn new(items: impl Iterator<Item = (f64, f64)>) -> Self {
        let mut items: Vec<(f64, f64)> = items.filter(|(d, w)| *d > 0.0 && *w > 0.0).collect();
        items.sort_by(|a, b| b.0.total_cmp(&a.0));
        Self::merge(items)
    }

    fn merge(items: Vec<(f64, f64)>) -> Self {
        let mut levels: Vec<(f64, f64)> = Vec::new();
        for (d, w) in items {
            match levels.last_mut() {
                Some(last) if last.0 == d => last.1 += w,
                _ => levels.push((d, w)),
            }
        }
        Self { levels }
    }

    fn normalised(mut self, total: f64) -> Self {
        for l in &mut self.levels {
            l.1 /= total;
        }
        self
    }

    /// Fraction of the group with positive benefit.
    fn capacity(&self) -> f64 {
        self.levels.iter().map(|l| l.1).sum()
    }

    /// Cumulative fractions at which the marginal benefit changes.
    fn breakpoints(&self) -> Vec<f64> {
        self.levels
            .iter()
            .scan(0.0, |acc, l| {
                *acc += l.1;
                Some(*acc)
            })
            .collect()
    }

    /// Treats the best `t` fraction. Returns (threshold, boundary
    /// probability, benefit per group member) following Eqs. 6.471–6.474.
    fn allocate(&self, t: f64) -> (f64, f64, f64) {
        let (mut cum, mut gain) = (0.0, 0.0);
        for &(d, m) in &self.levels {
            if cum + m <= t + 1e-12 {
                cum += m;
                gain += d * m;
            } else {
                let p = ((t - cum) / m).clamp(0.0, 1.0);
                return (d, p, gain + d * m * p);
            }
        }
        (0.0, 0.0, gain)
    }
}

fn group_codes(data: &Dataset, x: &str, x0: &str, x1: &str) -> Result<Vec<Option<usize>>, FairnessError> {
    let xcol = data.column(x)?;
    let (c0, c1) = contrast_codes(xcol, x, x0, x1)?;
    Ok(xcol
        .codes
        .iter()
        .map(|&c| match c as usize {
            c if c == c0 => Some(0),
            c if c == c1 => Some(1),
            _ => None,
        })
        .collect())
}

fn check(data: &Dataset, benefit: &[f64], budget: f64) -> Result<(), FairnessError> {
    if benefit.len() != data.len() {
        return Err(FairnessError::InvalidInput(format!(
            "{} benefit values for {} rows",
            benefit.len(),
            data.len()
        )));
    }
    if let Some(b) = benefit.iter().find(|b| !b.is_finite()) {
        return Err(FairnessError::InvalidInput(format!("benefit {b} is not finite")));
    }
    if !(0.0..=1.0).contains(&budget) {
        return Err(FairnessError::InvalidInput(format!("budget {budget} is not in [0, 1]")));
    }
    Ok(())
}

fn finish(
    data: &Dataset,
    groups: &[Option<usize>],
    benefit: &[f64],
    thresholds: [f64; 2],
    boundary: [f64; 2],
    shared: bool,
    levels: [&str; 2],
) -> Result<BenefitPolicy, FairnessError> {
    let mut probs = Vec::with_capacity(data.len());
    let (mut n, mut treated, mut gain) = ([0.0; 2], [0.0; 2], 0.0);
    let (mut total, mut total_treated) = (0.0, 0.0);
    for i in 0..data.len() {
        // Algorithm 17 treats every row by the shared threshold, including
        // rows outside the contrast; Algorithm 18 has none of those.
        let g = groups[i].unwrap_or(0);
        let (d, p) = if shared { (thresholds[0], boundary[0]) } else { (thresholds[g], boundary[g]) };
        let prob = if benefit[i] <= 0.0 {
            0.0
        } else if benefit[i] > d {
            1.0
        } else if benefit[i] == d {
            p
        } else {
            0.0
        };
        let w = data.weight(i);
        if let Some(g) = groups[i] {
            n[g] += w;
            treated[g] += w * prob;
        }
        total += w;
        total_treated += w * prob;
        gain += w * prob * benefit[i];
        probs.push(prob);
    }
    for (count, level) in n.iter().zip(levels) {
        if *count <= 0.0 {
            return Err(FairnessError::EmptyGroup {
                level: level.to_string(),
            });
        }
    }
    Ok(BenefitPolicy {
        treat_probability: probs,
        thresholds,
        boundary_probability: boundary,
        rates: [treated[0] / n[0], treated[1] / n[1]],
        rate: total_treated / total,
        expected_benefit: gain / total,
    })
}

/// Algorithm 17, decision-making with benefit fairness. Treats the units with
/// the largest benefit until a fraction `budget` of the population is
/// treated, randomising at the boundary level. The policy depends on the
/// covariates only through `Δ`, so it satisfies the benefit fairness
/// criterion (Def. 6.1.38) by construction.
///
/// `benefit` holds one `Δ` per row of `data`. `X` (column `x`, levels `x0`,
/// `x1`) is only used to report the group rates.
pub fn benefit_fairness(
    data: &Dataset,
    x: &str,
    x0: &str,
    x1: &str,
    benefit: &[f64],
    budget: f64,
) -> Result<BenefitPolicy, FairnessError> {
    check(data, benefit, budget)?;
    let groups = group_codes(data, x, x0, x1)?;
    let total: f64 = (0..data.len()).map(|i| data.weight(i)).sum();
    let ladder = Ladder::new((0..data.len()).map(|i| (benefit[i], data.weight(i)))).normalised(total);
    let (d, p, _) = ladder.allocate(budget);
    finish(data, &groups, benefit, [d, d], [p, p], true, [x0, x1])
}

/// Algorithm 18, decision-making with utilitarian adjusted benefit fairness.
/// Picks group-specific thresholds `δ(x0), δ(x1)` that maximise `E[Δ · d]`
/// subject to `P(d) ≤ budget` and `|P(d | x1) − P(d | x0)| ≤ max_disparity`.
///
/// Every row must have `X ∈ {x0, x1}`. Within a group the optimum treats
/// the highest-benefit units first, so the policy is fixed by the treatment
/// rates `(t0, t1)`. Each group's gain is concave and piecewise linear in
/// its rate, so the optimum lies at a vertex of the feasible polygon cut by
/// the gain breakpoints. All such vertices are enumerated, which makes the
/// result exact.
///
/// Eq. 6.494 writes the objective with the conditional means
/// `E[Δ | Δ > δ, x]`. This maximises the total benefit `E[Δ · d]` instead,
/// the quantity Prop. 6.1.41 and Algorithm 17 optimise.
pub fn utilitarian_benefit_fairness(
    data: &Dataset,
    x: &str,
    x0: &str,
    x1: &str,
    benefit: &[f64],
    budget: f64,
    max_disparity: f64,
) -> Result<BenefitPolicy, FairnessError> {
    check(data, benefit, budget)?;
    if max_disparity.is_nan() || max_disparity < 0.0 {
        return Err(FairnessError::InvalidInput(format!(
            "maximum disparity {max_disparity} is negative"
        )));
    }
    let groups = group_codes(data, x, x0, x1)?;
    if let Some(i) = groups.iter().position(Option::is_none) {
        return Err(FairnessError::InvalidInput(format!(
            "row {} has {x} outside {{{x0}, {x1}}}",
            i + 1
        )));
    }
    let mut n = [0.0; 2];
    let mut items: [Vec<(f64, f64)>; 2] = Default::default();
    for i in 0..data.len() {
        let g = groups[i].unwrap();
        n[g] += data.weight(i);
        items[g].push((benefit[i], data.weight(i)));
    }
    for (g, level) in [x0, x1].iter().enumerate() {
        if n[g] <= 0.0 {
            return Err(FairnessError::EmptyGroup {
                level: level.to_string(),
            });
        }
    }
    let [i0, i1] = items;
    let lad = [
        Ladder::new(i0.into_iter()).normalised(n[0]),
        Ladder::new(i1.into_iter()).normalised(n[1]),
    ];
    let p = [n[0] / (n[0] + n[1]), n[1] / (n[0] + n[1])];
    let u = [lad[0].capacity(), lad[1].capacity()];
    let m = max_disparity;
    let eps = 1e-12;

    let feasible = |t: [f64; 2]| {
        (0..2).all(|g| t[g] >= -eps && t[g] <= u[g] + eps)
            && p[0] * t[0] + p[1] * t[1] <= budget + eps
            && (t[0] - t[1]).abs() <= m + eps
    };
    let value = |t: [f64; 2]| p[0] * lad[0].allocate(t[0]).2 + p[1] * lad[1].allocate(t[1]).2;

    let mut candidates: Vec<[f64; 2]> = Vec::new();
    // Along each breakpoint line of one group, the best point pushes the
    // other group's rate as high as the constraints allow (gains increase).
    for g in 0..2 {
        let h = 1 - g;
        let mut ts = lad[g].breakpoints();
        ts.extend([0.0, u[g]]);
        for tg in ts {
            let hi = u[h].min(tg + m).min((budget - p[g] * tg) / p[h]);
            let lo = (tg - m).max(0.0);
            if hi >= lo - eps {
                let mut t = [0.0; 2];
                t[g] = tg;
                t[h] = hi.max(lo);
                candidates.push(t);
            }
        }
    }
    // Vertices of the feasible polygon itself.
    let lines = [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, u[0]),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, u[1]),
        (p[0], p[1], budget),
        (1.0, -1.0, m),
        (1.0, -1.0, -m),
    ];
    for (i, a) in lines.iter().enumerate() {
        for b in &lines[i + 1..] {
            let det = a.0 * b.1 - a.1 * b.0;
            if det.abs() > 1e-15 {
                candidates.push([(a.2 * b.1 - a.1 * b.2) / det, (a.0 * b.2 - a.2 * b.0) / det]);
            }
        }
    }

    let best = candidates
        .into_iter()
        .filter(|&t| feasible(t))
        .map(|t| [t[0].clamp(0.0, u[0]), t[1].clamp(0.0, u[1])])
        .max_by(|&a, &b| {
            value(a)
                .total_cmp(&value(b))
                // Ties: prefer the smaller disparity.
                .then((b[0] - b[1]).abs().total_cmp(&(a[0] - a[1]).abs()))
        })
        .unwrap_or([0.0, 0.0]);
    let (d0, b0, _) = lad[0].allocate(best[0]);
    let (d1, b1, _) = lad[1].allocate(best[1]);
    finish(data, &groups, benefit, [d0, d1], [b0, b1], false, [x0, x1])
}

/// Tight bounds on the canonical types (Def. 6.1.39) in one covariate cell,
/// from `m1 = P(y_{d1} | x, z, w)` and `m0 = P(y_{d0} | x, z, w)`
/// (Prop. 6.1.40).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanonicalBounds {
    /// Compliers, `Y_{d0} = 0, Y_{d1} = 1`: helped by the decision.
    pub complier: (f64, f64),
    /// Defiers, `Y_{d0} = 1, Y_{d1} = 0`: harmed by it.
    pub defier: (f64, f64),
    /// Safe, `Y_{d0} = Y_{d1} = 1`.
    pub safe: (f64, f64),
    /// Unsafe, `Y_{d0} = Y_{d1} = 0`.
    pub unsafe_: (f64, f64),
}

/// Bounds on the canonical types from the two interventional probabilities.
/// The book assumes `m1 ≥ m0`; this handles either order.
///
/// From `(s, d, c, u) = (m0 − d, d, d + m1 − m0, 1 − m1 − d)`, non-negativity
/// gives `d ∈ [max(0, m0 − m1), min(m0, 1 − m1)]`. Under monotonicity
/// (no defiers) the bounds collapse to `c = m1 − m0`.
///
/// ```
/// let b = why_data::fairness::canonical_type_bounds(0.7, 0.4);
/// assert!((b.complier.0 - 0.3).abs() < 1e-12 && (b.complier.1 - 0.6).abs() < 1e-12);
/// assert_eq!(b.defier.0, 0.0);
/// assert!((b.defier.1 - 0.3).abs() < 1e-12);
/// ```
pub fn canonical_type_bounds(m1: f64, m0: f64) -> CanonicalBounds {
    let d_lo = (m0 - m1).max(0.0);
    let d_hi = m0.min(1.0 - m1).max(d_lo);
    CanonicalBounds {
        defier: (d_lo, d_hi),
        complier: (d_lo + m1 - m0, d_hi + m1 - m0),
        safe: (m0 - d_hi, m0 - d_lo),
        unsafe_: (1.0 - m1 - d_hi, 1.0 - m1 - d_lo),
    }
}
