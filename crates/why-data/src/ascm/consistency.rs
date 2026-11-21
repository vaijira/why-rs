//! Ctf-consistent estimators (Def. 14.3.2).

use std::collections::BTreeMap;

use super::{AscmError, Interval, optimal_bounds};
use crate::fairness::Dataset;
use crate::graph::cinfer::counterfactual::CtfQuery;
use crate::graph::dseparation::Admg;

/// How a proxy model fares against Def. 14.3.2 on one F-ctf query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtfConsistency {
    /// The total variation distance between the true `P(V)` and the
    /// proxy's: condition (1), relaxed to a tolerance for a learned model.
    pub fit: f64,
    /// The optimal bounds `P(V)` and `G` imply for the query.
    pub bounds: Interval,
    /// The proxy's value of the query.
    pub value: f64,
    /// Whether both conditions hold within the tolerance.
    pub consistent: bool,
}

/// `½ Σ_v |P(v) − Q(v)|` over the joint levels of `vars`. Row weights are
/// frequencies, so either data set may be a sample or an exact table.
///
/// # Errors
///
/// [`AscmError::MissingColumn`] when a data set lacks one of `vars`, and
/// [`AscmError::Data`] when one has no mass.
pub fn total_variation(p: &Dataset, q: &Dataset, vars: &[&str]) -> Result<f64, AscmError> {
    let table = |d: &Dataset| -> Result<BTreeMap<Vec<String>, f64>, AscmError> {
        for v in vars {
            if d.levels(v).is_none() {
                return Err(AscmError::MissingColumn((*v).to_string()));
            }
        }
        let mut t: BTreeMap<Vec<String>, f64> = BTreeMap::new();
        for r in 0..d.len() {
            let key = vars
                .iter()
                .map(|v| d.value(v, r).expect("checked").to_string())
                .collect();
            *t.entry(key).or_default() += d.weights().map_or(1.0, |w| w[r]);
        }
        let total: f64 = t.values().sum();
        if total <= 0.0 {
            return Err(AscmError::Data("the distribution has no mass".into()));
        }
        t.values_mut().for_each(|w| *w /= total);
        Ok(t)
    };
    let (p, mut q) = (table(p)?, table(q)?);
    let mut sum = 0.0;
    for (k, a) in p {
        sum += (a - q.remove(&k).unwrap_or(0.0)).abs();
    }
    sum += q.values().sum::<f64>();
    Ok(sum / 2.0)
}

/// Def. 14.3.2 for one conditional F-ctf query `P(w′_{x′} | w)`.
///
/// `truth` is the true `P(V)` and `proxy` the proxy model's, `value` the
/// proxy's value of `query` (by Lemma 14.3.1, its generative-level
/// counterfactual). The proxy is ctf-consistent on the query when its
/// `P̂(V)` is within `tolerance` of `P(V)` in total variation and `value`
/// lies within `tolerance` of the optimal bounds from `truth` and `g`.
///
/// # Errors
///
/// Those of [`optimal_bounds`] and [`total_variation`].
pub fn ctf_consistency(
    g: &Admg,
    truth: &Dataset,
    proxy: &Dataset,
    query: &CtfQuery,
    values: &[(&str, &str)],
    value: f64,
    tolerance: f64,
) -> Result<CtfConsistency, AscmError> {
    let vars = g.variables();
    let fit = total_variation(truth, proxy, &vars)?;
    let bounds = optimal_bounds(g, truth, query, values)?;
    let consistent =
        fit <= tolerance && value >= bounds.lower - tolerance && value <= bounds.upper + tolerance;
    Ok(CtfConsistency {
        fit,
        bounds,
        value,
        consistent,
    })
}
