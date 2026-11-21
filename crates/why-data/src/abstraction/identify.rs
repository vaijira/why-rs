//! Abstract identification (Def. 15.3.2) by the dual abstract ID theorem
//! (Thm 15.3.2).

use super::{Abstraction, AbstractionError};
use crate::fairness::Dataset;
use crate::graph::cinfer::Input;
use crate::graph::cinfer::counterfactual::{CtfDerivation, CtfEstimand, CtfQuery, ctf_infer};
use crate::graph::dseparation::Admg;
use crate::transport::{Domain, EmpiricalDistributions, Estimand};

/// Decides whether a query is τ-ID from the C-DAG and the data (Def.
/// 15.3.2). By Thm 15.3.2 it is exactly when `τ(Q)` is identifiable in the
/// high-level space, so this runs C-INFER (chapter 5) on the cluster causal
/// diagram of `g`.
///
/// `query` is `τ(Q)`, written over the clusters, and should be parsed
/// against [`Abstraction::cdag`]. `inputs` are the available
/// distributions `τ(Z)`, also over clusters.
///
/// # Errors
///
/// [`AbstractionError::Identification`] when the query is not τ-ID (or is
/// malformed), and the errors of [`Abstraction::cdag`].
pub fn abstract_identify(
    g: &Admg,
    tau: &Abstraction,
    query: &CtfQuery,
    inputs: &[Input],
) -> Result<CtfDerivation, AbstractionError> {
    let gc = tau.cdag(g)?;
    Ok(ctf_infer(&gc, query, inputs)?)
}

/// Identifies `τ(Q)` from the C-DAG and observational data, then evaluates
/// the estimand on `τ(P(V_L))`, the low-level data `data` mapped through
/// `τ`. This computes the low-level `Q` (Corol. 15.3.3, symbolically).
/// Each value name of the query (`X`, `X'`, …) maps to a high-level level by
/// `values`.
///
/// ```
/// use why_data::abstraction::{Abstraction, abstract_estimate};
/// use why_data::fairness::Dataset;
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
///
/// // Two binary causes clustered into "both", and their effect.
/// let mut g = Admg::new();
/// g.directed("A", "Y");
/// g.directed("B", "Y");
/// let rows = [["0", "0", "0"], ["1", "1", "1"], ["1", "0", "0"], ["1", "1", "0"]];
/// let data = Dataset::from_records(["A", "B", "Y"], rows).unwrap();
/// let tau = Abstraction::new()
///     .cluster("X", &["A", "B"], |v| if v == ["1", "1"] { "1".into() } else { "0".into() })
///     .unwrap()
///     .keep("Y")
///     .unwrap();
/// let q = CtfQuery::parse(&tau.cdag(&g).unwrap(), "P(y_x)").unwrap();
/// let p = abstract_estimate(&g, &tau, &q, &[("X", "1"), ("Y", "1")], &data).unwrap();
/// assert!((p - 0.5).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// Those of [`abstract_identify`] and [`Abstraction::transform`], and
/// [`AbstractionError::Transport`] when the estimand divides by an empty
/// cell or a value name is missing.
pub fn abstract_estimate(
    g: &Admg,
    tau: &Abstraction,
    query: &CtfQuery,
    values: &[(&str, &str)],
    data: &Dataset,
) -> Result<f64, AbstractionError> {
    let d = abstract_identify(g, tau, query, &[Input::observational()])?;
    match d.estimand {
        CtfEstimand::Zero => Ok(0.0),
        CtfEstimand::One => Ok(1.0),
        CtfEstimand::Formula(f) => {
            let high = tau.transform(data)?;
            let dists = EmpiricalDistributions::new().with(Domain::Target, &high);
            Ok(Estimand::from_formula(&f, &Domain::Target).evaluate(&dists, values)?)
        }
    }
}
