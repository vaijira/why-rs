//! Algorithm 52 (NeuralAbstractID): neural identification across an
//! abstraction (chapter 15, Sec. 15.3).

use why_data::abstraction::Abstraction;
use why_data::graph::dseparation::Admg;

use crate::{Experiment, NcmConfig, NcmError, NeuralIdResult, Objective, neural_id};

/// Algorithm 52: decides whether a low-level query `Q` is τ-ID from the
/// cluster causal diagram and the data, and estimates it when it is.
///
/// `g` is the low-level diagram and `tau` the constructive abstraction
/// function. The C-DAG `G_C` is [`Abstraction::cdag`] of `g`, so only the
/// clusters' relations are used. `objective` is `τ(Q)`, written over the
/// clusters and parsed against that C-DAG. `experiments` are the low-level
/// data `Z`; each is mapped through `τ`, and its intervened variables must
/// cover whole clusters.
///
/// By Thm 15.3.2 and Corol. 15.3.3 this is NeuralID (Algorithm 46) on `G_C`
/// and `τ(Z)`. The two trained high-level models agree exactly when `Q` is
/// τ-ID, and then either one computes `Q` (Corol. 15.3.4).
///
/// # Errors
///
/// [`NcmError::Abstraction`] for a clustering that is not admissible, a
/// member the data or graph lack, or an intervention that splits a cluster.
/// Also the errors of [`neural_id`].
pub fn neural_abstract_id(
    g: &Admg,
    tau: &Abstraction,
    objective: &Objective,
    experiments: &[Experiment],
    config: &NcmConfig,
) -> Result<NeuralIdResult, NcmError> {
    let gc = tau.cdag(g).map_err(NcmError::Abstraction)?;
    let high = experiments
        .iter()
        .map(|e| {
            let data = tau.transform(&e.data).map_err(NcmError::Abstraction)?;
            let mut clusters: Vec<String> = Vec::new();
            for v in &e.intervened {
                let c = tau.cluster_of(v).ok_or_else(|| {
                    NcmError::Abstraction(why_data::abstraction::AbstractionError::Partial(
                        v.clone(),
                    ))
                })?;
                let members = tau.members(c).map_err(NcmError::Abstraction)?;
                if members.iter().any(|m| !e.intervened.contains(m)) {
                    return Err(NcmError::Abstraction(
                        why_data::abstraction::AbstractionError::Partial(c.to_string()),
                    ));
                }
                if !clusters.iter().any(|x| x == c) {
                    clusters.push(c.to_string());
                }
            }
            Ok(Experiment {
                intervened: clusters,
                data,
            })
        })
        .collect::<Result<Vec<_>, NcmError>>()?;
    neural_id(&gc, objective, &high, config)
}
