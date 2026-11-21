//! Data sets as weighted tables of distinct rows.

use std::collections::BTreeMap;

use crate::model::Spec;
use crate::{Experiment, NcmError};

/// One experiment's distinct rows over the model's variables, with their
/// relative frequencies.
#[derive(Clone, Debug)]
pub(crate) struct Table {
    /// Per variable, whether the experiment set it.
    pub intervened: Vec<bool>,
    /// Level indices, one row per distinct record.
    pub rows: Vec<Vec<usize>>,
    /// Frequencies, summing to 1.
    pub weights: Vec<f64>,
}

impl Table {
    pub fn new(spec: &Spec, e: &Experiment) -> Result<Self, NcmError> {
        let mut intervened = vec![false; spec.names.len()];
        for z in &e.intervened {
            intervened[spec.var(z)?] = true;
        }
        let n = e.data.len();
        if n == 0 {
            return Err(NcmError::NoData);
        }
        let mut counts: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for r in 0..n {
            let row = spec
                .names
                .iter()
                .enumerate()
                .map(|(v, name)| {
                    let x = e
                        .data
                        .value(name, r)
                        .ok_or_else(|| NcmError::MissingColumn(name.clone()))?;
                    spec.level(v, x)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let w = e.data.weights().map_or(1.0, |w| w[r]);
            *counts.entry(row).or_default() += w;
        }
        let total: f64 = counts.values().sum();
        if total <= 0.0 {
            return Err(NcmError::NoData);
        }
        let (rows, weights) = counts
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .map(|(r, w)| (r, w / total))
            .unzip();
        Ok(Self {
            intervened,
            rows,
            weights,
        })
    }

    /// `H(V \ Z | Z)` under the empirical distribution: the least negative
    /// log-likelihood any model can reach on it, since the likelihood of a
    /// row leaves out the factors of the intervened variables.
    pub fn entropy(&self) -> f64 {
        let h = |w: &[f64]| -w.iter().map(|w| w * w.ln()).sum::<f64>();
        let mut z: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for (row, w) in self.rows.iter().zip(&self.weights) {
            let key = row
                .iter()
                .zip(&self.intervened)
                .filter(|(_, i)| **i)
                .map(|(x, _)| *x)
                .collect();
            *z.entry(key).or_default() += w;
        }
        h(&self.weights) - h(&z.into_values().collect::<Vec<_>>())
    }
}
