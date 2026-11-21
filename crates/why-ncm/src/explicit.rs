//! The explicit (MLE) reading of an NCM: likelihoods and query
//! probabilities (Eqs. 13.142–13.147).

use std::collections::BTreeMap;

use burn::tensor::{Tensor, activation};
use why_data::ncm::{Expanded, Fix};

use crate::NcmError;
use crate::data::Table;
use crate::model::{Ncm, Noise};

/// A cell `(world, variable)` of an expanded query, and how its value is
/// set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellKind {
    /// Produced by the variable's mechanism.
    Free,
    /// Fixed to a level index.
    Level(usize),
    /// Copies another (earlier) cell.
    Copy(usize),
}

/// The cells of an expanded query that matter, in an order where every
/// cell comes after those it copies or reads.
#[derive(Clone, Debug)]
pub(crate) struct Cells {
    /// `(world, var)` per cell.
    pub at: Vec<(usize, usize)>,
    pub kind: Vec<CellKind>,
    pub index: BTreeMap<(usize, usize), usize>,
    /// `(cell, level)` the events require.
    pub events: Vec<(usize, usize)>,
    /// `(cell, level)` the condition requires.
    pub given: Vec<(usize, usize)>,
}

/// `relu(x)`, passing `leak` times the gradient of `x` where it is
/// negative too when `leak > 0`.
fn overlap(x: Tensor<2>, leak: f64) -> Tensor<2> {
    if leak > 0.0 {
        x.clone().clamp_min(0.0) + (x.clone() - x.detach()).mul_scalar(leak)
    } else {
        x.clamp_min(0.0)
    }
}

impl Ncm {
    /// The probability that one variable takes the chosen level in every
    /// world at once, given `Û`. In world `j` the level's CDF interval is
    /// `[lo_j, hi_j)`, or `[1 − hi_j, 1 − lo_j)` when the world reads the
    /// levels in reverse, which it does when `δ_V < g_j`. Summing over the
    /// set `S` of reversed worlds,
    /// `Σ_S relu(min_{j∈S} g_j − max_{j∉S} g_j) · relu(min_j hi_j^S − max_j lo_j^S)`.
    /// One world gives `hi − lo`, the level's probability.
    fn coupled(
        &self,
        his: Vec<Tensor<2>>,
        los: Vec<Tensor<2>>,
        gates: Vec<Tensor<2>>,
        leak: f64,
    ) -> Tensor<2> {
        let w = his.len();
        if w == 1 {
            return his[0].clone() - los[0].clone();
        }
        let rows = his[0].dims()[0];
        let one = || Tensor::<2>::ones([rows, 1], &self.device);
        let mut total = Tensor::<2>::zeros([rows, 1], &self.device);
        for set in 0..1usize << w {
            let inside = |j: usize| set >> j & 1 == 1;
            let mut lo_g = Vec::new();
            let mut hi_g = Vec::new();
            let mut hi = Vec::with_capacity(w);
            let mut lo = Vec::with_capacity(w);
            for j in 0..w {
                if inside(j) {
                    hi_g.push(gates[j].clone());
                    hi.push(one() - los[j].clone());
                    lo.push(one() - his[j].clone());
                } else {
                    lo_g.push(gates[j].clone());
                    hi.push(his[j].clone());
                    lo.push(los[j].clone());
                }
            }
            let top = if hi_g.is_empty() {
                one()
            } else {
                Tensor::cat(hi_g, 1).min_dim(1)
            };
            let bottom = if lo_g.is_empty() {
                Tensor::zeros([rows, 1], &self.device)
            } else {
                Tensor::cat(lo_g, 1).max_dim(1)
            };
            let weight = overlap(top - bottom, leak);
            let gap = Tensor::cat(hi, 1).min_dim(1) - Tensor::cat(lo, 1).max_dim(1);
            total = total + weight * overlap(gap, leak);
        }
        total
    }

    pub(crate) fn cells(&self, e: &Expanded) -> Result<Cells, NcmError> {
        let spec = &self.spec;
        for v in e.variables() {
            spec.var(&v)?;
        }
        let relevant = e.relevant(&spec.structure);
        let mut cells = Cells {
            at: Vec::new(),
            kind: Vec::new(),
            index: BTreeMap::new(),
            events: Vec::new(),
            given: Vec::new(),
        };
        for (j, vars) in relevant.iter().enumerate() {
            for name in vars {
                let v = spec.var(name)?;
                let kind = match e.worlds[j].fixed.get(name) {
                    None => CellKind::Free,
                    Some(Fix::Level(l)) => CellKind::Level(spec.level(v, l)?),
                    Some(Fix::Copy(k)) => CellKind::Copy(cells.index[&(*k, v)]),
                };
                cells.index.insert((j, v), cells.at.len());
                cells.at.push((j, v));
                cells.kind.push(kind);
            }
        }
        let resolve =
            |a: &why_data::ncm::Assignment, cells: &Cells| -> Result<(usize, usize), NcmError> {
                let v = spec.var(&a.var)?;
                Ok((cells.index[&(a.world, v)], spec.level(v, &a.level)?))
            };
        cells.events = e
            .events
            .iter()
            .map(|a| resolve(a, &cells))
            .collect::<Result<_, _>>()?;
        cells.given = e
            .given
            .iter()
            .map(|a| resolve(a, &cells))
            .collect::<Result<_, _>>()?;
        Ok(cells)
    }

    /// `Σ_r w_r log P̂(v_r | do(z_r))` over the table (Eqs. 13.142 and
    /// 13.144): each row's likelihood averages the product of its
    /// non-intervened factors over the draws of `Û`.
    pub(crate) fn table_log_likelihood(&self, t: &Table, noise: &Noise) -> Tensor<1> {
        let spec = &self.spec;
        let (n, m) = (t.rows.len(), noise.m);
        let mut total: Tensor<2> = Tensor::zeros([n * m, 1], &self.device);
        for v in 0..spec.names.len() {
            if t.intervened[v] {
                continue;
            }
            let parents: Vec<Vec<usize>> = t
                .rows
                .iter()
                .map(|r| spec.parents[v].iter().map(|p| r[*p]).collect())
                .collect();
            let (logits, _) = self.split(v, self.logits(v, &parents, noise));
            let logp = activation::log_softmax(logits, 1);
            let d = spec.levels[v].len();
            let mut mask = Vec::with_capacity(n * m * d);
            for r in &t.rows {
                for _ in 0..m {
                    mask.extend((0..d).map(|i| f32::from(u8::from(i == r[v]))));
                }
            }
            total = total + (logp * self.matrix(mask, n * m, d)).sum_dim(1);
        }
        let total = total.reshape([n, m]);
        let top = total.clone().max_dim(1).detach();
        let lme = (total - top.clone().expand([n, m])).exp().mean_dim(1).log() + top;
        let w = self.matrix(t.weights.iter().map(|w| *w as f32).collect(), n, 1);
        (lme * w).sum()
    }

    /// `P̂(y_* | x_*)` (Eqs. 13.145–13.147), differentiable in the
    /// parameters.
    ///
    /// The worlds' relevant cells are enumerated. Given `Û = û`, each
    /// variable takes the inverse CDF of `σ(φ_V)` at one `ε_V` shared by all
    /// worlds, so its joint across worlds is the overlap of the chosen
    /// levels' CDF intervals.
    ///
    /// An empty overlap has no gradient, so a model that starts with none
    /// could never be pushed toward one. With `leak > 0` (training only)
    /// the overlap keeps its exact value but passes `leak` times the
    /// gradient of the signed gap, a straight-through surrogate.
    pub(crate) fn query_tensor(
        &self,
        e: &Expanded,
        noise: &Noise,
        leak: f64,
    ) -> Result<Tensor<1>, NcmError> {
        let spec = &self.spec;
        let cells = self.cells(e)?;
        let n = cells.at.len();
        // Clamp the free cells the condition (or, for a joint, the events)
        // pins down.
        let pinned = if cells.given.is_empty() {
            &cells.events
        } else {
            &cells.given
        };
        let mut clamp: Vec<Option<usize>> = vec![None; n];
        for &(c, l) in pinned {
            match clamp[c] {
                Some(x) if x != l => {
                    if cells.given.is_empty() {
                        return Ok(Tensor::zeros([1], &self.device));
                    }
                    return Err(NcmError::ImpossibleCondition);
                }
                _ if cells.kind[c] == CellKind::Free => clamp[c] = Some(l),
                _ => {}
            }
        }
        let free: Vec<usize> = (0..n)
            .filter(|c| cells.kind[*c] == CellKind::Free && clamp[*c].is_none())
            .collect();
        let mut count = 1usize;
        for &c in &free {
            count = count
                .checked_mul(spec.levels[cells.at[c].1].len())
                .filter(|k| *k <= self.max_cells)
                .ok_or(NcmError::TooLarge(self.max_cells))?;
        }
        // Every configuration that satisfies the condition.
        let mut configs: Vec<Vec<usize>> = Vec::new();
        let mut hits: Vec<f32> = Vec::new();
        let mut digits = vec![0usize; free.len()];
        for _ in 0..count {
            let mut vals = vec![0usize; n];
            let mut f = 0;
            for c in 0..n {
                vals[c] = match cells.kind[c] {
                    CellKind::Level(l) => l,
                    CellKind::Copy(k) => vals[k],
                    CellKind::Free => match clamp[c] {
                        Some(l) => l,
                        None => {
                            f += 1;
                            digits[f - 1]
                        }
                    },
                };
            }
            if cells.given.iter().all(|&(c, l)| vals[c] == l) {
                hits.push(f32::from(u8::from(
                    cells.events.iter().all(|&(c, l)| vals[c] == l),
                )));
                configs.push(vals);
            }
            for (k, &c) in free.iter().enumerate() {
                digits[k] += 1;
                if digits[k] < spec.levels[cells.at[c].1].len() {
                    break;
                }
                digits[k] = 0;
            }
        }
        if configs.is_empty() {
            if cells.given.is_empty() {
                return Ok(Tensor::zeros([1], &self.device));
            }
            return Err(NcmError::ImpossibleCondition);
        }
        let (k, m) = (configs.len(), noise.m);
        let mut joint: Tensor<2> = Tensor::ones([k * m, 1], &self.device);
        for v in 0..spec.names.len() {
            let worlds: Vec<usize> = (0..n)
                .filter(|c| cells.at[*c].1 == v && cells.kind[*c] == CellKind::Free)
                .collect();
            if worlds.is_empty() {
                continue;
            }
            let d = spec.levels[v].len();
            let (mut his, mut los, mut gates) = (Vec::new(), Vec::new(), Vec::new());
            for &c in &worlds {
                let j = cells.at[c].0;
                let parents: Vec<Vec<usize>> = configs
                    .iter()
                    .map(|vals| {
                        spec.parents[v]
                            .iter()
                            .map(|p| vals[cells.index[&(j, *p)]])
                            .collect()
                    })
                    .collect();
                let (logits, gate) = self.split(v, self.logits(v, &parents, noise));
                let probs = activation::softmax(logits, 1);
                let cdf = probs.clone().cumsum(1);
                let lower = cdf.clone() - probs;
                let mut mask = Vec::with_capacity(k * m * d);
                for vals in &configs {
                    for _ in 0..m {
                        mask.extend((0..d).map(|i| f32::from(u8::from(i == vals[c]))));
                    }
                }
                let mask = self.matrix(mask, k * m, d);
                his.push((cdf * mask.clone()).sum_dim(1));
                los.push((lower * mask).sum_dim(1));
                gates.push(gate);
            }
            joint = joint * self.coupled(his, los, gates, leak);
        }
        let per_config = joint.reshape([k, m]).mean_dim(1);
        let num = (per_config.clone() * self.matrix(hits, k, 1)).sum();
        if cells.given.is_empty() {
            return Ok(num);
        }
        Ok(num / per_config.sum().clamp_min(1e-12))
    }
}
