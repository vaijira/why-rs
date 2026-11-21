//! Sampling from an NCM (Algorithm 48) and the implicit model's training
//! losses (Sec. 13.3.2, Eq. 13.154, Table 13.6).

use std::collections::BTreeMap;

use burn::tensor::{Tensor, activation};
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::ncm::{Expanded, expand};

use crate::data::Table;
use crate::explicit::{CellKind, Cells};
use crate::model::{Ncm, Noise, values};
use crate::{Kind, NcmError, Rng};

/// How a simulated cell gets its value.
#[derive(Clone, Debug)]
pub(crate) enum Source {
    /// The variable's mechanism.
    Free,
    /// A constant level.
    Level(usize),
    /// An earlier cell.
    Copy(usize),
    /// One level per draw, as an experiment's randomized treatment.
    PerDraw(Vec<usize>),
}

/// A batch of `m` simulated draws of some cells.
pub(crate) struct Sim {
    /// Per cell, the value as an `m × |D|` matrix: one-hot, with
    /// straight-through gradients when simulated that way.
    pub value: Vec<Tensor<2>>,
    /// Per cell, `σ(φ_V)`, the probabilities its value was drawn from given
    /// its parents and `Û` (a constant one-hot for fixed cells).
    pub soft: Vec<Tensor<2>>,
    /// Per cell, the level index of each draw.
    pub hard: Vec<Vec<usize>>,
}

impl Ncm {
    fn one_hot(&self, levels: &[usize], d: usize) -> Tensor<2> {
        let mut x = Vec::with_capacity(levels.len() * d);
        for &l in levels {
            x.extend((0..d).map(|i| f32::from(u8::from(i == l))));
        }
        self.matrix(x, levels.len(), d)
    }

    /// Simulates `cells` (`(world, var)` each, every reference pointing
    /// back) over the draws in `noise`. With `straight`, a free cell's
    /// value carries the softmax's gradient (straight-through); otherwise
    /// values are constants.
    ///
    /// The explicit model draws `V` by inverse CDF at the shared `ε_V`; the
    /// implicit one by the Gumbel-max trick at the shared `G_V`.
    pub(crate) fn simulate(
        &self,
        at: &[(usize, usize)],
        source: &[Source],
        noise: &Noise,
        straight: bool,
    ) -> Sim {
        let spec = &self.spec;
        let m = noise.m;
        let index: BTreeMap<(usize, usize), usize> =
            at.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let mut sim = Sim {
            value: Vec::with_capacity(at.len()),
            soft: Vec::with_capacity(at.len()),
            hard: Vec::with_capacity(at.len()),
        };
        for (c, &(j, v)) in at.iter().enumerate() {
            let d = spec.levels[v].len();
            let (value, soft, hard) = match &source[c] {
                Source::Level(l) => {
                    let h = vec![*l; m];
                    let t = self.one_hot(&h, d);
                    (t.clone(), t, h)
                }
                Source::PerDraw(h) => {
                    let t = self.one_hot(h, d);
                    (t.clone(), t, h.clone())
                }
                Source::Copy(k) => (
                    sim.value[*k].clone(),
                    sim.soft[*k].clone(),
                    sim.hard[*k].clone(),
                ),
                Source::Free => {
                    let parents = spec.parents[v]
                        .iter()
                        .map(|p| sim.value[index[&(j, *p)]].clone())
                        .collect();
                    let (logits, gate) = self.split(v, self.logits_soft(v, parents, noise));
                    let probs = activation::softmax(logits.clone(), 1);
                    let (h, relaxed) = match self.kind {
                        Kind::Explicit => {
                            // The inverse CDF at the shared ε_V, over the
                            // levels in reverse when δ_V < g_V.
                            let host = values(probs.clone());
                            let gates = values(gate);
                            let h: Vec<usize> = (0..m)
                                .map(|s| {
                                    let at = s * spec.names.len() + v;
                                    let row = &host[s * d..(s + 1) * d];
                                    let reverse = noise.delta[at] < gates[s];
                                    let mut acc = 0.0;
                                    let mut pick = |i: &usize| {
                                        acc += row[*i];
                                        noise.eps[at] < acc
                                    };
                                    if reverse {
                                        (0..d).rev().find(&mut pick).unwrap_or(0)
                                    } else {
                                        (0..d).find(&mut pick).unwrap_or(d - 1)
                                    }
                                })
                                .collect();
                            (h, probs.clone())
                        }
                        Kind::Implicit => {
                            // Gumbel-max at the shared G_V, relaxed by the
                            // Gumbel-softmax for the gradient.
                            let at = spec.offsets[v];
                            let mut g = Vec::with_capacity(m * d);
                            for s in 0..m {
                                let row = s * spec.width + at;
                                g.extend_from_slice(&noise.gumbel[row..row + d]);
                            }
                            let perturbed = logits + self.matrix(g, m, d);
                            let host = values(perturbed.clone());
                            let h: Vec<usize> = (0..m)
                                .map(|s| {
                                    host[s * d..(s + 1) * d]
                                        .iter()
                                        .enumerate()
                                        .max_by(|a, b| a.1.total_cmp(b.1))
                                        .map_or(0, |(i, _)| i)
                                })
                                .collect();
                            (
                                h,
                                activation::softmax(perturbed.div_scalar(self.temperature), 1),
                            )
                        }
                    };
                    let oh = self.one_hot(&h, d);
                    let value = if straight {
                        oh + relaxed.clone() - relaxed.detach()
                    } else {
                        oh
                    };
                    (value, probs, h)
                }
            };
            sim.value.push(value);
            sim.soft.push(soft);
            sim.hard.push(hard);
        }
        sim
    }

    pub(crate) fn query_sources(cells: &Cells) -> Vec<Source> {
        cells
            .kind
            .iter()
            .map(|k| match k {
                CellKind::Free => Source::Free,
                CellKind::Level(l) => Source::Level(*l),
                CellKind::Copy(c) => Source::Copy(*c),
            })
            .collect()
    }

    /// Algorithm 48 (NCM counterfactual sampling): draws `Û` until `m` draws
    /// satisfy the condition `x_*`, and returns the level each event's
    /// counterfactual variable takes in them, in the order of
    /// `query.events`. Value names map to levels as in
    /// [`why_data::ncm::expand`].
    ///
    /// # Errors
    ///
    /// [`NcmError::ImpossibleCondition`] when a thousand batches in a row
    /// accept nothing, and errors for unknown variables or levels.
    pub fn sample(
        &self,
        query: &CtfQuery,
        values: &[(&str, &str)],
        m: usize,
        seed: u64,
    ) -> Result<Vec<Vec<String>>, NcmError> {
        let e = expand(query, values)?;
        let cells = self.cells(&e)?;
        let sources = Self::query_sources(&cells);
        let mut rng = Rng::new(seed);
        let mut out = Vec::with_capacity(m);
        let mut dry = 0;
        let events: Vec<usize> = e
            .events
            .iter()
            .map(|a| cells.index[&(a.world, self.spec.index[&a.var])])
            .collect();
        while out.len() < m {
            let noise = self.draw(m.clamp(1, 4096), &mut rng);
            let sim = self.simulate(&cells.at, &sources, &noise, false);
            let before = out.len();
            for s in 0..noise.m {
                if out.len() == m {
                    break;
                }
                if cells.given.iter().all(|&(c, l)| sim.hard[c][s] == l) {
                    out.push(
                        events
                            .iter()
                            .map(|&c| self.spec.levels[cells.at[c].1][sim.hard[c][s]].clone())
                            .collect(),
                    );
                }
            }
            dry = if out.len() == before { dry + 1 } else { 0 };
            if dry >= 1000 {
                return Err(NcmError::ImpossibleCondition);
            }
        }
        Ok(out)
    }

    /// Eq. 13.151: the share of draws satisfying `x_*` that also satisfy
    /// `y_*`, over `m` draws of `Û`.
    pub(crate) fn monte_carlo(&self, e: &Expanded, m: usize, seed: u64) -> Result<f64, NcmError> {
        let cells = self.cells(e)?;
        let sources = Self::query_sources(&cells);
        let noise = self.draw(m, &mut Rng::new(seed));
        let sim = self.simulate(&cells.at, &sources, &noise, false);
        let (mut hit, mut kept) = (0usize, 0usize);
        for s in 0..m {
            if cells.given.iter().all(|&(c, l)| sim.hard[c][s] == l) {
                kept += 1;
                hit += usize::from(cells.events.iter().all(|&(c, l)| sim.hard[c][s] == l));
            }
        }
        if kept == 0 {
            return Err(NcmError::ImpossibleCondition);
        }
        Ok(hit as f64 / kept as f64)
    }

    /// The table's distribution and the model's, compared by the squared
    /// maximum mean discrepancy with a sum of Gaussian kernels over the
    /// concatenated one-hot rows. Intervened variables are set per draw to
    /// the values of rows resampled from the table.
    pub(crate) fn mmd(&self, t: &Table, noise: &Noise, rng: &mut Rng) -> Tensor<1> {
        let spec = &self.spec;
        let (m, nv) = (noise.m, spec.names.len());
        let cum: Vec<f64> = t
            .weights
            .iter()
            .scan(0.0, |a, w| {
                *a += w;
                Some(*a)
            })
            .collect();
        let picks: Vec<usize> = (0..m)
            .map(|_| {
                let u = f64::from(rng.unit());
                cum.iter().position(|c| u < *c).unwrap_or(cum.len() - 1)
            })
            .collect();
        let at: Vec<(usize, usize)> = (0..nv).map(|v| (0, v)).collect();
        let sources: Vec<Source> = (0..nv)
            .map(|v| {
                if t.intervened[v] {
                    Source::PerDraw(picks.iter().map(|&r| t.rows[r][v]).collect())
                } else {
                    Source::Free
                }
            })
            .collect();
        let sim = self.simulate(&at, &sources, noise, true);
        let x = Tensor::cat(sim.value, 1);
        let width: usize = spec.levels.iter().map(Vec::len).sum();
        let n = t.rows.len();
        let mut y = Vec::with_capacity(n * width);
        for r in &t.rows {
            for (v, l) in spec.levels.iter().enumerate() {
                y.extend((0..l.len()).map(|i| f32::from(u8::from(i == r[v]))));
            }
        }
        let y = self.matrix(y, n, width);
        let w = self.matrix(t.weights.iter().map(|w| *w as f32).collect(), n, 1);
        // The unbiased U-statistic: the diagonal k(x, x) would add a
        // penalty on the model's entropy. Rows are summed first, since one
        // f32 sum over m² terms loses most of its digits once m is in the
        // thousands.
        let kxx = kernel(x.clone(), x.clone()).sum_dim(1).sum();
        let diag: f32 = KERNELS.len() as f32 * m as f32;
        let xx = (kxx - diag).div_scalar((m * m.saturating_sub(1)).max(1) as f32);
        let xy = (kernel(x, y.clone()).mean_dim(0).transpose() * w.clone()).sum();
        let yy = (kernel(y.clone(), y).matmul(w.clone()) * w).sum();
        xx - xy.mul_scalar(2.0) + yy
    }

    /// One term of Table 13.6: `−log ŷ` per accepted draw to push `P(y_* |
    /// x_*)` up, `−log(1 − ŷ)` to push it down, where `ŷ` is the product of
    /// the events' softmax probabilities.
    pub(crate) fn query_log_loss(
        &self,
        e: &Expanded,
        noise: &Noise,
        up: bool,
    ) -> Result<Tensor<1>, NcmError> {
        let cells = self.cells(e)?;
        let sources = Self::query_sources(&cells);
        let sim = self.simulate(&cells.at, &sources, noise, true);
        let m = noise.m;
        let mut y: Tensor<2> = Tensor::ones([m, 1], &self.device);
        for &(c, l) in &cells.events {
            let d = self.spec.levels[cells.at[c].1].len();
            let pick = self.one_hot(&vec![l; m], d);
            y = y * (sim.soft[c].clone() * pick).sum_dim(1);
        }
        let keep: Vec<f32> = (0..m)
            .map(|s| {
                f32::from(u8::from(
                    cells.given.iter().all(|&(c, l)| sim.hard[c][s] == l),
                ))
            })
            .collect();
        let kept: f32 = keep.iter().sum();
        if kept == 0.0 {
            return Ok(Tensor::zeros([1], &self.device));
        }
        let y = y.clamp(1e-6, 1.0 - 1e-6);
        let loss = if up {
            y.log().neg()
        } else {
            y.neg().add_scalar(1.0).log().neg()
        };
        Ok((loss * self.matrix(keep, m, 1)).sum().div_scalar(kept))
    }
}

/// The bandwidths `σ²` of the kernel. One-hot rows differ by squared
/// distances 2, 4, …, so these range from nearly a point mass to smooth.
const KERNELS: [f32; 3] = [0.5, 2.0, 8.0];

/// `Σ_σ exp(−‖a − b‖² / 2σ²)` for every pair of rows.
fn kernel(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let (n, k) = (a.dims()[0], b.dims()[0]);
    let a2 = (a.clone() * a.clone()).sum_dim(1).expand([n, k]);
    let b2 = (b.clone() * b.clone())
        .sum_dim(1)
        .transpose()
        .expand([n, k]);
    let d = (a2 + b2 - a.matmul(b.transpose()).mul_scalar(2.0)).clamp_min(0.0);
    KERNELS
        .iter()
        .map(|s2| d.clone().div_scalar(-2.0 * s2).exp())
        .reduce(|x, y| x + y)
        .expect("bandwidths")
}
