//! Task 2, fair prediction by in-processing with causal constraints
//! (Thm 6.1.32).
//!
//! With categorical `X`, `Z` and `W`, any predictor `f(x, z, w)` is a lookup
//! table over the observed cells, and every identification expression in
//! Eqs. 6.366–6.370 is a *linear* functional of that table, since
//! `E[f | x, z, w] = f(x, z, w)`. Weighted least squares under linear equality
//! constraints has the closed form
//!
//! ```text
//! f = ȳ + D⁻¹ Aᵀ λ,    (A D⁻¹ Aᵀ) λ = b − A ȳ,
//! ```
//!
//! where `ȳ` holds the cell means, `D` the cell weights, and `A f = b` the
//! constraints. So the optimum of the theorem's program is computed exactly,
//! with no iterative optimiser.

use std::collections::HashMap;

use super::cookbook::BusinessNecessity;
use super::estimate::{Prepared, Sfm, named_columns};
use super::{Dataset, FairnessError};

/// Which constraints the predictor must satisfy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraints {
    /// No constraints: the cell means `E[Y | x, z, w]`.
    None,
    /// Only `TV_{x0,x1}(ŷ) = 0`: demographic parity, the target of standard
    /// fair-ML methods. Kept as a baseline; by the Fair Prediction Theorem
    /// (Thm 6.1.30), its causal measures are almost never all zero.
    TotalVariation,
    /// Thm 6.1.32: `x-DE_{x0,x1} = x-DE_{x1,x0} = 0` always. Pathways outside
    /// the business-necessity set get `x-IE_{x0,x1} = x-IE_{x1,x0} = 0` and
    /// `x-SE = 0`. A pathway in the set is instead required to transmit
    /// exactly what it transmits for the true outcome: `x-IE^sym(ŷ) =
    /// x-IE^sym(y)`, resp. `x-SE(ŷ) = x-SE(y)`.
    ///
    /// The symmetric IE is matched rather than both transitions because the
    /// two TV decompositions imply `IE₀₁ + IE₁₀ = DE₀₁ + DE₁₀`. Once both
    /// direct effects are zero, the two IE transitions are tied together, and
    /// matching each one to `Y` would usually be infeasible.
    Causal(BusinessNecessity),
}

/// A predictor fitted by constrained least squares on the cells of `(X, Z, W)`.
///
/// ```
/// use why_data::fairness::{BusinessNecessity, Constraints, Dataset, FairPredictor, Sfm, estimate};
///
/// let csv = "x,z,w,y\n\
///            a,0,0,0\na,0,1,1\na,1,0,0\na,1,1,1\na,1,1,0\n\
///            b,0,0,1\nb,0,1,1\nb,1,1,1\nb,1,0,0\nb,0,0,0\n";
/// let data = Dataset::from_csv(csv).unwrap();
/// let sfm = Sfm::new("x", ["z"], ["w"], "y");
/// let fair = FairPredictor::fit(&data, &sfm, "a", "b", Constraints::Causal(BusinessNecessity::NONE))
///     .unwrap();
/// let data = fair.with_predictions(&data, "y_fair").unwrap();
/// let m = estimate(&data, &sfm.with_outcome("y_fair"), "a", "b").unwrap();
/// assert_eq!((m.de_sym, m.ie_sym, m.se, m.tv), (0.0, 0.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone)]
pub struct FairPredictor {
    sfm: Sfm,
    table: HashMap<(String, Vec<String>, Vec<String>), f64>,
    /// Weighted mean squared error on the training rows.
    pub loss: f64,
    /// The same loss for the unconstrained cell means: the price of the
    /// constraints is `loss − unconstrained_loss`.
    pub unconstrained_loss: f64,
}

impl FairPredictor {
    /// Fits the predictor on the rows with `X ∈ {x0, x1}`.
    pub fn fit(
        data: &Dataset,
        sfm: &Sfm,
        x0: &str,
        x1: &str,
        constraints: Constraints,
    ) -> Result<Self, FairnessError> {
        let prep = Prepared::new(data, sfm, x0, x1, &sfm.y)?;
        let t = prep.tables(None)?;
        let nc = prep.cell_labels.len();

        // Every stratum the functionals read must satisfy positivity; this
        // reuses the estimator's check and its error messages.
        if constraints != Constraints::None {
            prep.x_specific(&t)?;
        }

        // One variable per (group, cell) with data.
        let mut var = [vec![usize::MAX; nc], vec![usize::MAX; nc]];
        let (mut ybar, mut dw) = (Vec::new(), Vec::new());
        for (g, vars) in var.iter_mut().enumerate() {
            for (c, slot) in vars.iter_mut().enumerate() {
                if t.n_c[g][c] > 0.0 {
                    *slot = ybar.len();
                    ybar.push(t.sy_c[g][c] / t.n_c[g][c]);
                    dw.push(t.n_c[g][c]);
                }
            }
        }
        let nv = ybar.len();

        // The five means as linear functionals of the table.
        let mut fact = [vec![0.0; nv], vec![0.0; nv]];
        let mut nested = vec![0.0; nv];
        let mut cross = vec![0.0; nv];
        let mut reverse = vec![0.0; nv];
        for c in 0..nc {
            let z = prep.cell_z[c];
            let (n0c, n1c) = (t.n_c[0][c], t.n_c[1][c]);
            for g in 0..2 {
                if t.n_c[g][c] > 0.0 {
                    fact[g][var[g][c]] += t.n_c[g][c] / t.n[g];
                }
            }
            // With `Constraints::None` positivity was not checked, so a cell
            // may lack one group; the functionals are unused then.
            let both = n0c > 0.0 && n1c > 0.0;
            if both {
                nested[var[1][c]] += n0c / t.n[0];
            }
            let pz0 = t.n_z[0][z] / t.n[0];
            if pz0 > 0.0 && both {
                let pw1 = n1c / t.n_z[1][z];
                cross[var[1][c]] += pw1 * pz0;
                reverse[var[0][c]] += pw1 * pz0;
            }
        }
        let sub = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>();
        let dot = |a: &[f64]| a.iter().zip(&ybar).map(|(x, y)| x * y).sum::<f64>();

        let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
        match constraints {
            Constraints::None => {}
            Constraints::TotalVariation => rows.push((sub(&fact[1], &fact[0]), 0.0)),
            Constraints::Causal(bn) => {
                rows.push((sub(&nested, &fact[0]), 0.0)); // x-DE_{x0,x1}
                rows.push((sub(&reverse, &cross), 0.0)); // x-DE_{x1,x0}
                let ie01 = sub(&reverse, &fact[0]);
                let ie10 = sub(&nested, &cross);
                if bn.w {
                    let sym: Vec<f64> = sub(&ie01, &ie10).iter().map(|v| v / 2.0).collect();
                    let target = dot(&sym);
                    rows.push((sym, target));
                } else {
                    rows.push((ie01, 0.0));
                    rows.push((ie10, 0.0));
                }
                let se = sub(&cross, &fact[1]);
                let target = if bn.z { dot(&se) } else { 0.0 };
                rows.push((se, target));
            }
        }

        let k = rows.len();
        let mut gram = vec![vec![0.0; k]; k];
        let mut rhs = vec![0.0; k];
        for i in 0..k {
            rhs[i] = rows[i].1 - dot(&rows[i].0);
            for j in 0..k {
                gram[i][j] = (0..nv).map(|v| rows[i].0[v] * rows[j].0[v] / dw[v]).sum();
            }
        }
        let lambda = solve_consistent(gram, rhs)?;
        let mut f = ybar.clone();
        for (row, l) in rows.iter().zip(&lambda) {
            for v in 0..nv {
                f[v] += l * row.0[v] / dw[v];
            }
        }

        let total: f64 = prep.weight.iter().sum();
        let mut loss = 0.0;
        let mut base = 0.0;
        for i in 0..prep.len() {
            let v = var[prep.group[i]][prep.cell[i]];
            loss += prep.weight[i] * (prep.y[i] - f[v]).powi(2);
            base += prep.weight[i] * (prep.y[i] - ybar[v]).powi(2);
        }

        let zcols = named_columns(data, &sfm.z)?;
        let wcols = named_columns(data, &sfm.w)?;
        let strings = |cols: &[(String, &super::dataset::Column)], key: &[u32]| {
            cols.iter()
                .zip(key)
                .map(|((_, col), &code)| col.levels[code as usize].clone())
                .collect::<Vec<_>>()
        };
        let mut table = HashMap::new();
        for g in 0..2 {
            for c in 0..nc {
                if var[g][c] != usize::MAX {
                    table.insert(
                        (
                            prep.x_levels[g].clone(),
                            strings(&zcols, &prep.z_keys[prep.cell_z[c]]),
                            strings(&wcols, &prep.cell_w[c]),
                        ),
                        f[var[g][c]],
                    );
                }
            }
        }

        Ok(Self {
            sfm: sfm.clone(),
            table,
            loss: loss / total,
            unconstrained_loss: base / total,
        })
    }

    /// Predictions for every row of `data`, which must have the SFM's `X`,
    /// `Z` and `W` columns. A row whose `(x, z, w)` was not seen in training
    /// is an error.
    pub fn predict(&self, data: &Dataset) -> Result<Vec<f64>, FairnessError> {
        let xcol = data.column(&self.sfm.x)?;
        let zcols = named_columns(data, &self.sfm.z)?;
        let wcols = named_columns(data, &self.sfm.w)?;
        (0..data.len())
            .map(|i| {
                let get = |cols: &[(String, &super::dataset::Column)]| {
                    cols.iter()
                        .map(|(_, c)| c.levels[c.codes[i] as usize].clone())
                        .collect::<Vec<_>>()
                };
                let key = (
                    xcol.levels[xcol.codes[i] as usize].clone(),
                    get(&zcols),
                    get(&wcols),
                );
                self.table.get(&key).copied().ok_or_else(|| {
                    FairnessError::UnseenCell(format!(
                        "row {}: {}={}, Z={:?}, W={:?}",
                        i + 1,
                        self.sfm.x,
                        key.0,
                        key.1,
                        key.2
                    ))
                })
            })
            .collect()
    }

    /// `data` with the predictions appended as a new column.
    pub fn with_predictions(&self, data: &Dataset, column: &str) -> Result<Dataset, FairnessError> {
        let values = self.predict(data)?;
        data.clone()
            .with_column(column, values.into_iter().map(|v| format!("{v}")))
    }
}

/// Solves `G λ = r` for a symmetric positive semi-definite `G` that may be
/// singular, as long as the system is consistent (redundant constraints are
/// fine). Uses full pivoting; free variables are set to zero.
fn solve_consistent(mut g: Vec<Vec<f64>>, mut r: Vec<f64>) -> Result<Vec<f64>, FairnessError> {
    let k = r.len();
    let scale = g
        .iter()
        .flatten()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let tol = 1e-12 * scale;
    let mut col: Vec<usize> = (0..k).collect();
    let mut rank = 0;
    while rank < k {
        let (mut bi, mut bj, mut best) = (rank, rank, 0.0);
        for (i, row) in g.iter().enumerate().skip(rank) {
            for (j, v) in row.iter().enumerate().skip(rank) {
                if v.abs() > best {
                    (bi, bj, best) = (i, j, v.abs());
                }
            }
        }
        if best <= tol {
            break;
        }
        g.swap(rank, bi);
        r.swap(rank, bi);
        for row in g.iter_mut() {
            row.swap(rank, bj);
        }
        col.swap(rank, bj);
        for i in rank + 1..k {
            let factor = g[i][rank] / g[rank][rank];
            if factor != 0.0 {
                let (top, bottom) = g.split_at_mut(i);
                for (gij, pj) in bottom[0][rank..].iter_mut().zip(&top[rank][rank..]) {
                    *gij -= factor * pj;
                }
                r[i] -= factor * r[rank];
            }
        }
        rank += 1;
    }
    let r_scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    if let Some(bad) = r[rank..].iter().find(|v| v.abs() > 1e-8 * r_scale) {
        return Err(FairnessError::InvalidInput(format!(
            "fairness constraints are inconsistent (residual {bad:e})"
        )));
    }
    let mut sol = vec![0.0; k];
    for i in (0..rank).rev() {
        let s: f64 = (i + 1..rank).map(|j| g[i][j] * sol[j]).sum();
        sol[i] = (r[i] - s) / g[i][i];
    }
    let mut out = vec![0.0; k];
    for (i, &c) in col.iter().enumerate() {
        out[c] = sol[i];
    }
    Ok(out)
}
