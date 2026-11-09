//! Plug-in estimators of the TV-family measures under the Standard Fairness
//! Model: x-specific (with symmetric variants), z-specific and general.

use std::collections::{HashMap, HashSet};

use super::dataset::Column;
use super::{Dataset, FairnessError};

/// Projection of a dataset onto the Standard Fairness Model: which columns
/// play the roles of `X`, `Z`, `W` and `Y`.
///
/// By the SFM invariance theorem (Thm 6.1.14), the edges *inside* `Z` and
/// inside `W` do not matter, so this is all the estimators need. `Z` must
/// hold no descendants of `X`, and nothing may be unobserved between the
/// blocks except `X ↔ Z`. Those assumptions are what make the estimands below
/// equal the counterfactual measures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sfm {
    /// Protected attribute.
    pub x: String,
    /// Confounders: may be associated with `X`, but are not caused by it.
    pub z: Vec<String>,
    /// Mediators: caused by `X` and lying on the way to `Y`.
    pub w: Vec<String>,
    /// Outcome (or prediction, or decision) column; must be numeric.
    pub y: String,
}

impl Sfm {
    /// Builds an SFM projection from column names.
    pub fn new<Z, W>(x: impl Into<String>, z: Z, w: W, y: impl Into<String>) -> Self
    where
        Z: IntoIterator,
        Z::Item: Into<String>,
        W: IntoIterator,
        W::Item: Into<String>,
    {
        Self {
            x: x.into(),
            z: z.into_iter().map(Into::into).collect(),
            w: w.into_iter().map(Into::into).collect(),
            y: y.into(),
        }
    }

    /// The same projection with another outcome column.
    pub fn with_outcome(&self, y: impl Into<String>) -> Self {
        Self {
            y: y.into(),
            ..self.clone()
        }
    }
}

/// The `x`-specific measures for the group `X = x0`: both transitions of the
/// direct and indirect effects, their symmetric averages (Def. 6.1.32), and
/// the counterfactual means they are built from.
///
/// Two exact decompositions hold, up to rounding:
///
/// * `tv = de − ie − se` (Eq. 6.304), see [`Measures::residual`];
/// * `tv = de_sym + ie_sym − se`, the average of Eqs. 6.304 and 6.305, see
///   [`Measures::residual_sym`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measures {
    /// `TV_{x0,x1}(y) = E[Y | x1] − E[Y | x0]`.
    pub tv: f64,
    /// `x-DE_{x0,x1}(y | x0) = E[Y_{x1,W_{x0}} | x0] − E[Y_{x0} | x0]`.
    pub de: f64,
    /// `x-IE_{x1,x0}(y | x0) = E[Y_{x1,W_{x0}} | x0] − E[Y_{x1} | x0]`.
    pub ie: f64,
    /// `x-SE_{x1,x0}(y) = E[Y_{x1} | x0] − E[Y_{x1} | x1]`.
    pub se: f64,
    /// `x-DE_{x1,x0}(y | x0) = E[Y_{x0,W_{x1}} | x0] − E[Y_{x1} | x0]`.
    pub de_reverse: f64,
    /// `x-IE_{x0,x1}(y | x0) = E[Y_{x0,W_{x1}} | x0] − E[Y_{x0} | x0]`.
    pub ie_reverse: f64,
    /// `x-DE^sym_x(y | x0) = (de − de_reverse) / 2`: the measure Algorithm 15
    /// tests for disparate treatment.
    pub de_sym: f64,
    /// `x-IE^sym_x(y | x0) = (ie_reverse − ie) / 2`. It has the sign of the
    /// `x0 → x1` transition, so it enters the TV with a plus.
    pub ie_sym: f64,
    /// `E[Y_{x1, W_{x0}} | x0]`.
    pub nested: f64,
    /// `E[Y | x0] = E[Y_{x0} | x0]`.
    pub factual_x0: f64,
    /// `E[Y_{x1} | x0]`.
    pub cross: f64,
    /// `E[Y_{x0, W_{x1}} | x0]`.
    pub reverse_nested: f64,
    /// `E[Y | x1] = E[Y_{x1} | x1]`.
    pub factual_x1: f64,
}

impl Measures {
    /// `tv − (de − ie − se)`, zero up to rounding.
    pub fn residual(&self) -> f64 {
        self.tv - (self.de - self.ie - self.se)
    }

    /// `tv − (de_sym + ie_sym − se)`, zero up to rounding.
    pub fn residual_sym(&self) -> f64 {
        self.tv - (self.de_sym + self.ie_sym - self.se)
    }
}

/// Population-level measures (Defs. 6.1.14–6.1.15): total effect, natural
/// effects and experimental spurious effects. Their decomposition
/// `tv = nde − nie + (exp_se_x1 − exp_se_x0)` is Eq. 6.75.
///
/// "Population" here means the rows with `X ∈ {x0, x1}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralMeasures {
    /// `E[Y | x1] − E[Y | x0]`.
    pub tv: f64,
    /// `TE_{x0,x1}(y) = E[Y_{x1}] − E[Y_{x0}]`.
    pub te: f64,
    /// `NDE_{x0,x1}(y) = E[Y_{x1,W_{x0}}] − E[Y_{x0}]`.
    pub nde: f64,
    /// `NIE_{x1,x0}(y) = E[Y_{x1,W_{x0}}] − E[Y_{x1}]`.
    pub nie: f64,
    /// `Exp-SE_{x0}(y) = E[Y | x0] − E[Y_{x0}]`.
    pub exp_se_x0: f64,
    /// `Exp-SE_{x1}(y) = E[Y | x1] − E[Y_{x1}]`.
    pub exp_se_x1: f64,
}

/// z-specific effects in one confounder stratum (Def. 6.1.19). Under the SFM
/// they equal the (x, z)-specific ones and are the most powerful of the
/// identifiable measures (Thm 6.1.7). `te = de_sym + ie_sym` exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZEffects {
    /// `z-TE_{x0,x1}(y | z) = E[Y_{x1} − Y_{x0} | z]`.
    pub te: f64,
    /// `z-DE_{x0,x1}(y | z) = E[Y_{x1,W_{x0}} − Y_{x0} | z]`.
    pub de: f64,
    /// `z-IE_{x1,x0}(y | z) = E[Y_{x1,W_{x0}} − Y_{x1} | z]`.
    pub ie: f64,
    /// `(z-DE_{x0,x1} − z-DE_{x1,x0}) / 2`.
    pub de_sym: f64,
    /// `(z-IE_{x0,x1} − z-IE_{x1,x0}) / 2`.
    pub ie_sym: f64,
}

/// One stratum of [`z_specific_measures`].
#[derive(Debug, Clone, PartialEq)]
pub struct ZMeasures {
    /// The stratum, as `name=value` pairs.
    pub stratum: String,
    /// `P(z)` among rows with `X ∈ {x0, x1}`.
    pub weight: f64,
    /// `None` when positivity fails inside the stratum: it lacks one of the
    /// groups, or a mediator cell seen under one group is missing under the
    /// other.
    pub effects: Option<ZEffects>,
}

/// Estimates the `x`-specific measures of `TV_{x0,x1}` for `sfm.y`.
///
/// Rows whose `X` is neither `x0` nor `x1` are ignored. Every `(z, w)` cell
/// that occurs in a stratum `z` of the `x0` group must occur under both
/// levels, since the symmetric measures read each cell in both directions.
///
/// ```
/// use why_data::fairness::{Dataset, Sfm, estimate};
///
/// let csv = "x,z,w,y\n\
///            a,0,0,0\na,0,1,1\na,1,0,0\na,1,1,1\n\
///            b,0,0,1\nb,0,1,1\nb,1,1,1\nb,1,0,0\n";
/// let data = Dataset::from_csv(csv).unwrap();
/// let m = estimate(&data, &Sfm::new("x", ["z"], ["w"], "y"), "a", "b").unwrap();
/// assert!(m.residual().abs() < 1e-12 && m.residual_sym().abs() < 1e-12);
/// assert!((m.tv - 0.25).abs() < 1e-12);
/// ```
pub fn estimate(data: &Dataset, sfm: &Sfm, x0: &str, x1: &str) -> Result<Measures, FairnessError> {
    Prepared::new(data, sfm, x0, x1, &sfm.y)?.measures(None)
}

/// Estimates TE, NDE, NIE and Exp-SE. Every confounder stratum must contain
/// both groups.
pub fn general_measures(
    data: &Dataset,
    sfm: &Sfm,
    x0: &str,
    x1: &str,
) -> Result<GeneralMeasures, FairnessError> {
    let prep = Prepared::new(data, sfm, x0, x1, &sfm.y)?;
    let t = prep.tables(None)?;
    prep.general(&t)
}

/// Estimates the z-specific effects in every confounder stratum.
///
/// With `Z = ∅` there is a single stratum, and its effects are the natural
/// effects.
pub fn z_specific_measures(
    data: &Dataset,
    sfm: &Sfm,
    x0: &str,
    x1: &str,
) -> Result<Vec<ZMeasures>, FairnessError> {
    let prep = Prepared::new(data, sfm, x0, x1, &sfm.y)?;
    let t = prep.tables(None)?;
    Ok(prep.z_specific(&t))
}

/// Weighted sums per group (`0` for `x0`, `1` for `x1`), per stratum and per
/// `(z, w)` cell.
#[derive(Debug, Clone)]
pub(crate) struct Tables {
    pub(crate) n: [f64; 2],
    pub(crate) sy: [f64; 2],
    pub(crate) n_z: [Vec<f64>; 2],
    pub(crate) sy_z: [Vec<f64>; 2],
    pub(crate) n_c: [Vec<f64>; 2],
    pub(crate) sy_c: [Vec<f64>; 2],
    /// Largest `|E[Y | x, z, w]|`; bounds every mean, so it sets the
    /// rounding tolerance.
    scale: f64,
}

impl Tables {
    fn snap(&self, d: f64) -> f64 {
        if d.abs() <= 1e-9 * self.scale { 0.0 } else { d }
    }
}

/// Per-stratum counterfactual means: `a = E[Y_{x1,W_{x0}} | z]`,
/// `b = E[Y | x0, z]`, `c = E[Y | x1, z]`, `r = E[Y_{x0,W_{x1}} | z]`.
#[derive(Debug, Clone, Copy)]
struct StratumMeans {
    a: f64,
    b: f64,
    c: f64,
    r: f64,
}

/// A dataset reduced to what the estimators touch: per row, its group, its
/// `Z` stratum, its `(Z, W)` cell, outcome and weight. The bootstrap reuses
/// one `Prepared` for every replicate.
#[derive(Debug)]
pub(crate) struct Prepared {
    pub(crate) group: Vec<usize>,
    pub(crate) zid: Vec<usize>,
    pub(crate) cell: Vec<usize>,
    pub(crate) y: Vec<f64>,
    pub(crate) weight: Vec<f64>,
    /// Dataset row of each prepared row.
    pub(crate) row: Vec<usize>,
    pub(crate) cell_z: Vec<usize>,
    pub(crate) z_cells: Vec<Vec<usize>>,
    pub(crate) z_keys: Vec<Vec<u32>>,
    pub(crate) cell_w: Vec<Vec<u32>>,
    pub(crate) z_labels: Vec<String>,
    pub(crate) cell_labels: Vec<String>,
    pub(crate) x_levels: [String; 2],
}

impl Prepared {
    pub(crate) fn new(
        data: &Dataset,
        sfm: &Sfm,
        x0: &str,
        x1: &str,
        outcome: &str,
    ) -> Result<Self, FairnessError> {
        let roles = std::iter::once(sfm.x.as_str())
            .chain(sfm.z.iter().map(String::as_str))
            .chain(sfm.w.iter().map(String::as_str))
            .chain(std::iter::once(outcome));
        let mut seen = HashSet::new();
        for name in roles {
            data.column(name)?;
            if !seen.insert(name) {
                return Err(FairnessError::OverlappingRoles(name.to_string()));
            }
        }

        let xcol = data.column(&sfm.x)?;
        let (c0, c1) = contrast_codes(xcol, &sfm.x, x0, x1)?;

        let zcols = named_columns(data, &sfm.z)?;
        let wcols = named_columns(data, &sfm.w)?;
        let ycol = data.column(outcome)?;
        let yvals = numeric_levels(ycol);

        let mut prep = Self {
            group: Vec::new(),
            zid: Vec::new(),
            cell: Vec::new(),
            y: Vec::new(),
            weight: Vec::new(),
            row: Vec::new(),
            cell_z: Vec::new(),
            z_cells: Vec::new(),
            z_keys: Vec::new(),
            cell_w: Vec::new(),
            z_labels: Vec::new(),
            cell_labels: Vec::new(),
            x_levels: [x0.to_string(), x1.to_string()],
        };
        let mut z_index: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut cell_index: HashMap<(usize, Vec<u32>), usize> = HashMap::new();

        for i in 0..data.len() {
            let xc = xcol.codes[i] as usize;
            let group = match xc {
                _ if xc == c0 => 0,
                _ if xc == c1 => 1,
                _ => continue,
            };

            let zkey: Vec<u32> = zcols.iter().map(|(_, c)| c.codes[i]).collect();
            let zid = match z_index.get(&zkey) {
                Some(&id) => id,
                None => {
                    let id = prep.z_labels.len();
                    prep.z_labels.push(label(&zcols, &zkey));
                    prep.z_cells.push(Vec::new());
                    prep.z_keys.push(zkey.clone());
                    z_index.insert(zkey, id);
                    id
                }
            };

            let wkey: Vec<u32> = wcols.iter().map(|(_, c)| c.codes[i]).collect();
            let cell = match cell_index.get(&(zid, wkey.clone())) {
                Some(&id) => id,
                None => {
                    let id = prep.cell_labels.len();
                    prep.cell_labels
                        .push(format!("{}; {}", prep.z_labels[zid], label(&wcols, &wkey)));
                    prep.cell_z.push(zid);
                    prep.z_cells[zid].push(id);
                    prep.cell_w.push(wkey.clone());
                    cell_index.insert((zid, wkey), id);
                    id
                }
            };

            let ycode = ycol.codes[i] as usize;
            let y = yvals[ycode].ok_or_else(|| FairnessError::NonNumericOutcome {
                column: outcome.to_string(),
                value: ycol.levels[ycode].clone(),
            })?;

            prep.group.push(group);
            prep.zid.push(zid);
            prep.cell.push(cell);
            prep.y.push(y);
            prep.weight.push(data.weight(i));
            prep.row.push(i);
        }
        Ok(prep)
    }

    /// Number of rows with `X ∈ {x0, x1}`.
    pub(crate) fn len(&self) -> usize {
        self.y.len()
    }

    /// Weighted sums, optionally with each row's weight multiplied by a
    /// bootstrap count.
    pub(crate) fn tables(&self, multiplier: Option<&[u32]>) -> Result<Tables, FairnessError> {
        let nz = self.z_labels.len();
        let nc = self.cell_labels.len();
        let mut t = Tables {
            n: [0.0; 2],
            sy: [0.0; 2],
            n_z: [vec![0.0; nz], vec![0.0; nz]],
            sy_z: [vec![0.0; nz], vec![0.0; nz]],
            n_c: [vec![0.0; nc], vec![0.0; nc]],
            sy_c: [vec![0.0; nc], vec![0.0; nc]],
            scale: 0.0,
        };
        for i in 0..self.len() {
            let w = match multiplier {
                Some(m) => self.weight[i] * f64::from(m[i]),
                None => self.weight[i],
            };
            if w == 0.0 {
                continue;
            }
            let (g, z, c) = (self.group[i], self.zid[i], self.cell[i]);
            let wy = w * self.y[i];
            t.n[g] += w;
            t.sy[g] += wy;
            t.n_z[g][z] += w;
            t.sy_z[g][z] += wy;
            t.n_c[g][c] += w;
            t.sy_c[g][c] += wy;
        }
        for (count, level) in t.n.iter().zip(&self.x_levels) {
            if *count <= 0.0 {
                return Err(FairnessError::EmptyGroup {
                    level: level.clone(),
                });
            }
        }
        for g in 0..2 {
            for c in 0..nc {
                if t.n_c[g][c] > 0.0 {
                    t.scale = t.scale.max((t.sy_c[g][c] / t.n_c[g][c]).abs());
                }
            }
        }
        Ok(t)
    }

    fn positivity(&self, group: usize, stratum: &str) -> FairnessError {
        FairnessError::Positivity {
            level: self.x_levels[group].clone(),
            stratum: stratum.to_string(),
        }
    }

    /// The four means in stratum `z`, or the positivity failure that
    /// prevents them.
    fn stratum(&self, t: &Tables, z: usize) -> Result<StratumMeans, FairnessError> {
        for g in 0..2 {
            if t.n_z[g][z] <= 0.0 {
                return Err(self.positivity(g, &self.z_labels[z]));
            }
        }
        let (mut a, mut r) = (0.0, 0.0);
        for &c in &self.z_cells[z] {
            let (n0, n1) = (t.n_c[0][c], t.n_c[1][c]);
            if n0 > 0.0 && n1 <= 0.0 {
                return Err(self.positivity(1, &self.cell_labels[c]));
            }
            if n1 > 0.0 && n0 <= 0.0 {
                return Err(self.positivity(0, &self.cell_labels[c]));
            }
            if n0 > 0.0 {
                a += t.sy_c[1][c] / n1 * n0;
                r += t.sy_c[0][c] / n0 * n1;
            }
        }
        Ok(StratumMeans {
            a: a / t.n_z[0][z],
            b: t.sy_z[0][z] / t.n_z[0][z],
            c: t.sy_z[1][z] / t.n_z[1][z],
            r: r / t.n_z[1][z],
        })
    }

    pub(crate) fn measures(&self, multiplier: Option<&[u32]>) -> Result<Measures, FairnessError> {
        let t = self.tables(multiplier)?;
        self.x_specific(&t)
    }

    pub(crate) fn x_specific(&self, t: &Tables) -> Result<Measures, FairnessError> {
        let (mut nested, mut cross, mut reverse) = (0.0, 0.0, 0.0);
        for z in 0..self.z_labels.len() {
            let pz = t.n_z[0][z] / t.n[0];
            if pz <= 0.0 {
                continue;
            }
            let s = self.stratum(t, z)?;
            nested += pz * s.a;
            cross += pz * s.c;
            reverse += pz * s.r;
        }
        let f0 = t.sy[0] / t.n[0];
        let f1 = t.sy[1] / t.n[1];
        let de = t.snap(nested - f0);
        let ie = t.snap(nested - cross);
        let de_reverse = t.snap(reverse - cross);
        let ie_reverse = t.snap(reverse - f0);
        Ok(Measures {
            tv: t.snap(f1 - f0),
            de,
            ie,
            se: t.snap(cross - f1),
            de_reverse,
            ie_reverse,
            de_sym: (de - de_reverse) / 2.0,
            ie_sym: (ie_reverse - ie) / 2.0,
            nested,
            factual_x0: f0,
            cross,
            reverse_nested: reverse,
            factual_x1: f1,
        })
    }

    pub(crate) fn general(&self, t: &Tables) -> Result<GeneralMeasures, FairnessError> {
        let total = t.n[0] + t.n[1];
        let (mut y_x0, mut y_x1, mut nested) = (0.0, 0.0, 0.0);
        for z in 0..self.z_labels.len() {
            let pz = (t.n_z[0][z] + t.n_z[1][z]) / total;
            if pz <= 0.0 {
                continue;
            }
            let s = self.stratum(t, z)?;
            y_x0 += pz * s.b;
            y_x1 += pz * s.c;
            nested += pz * s.a;
        }
        let (f0, f1) = (t.sy[0] / t.n[0], t.sy[1] / t.n[1]);
        Ok(GeneralMeasures {
            tv: t.snap(f1 - f0),
            te: t.snap(y_x1 - y_x0),
            nde: t.snap(nested - y_x0),
            nie: t.snap(nested - y_x1),
            exp_se_x0: t.snap(f0 - y_x0),
            exp_se_x1: t.snap(f1 - y_x1),
        })
    }

    pub(crate) fn z_specific(&self, t: &Tables) -> Vec<ZMeasures> {
        let total = t.n[0] + t.n[1];
        (0..self.z_labels.len())
            .map(|z| {
                let effects = self.stratum(t, z).ok().map(|s| {
                    let de = t.snap(s.a - s.b);
                    let ie = t.snap(s.a - s.c);
                    let de_rev = t.snap(s.r - s.c);
                    let ie_rev = t.snap(s.r - s.b);
                    ZEffects {
                        te: t.snap(s.c - s.b),
                        de,
                        ie,
                        de_sym: (de - de_rev) / 2.0,
                        ie_sym: (ie_rev - ie) / 2.0,
                    }
                });
                ZMeasures {
                    stratum: self.z_labels[z].clone(),
                    weight: (t.n_z[0][z] + t.n_z[1][z]) / total,
                    effects,
                }
            })
            .collect()
    }
}

pub(crate) fn contrast_codes(
    xcol: &Column,
    name: &str,
    x0: &str,
    x1: &str,
) -> Result<(usize, usize), FairnessError> {
    let code = |level: &str| {
        xcol.levels
            .iter()
            .position(|l| l == level)
            .ok_or_else(|| FairnessError::UnknownLevel {
                column: name.to_string(),
                level: level.to_string(),
            })
    };
    let (c0, c1) = (code(x0)?, code(x1)?);
    if c0 == c1 {
        return Err(FairnessError::IdenticalContrast(x0.to_string()));
    }
    Ok((c0, c1))
}

pub(crate) fn named_columns<'a>(
    data: &'a Dataset,
    names: &[String],
) -> Result<Vec<(String, &'a Column)>, FairnessError> {
    names
        .iter()
        .map(|n| Ok((n.clone(), data.column(n)?)))
        .collect()
}

pub(crate) fn numeric_levels(col: &Column) -> Vec<Option<f64>> {
    col.levels
        .iter()
        .map(|l| l.parse::<f64>().ok().filter(|v| v.is_finite()))
        .collect()
}

fn label(cols: &[(String, &Column)], key: &[u32]) -> String {
    if cols.is_empty() {
        return "∅".to_string();
    }
    cols.iter()
        .zip(key)
        .map(|((name, col), &code)| format!("{name}={}", col.levels[code as usize]))
        .collect::<Vec<_>>()
        .join(", ")
}
