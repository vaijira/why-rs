//! A small dense linear-program solver: two-phase tableau simplex with
//! Bland's rule.
//!
//! Chapter 9 needs linear programs in four places: the optimistic planning
//! of Algorithms 24 and 25, the zero-sum game of Causal MWAL
//! (Prop. 9.5.5), and support enumeration for Nash equilibria (Sec. 9.6).
//! All of them are tiny, so a dense tableau is enough. Bland's rule
//! guarantees termination on degenerate programs.

/// The sense of one constraint `a·x ⋚ b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cmp {
    Le,
    Ge,
    Eq,
}

/// `max c·x` subject to linear constraints and `x ≥ 0`.
#[derive(Debug, Clone)]
pub(crate) struct Lp {
    n: usize,
    objective: Vec<f64>,
    rows: Vec<(Vec<f64>, Cmp, f64)>,
}

/// The result of [`Lp::solve`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LpOutcome {
    Optimal { x: Vec<f64>, value: f64 },
    Infeasible,
    Unbounded,
}

const EPS: f64 = 1e-10;

impl Lp {
    /// A program over `n` non-negative variables with a zero objective.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            objective: vec![0.0; n],
            rows: Vec::new(),
        }
    }

    /// Sets the objective to maximise.
    pub(crate) fn maximize(&mut self, c: Vec<f64>) {
        assert_eq!(c.len(), self.n, "one coefficient per variable");
        self.objective = c;
    }

    /// Adds `a·x ⋚ b`.
    pub(crate) fn constraint(&mut self, a: Vec<f64>, cmp: Cmp, b: f64) {
        assert_eq!(a.len(), self.n, "one coefficient per variable");
        self.rows.push((a, cmp, b));
    }

    pub(crate) fn solve(&self) -> LpOutcome {
        // Normalise to b ≥ 0.
        let rows: Vec<(Vec<f64>, Cmp, f64)> = self
            .rows
            .iter()
            .map(|(a, cmp, b)| {
                if *b < 0.0 {
                    let flipped = match cmp {
                        Cmp::Le => Cmp::Ge,
                        Cmp::Ge => Cmp::Le,
                        Cmp::Eq => Cmp::Eq,
                    };
                    (a.iter().map(|v| -v).collect(), flipped, -b)
                } else {
                    (a.clone(), *cmp, *b)
                }
            })
            .collect();
        let m = rows.len();
        let n_slack = rows.iter().filter(|(_, c, _)| *c != Cmp::Eq).count();
        let n_art = rows.iter().filter(|(_, c, _)| *c != Cmp::Le).count();
        let cols = self.n + n_slack + n_art;
        let first_art = self.n + n_slack;

        let mut t = vec![vec![0.0; cols + 1]; m];
        let mut basis = vec![0usize; m];
        let (mut s, mut a) = (self.n, first_art);
        for (i, (coef, cmp, b)) in rows.iter().enumerate() {
            t[i][..self.n].copy_from_slice(coef);
            t[i][cols] = *b;
            match cmp {
                Cmp::Le => {
                    t[i][s] = 1.0;
                    basis[i] = s;
                    s += 1;
                }
                Cmp::Ge => {
                    t[i][s] = -1.0;
                    s += 1;
                    t[i][a] = 1.0;
                    basis[i] = a;
                    a += 1;
                }
                Cmp::Eq => {
                    t[i][a] = 1.0;
                    basis[i] = a;
                    a += 1;
                }
            }
        }

        // Phase 1: maximise −Σ artificials.
        if n_art > 0 {
            let mut cost = vec![0.0; cols];
            for c in cost.iter_mut().skip(first_art) {
                *c = -1.0;
            }
            let mut obj = objective_row(&t, &basis, &cost, cols);
            if simplex(&mut t, &mut basis, &mut obj, cols).is_err() {
                // Phase 1 is bounded by construction.
                return LpOutcome::Infeasible;
            }
            if obj[cols] < -1e-8 {
                return LpOutcome::Infeasible;
            }
            // Drive artificials out of the basis, dropping redundant rows.
            let mut i = 0;
            while i < t.len() {
                if basis[i] >= first_art {
                    if let Some(j) = (0..first_art).find(|&j| t[i][j].abs() > 1e-9) {
                        pivot(&mut t, &mut obj, i, j, cols);
                        basis[i] = j;
                        i += 1;
                    } else {
                        t.remove(i);
                        basis.remove(i);
                    }
                } else {
                    i += 1;
                }
            }
            // Forbid artificials from re-entering.
            for row in &mut t {
                for v in row.iter_mut().take(cols).skip(first_art) {
                    *v = 0.0;
                }
            }
        }

        let mut cost = vec![0.0; cols];
        cost[..self.n].copy_from_slice(&self.objective);
        let mut obj = objective_row(&t, &basis, &cost, cols);
        for v in obj.iter_mut().take(cols).skip(first_art) {
            *v = 0.0;
        }
        if simplex(&mut t, &mut basis, &mut obj, first_art).is_err() {
            return LpOutcome::Unbounded;
        }
        let mut x = vec![0.0; self.n];
        for (i, &b) in basis.iter().enumerate() {
            if b < self.n {
                x[b] = t[i][cols].max(0.0);
            }
        }
        let value = self.objective.iter().zip(&x).map(|(c, v)| c * v).sum();
        LpOutcome::Optimal { x, value }
    }
}

/// Reduced costs `c_B B⁻¹ A_j − c_j` and the objective value.
fn objective_row(t: &[Vec<f64>], basis: &[usize], cost: &[f64], cols: usize) -> Vec<f64> {
    let mut obj: Vec<f64> = cost.iter().map(|c| -c).collect();
    obj.push(0.0);
    for (i, &b) in basis.iter().enumerate() {
        let cb = cost[b];
        if cb != 0.0 {
            for j in 0..=cols {
                obj[j] += cb * t[i][j];
            }
        }
    }
    obj
}

fn pivot(t: &mut [Vec<f64>], obj: &mut [f64], r: usize, c: usize, cols: usize) {
    let p = t[r][c];
    for v in t[r].iter_mut() {
        *v /= p;
    }
    let pivot_row = t[r].clone();
    for (i, row) in t.iter_mut().enumerate() {
        if i != r {
            let f = row[c];
            if f != 0.0 {
                for j in 0..=cols {
                    row[j] -= f * pivot_row[j];
                }
            }
        }
    }
    let f = obj[c];
    if f != 0.0 {
        for j in 0..=cols {
            obj[j] -= f * pivot_row[j];
        }
    }
}

/// Primal simplex over the columns `0..enter_limit`. `Err` if unbounded.
fn simplex(
    t: &mut [Vec<f64>],
    basis: &mut [usize],
    obj: &mut [f64],
    enter_limit: usize,
) -> Result<(), ()> {
    let cols = obj.len() - 1;
    loop {
        // Bland: the lowest-index improving column.
        let Some(c) = (0..enter_limit).find(|&j| obj[j] < -EPS) else {
            return Ok(());
        };
        let mut best: Option<(usize, f64)> = None;
        for (i, row) in t.iter().enumerate() {
            if row[c] > EPS {
                let ratio = row[cols] / row[c];
                let better = match best {
                    None => true,
                    Some((bi, br)) => {
                        ratio < br - EPS || ((ratio - br).abs() <= EPS && basis[i] < basis[bi])
                    }
                };
                if better {
                    best = Some((i, ratio));
                }
            }
        }
        let Some((r, _)) = best else {
            return Err(());
        };
        pivot(t, obj, r, c, cols);
        basis[r] = c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optimum(lp: &Lp) -> (Vec<f64>, f64) {
        match lp.solve() {
            LpOutcome::Optimal { x, value } => (x, value),
            other => panic!("expected an optimum, got {other:?}"),
        }
    }

    #[test]
    fn textbook_programs() {
        // max 3x + 5y, x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18 → (2, 6), 36.
        let mut lp = Lp::new(2);
        lp.maximize(vec![3.0, 5.0]);
        lp.constraint(vec![1.0, 0.0], Cmp::Le, 4.0);
        lp.constraint(vec![0.0, 2.0], Cmp::Le, 12.0);
        lp.constraint(vec![3.0, 2.0], Cmp::Le, 18.0);
        let (x, v) = optimum(&lp);
        assert!((v - 36.0).abs() < 1e-9 && (x[0] - 2.0).abs() < 1e-9 && (x[1] - 6.0).abs() < 1e-9);

        // Equalities and ≥: min x + y s.t. x + 2y ≥ 4, x − y = 1 → (2, 1).
        let mut lp = Lp::new(2);
        lp.maximize(vec![-1.0, -1.0]);
        lp.constraint(vec![1.0, 2.0], Cmp::Ge, 4.0);
        lp.constraint(vec![1.0, -1.0], Cmp::Eq, 1.0);
        let (x, v) = optimum(&lp);
        assert!((v + 3.0).abs() < 1e-9 && (x[0] - 2.0).abs() < 1e-9);

        // A redundant equality survives phase 1.
        let mut lp = Lp::new(2);
        lp.maximize(vec![1.0, 0.0]);
        lp.constraint(vec![1.0, 1.0], Cmp::Eq, 1.0);
        lp.constraint(vec![2.0, 2.0], Cmp::Eq, 2.0);
        assert!((optimum(&lp).1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn infeasible_and_unbounded() {
        let mut lp = Lp::new(1);
        lp.constraint(vec![1.0], Cmp::Ge, 2.0);
        lp.constraint(vec![1.0], Cmp::Le, 1.0);
        assert_eq!(lp.solve(), LpOutcome::Infeasible);

        let mut lp = Lp::new(2);
        lp.maximize(vec![1.0, 1.0]);
        lp.constraint(vec![1.0, -1.0], Cmp::Le, 1.0);
        assert_eq!(lp.solve(), LpOutcome::Unbounded);
    }
}
