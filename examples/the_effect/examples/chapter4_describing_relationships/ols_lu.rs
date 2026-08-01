//! Solve the chapter's regressions -- `lwg ~ linc` and `lwg ~ linc + wc + k5`,
//! step 5 of `describing_relationships.rs` -- but exactly, by factoring the
//! normal equations with `burn::tensor::linalg::lu` instead of handing them to
//! an OLS routine.
//!
//! The normal equations for `y = X b` are
//!
//! ```text
//!     (X' X) b = X' y
//! ```
//!
//! `linalg::lu` gives `A = P L U`, so with `A = X'X` and `c = X'y`:
//!
//! ```text
//!     P L U b = c   =>   L U b = P' c   =>   L z = P' c,  U b = z
//! ```
//!
//! which is one forward and one back substitution. Burn has no triangular
//! solver of its own, so both are written out below with tensor ops.

use burn::data::dataset::Dataset;
use burn::prelude::*;
use burn::tensor::linalg;
use the_effect_examples::mroz::{self, MarriedWoman, Spec};

/// Solves `L z = b` for lower-triangular `L` by forward substitution.
fn forward_substitution(l: Tensor<2>, b: Tensor<1>) -> Tensor<1> {
    let n = l.dims()[0];
    let mut solved: Vec<Tensor<1>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut acc = b.clone().slice(i..i + 1);

        if i > 0 {
            // Row i of L, restricted to the columns already solved for.
            let row = l.clone().slice([i..i + 1, 0..i]).reshape([i]);
            let prefix = Tensor::cat(solved.clone(), 0);
            acc = acc - (row * prefix).sum();
        }

        let pivot = l.clone().slice([i..i + 1, i..i + 1]).reshape([1]);
        solved.push(acc / pivot);
    }

    Tensor::cat(solved, 0)
}

/// Solves `U x = z` for upper-triangular `U` by back substitution.
fn back_substitution(u: Tensor<2>, z: Tensor<1>) -> Tensor<1> {
    let n = u.dims()[0];
    let mut solved: Vec<Tensor<1>> = Vec::with_capacity(n);

    // Walk bottom-up; `solved` accumulates in reverse row order.
    for i in (0..n).rev() {
        let mut acc = z.clone().slice(i..i + 1);

        if i + 1 < n {
            let row = u.clone().slice([i..i + 1, i + 1..n]).reshape([n - i - 1]);
            let mut suffix = solved.clone();
            suffix.reverse();
            acc = acc - (row * Tensor::cat(suffix, 0)).sum();
        }

        let pivot = u.clone().slice([i..i + 1, i..i + 1]).reshape([1]);
        solved.push(acc / pivot);
    }

    solved.reverse();
    Tensor::cat(solved, 0)
}

/// Solves the square system `A x = b` via LU with partial pivoting.
fn lu_solve(a: Tensor<2>, b: Tensor<1>) -> Tensor<1> {
    let (p, l, u) = linalg::lu::<2, 1>(a.clone());

    println!("P =\n{p}");
    println!("L =\n{l}");
    println!("U =\n{u}");

    // Sanity check on the factors: the reconstruction error should be ~0.
    let reconstruction = p.clone().matmul(l.clone()).matmul(u.clone());
    let error = (reconstruction - a).abs().max().into_scalar::<f32>();
    println!("max |A - PLU| = {error:.3e}\n");

    // A = P L U, and P is a permutation, so P^-1 = P'.
    let n = b.dims()[0];
    let pb = p.transpose().matmul(b.reshape([n, 1])).reshape([n]);

    back_substitution(u, forward_substitution(l, pb))
}

/// Fits one specification by LU-solving its normal equations.
fn fit(spec: Spec, items: &[MarriedWoman], device: &Device) {
    // Design matrix X = [1, regressors...], outcome y = lwg.
    let k = spec.num_features() + 1;
    let mut design = Vec::with_capacity(items.len() * k);
    let mut outcome = Vec::with_capacity(items.len());

    for item in items {
        design.push(1.0);
        design.extend(spec.row(item));
        outcome.push(item.lwg);
    }
    let n = outcome.len();

    let x = Tensor::<1>::from_floats(design.as_slice(), device).reshape([n, k]);
    let y = Tensor::<1>::from_floats(outcome.as_slice(), device).reshape([n, 1]);

    // Normal equations: A = X'X (k x k), c = X'y (k).
    let a = x.clone().transpose().matmul(x.clone());
    let c = x.clone().transpose().matmul(y.clone()).reshape([k]);

    println!("=== lwg ~ {} ===\n", spec.regressors().join(" + "));

    let beta = lu_solve(a, c);
    let coefficients = beta.clone().into_data().to_vec::<f32>().expect("f32 betas");

    // Residual mean squared error at the solution.
    let residual = y - x.matmul(beta.reshape([k, 1]));
    let mse = residual.powf_scalar(2.0).sum().into_scalar::<f32>() / n as f32;

    println!("n = {n},  MSE = {mse:.6}");
    println!("  {:<12}{:>12.6}", "Intercept", coefficients[0]);
    for (name, value) in spec.regressors().iter().zip(&coefficients[1..]) {
        println!("  {name:<12}{value:>12.6}");
    }
    println!();
}

fn main() {
    let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    // The same rows step 5 of the chapter regresses on: working women with
    // positive `inc`.
    let data = mroz::load().unwrap();
    let items: Vec<MarriedWoman> = data.iter().filter_map(Result::ok).collect();

    for spec in [Spec::M1, Spec::M2] {
        fit(spec, &items, &device);
    }
}
