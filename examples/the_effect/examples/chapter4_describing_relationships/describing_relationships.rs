//! The Rust translation of `describing_relationships.py`, the Describing
//! Relationships chapter of *The Effect*.
//!
//! Each numbered step below is the same numbered step in the Python file:
//!
//! | Python | here |
//! | --- | --- |
//! | `causaldata.Mroz.load_pandas().data` | [`causaldata::datasets::mroz::load`] |
//! | `pandas` filtering/assign/groupby/cut | polars eager ops and plain folds |
//! | `seaborn` / `seaborn.objects` | `plotters`, written out as SVG |
//! | `statsmodels.formula.api.ols` | `inferust::regression::ols::Ols` |
//! | `lowess=True` | [`loess`] below |
//!
//! The two plots are written to `plots/` rather than shown in a window, since
//! there is no notebook to draw into.

use std::error::Error;
use std::fs;
use std::path::Path;

use causaldata::datasets::mroz;
use inferust::regression::ols::Ols;
use plotters::prelude::*;
use polars::prelude::*;

/// Where the two SVGs are written.
const PLOT_DIR: &str = "plots";

fn main() -> Result<(), Box<dyn Error>> {
    // Read in data. `load` already maps the yes/no columns to booleans, which
    // is what `lfp == True` compares against on the Python side.
    let df = mroz::load()?;
    println!("total rows: {}", df.height());

    // Keep just working women.
    let lfp = df.column("lfp")?.bool()?.clone();
    let df = df.filter(&lfp)?;
    println!("working women: {}", df.height());

    let inc: Vec<f64> = df.column("inc")?.f64()?.into_no_null_iter().collect();
    let lwg: Vec<f64> = df.column("lwg")?.f64()?.into_no_null_iter().collect();
    let wc: Vec<bool> = df.column("wc")?.bool()?.iter().flatten().collect();
    let k5: Vec<f64> = df
        .column("k5")?
        .cast(&DataType::Float64)?
        .f64()?
        .into_no_null_iter()
        .collect();

    // Create unlogged earnings.
    let earn: Vec<f64> = lwg.iter().map(|v| v.exp()).collect();

    fs::create_dir_all(PLOT_DIR)?;

    // 1. Draw a scatterplot, on log-log axes.
    scatter_log_log(&inc, &earn, Path::new(PLOT_DIR).join("1_scatter.svg"))?;
    println!("\n1. scatterplot -> {PLOT_DIR}/1_scatter.svg");

    // 2. Get the conditional mean by college attendance. `wc` is the college
    // variable.
    println!("\n2. mean earn by wc");
    let (attended, did_not): (Vec<f64>, Vec<f64>) = earn
        .iter()
        .zip(&wc)
        .partition_map(|(e, college)| (*e, *college));
    println!("   wc=false  n={:>3}  earn={:.6}", did_not.len(), mean(&did_not));
    println!("   wc=true   n={:>3}  earn={:.6}", attended.len(), mean(&attended));

    // 3. Get the conditional mean by bins -- ten equal-width bins over `inc`,
    // the same partition `pd.cut(x, 10)` makes.
    println!("\n3. mean earn by inc bin");
    for (lo, hi, n, m) in binned_means(&inc, &earn, 10) {
        if n == 0 {
            println!("   ({lo:>7.3}, {hi:>7.3}]  n={n:>3}  earn=");
        } else {
            println!("   ({lo:>7.3}, {hi:>7.3}]  n={n:>3}  earn={m:.6}");
        }
    }

    // From here the regressors are logged, so the one observation with a
    // non-positive `inc` has to go: log(inc) is undefined there. `statsmodels`
    // drops it implicitly when the NaN reaches the fit.
    let (linc, lwg, wc, k5) = {
        let keep: Vec<usize> = (0..inc.len()).filter(|&i| inc[i] > 0.0).collect();
        let pick = |src: &[f64]| -> Vec<f64> { keep.iter().map(|&i| src[i]).collect() };
        (
            keep.iter().map(|&i| inc[i].ln()).collect::<Vec<f64>>(),
            pick(&lwg),
            keep.iter().map(|&i| f64::from(wc[i])).collect::<Vec<f64>>(),
            pick(&k5),
        )
    };
    println!("\nrows entering the regressions: {}", linc.len());

    // 4. Draw the LOESS and linear regression curves. Do log beforehand for
    // these axes.
    loess_plot(&linc, &lwg, Path::new(PLOT_DIR).join("4_loess.svg"))?;
    println!("\n4. LOESS + fit -> {PLOT_DIR}/4_loess.svg");

    // 5. Run a linear regression, by itself and including controls.
    println!("\n5. m1: lwg ~ linc");
    let x1: Vec<Vec<f64>> = linc.iter().map(|v| vec![*v]).collect();
    Ols::new()
        .with_feature_names(vec!["linc".into()])
        .fit(&x1, &lwg)?
        .print_summary();

    // `k5` is the number of kids under 5 in the house.
    println!("\n   m2: lwg ~ linc + wc + k5");
    let x2: Vec<Vec<f64>> = (0..linc.len())
        .map(|i| vec![linc[i], wc[i], k5[i]])
        .collect();
    Ols::new()
        .with_feature_names(vec!["linc".into(), "wc".into(), "k5".into()])
        .fit(&x2, &lwg)?
        .print_summary();

    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Equal-width bins over the range of `xs`, with the mean of `ys` in each.
///
/// Returns `(lower, upper, count, mean)` per bin. `pd.cut` nudges the lowest
/// edge down by 0.1% of the range so the minimum falls inside the first
/// (half-open) interval; this does the same.
fn binned_means(xs: &[f64], ys: &[f64], bins: usize) -> Vec<(f64, f64, usize, f64)> {
    let (min, max) = xs.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let span = max - min;
    let lo = min - span * 0.001;
    let width = (max - lo) / bins as f64;

    let mut sums = vec![0.0; bins];
    let mut counts = vec![0usize; bins];

    for (x, y) in xs.iter().zip(ys) {
        // The last bin is closed on the right, so `max` lands in it rather
        // than one past the end.
        let idx = (((x - lo) / width).ceil() as usize).clamp(1, bins) - 1;
        sums[idx] += y;
        counts[idx] += 1;
    }

    (0..bins)
        .map(|i| {
            let m = if counts[i] == 0 { f64::NAN } else { sums[i] / counts[i] as f64 };
            (lo + width * i as f64, lo + width * (i + 1) as f64, counts[i], m)
        })
        .collect()
}

/// LOESS: locally weighted linear regression, evaluated at every `x`.
///
/// This is the smoother `seaborn` reaches for when `lowess=True`, with the same
/// defaults `statsmodels.nonparametric.smoothers_lowess.lowess` uses: a
/// neighbourhood of `frac` of the sample, tricube distance weights, and `it`
/// robustifying passes that down-weight points with large residuals by the
/// bisquare function.
fn loess(xs: &[f64], ys: &[f64], frac: f64, it: usize) -> Vec<f64> {
    let n = xs.len();
    let span = ((frac * n as f64).round() as usize).clamp(2, n);

    // Work in x-order so the neighbourhood of a point is a contiguous window.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| xs[*a].total_cmp(&xs[*b]));
    let x: Vec<f64> = order.iter().map(|&i| xs[i]).collect();
    let y: Vec<f64> = order.iter().map(|&i| ys[i]).collect();

    let mut robustness = vec![1.0; n];
    let mut fitted = vec![0.0; n];

    for pass in 0..=it {
        for i in 0..n {
            // The `span` nearest neighbours of x[i], as a sliding window.
            let mut lo = i.saturating_sub(span - 1);
            let mut hi = lo + span - 1;
            while hi < n - 1 && (x[i] - x[lo]) > (x[hi + 1] - x[i]) {
                lo += 1;
                hi += 1;
            }
            hi = hi.min(n - 1);

            let d_max = (x[i] - x[lo]).abs().max((x[hi] - x[i]).abs());

            // Weighted least squares on the window: tricube on distance,
            // times the robustness weight carried over from the last pass.
            let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for j in lo..=hi {
                let w = if d_max <= 0.0 { 1.0 } else { tricube((x[i] - x[j]).abs() / d_max) }
                    * robustness[j];
                if w <= 0.0 {
                    continue;
                }
                sw += w;
                swx += w * x[j];
                swy += w * y[j];
                swxx += w * x[j] * x[j];
                swxy += w * x[j] * y[j];
            }

            let denom = sw * swxx - swx * swx;
            fitted[i] = if denom.abs() < f64::EPSILON {
                swy / sw
            } else {
                let slope = (sw * swxy - swx * swy) / denom;
                let intercept = (swy - slope * swx) / sw;
                intercept + slope * x[i]
            };
        }

        if pass == it {
            break;
        }

        // Bisquare re-weighting on the residuals, scaled by their median.
        let mut resid: Vec<f64> = (0..n).map(|i| (y[i] - fitted[i]).abs()).collect();
        let mut sorted = resid.clone();
        sorted.sort_by(f64::total_cmp);
        let mad = sorted[n / 2];
        for (r, w) in resid.iter_mut().zip(robustness.iter_mut()) {
            *w = if mad <= 0.0 { 1.0 } else { bisquare(*r / (6.0 * mad)) };
        }
    }

    // Back to the caller's row order.
    let mut out = vec![0.0; n];
    for (slot, &i) in order.iter().enumerate() {
        out[i] = fitted[slot];
    }
    out
}

fn tricube(u: f64) -> f64 {
    if u >= 1.0 { 0.0 } else { (1.0 - u.powi(3)).powi(3) }
}

fn bisquare(u: f64) -> f64 {
    if u >= 1.0 { 0.0 } else { (1.0 - u * u).powi(2) }
}

/// Least squares slope and intercept of `ys` on `xs`.
fn fit_line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = mean(xs);
    let my = mean(ys);
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let slope = sxy / sxx;
    let _ = n;
    (slope, my - slope * mx)
}

/// Step 1: `so.Plot(dt, x='inc', y='earn').add(so.Dot()).scale(x='log', y='log')`.
fn scatter_log_log(
    inc: &[f64],
    earn: &[f64],
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    // A log scale cannot show a non-positive value; matplotlib drops those
    // points silently, so do the same rather than clamping them onto the axis.
    let points: Vec<(f64, f64)> = inc
        .iter()
        .zip(earn)
        .filter(|(x, y)| **x > 0.0 && **y > 0.0)
        .map(|(x, y)| (*x, *y))
        .collect();

    let root = SVGBackend::new(path.as_ref(), (720, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_lo, x_hi) = bounds(points.iter().map(|p| p.0));
    let (y_lo, y_hi) = bounds(points.iter().map(|p| p.1));

    let mut chart = ChartBuilder::on(&root)
        .caption("earn against inc (log-log)", ("sans-serif", 22))
        .margin(16)
        .x_label_area_size(44)
        .y_label_area_size(56)
        .build_cartesian_2d(
            (x_lo * 0.9..x_hi * 1.1).log_scale(),
            (y_lo * 0.9..y_hi * 1.1).log_scale(),
        )?;

    chart.configure_mesh().x_desc("inc").y_desc("earn").draw()?;
    chart.draw_series(
        points
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 2, BLUE.mix(0.5).filled())),
    )?;

    root.present()?;
    Ok(())
}

/// Step 4: `sns.regplot(x='linc', y='lwg', lowess=True, ci=None)`, with the
/// ordinary least squares line drawn alongside it for contrast.
fn loess_plot(linc: &[f64], lwg: &[f64], path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    let smoothed = loess(linc, lwg, 2.0 / 3.0, 3);
    let mut curve: Vec<(f64, f64)> = linc.iter().copied().zip(smoothed).collect();
    curve.sort_by(|a, b| a.0.total_cmp(&b.0));

    let (slope, intercept) = fit_line(linc, lwg);

    let root = SVGBackend::new(path.as_ref(), (720, 540)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_lo, x_hi) = bounds(linc.iter().copied());
    let (y_lo, y_hi) = bounds(lwg.iter().copied());
    let pad = (y_hi - y_lo) * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .caption("lwg against linc: LOESS and OLS", ("sans-serif", 22))
        .margin(16)
        .x_label_area_size(44)
        .y_label_area_size(56)
        .build_cartesian_2d(x_lo..x_hi, (y_lo - pad)..(y_hi + pad))?;

    chart.configure_mesh().x_desc("linc").y_desc("lwg").draw()?;

    chart.draw_series(
        linc.iter()
            .zip(lwg)
            .map(|(x, y)| Circle::new((*x, *y), 2, BLACK.mix(0.25).filled())),
    )?;

    chart
        .draw_series(LineSeries::new(curve, RED.stroke_width(2)))?
        .label("LOESS")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 18, y)], RED.stroke_width(2)));

    chart
        .draw_series(LineSeries::new(
            [x_lo, x_hi].map(|x| (x, intercept + slope * x)),
            BLUE.stroke_width(2),
        ))?
        .label("OLS")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 18, y)], BLUE.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.3))
        .draw()?;

    root.present()?;
    Ok(())
}

fn bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)))
}

/// `Iterator::partition` over a mapped pair, so the two conditional groups can
/// be split in one pass.
trait PartitionMap<T>: Iterator {
    fn partition_map<F>(self, f: F) -> (Vec<T>, Vec<T>)
    where
        Self: Sized,
        F: Fn(Self::Item) -> (T, bool),
    {
        let (mut yes, mut no) = (Vec::new(), Vec::new());
        for item in self {
            let (value, flag) = f(item);
            if flag { yes.push(value) } else { no.push(value) }
        }
        (yes, no)
    }
}

impl<I: Iterator, T> PartitionMap<T> for I {}
