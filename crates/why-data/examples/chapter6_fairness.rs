//! Chapter 6 on a synthetic hiring dataset: bias detection with the Fairness
//! Cookbook (Task 1, Algorithm 15), then fair prediction by in-processing
//! (Task 2, Thm 6.1.32) and by Causal IF (Task 2, Algorithm 16).
//!
//! The book has no chapter 6 notebook to port, so the ground truth comes from
//! the generating SCM. Every unit keeps its exogenous noise, which gives its
//! counterfactual outcomes `Y_{x1, W_{x0}}` and `Y_{x1}` exactly. The
//! estimates from the observed columns alone must match those sample
//! averages, and the run asserts that.
//!
//!   Z = age        (young / old)             not caused by gender
//!   X = gender     (female / male)           more men among older applicants
//!   W = education  (low / high)              men pushed towards "high"
//!   Y = hired      (0 / 1)                   direct + via education + via age
//!
//! Run with `cargo run -p why-data --example chapter6_fairness`.

use why_data::fairness::{
    BootstrapConfig, BusinessNecessity, CausalIf, Constraints, Cookbook, Dataset, FairPredictor,
    Sfm, Verdict, estimate,
};

/// Tiny deterministic generator so the example has no dependencies.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn education(male: bool, old: bool, u: f64) -> bool {
    u < 0.30 + 0.20 * f64::from(u8::from(male)) + 0.15 * f64::from(u8::from(old))
}

fn hired(male: bool, old: bool, high: bool, u: f64) -> bool {
    let p = 0.10
        + 0.08 * f64::from(u8::from(male)) // direct: disparate treatment
        + 0.30 * f64::from(u8::from(high))
        + 0.10 * f64::from(u8::from(old));
    u < p
}

fn main() {
    const N: usize = 50_000;
    let mut rng = Lcg(2024);
    let mut records = Vec::with_capacity(N);
    // Unit-level counterfactual sums over the x0 = female group.
    let (mut n0, mut n1) = (0.0, 0.0);
    let (mut nested, mut cross, mut fact0, mut fact1) = (0.0, 0.0, 0.0, 0.0);
    let mut reverse = 0.0;

    for _ in 0..N {
        let (uz, ux, uw, uy) = (rng.unit(), rng.unit(), rng.unit(), rng.unit());
        let old = uz < 0.5;
        let male = ux < if old { 0.70 } else { 0.45 };
        let high = education(male, old, uw);
        let y = hired(male, old, high, uy);
        let as_f = |b: bool| f64::from(u8::from(b));

        if male {
            n1 += 1.0;
            fact1 += as_f(y);
        } else {
            n0 += 1.0;
            fact0 += as_f(y);
            // Same noise, other world: X set to male, W kept / recomputed.
            nested += as_f(hired(true, old, high, uy));
            cross += as_f(hired(true, old, education(true, old, uw), uy));
            reverse += as_f(hired(false, old, education(true, old, uw), uy));
        }
        records.push([
            if old { "old" } else { "young" },
            if male { "male" } else { "female" },
            if high { "high" } else { "low" },
            if y { "1" } else { "0" },
        ]);
    }
    let (nested, cross, fact0, fact1) = (nested / n0, cross / n0, fact0 / n0, fact1 / n1);
    let reverse = reverse / n0;
    let truth = [
        ("TV", fact1 - fact0),
        ("x-DE", nested - fact0),
        ("x-IE", nested - cross),
        ("x-SE", cross - fact1),
        ("DEsym", (nested - fact0 - (reverse - cross)) / 2.0),
        ("IEsym", (reverse - fact0 - (nested - cross)) / 2.0),
    ];

    let kept = records.clone();
    let data = Dataset::from_records(["age", "gender", "education", "hired"], records).unwrap();
    let sfm = Sfm::new("gender", ["age"], ["education"], "hired");
    let m = estimate(&data, &sfm, "female", "male").unwrap();

    println!("Point estimates vs unit-level counterfactual truth (x0 = female, x1 = male)");
    for ((name, t), est) in truth.iter().zip([m.tv, m.de, m.ie, m.se, m.de_sym, m.ie_sym]) {
        println!("  {name:<5} estimate {est:+.4}   truth {t:+.4}");
        assert!((est - t).abs() < 0.015, "{name}: {est} vs {t}");
    }
    assert!(m.residual().abs() < 1e-9 && m.residual_sym().abs() < 1e-9);
    // Hand-derived population values, for orientation: x-DE = 0.08 exactly;
    // x-IE = −0.30 · 0.20 = −0.06; x-SE < 0 since men skew old and age helps.
    assert!((m.de - 0.08).abs() < 0.015);
    assert!((m.ie + 0.06).abs() < 0.015);
    assert!(m.se < 0.0);
    println!("  TV − (DE − IE − SE) = {:.1e}\n", m.residual());

    let bootstrap = BootstrapConfig {
        replicates: 300,
        seed: 1,
    };
    let audit = Cookbook::new(sfm.clone(), "female", "male").with_bootstrap(bootstrap);

    let strict = audit.run(&data).unwrap();
    println!("{strict}\n");
    assert_eq!(strict.disparate_treatment, Verdict::Discrimination);
    assert_eq!(strict.indirect, Verdict::Discrimination);

    // Education and age accepted as job-relevant: only the direct path is
    // still judged.
    let lenient = audit
        .clone()
        .with_business_necessity(BusinessNecessity::ZW)
        .run(&data)
        .unwrap();
    println!("{lenient}\n");

    // Auditing a predictor: "fairness through unawareness" drops gender and
    // predicts E[hired | age, education]. That makes x-DE exactly zero, and
    // with education in the BN-set its indirect effect is compared with the
    // true outcome's. But the prediction still reads age, which men skew
    // towards, so the spurious pathway (not in the BN-set) stays non-zero.
    let mut sums = std::collections::HashMap::<(&str, &str), (f64, f64)>::new();
    for r in &kept {
        let e = sums.entry((r[0], r[2])).or_default();
        e.0 += r[3].parse::<f64>().unwrap();
        e.1 += 1.0;
    }
    let rows: Vec<Vec<String>> = kept
        .iter()
        .map(|r| {
            let (s, n) = sums[&(r[0], r[2])];
            let mut row: Vec<String> = r.iter().map(|v| v.to_string()).collect();
            row.push(format!("{}", s / n));
            row
        })
        .collect();
    let with_pred =
        Dataset::from_records(["age", "gender", "education", "hired", "unaware"], rows).unwrap();
    let unaware = Cookbook::new(
        Sfm::new("gender", ["age"], ["education"], "unaware"),
        "female",
        "male",
    )
    .with_reference("hired")
    .with_business_necessity(BusinessNecessity::W)
    .with_bootstrap(bootstrap)
    .run(&with_pred)
    .unwrap();
    println!("Predictor `unaware` = E[hired | age, education], audited against `hired`:");
    println!("{unaware}");
    assert_eq!(unaware.de.estimate, 0.0);
    assert!(!unaware.de.rejects());
    assert_eq!(unaware.spurious, Verdict::Discrimination);

    // Task 2. The Fair Prediction Theorem: forcing demographic parity alone
    // leaves the causal disparities in place, just rearranged.
    println!("\nTask 2: fair predictors of `hired` (x-DE^sym, x-IE^sym, x-SE, TV; MSE)");
    let rows = [
        ("unconstrained", Constraints::None),
        ("TV = 0 only", Constraints::TotalVariation),
        ("causal, BN = ∅", Constraints::Causal(BusinessNecessity::NONE)),
        ("causal, BN = {W}", Constraints::Causal(BusinessNecessity::W)),
    ];
    for (name, c) in rows {
        let fit = FairPredictor::fit(&data, &sfm, "female", "male", c).unwrap();
        let out = fit.with_predictions(&data, "pred").unwrap();
        let p = estimate(&out, &sfm.with_outcome("pred"), "female", "male").unwrap();
        println!(
            "  {name:<18} {:+.4} {:+.4} {:+.4} {:+.4}   MSE {:.5}",
            p.de_sym, p.ie_sym, p.se, p.tv, fit.loss
        );
        match c {
            Constraints::TotalVariation => {
                assert_eq!(p.tv, 0.0);
                assert!(p.de_sym.abs() > 0.01, "FPT: TV = 0 should not remove x-DE here");
            }
            Constraints::Causal(bn) => {
                assert_eq!((p.de_sym, p.se), (0.0, 0.0));
                if bn.w {
                    assert!((p.ie_sym - m.ie_sym).abs() < 1e-9);
                } else {
                    assert_eq!(p.ie_sym, 0.0);
                }
            }
            Constraints::None => {}
        }
    }

    // Algorithm 16: transport the female applicants' data onto the male
    // applicants', one variable at a time.
    println!("\nTask 2: Causal IF pre-processing (x-DE^sym, x-IE^sym, x-SE)");
    for bn in [BusinessNecessity::NONE, BusinessNecessity::W, BusinessNecessity::Z] {
        let fair = CausalIf::new(sfm.clone(), "female", "male")
            .with_business_necessity(bn)
            .transform(&data)
            .unwrap();
        let p = estimate(&fair, &sfm, "female", "male").unwrap();
        println!(
            "  BN-set {bn:<7} {:+.4} {:+.4} {:+.4}",
            p.de_sym, p.ie_sym, p.se
        );
        assert_eq!(p.de_sym, 0.0);
        if !bn.w {
            assert_eq!(p.ie_sym, 0.0);
        }
        if !bn.z {
            assert_eq!(p.se, 0.0);
        } else {
            assert!((p.se - m.se).abs() < 1e-9);
        }
    }
}
