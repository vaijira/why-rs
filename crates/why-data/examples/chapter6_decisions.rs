//! Chapter 6, Task 3: fair decision-making in the outcome-control setting.
//!
//! Ports the cancer-surgery examples (Ex. 6.31, 6.32, 6.35) and checks the
//! numbers the book reports:
//!
//!   X ← U_X                         x0 = male, x1 = female, P(x1) = 0.5
//!   W ← X + (−1)^X √U_W             illness severity, men more severe
//!   Y ← 1(U_Y + W·D/3 − W/5 > 0.5)  2-year survival
//!
//! so the degree of benefit is `Δ(x, w) = w/3` (Eqs. 6.448–6.449). It then
//! shows Causal Benefit Fairness on the professional-development example
//! (Ex. 6.34) and the canonical-type bounds (Prop. 6.1.40).
//!
//! Run with `cargo run -p why-data --example chapter6_decisions`.

use why_data::fairness::{
    Dataset, Sfm, benefit_fairness, canonical_type_bounds, estimate, estimate_benefit,
    utilitarian_benefit_fairness,
};

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    const N: usize = 100_000;
    let mut rng = Lcg(6);
    let mut records = Vec::with_capacity(N);
    let mut delta = Vec::with_capacity(N);
    for _ in 0..N {
        let female = rng.unit() < 0.5;
        let u = rng.unit().sqrt();
        let w = if female { 1.0 - u } else { u };
        // Historical decisions were randomised, so Δ is estimable from them.
        let d = rng.unit() < 0.5;
        let y = rng.unit() + w * f64::from(u8::from(d)) / 3.0 - w / 5.0 > 0.5;
        records.push([
            if female { "female" } else { "male" }.to_string(),
            // Severity in tenths: a categorical mediator for the estimators.
            format!("{}", (w * 10.0).floor().min(9.0)),
            u8::from(d).to_string(),
            u8::from(y).to_string(),
        ]);
        delta.push(w / 3.0);
    }
    let data = Dataset::from_records(["sex", "severity", "d_hist", "alive"], records).unwrap();

    // Algorithm 17 with budget 50%: treat W > 1/2, i.e. Δ > 1/6 (Ex. 6.31).
    let a17 = benefit_fairness(&data, "sex", "male", "female", &delta, 0.5).unwrap();
    println!("Algorithm 17 (budget 50%)");
    println!(
        "  threshold Δ > {:.4} (W > {:.3})   P(d) = {:.3}",
        a17.thresholds[0],
        3.0 * a17.thresholds[0],
        a17.rate
    );
    println!(
        "  P(d | male) = {:.3}   P(d | female) = {:.3}   disparity {:+.3}",
        a17.rates[0],
        a17.rates[1],
        a17.disparity()
    );
    assert!((a17.thresholds[0] - 1.0 / 6.0).abs() < 0.005);
    assert!((a17.disparity() + 0.5).abs() < 0.01, "book: −50%");

    // Ex. 6.32: decompose the disparity. The policy reads only severity,
    // so it is entirely indirect.
    let with_d = data.clone().with_column("d", a17.column()).unwrap();
    let sfm = Sfm::new("sex", Vec::<String>::new(), ["severity"], "d");
    let m = estimate(&with_d, &sfm, "male", "female").unwrap();
    println!(
        "  P(d|x1) − P(d|x0) = {:+.3} = DE {:+.3} + IE {:+.3} − SE {:+.3}",
        m.tv, m.de_sym, m.ie_sym, m.se
    );
    // Not exactly zero: the sample threshold (W > 0.499…) falls just inside
    // the [0.4, 0.5) severity bin, so within that bin the decision is not
    // constant.
    assert!(m.de_sym.abs() < 0.005 && m.se == 0.0);
    assert!((m.ie_sym + 0.5).abs() < 0.01, "book: 0% + −50% + 0%");

    // Algorithm 18 with a maximum disparity of 20% (Ex. 6.35).
    let a18 = utilitarian_benefit_fairness(&data, "sex", "male", "female", &delta, 0.5, 0.2).unwrap();
    println!("\nAlgorithm 18 (budget 50%, M = 20%)");
    println!(
        "  δ(male) = {:.3}   δ(female) = {:.3}   (book: 0.21, 0.12)",
        a18.thresholds[0], a18.thresholds[1]
    );
    println!(
        "  P(d | male) = {:.3}   P(d | female) = {:.3}   P(d) = {:.3}   (book: 60%, 40%, 50%)",
        a18.rates[0], a18.rates[1], a18.rate
    );
    println!(
        "  E[Δ·d]: {:.4} vs {:.4} under Algorithm 17",
        a18.expected_benefit, a17.expected_benefit
    );
    assert!((a18.thresholds[0] - 0.21).abs() < 0.01 && (a18.thresholds[1] - 0.12).abs() < 0.01);
    assert!((a18.rates[0] - 0.6).abs() < 0.01 && (a18.rates[1] - 0.4).abs() < 0.01);
    assert!((a18.disparity().abs() - 0.2).abs() < 1e-6);
    assert!(a18.expected_benefit <= a17.expected_benefit);

    // The benefit estimated from the randomised historical decisions gives
    // nearly the same allocation (it only sees severity in tenths).
    let est = estimate_benefit(
        &data,
        &Sfm::new("sex", Vec::<String>::new(), ["severity"], "alive"),
        "d_hist",
        "0",
        "1",
    )
    .unwrap();
    let a17e = benefit_fairness(&data, "sex", "male", "female", &est, 0.5).unwrap();
    println!(
        "\nWith Δ estimated from historical decisions: P(d|male) = {:.3}, P(d|female) = {:.3}",
        a17e.rates[0], a17e.rates[1]
    );
    assert!((a17e.disparity() - a17.disparity()).abs() < 0.05);

    // Ex. 6.34, Causal Benefit Fairness: the course helps men more
    // (Δ = β + αX) purely through societal bias. Removing the X-specific term
    // from the benefit before Algorithm 17 equalises the allocation.
    let (alpha, beta) = (0.1, 0.2);
    let mut rows = Vec::new();
    let mut biased = Vec::new();
    let mut fair = Vec::new();
    for i in 0..10_000 {
        let male = i % 2 == 1;
        rows.push([if male { "male" } else { "female" }]);
        biased.push(beta + alpha * f64::from(u8::from(male)));
        fair.push(beta);
    }
    let staff = Dataset::from_records(["sex"], rows).unwrap();
    let pb = benefit_fairness(&staff, "sex", "female", "male", &biased, 0.3).unwrap();
    let pf = benefit_fairness(&staff, "sex", "female", "male", &fair, 0.3).unwrap();
    println!("\nEx. 6.34 (budget 30%)");
    println!(
        "  benefit β + αX:   P(d | female) = {:.2}, P(d | male) = {:.2}",
        pb.rates[0], pb.rates[1]
    );
    println!(
        "  fair benefit β:   P(d | female) = {:.2}, P(d | male) = {:.2}  (boundary p = {:.2})",
        pf.rates[0], pf.rates[1], pf.boundary_probability[0]
    );
    assert!((pb.rates[1] - 0.6).abs() < 1e-9 && pb.rates[0] == 0.0);
    assert!((pf.rates[0] - 0.3).abs() < 1e-9 && (pf.rates[1] - 0.3).abs() < 1e-9);

    // Prop. 6.1.40 for a patient with severity 0.75: m1 = 1/2 + 2w/15,
    // m0 = 1/2 − w/5. Monotonicity holds in this SCM, so the true complier
    // share is the lower bound m1 − m0 = Δ = w/3.
    let w = 0.75;
    let b = canonical_type_bounds(0.5 + 2.0 * w / 15.0, 0.5 - w / 5.0);
    println!("\nCanonical types at w = 0.75: compliers in [{:.3}, {:.3}], defiers in [{:.3}, {:.3}]",
        b.complier.0, b.complier.1, b.defier.0, b.defier.1);
    assert!((b.complier.0 - w / 3.0).abs() < 1e-12 && b.defier.0 == 0.0);
}
