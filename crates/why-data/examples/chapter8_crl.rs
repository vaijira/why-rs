//! Chapter 8: reinforcement learning through causal lenses.
//!
//! Ports the worked examples of Sec. 8.1–8.2 and checks the numbers the book
//! reports. It covers the DTR planning, IPW and DP of Ex. 8.6, 8.16 and 8.17,
//! the failure of NUC in Ex. 8.18, and the sequential back-door without NUC
//! of Ex. 8.23–8.25. It ends with a smaller run of Experiments 8.1 and 8.2
//! (RCT vs UCB). There is no companion notebook for this chapter.
//!
//! Run with `cargo run -p why-data --example chapter8_crl`.

use why_data::crl::{
    Assumptions, Cdm, Dataset, Deterministic, PolicySpace, Rct, Regime, Reward, State, Task, Ucb,
    dp, ipw, optimize, rct_optimal_trials, rct_regret_bound, sequential_backdoor,
    ucb_regret_bound,
};
use why_data::scm::{Dist, Model, expr};

fn check(label: &str, got: f64, want: f64, tol: f64) {
    println!("  {label:<44} {got:.4}   (book: {want})");
    assert!((got - want).abs() <= tol, "{label}: {got} vs {want}");
}

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The exact P(S1, X1, S2, X2, Y) of the DTR in Eq. 8.2 with α = 0
/// (Table 8.5). Y's extra logistic noise U5 (Eq. 8.80) makes
/// P(Y = 1 | k) = ∫₀¹ σ(3u − 3k) du.
fn dtr_table() -> Dataset {
    let p_y = |k: f64| ((1.0 + (3.0 - 3.0 * k).exp()).ln() - (1.0 + (-3.0 * k).exp()).ln()) / 3.0;
    let bern = |p: f64, v: u8| if v == 1 { p } else { 1.0 - p };
    let mut records = Vec::new();
    let mut weights = Vec::new();
    for bits in 0..32u8 {
        let [s1, x1, s2, x2, y] = [4, 3, 2, 1, 0].map(|b| (bits >> b) & 1);
        let f = |v: u8| f64::from(v);
        let k = f(s1) + f(x1) + f(s1) * f(x1) - f(x2) + f(s2) * f(x2) - f(x1) * f(x2);
        weights.push(
            0.5 * bern(sigma(3.0 * f(s1)), x1)
                * bern(sigma(0.1 + 0.1 * f(s1) + 0.1 * f(x1)), s2)
                * bern(sigma(3.0 * f(s2)), x2)
                * bern(p_y(k), y),
        );
        records.push([s1, x1, s2, x2, y].map(|v| v.to_string()));
    }
    Dataset::from_records(["S1", "X1", "S2", "X2", "Y"], records)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

fn main() {
    let y = Reward::total(["Y"]);
    let full = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1", "X1", "S2"])]).unwrap();

    println!("Ex. 8.6: planning in the DTR, Π = {full}");
    let data = dtr_table();
    let plan = optimize(&data, &full, &y).unwrap();
    println!("{}\n", plan.q[0]);
    println!("{}\n", plan.q[1]);
    println!("π* =\n{}\n", plan.policy);
    // Exactly 0.67574; the book sums its rounded Q-values.
    check("E_π*[Y]", plan.value, 0.6758, 1e-4);

    println!("\nEx. 8.16–8.17: evaluating π = (X1 ← 0, X2 ← 1) off-policy");
    let pi = Deterministic(|x: &str, _: &State<'_>| if x == "X1" { "0" } else { "1" }.to_string());
    check("IPW (Thm 8.2.1)", ipw(&data, &full, &y, &pi).unwrap(), 0.6757, 1e-4);
    check("DP (Thm 8.2.2)", dp(&data, &full, &y, &pi).unwrap().value, 0.6757, 1e-4);

    println!("\nEx. 8.18: the confounded bandit X ← U < 0.8, Y ← U < 0.4 − 0.1·X");
    let bandit = Model::new(
        [
            ("X".to_string(), expr::lt(expr::var("U"), expr::int(8))),
            (
                "Y".to_string(),
                expr::lt(expr::var("U"), expr::sub(expr::int(4), expr::var("X"))),
            ),
        ],
        [("U".to_string(), Dist::Categorical(vec![0.1; 10]))],
    )
    .unwrap();
    let mut mab = Cdm::new(bandit, PolicySpace::bandit("X"), y.clone()).unwrap();
    for v in mab.nuc().unwrap().violations {
        println!("  NUC fails: {v}");
    }
    let arm0 = Deterministic(|_: &str, _: &State<'_>| "0".to_string());
    let obs = mab.observational_data().unwrap();
    check("IPW E_{X←0}[Y] from P(X, Y)", ipw(&obs, mab.space(), &y, &arm0).unwrap(), 0.0, 1e-12);
    check("true E_{X←0}[Y]", mab.evaluate(&arm0).unwrap(), 0.4, 1e-12);
    let exp = mab.interventional_data(&mab.uniform()).unwrap();
    check(
        "IPW E_{X←0}[Y] from P_UNIF(X, Y) (Lem. 8.2.3)",
        ipw(&exp, mab.space(), &y, &arm0).unwrap(),
        0.4,
        1e-12,
    );

    println!("\nEx. 8.23–8.25: sequential back-door without NUC");
    // Eq. 8.119 prints Y ← S1 ⊕ X1 ⊕ U3; Tables 8.7–8.8 need S1 ⊕ X2 ⊕ U3.
    let rare = || Dist::Bernoulli(0.1);
    let fair = || Dist::Bernoulli(0.5);
    let v = expr::var;
    let m = Model::new(
        [
            ("S1".to_string(), v("U1")),
            ("X1".to_string(), expr::xor(v("U1"), v("U4"))),
            ("S2".to_string(), expr::xor(expr::xor(v("S1"), v("X1")), v("U2"))),
            ("X2".to_string(), expr::xor(v("U2"), v("U5"))),
            ("Y".to_string(), expr::xor(expr::xor(v("S1"), v("X2")), v("U3"))),
        ],
        [
            ("U1".to_string(), rare()),
            ("U2".to_string(), rare()),
            ("U3".to_string(), rare()),
            ("U4".to_string(), fair()),
            ("U5".to_string(), fair()),
        ],
    )
    .unwrap();
    let space = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1"])]).unwrap();
    let mut cdm = Cdm::new(m, space.clone(), y.clone()).unwrap();
    for v in cdm.nuc().unwrap().violations {
        println!("  NUC fails: {v}");
    }
    let report = sequential_backdoor(&cdm.diagram(), &space, &["Y"]).unwrap();
    for s in &report.stages {
        println!(
            "  ({} ⊥ Y | {{{}}}) in G_(X̲, π…): {}",
            s.action,
            s.conditioning.join(", "),
            s.holds
        );
    }
    assert!(report.holds());
    let pi = Deterministic(|x: &str, s: &State<'_>| match x {
        "X1" => s["S1"].to_string(),
        _ => if &s["S1"] == "1" { "0" } else { "1" }.to_string(),
    });
    let obs = cdm.observational_data().unwrap();
    check("E_π[Y] in M*", cdm.evaluate(&pi).unwrap(), 0.9, 1e-12);
    check("IPW from P(V)", ipw(&obs, &space, &y, &pi).unwrap(), 0.9, 1e-12);
    check("DP from P(V)", dp(&obs, &space, &y, &pi).unwrap().value, 0.9, 1e-12);

    let task = Task::new(Regime::See, Assumptions::Diagram(cdm.diagram()), space, y.clone());
    let learned = task.run(&mut cdm, 20_000, 1).unwrap();
    println!(
        "  Algorithm 19 from 20 000 observations, justified by {}:",
        learned.justification
    );
    println!("{}", learned.plan.policy);
    check("  its estimated value", learned.plan.value, 0.9, 0.02);

    println!("\nExperiments 8.1–8.2, smaller: RCT vs UCB, Δ = 0.1, T = 5000, 20 runs");
    let t = 5000;
    let runs = 20;
    let arms = [("0", 0.4), ("1", 0.3)];
    let gaps = [0.0, 0.1];
    let n_star = rct_optimal_trials(0.1, t);
    println!("  Eq. 8.101: N* = {n_star} (natural log; the book's 878 uses log₁₀)");
    for n in [100, 300, 500, 900] {
        let mut regret = 0.0;
        for seed in 0..runs {
            let run = Rct { trials: n }
                .run(&mut mab, &PolicySpace::bandit("X"), &y, t, seed)
                .unwrap();
            regret += run.pseudo_regret(&arms).unwrap()[t - 1];
        }
        println!(
            "  RCT N = {n:>3}: mean regret {:>6.1}   (Thm 8.2.4 bound {:.1})",
            regret / runs as f64,
            rct_regret_bound(&gaps, n, t)
        );
    }
    let mut regret = 0.0;
    for seed in 0..runs {
        regret += Ucb.run(&mut mab, "X", &y, t, seed).unwrap().pseudo_regret(&arms).unwrap()[t - 1];
    }
    let regret = regret / runs as f64;
    println!(
        "  UCB:        mean regret {regret:>6.1}   (Thm 8.2.5 bound {:.1})",
        ucb_regret_bound(&gaps, t)
    );
    assert!(regret < ucb_regret_bound(&gaps, t));
}
