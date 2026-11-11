//! Chapter 9: causal reinforcement learning tasks.
//!
//! Walks through the four CRL tasks and causal game theory, checking the
//! numbers the book reports:
//!
//! 1. Causal offline-to-online learning (Sec. 9.2): natural bounds
//!    (Table 9.2), UCB+ against UCB and UCB− (Experiments 9.1–9.2), and the
//!    sequential bounds of Table 9.3.
//! 2. Where to intervene (Sec. 9.3): Ex. 9.5's POMISes.
//! 3. Counterfactual decision-making (Sec. 9.4): the greedy casino
//!    (Tables 9.6–9.7, Ex. 9.12), Ctf-UCB against UCB (Experiment 9.7), and
//!    the inventory MDP of Ex. 9.15.
//! 4. Causal imitation learning (Sec. 9.5): behavioural cloning with and
//!    without confounding (Exs. 9.18–9.20) and causal MWAL (Ex. 9.24).
//! 5. Causal game theory (Sec. 9.6): the causal prisoner's dilemma
//!    (Fig. 9.36, Table 9.15) and its causal Nash equilibrium.
//!
//! The book's companion Python has no chapter 9 code.
//!
//! Run with `cargo run -p why-data --example chapter9_crl`.

use why_data::crl::{
    ActionSpace, Cdm, CausalGame, CtfBandit, CtfMdp, CtfUcb, Dataset, Environment, PolicySpace,
    Reward, StageVars, Ucb, UcbPlus, UcbTransfer, behavioral_cloning,
    causal_bounds, causal_mwal, diagram, ett, imitation_backdoor, natural_bounds, optimize, pomis,
};
use rssn::symbolic::core::Expr;
use why_data::scm::{Dist, Model, expr};

fn check(label: &str, got: f64, want: f64, tol: f64) {
    println!("  {label:<52} {got:.4}   (book: {want})");
    assert!((got - want).abs() <= tol, "{label}: {got} vs {want}");
}

fn model(f: Vec<(&str, Expr)>, u: Vec<(&str, Dist)>) -> Model {
    Model::new(
        f.into_iter().map(|(k, e)| (k.to_string(), e)),
        u.into_iter().map(|(k, d)| (k.to_string(), d)),
    )
    .unwrap()
}

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The exact P(S1, X1, S2, X2, Y) of the DTR of Eq. 8.2 with confounding
/// α₁, α₂, integrating U ~ Unif(0, 1) by Simpson's rule.
fn dtr_table(alpha1: f64, alpha2: f64) -> Dataset {
    let bern = |p: f64, v: u8| if v == 1 { p } else { 1.0 - p };
    let n = 2000;
    let mut records = Vec::new();
    let mut weights = Vec::new();
    for bits in 0..32u8 {
        let [s1, x1, s2, x2, y] = [4, 3, 2, 1, 0].map(|b| (bits >> b) & 1);
        let f = |v: u8| f64::from(v);
        let k = f(s1) + f(x1) + f(s1) * f(x1) - f(x2) + f(s2) * f(x2) - f(x1) * f(x2);
        let g = |u: f64| {
            bern(sigma(3.0 * f(s1) + alpha1 * u), x1)
                * bern(sigma(3.0 * f(s2) + alpha2 * u), x2)
                * bern(sigma(3.0 * u - 3.0 * k), y)
        };
        let h = 1.0 / n as f64;
        let mut integral = g(0.0) + g(1.0);
        for i in 1..n {
            integral += if i % 2 == 1 { 4.0 } else { 2.0 } * g(i as f64 * h);
        }
        integral *= h / 3.0;
        weights.push(0.5 * bern(sigma(0.1 + 0.1 * f(s1) + 0.1 * f(x1)), s2) * integral);
        records.push([s1, x1, s2, x2, y].map(|v| v.to_string()));
    }
    Dataset::from_records(["S1", "X1", "S2", "X2", "Y"], records)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

fn task1() {
    println!("\n=== Task 1: causal offline-to-online learning (Sec. 9.2) ===");
    let y = Reward::total(["Y"]);
    // Ex. 7.1 in tenths: X ← U < 8, Y ← U < 4 − dX.
    let mab = |d: i64| {
        let m = model(
            vec![
                ("X", expr::lt(expr::var("U"), expr::int(8))),
                (
                    "Y",
                    expr::lt(expr::var("U"), expr::sub(expr::int(4), expr::mul(expr::int(d), expr::var("X")))),
                ),
            ],
            vec![("U", Dist::Categorical(vec![0.1; 10]))],
        );
        Cdm::new(m, PolicySpace::bandit("X"), y.clone()).unwrap()
    };

    println!("Table 9.2 (Δ = 0.1): natural bounds from P(X, Y)");
    let cdm = mab(1);
    let b = natural_bounds(&cdm.observational_data().unwrap(), "X", &y).unwrap();
    check("r_0 = E[Y|0]P(0) + 1 − P(0)", b[0].upper, 0.8, 1e-9);
    check("l_1 = E[Y|1]P(1) = 0.4 − Δ", b[1].lower, 0.3, 1e-9);
    check("r_1 = 0.6 − Δ", b[1].upper, 0.5, 1e-9);

    println!("Experiments 9.1–9.2: cumulative regret after 3000 episodes, 5000 observations");
    for d in [3, 1] {
        let mut env = mab(d);
        let mut rng = 7u64;
        let mut unit = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let rows: Vec<Vec<String>> = (0..5000).map(|_| env.observe(&mut unit).unwrap()).collect();
        let obs = Dataset::from_records(env.variables(), rows).unwrap();
        let bounds = natural_bounds(&obs, "X", &y).unwrap();
        let arms = [("0", 0.4), ("1", 0.4 - d as f64 / 10.0)];
        let mut total = [0.0; 3];
        for seed in 0..10 {
            let runs = [
                Ucb.run(&mut env, "X", &y, 3000, seed).unwrap(),
                UcbPlus { bounds: bounds.clone() }.run(&mut env, "X", &y, 3000, seed).unwrap(),
                UcbTransfer { observational: obs.clone() }.run(&mut env, "X", &y, 3000, seed).unwrap(),
            ];
            for (t, r) in total.iter_mut().zip(&runs) {
                *t += r.pseudo_regret(&arms).unwrap().last().unwrap() / 10.0;
            }
        }
        println!(
            "  Δ = 0.{d}: UCB {:7.2}   UCB+ {:7.2}   UCB− {:7.2}",
            total[0], total[1], total[2]
        );
        if d == 3 {
            assert!(total[1] * 5.0 < total[0]);
        }
    }

    println!("Table 9.3 (α₁ = 3, α₂ = −3): sequential causal bounds");
    let space = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1", "X1", "S2"])]).unwrap();
    let bounds = causal_bounds(&dtr_table(3.0, -3.0), &space, &y).unwrap();
    let t = bounds.transition(1, &["0", "1", "0"]).unwrap();
    check("l for P_{X1=1}(S2=0 | S1=0)", t.lower, 0.3534, 5e-5);
    check("r for P_{X1=1}(S2=0 | S1=0)", t.upper, 0.5683, 5e-5);
    let r = bounds.reward(&["1", "1", "0", "0"]).unwrap();
    check("r for E_{X1=1,X2=0}[Y | S1=1, S2=0]", r.upper, 0.2417, 5e-5);
    let plan = optimize(&dtr_table(0.0, 0.0), &space, &y).unwrap();
    let pi = bounds.policy_bounds(&plan.policy).unwrap();
    println!(
        "  optimal value {:.4} lies in its bounds [{:.4}, {:.4}]",
        plan.value, pi.lower, pi.upper
    );
    assert!(pi.contains(plan.value, 1e-9));
}

fn task2() {
    println!("\n=== Task 2: where to intervene (Sec. 9.3) ===");
    // Ex. 9.5: X1 ← U1, X2 ← X1 ⊕ U2, Y ← X2 ⊕ U2.
    let m = model(
        vec![
            ("X1", expr::var("U1")),
            ("X2", expr::xor(expr::var("X1"), expr::var("U2"))),
            ("Y", expr::xor(expr::var("X2"), expr::var("U2"))),
        ],
        vec![("U1", Dist::Bernoulli(0.5)), ("U2", Dist::Bernoulli(0.5))],
    );
    let sets = pomis(&diagram(&m), &["X1", "X2"], "Y").unwrap();
    println!("  POMISes of Fig. 9.8(a): {sets:?} (∅ and {{X1, X2}} are never needed)");
    let y = Reward::total(["Y"]);
    for (set, want) in [(vec!["X1"], 1.0), (vec!["X2"], 0.5), (vec!["X1", "X2"], 0.5)] {
        let space = PolicySpace::new(set.iter().map(|x| (*x, &[] as &[&str]))).unwrap();
        let best = Cdm::new(m.clone(), space, y.clone()).unwrap().optimal().unwrap().value;
        check(&format!("max E[Y | do({})]", set.join(", ")), best, want, 1e-12);
    }
}

fn task3() {
    println!("\n=== Task 3: counterfactual decision-making (Sec. 9.4) ===");
    // The greedy casino, Table 9.6a, rates in tenths; X ← UD ⊕ UB (see the
    // crate docs on Eq. 9.98).
    let lit = |v: &str, b: bool| if b { expr::var(v) } else { expr::not(expr::var(v)) };
    let mut rate = expr::int(0);
    for (bits, r) in [(0u8, 1), (1, 5), (2, 5), (3, 1), (4, 4), (5, 2), (6, 2), (7, 4)] {
        let cell = expr::and([lit("UD", bits & 4 != 0), lit("UB", bits & 2 != 0), lit("X", bits & 1 != 0)]);
        rate = expr::add(rate, expr::mul(expr::int(r), cell));
    }
    let casino = model(
        vec![
            ("X", expr::xor(expr::var("UD"), expr::var("UB"))),
            ("Y", expr::lt(expr::var("UY"), rate)),
        ],
        vec![
            ("UD", Dist::Bernoulli(0.5)),
            ("UB", Dist::Bernoulli(0.5)),
            ("UY", Dist::Categorical(vec![0.1; 10])),
        ],
    );
    let y = Reward::total(["Y"]);
    let t = ett(&casino, "X", &y).unwrap();
    check("E[Y | X = x] (Table 9.6b)", t.observational()[0], 0.15, 1e-12);
    check("E_x[Y] (Table 9.6b)", t.interventional()[0], 0.3, 1e-12);
    check("E[Y_{X←1} | X = 0] (Table 9.7)", t.values[0][1], 0.45, 1e-12);
    check("E_π[Y], π ≜ X ← ¬X′ (Ex. 9.12)", t.optimal().1, 0.45, 1e-12);

    let mut env = CtfBandit::new(casino.clone(), "X", y.clone()).unwrap();
    let ctf = CtfUcb.run(&mut env, 2000, 1).unwrap();
    let ctf_regret = ctf.pseudo_regret(&t).unwrap()[1999];
    let mut cdm = Cdm::new(casino, PolicySpace::bandit("X"), y.clone()).unwrap();
    let ucb = Ucb.run(&mut cdm, "X", &y, 2000, 1).unwrap();
    let ucb_regret = 0.15 * ucb.actions.len() as f64;
    println!("  Experiment 9.7, 2000 episodes: Ctf-UCB regret {ctf_regret:.1}, UCB {ucb_regret:.1}");
    assert!(ctf_regret < 0.2 * ucb_regret);

    // Eq. 7.4: the inventory MDP, one stage.
    let stage = model(
        vec![
            ("S", expr::var("US")),
            ("X", expr::xor(expr::var("S"), expr::var("U1"))),
            (
                "Y",
                expr::xor(expr::xor(expr::var("S"), expr::var("X")), expr::xor(expr::var("U1"), expr::var("U3"))),
            ),
            (
                "S2",
                expr::xor(expr::or([expr::var("S"), expr::var("X")]), expr::xor(expr::var("U1"), expr::var("U2"))),
            ),
        ],
        vec![
            ("US", Dist::Bernoulli(0.5)),
            ("U1", Dist::Bernoulli(0.9)),
            ("U2", Dist::Bernoulli(0.1)),
            ("U3", Dist::Bernoulli(0.1)),
        ],
    );
    let mdp = CtfMdp::from_stage_model(
        &stage,
        StageVars {
            state: "S",
            action: "X",
            reward: "Y",
            next_state: "S2",
        },
    )
    .unwrap();
    let q = mdp.value_iteration(0.9, 1e-12).unwrap();
    check("Ex. 9.15: E_π*[Σ γ^{i−1} Y_i], π* ≜ X ← ¬X′", q.value, 9.0, 1e-9);
    let e = mdp.experimental_value_iteration(0.9, 1e-12).unwrap();
    check("Ex. 8.5: best experimental policy X ← S", e.value, 8.2, 1e-9);
    // Table 9.11 prints 1 for Q*(s, x′, x′); the Bellman equation gives 8.2.
    println!("  Q*(s, x′, x′) = {:.4} (Table 9.11 prints 1)", q.q[0][0][0]);
}

fn task4() {
    println!("\n=== Task 4: causal imitation learning (Sec. 9.5) ===");
    let y = Reward::total(["Y"]);
    let space = PolicySpace::bandit("X");
    // Ex. 9.17: X ← U, Y ← X, P(U = 1) = 0.9.
    let clean = model(
        vec![("X", expr::var("U")), ("Y", expr::var("X"))],
        vec![("U", Dist::Bernoulli(0.9))],
    );
    let cdm = Cdm::new(clean, space.clone(), y.clone()).unwrap();
    let obs = cdm.observational_data().unwrap();
    let bc = behavioral_cloning(&obs, &space).unwrap();
    check("Ex. 9.18: E_{π_BC}[Y] = E[Y]", cdm.evaluate(&bc).unwrap(), 0.9, 1e-12);
    let identity: &dyn Fn(&[f64]) -> f64 = &|v| v[0];
    let irl = causal_mwal(&obs, &space, &["Y"], &[identity]).unwrap();
    check("Ex. 9.24: ν* of inverse RL", irl.value, -0.1, 1e-12);
    check("Ex. 9.24: E_{π_IRL}[Y]", cdm.evaluate(&irl.mixture[0].1).unwrap(), 1.0, 1e-12);

    // Ex. 9.20: X ← ¬U, Y ← X ⊕ U.
    let confounded = model(
        vec![("X", expr::not(expr::var("U"))), ("Y", expr::xor(expr::var("X"), expr::var("U")))],
        vec![("U", Dist::Bernoulli(0.5))],
    );
    let cdm = Cdm::new(confounded, space.clone(), y).unwrap();
    let bc = behavioral_cloning(&cdm.observational_data().unwrap(), &space).unwrap();
    check("Ex. 9.20: E_{π_BC}[Y] under confounding", cdm.evaluate(&bc).unwrap(), 0.5, 1e-12);
    let report = imitation_backdoor(&cdm.diagram(), &space, &["Y"]).unwrap();
    println!("  imitation backdoor holds: {} ({:?})", report.holds(), report.stages[0].condition);
}

fn task5() {
    println!("\n=== Causal game theory (Sec. 9.6) ===");
    // The causal prisoner's dilemma, scenario M2, payoffs in tenths.
    let lit = |v: &str, b: bool| if b { expr::var(v) } else { expr::not(expr::var(v)) };
    let table = [
        ((0, 0, 0, 0), (-14, -14)),
        ((0, 0, 0, 1), (-10, -10)),
        ((0, 0, 1, 0), (-140, 0)),
        ((0, 0, 1, 1), (-80, 0)),
        ((0, 1, 0, 0), (-10, -10)),
        ((0, 1, 0, 1), (0, 0)),
        ((0, 1, 1, 0), (0, 0)),
        ((0, 1, 1, 1), (0, -30)),
        ((1, 0, 0, 0), (0, -140)),
        ((1, 0, 0, 1), (0, 0)),
        ((1, 0, 1, 0), (0, 0)),
        ((1, 0, 1, 1), (0, -80)),
        ((1, 1, 0, 0), (0, -80)),
        ((1, 1, 0, 1), (-30, 0)),
        ((1, 1, 1, 0), (-80, 0)),
        ((1, 1, 1, 1), (0, 0)),
    ];
    let (mut y1, mut y2) = (expr::int(0), expr::int(0));
    for ((x1, u1, x2, u2), (a, b)) in table {
        let cell = || expr::and([lit("X1", x1 == 1), lit("U1", u1 == 1), lit("X2", x2 == 1), lit("U2", u2 == 1)]);
        y1 = expr::add(y1, expr::mul(expr::int(a), cell()));
        y2 = expr::add(y2, expr::mul(expr::int(b), cell()));
    }
    let cpd = model(
        vec![
            ("X1", expr::var("U1")),
            ("X2", expr::var("U2")),
            ("Y1", y1),
            ("Y2", y2),
        ],
        vec![("U1", Dist::Bernoulli(0.4)), ("U2", Dist::Bernoulli(0.4))],
    );
    let game = CausalGame::from_model(&cpd, [("X1", "Y1"), ("X2", "Y2")]).unwrap();
    let g = &game.layered.game;
    check("Fig. 9.36: u₁ when both follow instinct", g.payoffs[0][0].0 / 10.0, -2.4, 0.05);
    check("Fig. 9.36: u₁ at the standard NE (do(1), do(1))", g.payoffs[2][2].0 / 10.0, -1.9, 0.05);
    let cne = game.causal_nash((ActionSpace::L3, ActionSpace::L3)).unwrap();
    let mut table = cne.selection.clone();
    for cell in table.payoffs.iter_mut().flatten() {
        *cell = (cell.0 / 10.0, cell.1 / 10.0);
    }
    println!("PCH layer selection game (Table 9.15):\n{table}");
    let (rows, cols) = cne.selection_equilibrium.support();
    println!(
        "  layers chosen: {} / {}",
        cne.selection.rows[rows[0]], cne.selection.cols[cols[0]]
    );
    check("CNE payoff u₁ (both play X = 1 − X′)", cne.equilibrium.payoff.0 / 10.0, 0.0, 1e-9);
}

fn main() {
    task1();
    task2();
    task3();
    task4();
    task5();
}
