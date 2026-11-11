use super::offpolicy::optimize_exhaustive;
use super::*;
use crate::fairness::SplitMix64;
use crate::scm::{Dist, expr};
use rssn::symbolic::core::Expr;

fn model(f: Vec<(&str, Expr)>, u: Vec<(&str, Dist)>) -> Model {
    Model::new(
        f.into_iter().map(|(k, e)| (k.to_string(), e)),
        u.into_iter().map(|(k, d)| (k.to_string(), d)),
    )
    .unwrap()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
}

fn constant(values: &'static [(&'static str, &'static str)]) -> impl Policy {
    Deterministic(move |x: &str, _: &State<'_>| {
        values
            .iter()
            .find(|(a, _)| *a == x)
            .map(|(_, v)| (*v).to_string())
            .unwrap()
    })
}

// ---------------------------------------------------------------------------
// The 2-stage DTR of Eq. 8.2 with α₁ = α₂ = 0 (Ex. 8.1, 8.6, 8.16, 8.17).
//
// Its noise is continuous, so instead of an `scm::Model` the exact table of
// P(S1, X1, S2, X2, Y) is built in closed form. U ~ Unif(0, 1), and Y carries
// a further logistic noise U5 (listed in Eq. 8.80), without which Table 8.5
// cannot be reproduced:
//   P(Y = 1 | s1, x1, s2, x2) = ∫₀¹ σ(3u − 3k) du,
//   k = s1 + x1 + s1·x1 − x2 + s2·x2 − x1·x2.
// ---------------------------------------------------------------------------

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn dtr_table() -> Dataset {
    let p_y = |k: f64| ((1.0 + (3.0 - 3.0 * k).exp()).ln() - (1.0 + (-3.0 * k).exp()).ln()) / 3.0;
    let bern = |p: f64, v: u8| if v == 1 { p } else { 1.0 - p };
    let mut records = Vec::new();
    let mut weights = Vec::new();
    for s1 in 0..2u8 {
        for x1 in 0..2u8 {
            for s2 in 0..2u8 {
                for x2 in 0..2u8 {
                    for y in 0..2u8 {
                        let (fs1, fx1, fs2, fx2) =
                            (f64::from(s1), f64::from(x1), f64::from(s2), f64::from(x2));
                        let k = fs1 + fx1 + fs1 * fx1 - fx2 + fs2 * fx2 - fx1 * fx2;
                        let w = 0.5
                            * bern(sigma(3.0 * fs1), x1)
                            * bern(sigma(0.1 + 0.1 * fs1 + 0.1 * fx1), s2)
                            * bern(sigma(3.0 * fs2), x2)
                            * bern(p_y(k), y);
                        records.push([s1, x1, s2, x2, y].map(|v| v.to_string()));
                        weights.push(w);
                    }
                }
            }
        }
    }
    Dataset::from_records(["S1", "X1", "S2", "X2", "Y"], records)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

fn dtr_full() -> PolicySpace {
    PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1", "X1", "S2"])]).unwrap()
}

#[test]
fn dtr_observational_table_matches_table_8_5() {
    let data = dtr_table();
    let w = data.weights().unwrap();
    let find = |want: [&str; 5]| {
        (0..data.len())
            .find(|&r| {
                ["S1", "X1", "S2", "X2", "Y"]
                    .iter()
                    .zip(want)
                    .all(|(c, v)| data.value(c, r) == Some(v))
            })
            .map(|r| w[r])
            .unwrap()
    };
    close(find(["0", "0", "0", "0", "0"]), 0.0128, 5e-5);
    close(find(["0", "1", "1", "1", "1"]), 0.1028, 5e-5);
    close(find(["1", "1", "1", "1", "0"]), 0.2566, 5e-5);
    close(find(["1", "0", "1", "1", "1"]), 0.0027, 5e-5);
}

#[test]
fn dtr_planning_matches_example_8_6() {
    let plan = optimize(&dtr_table(), &dtr_full(), &Reward::total(["Y"])).unwrap();
    // Exactly 0.67574. The book's 0.6758 is Eq. 8.41 on its rounded
    // Table 8.2: 0.5 · 0.8799 + 0.5 · 0.4716 = 0.67575.
    close(plan.value, 0.6758, 1e-4);

    // π₁* ≜ X₁ ← 0 whatever S₁.
    let r1 = plan.policy.rule("X1").unwrap();
    assert_eq!(r1.action_for(&["0"]), Some("0"));
    assert_eq!(r1.action_for(&["1"]), Some("0"));
    // π₂* ≜ X₂ ← 1 wherever the choice is not a tie.
    let r2 = plan.policy.rule("X2").unwrap();
    for s in [["0", "0", "0"], ["0", "1", "0"], ["1", "0", "0"], ["1", "1", "1"]] {
        assert_eq!(r2.action_for(&s), Some("1"), "{s:?}");
    }

    // Table 8.2.
    let q1 = &plan.q[0];
    assert_eq!(q1.columns, ["S1", "X1"]);
    for (cell, v) in [
        (["0", "0"], 0.8799),
        (["0", "1"], 0.8749),
        (["1", "0"], 0.4716),
        (["1", "1"], 0.1003),
    ] {
        close(q1.get(&cell).unwrap(), v, 5e-5);
    }
    let q2 = &plan.q[1];
    assert_eq!(q2.columns, ["S1", "X1", "S2", "X2"]);
    for (cell, v) in [
        (["0", "0", "0", "0"], 0.7851),
        (["0", "0", "0", "1"], 0.9846),
        (["0", "1", "0", "0"], 0.2149),
        (["1", "0", "0", "1"], 0.7851),
        (["1", "1", "0", "0"], 0.0008),
        (["1", "1", "1", "1"], 0.0154),
    ] {
        close(q2.get(&cell).unwrap(), v, 5e-5);
    }
}

#[test]
fn dtr_ipw_and_dp_match_examples_8_16_and_8_17() {
    let data = dtr_table();
    let pi = constant(&[("X1", "0"), ("X2", "1")]);
    let reward = Reward::total(["Y"]);
    let v_ipw = ipw(&data, &dtr_full(), &reward, &pi).unwrap();
    let eval = dp(&data, &dtr_full(), &reward, &pi).unwrap();
    // The book prints 0.6757 for both, from its 4-digit Table 8.5, and
    // 0.6758 for the same policy evaluated in M*.
    close(v_ipw, 0.6758, 1e-4);
    close(eval.value, v_ipw, 1e-12);

    // Table 8.6: Q^(1)_π also covers x1 = 1, which π never takes.
    for (cell, v) in [
        (["0", "0"], 0.8799),
        (["0", "1"], 0.8749),
        (["1", "0"], 0.4716),
        (["1", "1"], 0.1003),
    ] {
        close(eval.q[0].get(&cell).unwrap(), v, 5e-5);
    }
    close(eval.q[1].get(&["0", "1", "1", "1"]).unwrap(), 0.7851, 5e-5);

    // A Markov space evaluates the same policy identically, since it reads
    // no state at all.
    let markov = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S2"])]).unwrap();
    close(ipw(&data, &markov, &reward, &pi).unwrap(), v_ipw, 1e-12);
    close(dp(&data, &markov, &reward, &pi).unwrap().value, v_ipw, 1e-12);
}

// ---------------------------------------------------------------------------
// Ex. 8.3: Y ← X2 ⊕ S1. A general policy reaches 1, a Markov one only 0.5.
// ---------------------------------------------------------------------------

fn example_8_3() -> Model {
    let b = || Dist::Bernoulli(0.5);
    model(
        vec![
            ("S1", expr::var("U1")),
            ("X1", expr::var("U2")),
            ("S2", expr::var("U3")),
            ("X2", expr::var("U4")),
            ("Y", expr::xor(expr::var("X2"), expr::var("S1"))),
        ],
        vec![("U1", b()), ("U2", b()), ("U3", b()), ("U4", b())],
    )
}

#[test]
fn general_policies_beat_markov_ones_in_example_8_3() {
    let full = Cdm::new(example_8_3(), dtr_full(), Reward::total(["Y"])).unwrap();
    let plan = full.optimal().unwrap();
    close(plan.value, 1.0, 1e-12);
    close(full.evaluate(&plan.policy).unwrap(), 1.0, 1e-12);
    // π₂* ≜ X₂ ← ¬S₁.
    let r2 = plan.policy.rule("X2").unwrap();
    assert_eq!(r2.action_for(&["0", "1", "1"]), Some("1"));
    assert_eq!(r2.action_for(&["1", "0", "0"]), Some("0"));

    let markov = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S2"])]).unwrap();
    let cdm = Cdm::new(example_8_3(), markov, Reward::total(["Y"])).unwrap();
    close(cdm.optimal().unwrap().value, 0.5, 1e-12);

    // R(Y) = −Y: the old optimum is now the worst policy.
    let neg = Reward::custom(["Y"], |y| -y[0]);
    let cdm = Cdm::new(example_8_3(), dtr_full(), neg).unwrap();
    close(cdm.evaluate(&plan.policy).unwrap(), -1.0, 1e-12);
    close(cdm.optimal().unwrap().value, 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// The MAB of Ex. 7.1 / 8.4 / 8.18 / 8.20 with Δ = 0.1, in tenths:
//   X ← U < 8,  Y ← U < 4 − X,  U uniform on {0, …, 9}.
// ---------------------------------------------------------------------------

fn confounded_bandit() -> Model {
    model(
        vec![
            ("X", expr::lt(expr::var("U"), expr::int(8))),
            (
                "Y",
                expr::lt(expr::var("U"), expr::sub(expr::int(4), expr::var("X"))),
            ),
        ],
        vec![("U", Dist::Categorical(vec![0.1; 10]))],
    )
}

#[test]
fn bandit_planning_matches_example_8_4() {
    let cdm = Cdm::new(confounded_bandit(), PolicySpace::bandit("X"), Reward::total(["Y"])).unwrap();
    // E_π[Y] = 0.4 − Δ π(X = 1).
    for p1 in [0.0, 0.3, 1.0] {
        let pi = Stochastic(move |_: &str, _: &State<'_>, x: &str| if x == "1" { p1 } else { 1.0 - p1 });
        close(cdm.evaluate(&pi).unwrap(), 0.4 - 0.1 * p1, 1e-12);
    }
    let plan = cdm.optimal().unwrap();
    assert_eq!(plan.policy.rule("X").unwrap().action_for(&[]), Some("0"));
    close(plan.value, 0.4, 1e-12);
}

#[test]
fn off_policy_methods_fail_without_nuc_in_example_8_18() {
    let cdm = Cdm::new(confounded_bandit(), PolicySpace::bandit("X"), Reward::total(["Y"])).unwrap();
    let report = cdm.nuc().unwrap();
    assert_eq!(
        report.violations,
        [NucViolation::SharedNoise {
            action: "X".into(),
            noise: "U".into(),
            with: "Y".into()
        }]
    );
    assert!(!cdm.sequential_backdoor().unwrap().holds());

    let obs = cdm.observational_data().unwrap();
    let arm0 = constant(&[("X", "0")]);
    let reward = Reward::total(["Y"]);
    // Both estimate P(Y = 1 | X = 0) = 0, against E_{X←0}[Y] = 0.4.
    close(ipw(&obs, cdm.space(), &reward, &arm0).unwrap(), 0.0, 1e-12);
    close(dp(&obs, cdm.space(), &reward, &arm0).unwrap().value, 0.0, 1e-12);
    close(cdm.evaluate(&arm0).unwrap(), 0.4, 1e-12);

    // Intervening restores NUC (Lemma 8.2.3): data under π_UNIF evaluate
    // every arm correctly.
    let exp = cdm.interventional_data(&cdm.uniform()).unwrap();
    close(ipw(&exp, cdm.space(), &reward, &arm0).unwrap(), 0.4, 1e-12);
    close(
        ipw(&exp, cdm.space(), &reward, &constant(&[("X", "1")])).unwrap(),
        0.3,
        1e-12,
    );
}

// ---------------------------------------------------------------------------
// Ex. 8.23–8.25: the sequential back-door holds although NUC does not.
//
// Eq. 8.119 prints Y ← S1 ⊕ X1 ⊕ U3. Tables 8.7 and 8.8, Eq. 8.124 and the
// back-door path the text blocks (X2 ↔ S2 ← X1 ↔ S1 → Y) all need
// Y ← S1 ⊕ X2 ⊕ U3 instead, which is what is used here.
// ---------------------------------------------------------------------------

fn example_8_23() -> Model {
    let rare = || Dist::Bernoulli(0.1);
    let fair = || Dist::Bernoulli(0.5);
    model(
        vec![
            ("S1", expr::var("U1")),
            ("X1", expr::xor(expr::var("U1"), expr::var("U4"))),
            (
                "S2",
                expr::xor(
                    expr::xor(expr::var("S1"), expr::var("X1")),
                    expr::var("U2"),
                ),
            ),
            ("X2", expr::xor(expr::var("U2"), expr::var("U5"))),
            (
                "Y",
                expr::xor(
                    expr::xor(expr::var("S1"), expr::var("X2")),
                    expr::var("U3"),
                ),
            ),
        ],
        vec![
            ("U1", rare()),
            ("U2", rare()),
            ("U3", rare()),
            ("U4", fair()),
            ("U5", fair()),
        ],
    )
}

fn example_8_23_space() -> PolicySpace {
    PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1"])]).unwrap()
}

fn not_s1_policy() -> impl Policy {
    // π = (X1 ← S1, X2 ← ¬S1).
    Deterministic(|x: &str, s: &State<'_>| match x {
        "X1" => s["S1"].to_string(),
        _ => if &s["S1"] == "1" { "0" } else { "1" }.to_string(),
    })
}

#[test]
fn backdoor_without_nuc_in_examples_8_23_to_8_25() {
    let cdm = Cdm::new(example_8_23(), example_8_23_space(), Reward::total(["Y"])).unwrap();
    let nuc = cdm.nuc().unwrap();
    assert!(!nuc.holds());
    assert!(nuc.violations.contains(&NucViolation::SharedNoise {
        action: "X1".into(),
        noise: "U1".into(),
        with: "S1".into()
    }));
    assert!(nuc.violations.contains(&NucViolation::SharedNoise {
        action: "X2".into(),
        noise: "U2".into(),
        with: "S2".into()
    }));
    let sbc = cdm.sequential_backdoor().unwrap();
    assert!(sbc.holds(), "{sbc:?}");
    assert_eq!(sbc.stages[1].conditioning, ["S1", "X1"]);

    // Table 8.7, marginalising S2.
    let obs = cdm.observational_data().unwrap();
    let marginal = |want: [&str; 4]| -> f64 {
        (0..obs.len())
            .filter(|&r| {
                ["S1", "X1", "X2", "Y"]
                    .iter()
                    .zip(want)
                    .all(|(c, v)| obs.value(c, r) == Some(v))
            })
            .map(|r| obs.weights().unwrap()[r])
            .sum()
    };
    close(marginal(["0", "0", "0", "0"]), 0.2025, 1e-12);
    close(marginal(["0", "0", "1", "0"]), 0.0225, 1e-12);
    close(marginal(["1", "1", "1", "0"]), 0.0225, 1e-12);
    close(marginal(["1", "0", "0", "1"]), 0.0225, 1e-12);

    let pi = not_s1_policy();
    let reward = Reward::total(["Y"]);
    close(cdm.evaluate(&pi).unwrap(), 0.9, 1e-12);
    close(ipw(&obs, cdm.space(), &reward, &pi).unwrap(), 0.9, 1e-12);
    let eval = dp(&obs, cdm.space(), &reward, &pi).unwrap();
    close(eval.value, 0.9, 1e-12);
    // Table 8.8.
    for cell in [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]] {
        close(eval.q[0].get(&cell).unwrap(), 0.9, 1e-12);
    }
    for (cell, v) in [
        (["0", "0", "0"], 0.1),
        (["0", "0", "1"], 0.9),
        (["0", "1", "0"], 0.1),
        (["1", "0", "0"], 0.9),
        (["1", "1", "1"], 0.1),
    ] {
        close(eval.q[1].get(&cell).unwrap(), v, 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Def. 8.2.3 on the DTR diagram of Fig. 8.10(a) (Ex. 8.22), and how it fails.
// ---------------------------------------------------------------------------

fn dtr_diagram(confounded: bool) -> Admg {
    let mut g = Admg::new();
    for (a, b) in [
        ("S1", "X1"),
        ("S1", "S2"),
        ("X1", "S2"),
        ("S2", "X2"),
        ("S1", "Y"),
        ("X1", "Y"),
        ("S2", "Y"),
        ("X2", "Y"),
    ] {
        g.directed(a, b);
    }
    if confounded {
        g.bidirected("X1", "Y");
        g.bidirected("X2", "Y");
        g.bidirected("X1", "X2");
    }
    g
}

#[test]
fn sequential_backdoor_on_the_dtr_of_example_8_22() {
    let g = dtr_diagram(false);
    dtr_full().validate(&g).unwrap();
    let report = sequential_backdoor(&g, &dtr_full(), &["Y"]).unwrap();
    assert!(report.holds());

    // Fig. 8.10(c): G_{π₂} keeps S1 → X2 and S2 → X2 and adds X1 → X2.
    let g2 = dtr_full().manipulated(&g, 1).unwrap();
    assert!(!g2.is_d_separated(&["X1"], &["X2"], &["S1", "S2"]).unwrap());

    // α ≠ 0: U confounds both treatments with Y (Ex. 8.15).
    let report = sequential_backdoor(&dtr_diagram(true), &dtr_full(), &["Y"]).unwrap();
    assert!(!report.stages[0].holds && !report.stages[1].holds);
    assert!(report.stages[0].open_path.as_deref().unwrap().contains("<->"));

    // Π' = {⟨X1, ∅⟩, ⟨X2, ∅⟩} leaves the back-door through S1 open.
    let blind = PolicySpace::new([("X1", &[] as &[&str]), ("X2", &[])]).unwrap();
    let report = sequential_backdoor(&g, &blind, &["Y"]).unwrap();
    assert!(!report.stages[0].holds);
    assert_eq!(report.stages[0].open_path.as_deref(), Some("X1 <- S1 -> Y"));
}

#[test]
fn nuc_conditions_of_examples_8_14_and_8_15() {
    // A binary DTR whose treatments may share the latent U with Y.
    let dtr = |alpha: bool| {
        let treat = |s: &str, u: &str| {
            let base = expr::xor(expr::var(s), expr::var(u));
            if alpha {
                expr::xor(base, expr::var("U"))
            } else {
                base
            }
        };
        model(
            vec![
                ("S1", expr::var("U3")),
                ("X1", treat("S1", "U1")),
                ("S2", expr::xor(expr::and([expr::var("S1"), expr::var("X1")]), expr::var("U4"))),
                ("X2", treat("S2", "U2")),
                (
                    "Y",
                    expr::or([
                        expr::and([expr::var("U"), expr::not(expr::var("X1"))]),
                        expr::xor(expr::var("X2"), expr::var("S2")),
                    ]),
                ),
            ],
            vec![
                ("U", Dist::Bernoulli(0.5)),
                ("U1", Dist::Bernoulli(0.3)),
                ("U2", Dist::Bernoulli(0.3)),
                ("U3", Dist::Bernoulli(0.5)),
                ("U4", Dist::Bernoulli(0.2)),
            ],
        )
    };
    assert!(nuc(&dtr(false), &dtr_full()).unwrap().holds());

    let blind = PolicySpace::new([("X1", &[] as &[&str]), ("X2", &[])]).unwrap();
    let report = nuc(&dtr(false), &blind).unwrap();
    assert_eq!(
        report.violations,
        [
            NucViolation::UnobservedParent {
                action: "X1".into(),
                parent: "S1".into()
            },
            NucViolation::UnobservedParent {
                action: "X2".into(),
                parent: "S2".into()
            },
        ]
    );

    let report = nuc(&dtr(true), &dtr_full()).unwrap();
    assert!(report.violations.contains(&NucViolation::SharedNoise {
        action: "X1".into(),
        noise: "U".into(),
        with: "Y".into()
    }));

    // Under α ≠ 0 off-policy evaluation is biased, but data collected under
    // any policy evaluate every other one (Ex. 8.19).
    let cdm = Cdm::new(dtr(true), dtr_full(), Reward::total(["Y"])).unwrap();
    let pi = constant(&[("X1", "0"), ("X2", "1")]);
    let truth = cdm.evaluate(&pi).unwrap();
    let biased = ipw(&cdm.observational_data().unwrap(), cdm.space(), cdm.reward(), &pi).unwrap();
    assert!((biased - truth).abs() > 1e-3, "{biased} vs {truth}");
    let behaviour = Stochastic(|_: &str, s: &State<'_>, x: &str| {
        // π_i ≜ S_i-dependent coin, like Eq. 8.79.
        let last = s.values().last().unwrap();
        let p1 = if last == "1" { 0.8 } else { 0.3 };
        if x == "1" { p1 } else { 1.0 - p1 }
    });
    let exp = cdm.interventional_data(&behaviour).unwrap();
    close(ipw(&exp, cdm.space(), cdm.reward(), &pi).unwrap(), truth, 1e-12);
}

// ---------------------------------------------------------------------------
// Validation, positivity, rewards.
// ---------------------------------------------------------------------------

#[test]
fn policy_spaces_are_validated() {
    assert!(matches!(
        PolicySpace::new([("X1", &["X1"][..])]),
        Err(CrlError::InvalidSpace(_))
    ));
    assert!(matches!(
        PolicySpace::new([("X1", &["X2"][..]), ("X2", &[])]),
        Err(CrlError::InvalidSpace(_))
    ));
    assert!(matches!(
        PolicySpace::new([("X", &[] as &[&str]), ("X", &[])]),
        Err(CrlError::InvalidSpace(_))
    ));

    let g = dtr_diagram(false);
    // X2 is a descendant of X1, so X1 cannot come second.
    let swapped = PolicySpace::new([("X2", &["S2"][..]), ("X1", &["S1"])]).unwrap();
    assert!(matches!(swapped.validate(&g), Err(CrlError::InvalidSpace(_))));
    // S2 is a descendant of X1.
    let peeking = PolicySpace::new([("X1", &["S2"][..])]).unwrap();
    assert!(matches!(peeking.validate(&g), Err(CrlError::InvalidSpace(_))));
    let unknown = PolicySpace::new([("X1", &["Z"][..])]).unwrap();
    assert_eq!(unknown.validate(&g), Err(CrlError::Unknown("Z".into())));
}

#[test]
fn positivity_and_normalization_are_reported() {
    // X = 1 never occurs after S = 1.
    let data = Dataset::from_csv("S,X,Y\n0,0,1\n0,1,0\n1,0,1\n1,0,0\n").unwrap();
    let space = PolicySpace::new([("X", &["S"][..])]).unwrap();
    let reward = Reward::total(["Y"]);
    let always1 = constant(&[("X", "1")]);
    let expected = CrlError::Positivity {
        stage: 1,
        history: vec![("S".into(), "1".into())],
        action: "1".into(),
    };
    assert_eq!(ipw(&data, &space, &reward, &always1), Err(expected.clone()));
    assert_eq!(dp(&data, &space, &reward, &always1).unwrap_err(), expected);

    // A rule that is unsupported only where the policy never goes is fine.
    let follow = Deterministic(|_: &str, s: &State<'_>| {
        if &s["S"] == "1" { "0" } else { "1" }.to_string()
    });
    close(ipw(&data, &space, &reward, &follow).unwrap(), 0.25, 1e-12);
    close(dp(&data, &space, &reward, &follow).unwrap().value, 0.25, 1e-12);

    let broken = Stochastic(|_: &str, _: &State<'_>, _: &str| 0.7);
    assert!(matches!(
        ipw(&data, &space, &reward, &broken),
        Err(CrlError::PolicyNotNormalized { .. })
    ));

    let bad = Dataset::from_csv("S,X,Y\n0,0,yes\n").unwrap();
    assert!(matches!(
        dp(&bad, &space, &reward, &follow),
        Err(CrlError::NonNumericReward { .. })
    ));
    assert_eq!(
        ipw(&data, &space, &Reward::total(["Z"]), &follow),
        Err(CrlError::Unknown("Z".into()))
    );
}

#[test]
fn rewards_aggregate_as_defined() {
    assert_eq!(Reward::total(["a", "b"]).apply(&[1.0, 2.0]), 3.0);
    assert_eq!(Reward::average(["a", "b"]).apply(&[1.0, 2.0]), 1.5);
    close(
        Reward::discounted(["a", "b", "c"], 0.9).unwrap().apply(&[1.0, 1.0, 1.0]),
        2.71,
        1e-12,
    );
    assert!(Reward::discounted(["a"], 1.0).is_err());
}

// ---------------------------------------------------------------------------
// Random SCMs: IPW = DP always; NUC ⇒ SBC (Cor. 8.2.7); SBC ⇒ both equal
// E_π[R(Y); M*] (Thm 8.2.8); Exp-NUC (Lemma 8.2.3); planning is optimal.
// ---------------------------------------------------------------------------

/// A random boolean function of `inputs`, as an OR of minterms.
fn random_function(rng: &mut SplitMix64, inputs: &[Expr]) -> Expr {
    let mut minterms = Vec::new();
    for row in 0..(1usize << inputs.len()) {
        if rng.unit() < 0.5 {
            let literals: Vec<Expr> = inputs
                .iter()
                .enumerate()
                .map(|(j, e)| {
                    if row >> j & 1 == 1 {
                        e.clone()
                    } else {
                        expr::not(e.clone())
                    }
                })
                .collect();
            minterms.push(expr::and(literals));
        }
    }
    if minterms.is_empty() {
        expr::boolean(false)
    } else {
        expr::or(minterms)
    }
}

struct RandomCase {
    model: Model,
    space: PolicySpace,
}

/// Every V_k := g_k(earlier parents, latents) ⊕ N_k with N_k ~ Bern(p),
/// p ∈ [0.1, 0.4], so every conditional has full support. Two latents each
/// feed a random pair of variables.
fn random_case(rng: &mut SplitMix64) -> RandomCase {
    let n = 6;
    let names: Vec<String> = (0..n).map(|k| format!("V{k}")).collect();
    let mut latent_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for l in 0..2 {
        let a = rng.below(n);
        let mut b = rng.below(n);
        while b == a {
            b = rng.below(n);
        }
        latent_children[a].push(l);
        latent_children[b].push(l);
    }
    let mut f = Vec::new();
    let mut u = Vec::new();
    for k in 0..n {
        let mut inputs: Vec<Expr> = Vec::new();
        for name in &names[..k] {
            if inputs.len() < 2 && rng.unit() < 0.45 {
                inputs.push(expr::var(name));
            }
        }
        for l in &latent_children[k] {
            inputs.push(expr::var(&format!("L{l}")));
        }
        let g = random_function(rng, &inputs);
        f.push((
            names[k].clone(),
            expr::xor(g, expr::var(&format!("N{k}"))),
        ));
        u.push((format!("N{k}"), Dist::Bernoulli(0.1 + 0.3 * rng.unit())));
    }
    for l in 0..2 {
        u.push((format!("L{l}"), Dist::Bernoulli(0.2 + 0.6 * rng.unit())));
    }
    let model = Model::new(f, u).unwrap();

    // Actions V_a < V_b strictly between the first variable and Y = V5.
    let a = 1 + rng.below(n - 3);
    let b = a + 1 + rng.below(n - 2 - a);
    let g = diagram(&model);
    let actions = [names[a].as_str(), names[b].as_str()];
    let mut stages = Vec::new();
    for i in 0..2 {
        let blocked: Vec<&str> = g.descendants_of(&actions[i..]).unwrap();
        // At most two states per stage keeps exhaustive search small.
        let states: Vec<String> = names
            .iter()
            .filter(|v| {
                v.as_str() != names[n - 1]
                    && !actions[i..].contains(&v.as_str())
                    && !blocked.contains(&v.as_str())
                    && rng.unit() < 0.5
            })
            .take(2)
            .cloned()
            .collect();
        stages.push(Stage {
            action: actions[i].to_string(),
            states,
        });
    }
    RandomCase {
        model,
        space: PolicySpace::from_stages(stages).unwrap(),
    }
}

/// A stochastic policy keyed on a hash of the action and state.
fn hashed_policy(seed: u64) -> impl Policy {
    Stochastic(move |x: &str, s: &State<'_>, v: &str| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
        for b in x.bytes().chain(s.to_string().bytes()) {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
        let p1 = SplitMix64::new(h).unit();
        if v == "1" { p1 } else { 1.0 - p1 }
    })
}

#[test]
fn random_models_satisfy_the_chapter_8_theorems() {
    let mut rng = SplitMix64::new(8);
    let (mut sbc_cases, mut biased_without_sbc, mut nuc_cases, mut recall_cases) = (0, 0, 0, 0);
    for case in 0..150 {
        let RandomCase { model, space } = random_case(&mut rng);
        let reward = Reward::total(["V5"]);
        let cdm = Cdm::new(model, space.clone(), reward.clone()).unwrap();
        let obs = cdm.observational_data().unwrap();
        let pi = hashed_policy(case);
        let truth = cdm.evaluate(&pi).unwrap();

        let v_ipw = ipw(&obs, &space, &reward, &pi).unwrap();
        let v_dp = dp(&obs, &space, &reward, &pi).unwrap().value;
        close(v_ipw, v_dp, 1e-10);

        let nuc_holds = cdm.nuc().unwrap().holds();
        let sbc_holds = cdm.sequential_backdoor().unwrap().holds();
        if nuc_holds {
            nuc_cases += 1;
            assert!(sbc_holds, "case {case}: NUC holds but SBC fails for {space}");
        }
        if sbc_holds {
            sbc_cases += 1;
            close(v_ipw, truth, 1e-10);
        } else if (v_ipw - truth).abs() > 1e-6 {
            biased_without_sbc += 1;
        }

        // Exp-NUC: data under any policy evaluate any other.
        let exp = cdm.interventional_data(&hashed_policy(case + 1000)).unwrap();
        close(ipw(&exp, &space, &reward, &pi).unwrap(), truth, 1e-10);

        // Planning: the plan is worth what it claims and beats π.
        let plan = cdm.optimal().unwrap();
        close(cdm.evaluate(&plan.policy).unwrap(), plan.value, 1e-10);
        assert!(plan.value >= truth - 1e-10);
        if space.has_perfect_recall() {
            let exp = cdm.interventional_data(&cdm.uniform()).unwrap();
            match optimize_exhaustive(&exp, &space, &reward, 1 << 12) {
                Ok(brute) => {
                    recall_cases += 1;
                    close(brute.value, plan.value, 1e-10);
                }
                Err(CrlError::TooManyPolicies { .. }) => {}
                Err(e) => panic!("{e}"),
            }
        }
    }
    assert!(nuc_cases >= 5, "only {nuc_cases} NUC cases");
    assert!(sbc_cases >= 20, "only {sbc_cases} SBC cases");
    assert!(biased_without_sbc >= 20, "only {biased_without_sbc} biased cases");
    assert!(recall_cases >= 5, "only {recall_cases} perfect-recall cases");
}

#[test]
fn backward_induction_agrees_with_enumeration_on_the_dtr() {
    let data = dtr_table();
    let reward = Reward::total(["Y"]);
    let smart = optimize(&data, &dtr_full(), &reward).unwrap();
    let brute = optimize_exhaustive(&data, &dtr_full(), &reward, 1 << 12).unwrap();
    close(smart.value, brute.value, 1e-12);
}

// ---------------------------------------------------------------------------
// Online learning (Sec. 8.2.2) and Algorithm 19.
// ---------------------------------------------------------------------------

#[test]
fn regret_bounds_match_theorems_8_2_4_and_8_2_5() {
    // Eq. 8.101 with the natural log; the book's 878 is 800 · log₁₀ 12.5.
    assert_eq!(rct_optimal_trials(0.1, 5000), 2021);
    assert_eq!((800.0 * 12.5f64.log10()).ceil(), 878.0);
    assert_eq!(rct_optimal_trials(0.1, 100), 0);

    let gaps = [0.0, 0.1];
    // Eq. 8.100: N/2 Δ + T Δ exp(−NΔ²/8), with T − N in place of T.
    close(
        rct_regret_bound(&gaps, 1000, 5000),
        500.0 * 0.1 + 4000.0 * 0.1 * (-1000.0 * 0.01 / 8.0f64).exp(),
        1e-9,
    );
    close(
        ucb_regret_bound(&gaps, 10_000),
        8.0 * 10_000f64.ln() / 0.1 + (1.0 + std::f64::consts::PI.powi(2) / 3.0) * 0.1,
        1e-9,
    );
}

#[test]
fn ucb_and_rct_learn_the_bandit_of_example_8_20() {
    let mut cdm =
        Cdm::new(confounded_bandit(), PolicySpace::bandit("X"), Reward::total(["Y"])).unwrap();
    let reward = Reward::total(["Y"]);
    let arms = [("0", 0.4), ("1", 0.3)];
    let t = 3000;
    let runs = 8;

    let mut ucb_regret = 0.0;
    for seed in 0..runs {
        let run = Ucb.run(&mut cdm, "X", &reward, t, seed).unwrap();
        assert_eq!(run.rewards.len(), t);
        ucb_regret += run.pseudo_regret(&arms).unwrap()[t - 1];
    }
    ucb_regret /= runs as f64;
    let bound = ucb_regret_bound(&[0.0, 0.1], t);
    assert!(ucb_regret < bound, "UCB regret {ucb_regret} above bound {bound}");
    // Far below linear: always pulling the bad arm costs 300.
    assert!(ucb_regret < 0.6 * t as f64 * 0.1, "{ucb_regret}");

    let mut rct_regret = 0.0;
    for seed in 0..runs {
        let run = Rct { trials: 1000 }
            .run(&mut cdm, &PolicySpace::bandit("X"), &reward, t, seed)
            .unwrap();
        let committed = run.committed.as_ref().unwrap();
        assert_eq!(committed.policy.rule("X").unwrap().action_for(&[]), Some("0"));
        assert!(run.actions[1000..].iter().all(|a| a[0] == "0"));
        rct_regret += run.pseudo_regret(&arms).unwrap()[t - 1];
    }
    rct_regret /= runs as f64;
    // All of it is exploration: about N/2 · Δ = 50.
    close(rct_regret, 50.0, 10.0);
    assert!(rct_regret <= rct_regret_bound(&[0.0, 0.1], 1000, t));
}

#[test]
fn rct_learns_a_confounded_two_stage_policy() {
    // Example 8.3's environment, plus a latent U that confounds X1 and Y.
    // Treated patients recover iff U = 1 (50%), untreated ones with
    // probability 0.7; but the physician treats mostly when U = 1, so
    // treatment looks 80% effective in the records.
    let b = |p| Dist::Bernoulli(p);
    let m = model(
        vec![
            ("S1", expr::var("U1")),
            ("X1", expr::xor(expr::var("U"), expr::var("U2"))),
            ("S2", expr::xor(expr::var("X1"), expr::var("U3"))),
            ("X2", expr::var("U4")),
            (
                "Y",
                expr::and([
                    expr::xor(expr::var("X2"), expr::var("S1")),
                    expr::or([
                        expr::and([expr::not(expr::var("X1")), expr::var("U5")]),
                        expr::and([expr::var("X1"), expr::var("U")]),
                    ]),
                ]),
            ),
        ],
        vec![
            ("U", b(0.5)),
            ("U1", b(0.5)),
            ("U2", b(0.2)),
            ("U3", b(0.3)),
            ("U4", b(0.5)),
            ("U5", b(0.7)),
        ],
    );
    let mut cdm = Cdm::new(m, dtr_full(), Reward::total(["Y"])).unwrap();
    assert!(!cdm.nuc().unwrap().holds());
    let best = cdm.optimal().unwrap();
    close(best.value, 0.7, 1e-12);
    assert_eq!(best.policy.rule("X1").unwrap().action_for(&["0"]), Some("0"));

    let run = Rct { trials: 2000 }
        .run(&mut cdm.clone(), &dtr_full(), &Reward::total(["Y"]), 2500, 3)
        .unwrap();
    let committed = run.committed.unwrap();
    close(cdm.evaluate(&committed.policy).unwrap(), 0.7, 1e-12);
    close(committed.value, 0.7, 0.05);

    // Algorithm 19 in each regime.
    let see_nuc = Task::new(Regime::See, Assumptions::Nuc, dtr_full(), Reward::total(["Y"]));
    let learned = see_nuc.run(&mut cdm, 4000, 1).unwrap();
    assert_eq!(learned.justification, Justification::Nuc);
    // NUC was wrong: the records make treatment look better (≈ 0.8), and
    // the plan that follows them is worth only 0.5.
    close(learned.plan.value, 0.8, 0.05);
    close(cdm.evaluate(&learned.plan.policy).unwrap(), 0.5, 1e-12);

    let see_g = Task::new(
        Regime::See,
        Assumptions::Diagram(cdm.diagram()),
        dtr_full(),
        Reward::total(["Y"]),
    );
    assert!(matches!(see_g.run(&mut cdm, 10, 1), Err(CrlError::NotIdentifiable(_))));

    let online = Task::new(Regime::Do, Assumptions::None, dtr_full(), Reward::total(["Y"]));
    let learned = online.run(&mut cdm, 4000, 1).unwrap();
    assert_eq!(learned.justification, Justification::ExperimentalNuc);
    close(cdm.evaluate(&learned.plan.policy).unwrap(), 0.7, 1e-12);
}

#[test]
fn algorithm_19_with_a_diagram_recovers_example_8_24() {
    let mut cdm = Cdm::new(example_8_23(), example_8_23_space(), Reward::total(["Y"])).unwrap();
    let task = Task::new(
        Regime::See,
        Assumptions::Diagram(cdm.diagram()),
        example_8_23_space(),
        Reward::total(["Y"]),
    );
    let learned = task.run(&mut cdm, 20_000, 7).unwrap();
    assert_eq!(learned.justification, Justification::SequentialBackdoor);
    close(learned.plan.value, 0.9, 0.02);
    close(cdm.evaluate(&learned.plan.policy).unwrap(), 0.9, 1e-12);
    // The sampled estimate of Ex. 8.24's policy, too.
    close(
        ipw(&learned.data, cdm.space(), cdm.reward(), &not_s1_policy()).unwrap(),
        0.9,
        0.02,
    );

    let blind = Task::new(Regime::See, Assumptions::None, example_8_23_space(), Reward::total(["Y"]));
    assert!(matches!(blind.justification(), Err(CrlError::NotIdentifiable(_))));
}
