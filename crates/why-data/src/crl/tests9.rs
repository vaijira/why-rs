//! Chapter 9 tests.
use super::*;
use crate::fairness::SplitMix64;
use crate::graph::dseparation::Admg;
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

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The exact P(S1, X1, S2, X2, Y) of the DTR of Eq. 8.2 with confounding
/// coefficients α₁, α₂, by Gauss–Legendre quadrature over U ~ Unif(0, 1).
fn dtr_table(alpha1: f64, alpha2: f64) -> Dataset {
    // 20-point Gauss–Legendre on [0, 1], composite over 50 panels.
    let nodes = gauss_legendre(20);
    let panels = 50;
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
                        let bern = |p: f64, v: u8| if v == 1 { p } else { 1.0 - p };
                        let mut integral = 0.0;
                        for panel in 0..panels {
                            let (a, b) = (panel as f64 / panels as f64, (panel + 1) as f64 / panels as f64);
                            for (t, w) in &nodes {
                                let u = 0.5 * (b - a) * t + 0.5 * (a + b);
                                let f = bern(sigma(3.0 * fs1 + alpha1 * u), x1)
                                    * bern(sigma(3.0 * fs2 + alpha2 * u), x2)
                                    * bern(sigma(3.0 * u - 3.0 * k), y);
                                integral += 0.5 * (b - a) * w * f;
                            }
                        }
                        let w = 0.5 * bern(sigma(0.1 + 0.1 * fs1 + 0.1 * fx1), s2) * integral;
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

/// Nodes and weights of n-point Gauss–Legendre quadrature on [−1, 1].
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let mut x = (std::f64::consts::PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        out.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}

fn dtr_full() -> PolicySpace {
    PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S1", "X1", "S2"])]).unwrap()
}

fn constant(x1: &'static str, x2: &'static str) -> impl Policy {
    Deterministic(move |x: &str, _: &State<'_>| if x == "X1" { x1 } else { x2 }.to_string())
}

/// Samples `n` rows from an environment's observational distribution.
fn observe(env: &mut dyn Environment, n: usize, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let rows: Vec<Vec<String>> = (0..n).map(|_| env.observe(&mut || rng.unit()).unwrap()).collect();
    Dataset::from_records(env.variables(), rows).unwrap()
}

/// A random boolean function of `inputs`, as a disjunction of minterms.
fn random_fn(inputs: &[&str], rng: &mut SplitMix64) -> Expr {
    if inputs.is_empty() {
        return expr::boolean(rng.unit() < 0.5);
    }
    let k = inputs.len();
    let mut terms = Vec::new();
    for m in 0..(1u32 << k) {
        if rng.unit() < 0.5 {
            terms.push(expr::and((0..k).map(|i| {
                if m >> i & 1 == 1 {
                    expr::var(inputs[i])
                } else {
                    expr::not(expr::var(inputs[i]))
                }
            })));
        }
    }
    if terms.is_empty() { expr::boolean(false) } else { expr::or(terms) }
}

/// A random binary SCM over `vars` in temporal order: each variable reads
/// a random subset of the earlier ones, its own noise, and the latent
/// confounders it shares with other variables.
fn random_scm(vars: &[&str], rng: &mut SplitMix64, confounders: usize) -> Model {
    let mut f = Vec::new();
    let mut u: Vec<(String, Dist)> = Vec::new();
    let mut shared: Vec<Vec<String>> = vec![Vec::new(); vars.len()];
    for c in 0..confounders {
        let name = format!("C{c}");
        let a = rng.below(vars.len());
        let mut b = rng.below(vars.len());
        while b == a {
            b = rng.below(vars.len());
        }
        shared[a].push(name.clone());
        shared[b].push(name.clone());
        u.push((name, Dist::Bernoulli(0.2 + 0.6 * rng.unit())));
    }
    for (i, v) in vars.iter().enumerate() {
        let noise = format!("U{v}");
        u.push((noise.clone(), Dist::Bernoulli(0.1 + 0.8 * rng.unit())));
        let mut inputs: Vec<String> = vars[..i].iter().filter(|_| rng.unit() < 0.6).map(|s| (*s).to_string()).collect();
        inputs.push(noise);
        inputs.extend(shared[i].iter().cloned());
        let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
        f.push(((*v).to_string(), random_fn(&refs, rng)));
    }
    Model::new(f, u).unwrap()
}

// ===========================================================================
// Task 1: causal offline-to-online learning (Sec. 9.2)
// ===========================================================================

/// The MAB of Ex. 7.1 in tenths: X ← U < 8, Y ← U < 4 − dX, U uniform on
/// {0, …, 9}, so Δ = d/10.
fn mab(d: i64) -> Model {
    model(
        vec![
            ("X", expr::lt(expr::var("U"), expr::int(8))),
            (
                "Y",
                expr::lt(
                    expr::var("U"),
                    expr::sub(expr::int(4), expr::mul(expr::int(d), expr::var("X"))),
                ),
            ),
        ],
        vec![("U", Dist::Categorical(vec![0.1; 10]))],
    )
}

fn mab_cdm(d: i64) -> Cdm {
    Cdm::new(mab(d), PolicySpace::bandit("X"), Reward::total(["Y"])).unwrap()
}

#[test]
fn natural_bounds_match_table_9_2() {
    for d in 1..=3 {
        let delta = d as f64 / 10.0;
        let cdm = mab_cdm(d);
        let obs = cdm.observational_data().unwrap();
        let b = natural_bounds(&obs, "X", &Reward::total(["Y"])).unwrap();
        assert_eq!(b[0].arm, "0");
        close(b[0].lower, 0.0, 1e-12);
        close(b[0].upper, 0.8, 1e-12);
        close(b[1].lower, 0.4 - delta, 1e-12);
        close(b[1].upper, 0.6 - delta, 1e-12);
        // E[Y | X = 1] = 0.5 − 1.25Δ, and the bounds contain E_x[Y] = 0.4 − Δx.
        let ey1 = dp(&obs, cdm.space(), cdm.reward(), &Deterministic(|_: &str, _: &State<'_>| "1".to_string()))
            .unwrap()
            .value;
        close(ey1, 0.5 - 1.25 * delta, 1e-12);
        for (arm, truth) in [(0, 0.4), (1, 0.4 - delta)] {
            assert!(b[arm].lower <= truth + 1e-12 && truth <= b[arm].upper + 1e-12);
        }
    }

    // With one stage, Thms 9.2.5–9.2.6 are the natural bounds.
    let obs = mab_cdm(1).observational_data().unwrap();
    let seq = causal_bounds(&obs, &PolicySpace::bandit("X"), &Reward::total(["Y"])).unwrap();
    let iv = seq.reward(&["1"]).unwrap();
    close(iv.lower, 0.3, 1e-12);
    close(iv.upper, 0.5, 1e-12);

    assert!(matches!(
        natural_bounds(&obs, "X", &Reward::custom(["Y"], |y| 2.0 * y[0])),
        Err(CrlError::InvalidInput(_))
    ));
}

fn bandit_regret(run: &OnlineRun, d: i64) -> f64 {
    let delta = d as f64 / 10.0;
    *run.pseudo_regret(&[("0", 0.4), ("1", 0.4 - delta)]).unwrap().last().unwrap()
}

#[test]
fn ucb_plus_never_transfers_negatively_experiments_9_1_and_9_2() {
    let reward = Reward::total(["Y"]);
    let episodes = 3000;
    let mut totals = [[0.0; 3]; 2];
    for (k, d) in [3, 1].into_iter().enumerate() {
        let mut env = mab_cdm(d);
        let obs = observe(&mut env, 5000, 7);
        let bounds = natural_bounds(&obs, "X", &reward).unwrap();
        for seed in 0..20 {
            let ucb = Ucb.run(&mut env, "X", &reward, episodes, seed).unwrap();
            let plus = UcbPlus { bounds: bounds.clone() }
                .run(&mut env, "X", &reward, episodes, seed)
                .unwrap();
            let minus = UcbTransfer { observational: obs.clone() }
                .run(&mut env, "X", &reward, episodes, seed)
                .unwrap();
            totals[k][0] += bandit_regret(&ucb, d) / 20.0;
            totals[k][1] += bandit_regret(&plus, d) / 20.0;
            totals[k][2] += bandit_regret(&minus, d) / 20.0;
        }
    }
    let [[ucb3, plus3, _], [ucb1, plus1, minus1]] = totals;
    // Δ = 0.3: r₁ = 0.3 < µ* = 0.4, so UCB+ almost never pulls arm 1.
    assert!(plus3 < 2.0 && plus3 * 5.0 < ucb3, "{plus3} vs {ucb3}");
    // Δ = 0.1: r₁ = 0.5 ≥ µ*, so UCB+ behaves like UCB.
    assert!(plus1 <= ucb1 * 1.15 + 2.0, "{plus1} vs {ucb1}");
    // Transferring confounded data as if it were experimental backfires.
    assert!(minus1 > 2.0 * ucb1, "{minus1} vs {ucb1}");

    // Thm 9.2.3 against Thm 8.2.5.
    let means = [0.4, 0.1];
    assert!(ucb_plus_regret_bound(&means, &[0.8, 0.3], episodes) < ucb_regret_bound(&[0.0, 0.3], episodes));
    close(
        ucb_plus_regret_bound(&means, &[0.8, 0.45], episodes),
        ucb_regret_bound(&[0.0, 0.3], episodes),
        1e-9,
    );
}

/// Every causal bound of Table 9.3 (α₁ = 3, α₂ = −3), to the book's four
/// decimals.
#[test]
fn sequential_bounds_match_table_9_3() {
    let data = dtr_table(3.0, -3.0);
    let b = causal_bounds(&data, &dtr_full(), &Reward::total(["Y"])).unwrap();
    // (S1, X1, S2): P_{X1}(S2 | S1), l, r.
    let transitions = [
        ("000", 0.4750, 0.1021, 0.8872),
        ("001", 0.5250, 0.1128, 0.8979),
        ("010", 0.4502, 0.3534, 0.5683),
        ("011", 0.5498, 0.4317, 0.6466),
        ("100", 0.4502, 0.0069, 0.9915),
        ("101", 0.5498, 0.0085, 0.9931),
        ("110", 0.4256, 0.4190, 0.4344),
        ("111", 0.5744, 0.5656, 0.5810),
    ];
    for (key, p, l, r) in transitions {
        let v: Vec<&str> = key.split("").filter(|s| !s.is_empty()).collect();
        let iv = b.transition(1, &v).unwrap();
        close(iv.lower, l, 5e-5);
        close(iv.upper, r, 5e-5);
        // The truth, σ(0.1 + 0.1 s1 + 0.1 x1), lies inside.
        let (s1, x1, s2): (f64, f64, u8) = (v[0].parse().unwrap(), v[1].parse().unwrap(), v[2].parse().unwrap());
        let q = sigma(0.1 + 0.1 * s1 + 0.1 * x1);
        let truth = if s2 == 1 { q } else { 1.0 - q };
        close(truth, p, 5e-5);
        assert!(iv.contains(truth, 1e-12));
    }
    // (S1, X1, S2, X2): E_{X1,X2}[Y | S1, S2], l, r.
    let rewards = [
        ("0000", 0.7851, 0.0587, 0.9779),
        ("0001", 0.9846, 0.0333, 0.9990),
        ("0010", 0.7851, 0.0138, 0.9963),
        ("0011", 0.7851, 0.0744, 0.9663),
        ("0100", 0.2149, 0.1279, 0.6254),
        ("0101", 0.9846, 0.1170, 0.9975),
        ("0110", 0.2149, 0.0490, 0.8917),
        ("0111", 0.7851, 0.4021, 0.8917),
        ("1000", 0.2149, 0.0007, 0.9959),
        ("1001", 0.7851, 0.0014, 0.9992),
        ("1010", 0.2149, 0.0002, 0.9991),
        ("1011", 0.2149, 0.0009, 0.9934),
        ("1100", 0.0008, 0.0007, 0.2417),
        ("1101", 0.2149, 0.0288, 0.8232),
        ("1110", 0.0008, 0.0002, 0.7897),
        ("1111", 0.0154, 0.0102, 0.2472),
    ];
    let p_y = |k: f64| ((1.0 + (3.0 - 3.0 * k).exp()).ln() - (1.0 + (-3.0 * k).exp()).ln()) / 3.0;
    for (key, e, l, r) in rewards {
        let v: Vec<&str> = key.split("").filter(|s| !s.is_empty()).collect();
        let iv = b.reward(&v).unwrap();
        close(iv.lower, l, 5e-5);
        close(iv.upper, r, 5e-5);
        let f: Vec<f64> = v.iter().map(|s| s.parse().unwrap()).collect();
        let k = f[0] + f[1] + f[0] * f[1] - f[3] + f[2] * f[3] - f[1] * f[3];
        close(p_y(k), e, 5e-5);
        assert!(iv.contains(p_y(k), 1e-12));
    }

    // Eq. 9.67 contains every policy's true value. Interventional values do
    // not depend on α, and with α = 0 NUC holds, so DP on that table gives
    // them exactly.
    let unconfounded = dtr_table(0.0, 0.0);
    let reward = Reward::total(["Y"]);
    for (x1, x2) in [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")] {
        let pi = constant(x1, x2);
        let truth = dp(&unconfounded, &dtr_full(), &reward, &pi).unwrap().value;
        assert!(b.policy_bounds(&pi).unwrap().contains(truth, 1e-9));
    }
    let plan = optimize(&unconfounded, &dtr_full(), &reward).unwrap();
    close(plan.value, 0.6757, 1e-4);
    // Experiment 9.4 reports r_π = 0.6283 for π = (X1 ← 1, X2 ← 0). The
    // exact upper bound is 0.6788, and no deterministic policy has 0.6283
    // (the nearest are ≈ 0.62), so the experiment's figure must come from
    // something other than these bounds, e.g. finite samples.
    close(b.policy_bounds(&constant("1", "0")).unwrap().upper, 0.6788, 5e-5);
}

/// Thms 9.2.5–9.2.6 and Eq. 9.67 on random confounded two-stage SCMs: the
/// bounds always contain the truth, for every transition, reward and
/// policy.
#[test]
fn sequential_bounds_are_sound_on_random_scms() {
    let mut rng = SplitMix64::new(2024);
    let vars = ["S1", "X1", "S2", "X2", "Y"];
    for _ in 0..25 {
        let m = random_scm(&vars, &mut rng, 3);
        let cdm = match Cdm::new(m, dtr_full(), Reward::total(["Y"])) {
            Ok(c) => c,
            Err(CrlError::InvalidSpace(_)) => continue,
            Err(e) => panic!("{e}"),
        };
        let obs = cdm.observational_data().unwrap();
        let b = causal_bounds(&obs, cdm.space(), cdm.reward()).unwrap();
        for _ in 0..20 {
            let table: Vec<bool> = (0..10).map(|_| rng.unit() < 0.5).collect();
            let pi = Deterministic(move |x: &str, s: &State<'_>| {
                let bit = |v: &str| usize::from(&s[v] == "1");
                let i = if x == "X1" {
                    bit("S1")
                } else {
                    2 + bit("S1") * 4 + bit("X1") * 2 + bit("S2")
                };
                if table[i] { "1" } else { "0" }.to_string()
            });
            let truth = cdm.evaluate(&pi).unwrap();
            let iv = b.policy_bounds(&pi).unwrap();
            assert!(iv.contains(truth, 1e-9), "{truth} not in {iv:?}");
        }
        let opt = cdm.optimal().unwrap();
        assert!(b.policy_bounds(&opt.policy).unwrap().contains(opt.value, 1e-9));
    }
}

/// A two-stage environment whose observational data rule out the stage-1
/// action 1: the behaviour policy takes it almost always, and it pays off
/// rarely.
fn ucb_environment() -> Cdm {
    let m = model(
        vec![
            ("S1", expr::var("U1")),
            ("X1", expr::or([expr::var("UX"), expr::var("C")])),
            ("S2", expr::xor(expr::var("S1"), expr::var("U2"))),
            ("X2", expr::xor(expr::var("S2"), expr::var("C"))),
            (
                "Y",
                expr::and([
                    expr::not(expr::var("X1")),
                    expr::or([
                        expr::eq(expr::var("X2"), expr::var("S2")),
                        expr::var("UY"),
                    ]),
                ]),
            ),
        ],
        vec![
            ("U1", Dist::Bernoulli(0.5)),
            ("UX", Dist::Bernoulli(0.9)),
            ("C", Dist::Bernoulli(0.5)),
            ("U2", Dist::Bernoulli(0.2)),
            ("UY", Dist::Bernoulli(0.3)),
        ],
    );
    Cdm::new(m, dtr_full(), Reward::total(["Y"])).unwrap()
}

#[test]
fn ucb_plus_for_cdms_beats_ucb_when_bounds_are_informative() {
    let mut env = ucb_environment();
    let space = dtr_full();
    let reward = Reward::total(["Y"]);
    let obs = env.observational_data().unwrap();
    let bounds = causal_bounds(&obs, &space, &reward).unwrap();
    let opt = env.optimal().unwrap();
    // Every policy that plays X1 = 1 has r_π < µ*.
    assert!(bounds.policy_bounds(&constant("1", "0")).unwrap().upper < opt.value);

    let episodes = 300;
    let plain = UcbCdm::default().run(&mut env, &space, &reward, episodes, 3).unwrap();
    let plus = UcbCdm { bounds: Some(bounds) }.run(&mut env, &space, &reward, episodes, 3).unwrap();
    let r_plain = *plain.regret(&env).unwrap().last().unwrap();
    let r_plus = *plus.regret(&env).unwrap().last().unwrap();
    assert!(r_plus < r_plain, "{r_plus} vs {r_plain}");
    // UCB+ stops paying for X1 = 1 at once; UCB keeps learning slowly.
    let late_plus = plus.regret(&env).unwrap();
    assert!(late_plus[episodes - 1] - late_plus[episodes / 2] < 0.5 * late_plus[episodes / 2] + 1.0);
    assert_eq!(plus.policies.len(), episodes);

    // A space without perfect recall takes the enumeration path.
    let markov = PolicySpace::new([("X1", &["S1"][..]), ("X2", &["S2"])]).unwrap();
    let mut env2 = Cdm::new(env.model().clone(), markov.clone(), reward.clone()).unwrap();
    let run = UcbCdm::default().run(&mut env2, &markov, &reward, 60, 1).unwrap();
    assert_eq!(run.run.rewards.len(), 60);
    let mismatched = UcbCdm {
        bounds: Some(causal_bounds(&obs, &space, &reward).unwrap()),
    };
    assert!(matches!(
        mismatched.run(&mut env2, &markov, &reward, 1, 1),
        Err(CrlError::InvalidInput(_))
    ));
}

// ===========================================================================
// Task 2: where to intervene (Sec. 9.3)
// ===========================================================================

fn example_9_5(additive: bool) -> Model {
    let y = if additive {
        expr::add(expr::var("X2"), expr::var("U2"))
    } else {
        expr::xor(expr::var("X2"), expr::var("U2"))
    };
    model(
        vec![
            ("X1", expr::var("U1")),
            ("X2", expr::xor(expr::var("X1"), expr::var("U2"))),
            ("Y", y),
        ],
        vec![("U1", Dist::Bernoulli(0.5)), ("U2", Dist::Bernoulli(0.5))],
    )
}

/// `max_x E[Y | do(x)]` over the arms of an intervention set, exactly.
fn best_arm(m: &Model, set: &[&str]) -> f64 {
    if set.is_empty() {
        return m
            .exogenous_assignments()
            .into_iter()
            .map(|(u, p)| p * super::cdm::evaluate_world(m, &u, &[]).unwrap()["Y"].as_int() as f64)
            .sum();
    }
    let space = PolicySpace::new(set.iter().map(|x| (*x, &[] as &[&str]))).unwrap();
    Cdm::new(m.clone(), space, Reward::total(["Y"])).unwrap().optimal().unwrap().value
}

#[test]
fn example_9_5_where_to_intervene() {
    let m = example_9_5(false);
    let g = diagram(&m);
    // Eqs. 9.76–9.80.
    close(best_arm(&m, &[]), 0.5, 1e-12);
    close(best_arm(&m, &["X1"]), 1.0, 1e-12);
    close(best_arm(&m, &["X2"]), 0.5, 1e-12);
    close(best_arm(&m, &["X1", "X2"]), 0.5, 1e-12);
    // Fig. 9.8(c): {X1, X2} is not minimal (µ_{x1,x2} = µ_{x2}).
    let mis = minimal_intervention_sets(&g, &["X1", "X2"], "Y").unwrap();
    assert_eq!(mis, [vec![], vec!["X1".to_string()], vec!["X2".to_string()]]);
    // ∅ is dominated by X1 (Eq. 9.82); both singletons can be optimal.
    let sets = pomis(&g, &["X1", "X2"], "Y").unwrap();
    assert_eq!(sets, [vec!["X1".to_string()], vec!["X2".to_string()]]);
    assert_eq!(muct(&g, "Y").unwrap(), ["X2", "Y"]);
    assert_eq!(interventional_border(&g, "Y").unwrap(), ["X1"]);
    // M′ (Eq. 9.83) makes X2 the better scope.
    let m2 = example_9_5(true);
    close(best_arm(&m2, &["X1"]), 1.0, 1e-12);
    close(best_arm(&m2, &["X2"]), 1.5, 1e-12);
}

/// Fig. 9.12, reconstructed from its c-factorisation `P_b(C) P_c(D)
/// P_{a,c}(E) P_{d,e}(Y, A, B)`: B → C → D, A → E ← C, D → Y ← E, and
/// A ↔ Y ↔ B. Of the graphs that factorisation allows, this is the
/// sparsest whose POMISes are the book's.
#[test]
fn pomis_of_figure_9_12() {
    let mut g = Admg::new();
    for (a, b) in [("B", "C"), ("C", "D"), ("A", "E"), ("C", "E"), ("D", "Y"), ("E", "Y")] {
        g.directed(a, b);
    }
    g.bidirected("A", "Y");
    g.bidirected("B", "Y");
    let sets = pomis(&g, &["B", "D", "E"], "Y").unwrap();
    let want: Vec<Vec<String>> = [&[][..], &["B"], &["D"], &["E"], &["B", "D"], &["D", "E"]]
        .iter()
        .map(|v| v.iter().map(|s| (*s).to_string()).collect())
        .collect();
    assert_eq!(sets, want);

    // The projection onto {B, D, E, Y} turns the latent C into D ↔ E.
    let h = latent_projection(&g, &["B", "D", "E", "Y"]).unwrap();
    assert!(!h.is_d_separated(&["D"], &["E"], &["B"]).unwrap());
    assert!(h.ancestors_of(&["D"]).unwrap().contains(&"B"));
}

/// POMIS completeness on random SCMs: some POMIS always attains the best
/// reward over every intervention set (Def. 9.3.3, Thm 9.3.2).
#[test]
fn some_pomis_is_optimal_on_random_scms() {
    let mut rng = SplitMix64::new(99);
    let vars = ["A", "B", "C", "Y"];
    for _ in 0..30 {
        let m = random_scm(&vars, &mut rng, 2);
        let g = diagram(&m);
        let candidates = ["A", "B", "C"];
        let all = (0..8u32)
            .map(|mask| {
                let set: Vec<&str> = (0..3).filter(|i| mask >> i & 1 == 1).map(|i| candidates[i]).collect();
                best_arm(&m, &set)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let best_pomis = pomis(&g, &candidates, "Y")
            .unwrap()
            .iter()
            .map(|s| best_arm(&m, &s.iter().map(String::as_str).collect::<Vec<_>>()))
            .fold(f64::NEG_INFINITY, f64::max);
        close(best_pomis, all, 1e-12);
    }
}

// ===========================================================================
// Task 3: counterfactual decision-making (Sec. 9.4)
// ===========================================================================

/// The greedy casino of Ex. 9.10. Table 9.6a's payout rates in tenths,
/// drawn with UY uniform on {0, …, 9}. The natural choice is
/// X ← UD ⊕ UB: Eq. 9.98 prints X ← ¬(UD ⊕ UB), but Table 9.6a's asterisks,
/// Table 9.6b (E[Y | X] = 0.15), Eqs. 9.108–9.113 and Ex. 9.12
/// (X′ ← UD ⊕ UB) all need the version without the negation.
fn greedy_casino() -> Model {
    let lit = |v: &str, b: bool| if b { expr::var(v) } else { expr::not(expr::var(v)) };
    let rates = [
        ((false, false, false), 1),
        ((false, false, true), 5),
        ((false, true, false), 5),
        ((false, true, true), 1),
        ((true, false, false), 4),
        ((true, false, true), 2),
        ((true, true, false), 2),
        ((true, true, true), 4),
    ];
    let mut rate = expr::int(0);
    for ((d, b, x), r) in rates {
        rate = expr::add(
            rate,
            expr::mul(expr::int(r), expr::and([lit("UD", d), lit("UB", b), lit("X", x)])),
        );
    }
    model(
        vec![
            ("X", expr::xor(expr::var("UD"), expr::var("UB"))),
            ("Y", expr::lt(expr::var("UY"), rate)),
        ],
        vec![
            ("UD", Dist::Bernoulli(0.5)),
            ("UB", Dist::Bernoulli(0.5)),
            ("UY", Dist::Categorical(vec![0.1; 10])),
        ],
    )
}

#[test]
fn greedy_casino_tables_9_6_to_9_8() {
    let t = ett(&greedy_casino(), "X", &Reward::total(["Y"])).unwrap();
    assert_eq!(t.arms, ["0", "1"]);
    close(t.intended[1], 0.5, 1e-12);
    // Table 9.6b.
    for v in t.observational() {
        close(v, 0.15, 1e-12);
    }
    for v in t.interventional() {
        close(v, 0.3, 1e-12);
    }
    // Table 9.7.
    close(t.values[0][0], 0.15, 1e-12);
    close(t.values[0][1], 0.45, 1e-12);
    close(t.values[1][0], 0.45, 1e-12);
    close(t.values[1][1], 0.15, 1e-12);
    // Ex. 9.12: π ≜ X ← ¬X′ earns 0.45, the optimum; Table 9.8 has
    // P(X′ = 0, X = 1, Y = 1) = 0.225.
    let (best, value) = t.optimal();
    assert_eq!(best, [1, 0]);
    close(value, 0.45, 1e-12);
    close(t.intended[0] * t.values[0][1], 0.225, 1e-12);
    close(t.best_interventional().1, 0.3, 1e-12);
    // Eq. 9.129: a policy ignoring X′ earns Σ π(x) E_x[Y].
    close(t.evaluate(|x, _| if x == 1 { 0.7 } else { 0.3 }), 0.3, 1e-12);
}

#[test]
fn counterfactual_bandits_beat_ucb_in_the_greedy_casino() {
    let m = greedy_casino();
    let reward = Reward::total(["Y"]);
    let t = ett(&m, "X", &reward).unwrap();
    let episodes = 3000;

    let mut env = CtfBandit::new(m.clone(), "X", reward.clone()).unwrap();
    let ctf = CtfUcb.run(&mut env, episodes, 5).unwrap();
    let ctf_regret = *ctf.pseudo_regret(&t).unwrap().last().unwrap();
    // UCB plays atomic interventions, worth 0.3 against 0.45: linear
    // regret 0.15 T (Cor. 9.4.5).
    let mut cdm = Cdm::new(m, PolicySpace::bandit("X"), reward.clone()).unwrap();
    let ucb = Ucb.run(&mut cdm, "X", &reward, episodes, 5).unwrap();
    let ucb_regret: f64 = ucb.actions.iter().map(|a| 0.45 - t.interventional()[usize::from(a[0] == "1")]).sum();
    close(ucb_regret, 0.15 * episodes as f64, 1e-9);
    assert!(ctf_regret < 0.2 * ucb_regret, "{ctf_regret} vs {ucb_regret}");

    // Ctf-RCT commits to X ← ¬X′ after exploring.
    let rct = CtfRct { trials: 400 }.run(&mut env, episodes, 6).unwrap();
    for (xp, x) in rct.intended.iter().zip(&rct.realized).skip(400) {
        assert_ne!(xp, x);
    }
    let r = rct.pseudo_regret(&t).unwrap();
    close(r[episodes - 1], r[399], 1e-12);
    assert!(matches!(CtfRct { trials: 0 }.run(&mut env, 10, 1), Err(CrlError::InvalidInput(_))));
}

/// One stage of the inventory MDP of Ex. 7.2 (Eq. 7.4), with a fair coin
/// for the current state.
fn inventory_stage() -> Model {
    model(
        vec![
            ("S", expr::var("US")),
            ("X", expr::xor(expr::var("S"), expr::var("U1"))),
            (
                "Y",
                expr::xor(
                    expr::xor(expr::var("S"), expr::var("X")),
                    expr::xor(expr::var("U1"), expr::var("U3")),
                ),
            ),
            (
                "S2",
                expr::xor(
                    expr::or([expr::var("S"), expr::var("X")]),
                    expr::xor(expr::var("U1"), expr::var("U2")),
                ),
            ),
        ],
        vec![
            ("US", Dist::Bernoulli(0.5)),
            ("U1", Dist::Bernoulli(0.9)),
            ("U2", Dist::Bernoulli(0.1)),
            ("U3", Dist::Bernoulli(0.1)),
        ],
    )
}

fn inventory() -> CtfMdp {
    CtfMdp::from_stage_model(
        &inventory_stage(),
        StageVars {
            state: "S",
            action: "X",
            reward: "Y",
            next_state: "S2",
        },
    )
    .unwrap()
}

#[test]
fn inventory_mdp_tables_9_9_to_9_11_and_8_1() {
    let mdp = inventory();
    // Table 9.9, transition(s, x′, x, s″, x″), X′ intended and X realised.
    // The book's block with S″ = 0, X″ = 1 repeats the X″ = 0 block, so its
    // rows do not sum to one; the other three blocks are checked.
    let rows = [
        ((0, 0, 0, 0, 0), 0.09),
        ((0, 0, 1, 0, 0), 0.01),
        ((0, 1, 0, 0, 0), 0.01),
        ((1, 0, 1, 0, 0), 0.09),
        ((1, 1, 0, 0, 0), 0.01),
        ((0, 0, 1, 1, 0), 0.81),
        ((0, 1, 0, 1, 0), 0.81),
        ((1, 0, 0, 1, 0), 0.09),
        ((1, 1, 1, 1, 0), 0.81),
        ((0, 0, 0, 1, 1), 0.01),
        ((0, 0, 1, 1, 1), 0.09),
        ((1, 1, 1, 1, 1), 0.09),
    ];
    for ((s, xp, x, s2, x2), p) in rows {
        close(mdp.transition(s, xp, x, s2, x2), p, 1e-12);
    }
    for s in 0..2 {
        for xp in 0..2 {
            for x in 0..2 {
                let total: f64 = (0..2).flat_map(|s2| (0..2).map(move |x2| (s2, x2))).map(|(s2, x2)| mdp.transition(s, xp, x, s2, x2)).sum();
                close(total, 1.0, 1e-12);
                // Table 9.10: E[Y_x | s, x′] is 0.9 against intuition, 0.1 with it.
                close(mdp.reward[s][xp][x], if x == xp { 0.1 } else { 0.9 }, 1e-12);
            }
        }
    }
    // Ex. 9.15 with γ = 0.9: always defy the intuition, worth 0.9/(1 − γ) = 9.
    let q = mdp.value_iteration(0.9, 1e-12).unwrap();
    close(q.value, 9.0, 1e-9);
    for s in 0..2 {
        for xp in 0..2 {
            assert_eq!(q.policy[s][xp], 1 - xp);
            close(q.q[s][xp][1 - xp], 9.0, 1e-9);
            // Table 9.11 prints 1 here; the Bellman equation gives 0.1 + 0.9 · 9.
            close(q.q[s][xp][xp], 8.2, 1e-9);
        }
    }
    // Ex. 8.5 / Table 8.1: the best experimental policy is X ← S, worth 8.2
    // (Ex. 9.15 writes X ← ¬S).
    let e = mdp.experimental_value_iteration(0.9, 1e-12).unwrap();
    close(e.value, 8.2, 1e-9);
    assert_eq!(e.policy, [0, 1]);
    close(e.q[0][0], 8.2, 1e-9);
    close(e.q[0][1], 7.56, 1e-9);
    close(e.q[1][0], 7.56, 1e-9);

    // Finite horizon: 0.9 per step against 0.82.
    let h = 10;
    close(mdp.finite_horizon(h)[0].value, 9.0, 1e-9);
    close(mdp.experimental_finite_horizon(h)[0].value, 8.2, 1e-9);
    let defy = vec![vec![vec![1, 0]; 2]; h];
    close(mdp.evaluate_finite(&defy, h), 9.0, 1e-9);
}

#[test]
fn ctf_ucbvi_learns_what_ucbvi_cannot() {
    let mdp = inventory();
    let h = 5;
    let episodes = 1500;
    // Eq. 9.197's constant 7 keeps every Q at its cap H for far longer
    // than a test can run, so use a smaller one.
    let mut env = CtfMdpSimulator::new(mdp.clone()).unwrap();
    let ctf = CtfUcbvi { bonus: 0.3, ..CtfUcbvi::default() }.run(&mut env, h, episodes, 11).unwrap();
    let exp = CtfUcbvi { bonus: 0.3, use_intent: false, ..CtfUcbvi::default() }
        .run(&mut env, h, episodes, 11)
        .unwrap();
    let r_ctf = ctf.regret(&mdp, h);
    let r_exp = exp.regret(&mdp, h);
    // UCBVI over (s, x) loses at least 0.08 per step forever.
    assert!(r_exp[episodes - 1] > 0.08 * (h * episodes) as f64 * 0.9);
    // Ctf-UCBVI has converged by mid-run; UCBVI keeps paying.
    let half = episodes / 2;
    let late_ctf = r_ctf[episodes - 1] - r_ctf[half];
    let late_exp = r_exp[episodes - 1] - r_exp[half];
    assert!(late_ctf < 0.1 * late_exp, "{late_ctf} vs {late_exp}");
    assert!(r_ctf[episodes - 1] < 0.5 * r_exp[episodes - 1]);
    // Late episodes play the optimal counterfactual policy.
    assert_eq!(ctf.policies[episodes - 1][0], vec![vec![1, 0]; 2]);
    assert_eq!(ctf.returns.len(), episodes);
}

// ===========================================================================
// Task 4: causal imitation learning (Sec. 9.5)
// ===========================================================================

fn bandit_model(confounded: bool) -> Model {
    if confounded {
        // Ex. 9.20: X ← ¬U, Y ← X ⊕ U.
        model(
            vec![
                ("X", expr::not(expr::var("U"))),
                ("Y", expr::xor(expr::var("X"), expr::var("U"))),
            ],
            vec![("U", Dist::Bernoulli(0.5))],
        )
    } else {
        // Ex. 9.17: X ← U, Y ← X.
        model(
            vec![("X", expr::var("U")), ("Y", expr::var("X"))],
            vec![("U", Dist::Bernoulli(0.9))],
        )
    }
}

#[test]
fn behavioral_cloning_and_irl_in_bandits_examples_9_17_to_9_25() {
    let y = Reward::total(["Y"]);
    let space = PolicySpace::bandit("X");
    let identity: &dyn Fn(&[f64]) -> f64 = &|v| v[0];

    // Ex. 9.18: BC matches the expert, 0.9.
    let cdm = Cdm::new(bandit_model(false), space.clone(), y.clone()).unwrap();
    let obs = cdm.observational_data().unwrap();
    let bc = behavioral_cloning(&obs, &space).unwrap();
    close(bc.rule("X").unwrap().table[&Vec::new()][1], 0.9, 1e-12);
    close(cdm.evaluate(&bc).unwrap(), 0.9, 1e-12);
    assert!(imitation_backdoor(&cdm.diagram(), &space, &["Y"]).unwrap().holds());
    // Ex. 9.24: IRL with R = αY, α > 0, beats the expert: X ← 1, ν* = −0.1.
    let irl = causal_mwal(&obs, &space, &["Y"], &[identity]).unwrap();
    close(irl.value, -0.1, 1e-12);
    assert_eq!(irl.mixture.len(), 1);
    close(cdm.evaluate(&irl.mixture[0].1).unwrap(), 1.0, 1e-12);

    // Ex. 9.20: under confounding, BC earns 0.5 against the expert's 1.
    let cdm = Cdm::new(bandit_model(true), space.clone(), y.clone()).unwrap();
    let obs = cdm.observational_data().unwrap();
    let bc = behavioral_cloning(&obs, &space).unwrap();
    close(cdm.evaluate(&bc).unwrap(), 0.5, 1e-12);
    let report = imitation_backdoor(&cdm.diagram(), &space, &["Y"]).unwrap();
    assert!(!report.holds());
    assert_eq!(report.stages[0].condition, ImitationCondition::Open("X <-> Y".into()));
    assert_eq!(imitation_subspace(&cdm.diagram(), &space, &["Y"]).unwrap(), None);
    // Ex. 9.25: naive IRL believes ν* = 0, yet its policy earns 0.5.
    let irl = causal_mwal(&obs, &space, &["Y"], &[identity]).unwrap();
    close(irl.value, 0.0, 1e-12);
    for (_, pi) in &irl.mixture {
        close(cdm.evaluate(pi).unwrap(), 0.5, 1e-12);
    }
}

/// Fig. 9.30(a), reconstructed from Exs. 9.21–9.23: X1 → W → X2 → Y,
/// Z → W, Z → Y, with X1 ↔ Z and X2 ↔ W.
fn figure_9_30() -> Admg {
    let mut g = Admg::new();
    for (a, b) in [("X1", "W"), ("W", "X2"), ("X2", "Y"), ("Z", "W"), ("Z", "Y")] {
        g.directed(a, b);
    }
    g.bidirected("X1", "Z");
    g.bidirected("X2", "W");
    g
}

#[test]
fn imitation_backdoor_examples_9_21_to_9_23() {
    let g = figure_9_30();
    // Ex. 9.21: Π₁ is imitation admissible but not sequentially back-door.
    let pi1 = PolicySpace::new([("X1", &[] as &[&str]), ("X2", &["Z"])]).unwrap();
    let report = imitation_backdoor(&g, &pi1, &["Y"]).unwrap();
    assert!(report.holds());
    assert_eq!(report.stages[0].condition, ImitationCondition::NotAncestor);
    assert_eq!(report.stages[1].condition, ImitationCondition::Blocked);
    assert!(!sequential_backdoor(&g, &pi1, &["Y"]).unwrap().holds());
    // Ex. 9.22: Π₂ is sequentially back-door but not imitation admissible.
    let pi2 = PolicySpace::new([("X1", &["Z"][..]), ("X2", &["W"])]).unwrap();
    assert!(sequential_backdoor(&g, &pi2, &["Y"]).unwrap().holds());
    let report = imitation_backdoor(&g, &pi2, &["Y"]).unwrap();
    assert!(!report.holds());
    assert!(matches!(&report.stages[1].condition, ImitationCondition::Open(p) if p.contains("W")));
    // Ex. 9.23: the subspace search finds Π₁ inside {⟨X1, Z⟩, ⟨X2, {Z, W}⟩}.
    let big = PolicySpace::new([("X1", &["Z"][..]), ("X2", &["Z", "W"])]).unwrap();
    assert_eq!(imitation_subspace(&g, &big, &["Y"]).unwrap(), Some(pi1.clone()));
    assert!(is_minimal_imitation_admissible(&g, &pi1, &["Y"]).unwrap());
}

/// Experiment 9.11's SCM on Fig. 9.31(a). Its Y is a feature vector the
/// book does not specify; here Y ← X1 ⊕ X2 ⊕ Z2 ⊕ U3. Y does not read Z1:
/// with Z1 → Y the path X2 ↔ Z2 ↔ Z1 → Y would stay open given Z2, and
/// Ex. 9.26's claim that {Z2} blocks X2 would fail.
fn experiment_9_11() -> Model {
    let x = |a: &str, b: &str| expr::xor(expr::var(a), expr::var(b));
    model(
        vec![
            ("Z1", x("U1", "U3")),
            ("X1", expr::var("U5")),
            ("Z2", expr::xor(x("U1", "U2"), expr::var("U4"))),
            ("X2", x("U2", "Z2")),
            ("Y", expr::xor(x("X1", "X2"), x("Z2", "U3"))),
        ],
        vec![
            ("U1", Dist::Bernoulli(0.8)),
            ("U2", Dist::Bernoulli(0.8)),
            ("U3", Dist::Bernoulli(0.2)),
            ("U4", Dist::Bernoulli(0.1)),
            ("U5", Dist::Bernoulli(0.68)),
        ],
    )
}

#[test]
fn causal_bc_and_identification_examples_9_26_and_9_27() {
    let m = experiment_9_11();
    let full = PolicySpace::new([("X1", &["Z1"][..]), ("X2", &["Z1", "X1", "Z2"])]).unwrap();
    let cdm = Cdm::new(m.clone(), full.clone(), Reward::total(["Y"])).unwrap();
    let g = cdm.diagram();
    let obs = cdm.observational_data().unwrap();

    // Ex. 9.26: Π₁ is admissible but not minimal; Π₂ is the minimal one.
    let pi1 = PolicySpace::new([("X1", &["Z1"][..]), ("X2", &["Z2"])]).unwrap();
    let pi2 = PolicySpace::new([("X1", &[] as &[&str]), ("X2", &["Z2"])]).unwrap();
    assert!(imitation_backdoor(&g, &pi1, &["Y"]).unwrap().holds());
    assert!(!is_minimal_imitation_admissible(&g, &pi1, &["Y"]).unwrap());
    assert_eq!(imitation_subspace(&g, &full, &["Y"]).unwrap(), Some(pi2.clone()));
    assert!(!imitation_backdoor(&g, &full, &["Y"]).unwrap().holds());

    // Thm 9.5.3: causal BC matches the expert exactly; BC on everything
    // the imitator sees does not.
    let expert = expert_distribution(&obs, &["Y"]).unwrap();
    let e_y = expert.iter().find(|(y, _)| y[0] == "1").unwrap().1;
    let causal = behavioral_cloning(&obs, &pi2).unwrap();
    let in_pi2 = Cdm::new(m.clone(), pi2.clone(), Reward::total(["Y"])).unwrap();
    close(in_pi2.evaluate(&causal).unwrap(), e_y, 1e-12);
    let naive = behavioral_cloning(&obs, &full).unwrap();
    assert!((cdm.evaluate(&naive).unwrap() - e_y).abs() > 1e-3);

    // Thm 9.5.4 (Ex. 9.27): Eq. 9.266 equals the true P_π(Y) for policies
    // in Π₂, here causal BC and an arbitrary stochastic rule.
    let other = Stochastic(|x: &str, s: &State<'_>, v: &str| {
        let p1 = if x == "X1" { 0.3 } else if &s["Z2"] == "1" { 0.9 } else { 0.2 };
        if v == "1" { p1 } else { 1.0 - p1 }
    });
    for pi in [&causal as &dyn Policy, &other] {
        let identified = imitation_distribution(&obs, &pi2, &["Y"], pi).unwrap();
        let truth = in_pi2.evaluate(pi).unwrap();
        let p1 = identified.iter().find(|(y, _)| y[0] == "1").unwrap().1;
        close(p1, truth, 1e-12);
    }
}

/// Thm 9.5.4 fails for a stage admitted by condition 1 alone: here X does
/// not cause Y but shares a confounder with it, so every policy has
/// `P_π(Y) = P(Y)`, while Eq. 9.266 conditions on X.
#[test]
fn condition_1_breaks_theorem_9_5_4() {
    let m = model(
        vec![("X", expr::var("U")), ("Y", expr::or([expr::var("U"), expr::var("V")]))],
        vec![("U", Dist::Bernoulli(0.5)), ("V", Dist::Bernoulli(0.2))],
    );
    let space = PolicySpace::bandit("X");
    let cdm = Cdm::new(m, space.clone(), Reward::total(["Y"])).unwrap();
    let g = cdm.diagram();
    assert!(is_minimal_imitation_admissible(&g, &space, &["Y"]).unwrap());
    let report = imitation_backdoor(&g, &space, &["Y"]).unwrap();
    assert_eq!(report.stages[0].condition, ImitationCondition::NotAncestor);
    let obs = cdm.observational_data().unwrap();
    let always1 = Deterministic(|_: &str, _: &State<'_>| "1".to_string());
    close(cdm.evaluate(&always1).unwrap(), 0.6, 1e-12);
    let formula = imitation_distribution(&obs, &space, &["Y"], &always1).unwrap();
    close(formula.iter().find(|(y, _)| y[0] == "1").unwrap().1, 1.0, 1e-12);
    // BC still matches the expert, as Thm 9.5.3 promises.
    let bc = behavioral_cloning(&obs, &space).unwrap();
    close(cdm.evaluate(&bc).unwrap(), 0.6, 1e-12);
}

/// Thms 9.5.3 and 9.5.4 on random SCMs: whenever a minimal admissible
/// subspace exists, BC on it matches the expert, and when every stage
/// blocks its back-door paths Eq. 9.266 identifies every policy in it.
#[test]
fn imitation_theorems_hold_on_random_scms() {
    let mut rng = SplitMix64::new(77);
    let vars = ["S1", "X1", "S2", "X2", "Y"];
    let full = dtr_full();
    let (mut found, mut identified) = (0, 0);
    for _ in 0..200 {
        let m = random_scm(&vars, &mut rng, 3);
        let Ok(cdm) = Cdm::new(m.clone(), full.clone(), Reward::total(["Y"])) else {
            continue;
        };
        let g = cdm.diagram();
        let Some(sub) = imitation_subspace(&g, &full, &["Y"]).unwrap() else {
            continue;
        };
        found += 1;
        let obs = cdm.observational_data().unwrap();
        let e_y = expert_distribution(&obs, &["Y"]).unwrap().iter().find(|(y, _)| y[0] == "1").map_or(0.0, |p| p.1);
        let in_sub = Cdm::new(m, sub.clone(), Reward::total(["Y"])).unwrap();
        let bc = behavioral_cloning(&obs, &sub).unwrap();
        close(in_sub.evaluate(&bc).unwrap(), e_y, 1e-12);
        let coin: Vec<f64> = (0..32).map(|_| rng.unit()).collect();
        let pi = Stochastic(move |x: &str, s: &State<'_>, v: &str| {
            let mut i = usize::from(x == "X2") * 16;
            for (k, val) in s.values().enumerate() {
                i += usize::from(val == "1") << k;
            }
            let p1 = coin[i % 32];
            if v == "1" { p1 } else { 1.0 - p1 }
        });
        // Thm 9.5.4 needs every stage to block its back-door paths
        // (condition 2); see `condition_1_breaks_theorem_9_5_4`.
        let blocked = imitation_backdoor(&g, &sub, &["Y"])
            .unwrap()
            .stages
            .iter()
            .all(|s| s.condition == ImitationCondition::Blocked);
        if !blocked {
            continue;
        }
        identified += 1;
        match imitation_distribution(&obs, &sub, &["Y"], &pi) {
            Ok(d) => {
                let p1 = d.iter().find(|(y, _)| y[0] == "1").map_or(0.0, |p| p.1);
                close(p1, in_sub.evaluate(&pi).unwrap(), 1e-12);
            }
            // An action the data never show (a constant X_i) or never show
            // after some history cannot be evaluated.
            Err(CrlError::Positivity { .. } | CrlError::PolicyNotNormalized { .. }) => {}
            Err(e) => panic!("{e}"),
        }
    }
    assert!(found >= 20 && identified >= 10, "only {found} imitable models, {identified} identified");
}

// ===========================================================================
// Causal game theory (Sec. 9.6)
// ===========================================================================

/// The causal prisoner's dilemma of Ex. 9.28, payoffs in tenths. Scenario
/// M1 has R1 = R2 = 1, M2 has R1 = R2 = 0.
fn prisoners(read_well: bool) -> Model {
    let lit = |v: &str, b: bool| if b { expr::var(v) } else { expr::not(expr::var(v)) };
    // Table 9.14: ((X1, U1, X2, U2), (Y1, Y2)).
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
    let mut y1 = expr::int(0);
    let mut y2 = expr::int(0);
    for ((x1, u1, x2, u2), (a, b)) in table {
        let cell = || expr::and([lit("X1", x1 == 1), lit("U1", u1 == 1), lit("X2", x2 == 1), lit("U2", u2 == 1)]);
        y1 = expr::add(y1, expr::mul(expr::int(a), cell()));
        y2 = expr::add(y2, expr::mul(expr::int(b), cell()));
    }
    let r = expr::boolean(read_well);
    model(
        vec![
            ("X1", expr::xor(r.clone(), expr::var("U1"))),
            ("X2", expr::xor(r, expr::var("U2"))),
            ("Y1", y1),
            ("Y2", y2),
        ],
        vec![("U1", Dist::Bernoulli(0.4)), ("U2", Dist::Bernoulli(0.4))],
    )
}

fn tenths(p: (f64, f64)) -> (f64, f64) {
    (p.0 / 10.0, p.1 / 10.0)
}

fn close2(a: (f64, f64), b: (f64, f64), tol: f64) {
    assert!((a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol, "{a:?} vs {b:?} (tol {tol})");
}

/// Table 9.17, the payoffs player 1 learned from 100 000 samples. Rows and
/// columns: natural, do(0), do(1), 1 − X′.
const TABLE_9_17: [[(f64, f64); 4]; 4] = [
    [(-2.4428, -2.4450), (-1.2184, -2.6837), (-8.8923, 0.0), (-7.6678, -0.2388)],
    [(-2.6828, -1.2345), (-0.9831, -0.9831), (-6.9321, -0.4901), (-5.2324, -0.2388)],
    [(0.0, -8.8479), (-0.4753, -6.9509), (-1.9602, -1.8970), (-2.4354, 0.0)],
    [(-0.2400, -7.6374), (-0.2400, -5.2502), (0.0, -2.3872), (0.0, 0.0)],
];

/// Fig. 9.36 as printed. Its entries are not consistently rounded: the
/// exact (natural, do(0)) is (−1.224, −2.664), printed (−1.2, −2.6).
const FIGURE_9_36: [[(f64, f64); 4]; 4] = [
    [(-2.4, -2.4), (-1.2, -2.6), (-8.9, 0.0), (-7.7, -0.2)],
    [(-2.6, -1.2), (-1.0, -1.0), (-7.0, -0.5), (-5.3, -0.2)],
    [(0.0, -8.9), (-0.5, -7.0), (-1.9, -1.9), (-2.4, 0.0)],
    [(-0.2, -7.7), (-0.2, -5.3), (0.0, -2.4), (0.0, 0.0)],
];

#[test]
fn causal_prisoners_dilemma_figures_9_33_and_9_36() {
    let m2 = CausalGame::from_model(&prisoners(false), [("X1", "Y1"), ("X2", "Y2")]).unwrap();
    let g = &m2.layered.game;
    assert_eq!(m2.layered.row_layers, [Layer::Natural, Layer::Do, Layer::Do, Layer::Counterfactual]);
    for i in 0..4 {
        for j in 0..4 {
            close2(tenths(g.payoffs[i][j]), FIGURE_9_36[i][j], 0.07);
            close2(tenths(g.payoffs[i][j]), TABLE_9_17[i][j], 0.06);
        }
    }
    close2(tenths(g.payoffs[0][0]), (-2.424, -2.424), 1e-9);
    close2(tenths(g.payoffs[1][2]), (-6.96, -0.48), 1e-9);

    // Fig. 9.33: M1 and M2 share the standard game, but instinct pays (0, 0)
    // in M1 (Thm 9.6.1).
    let m1 = CausalGame::from_model(&prisoners(true), [("X1", "Y1"), ("X2", "Y2")]).unwrap();
    close2(tenths(m1.layered.game.payoffs[0][0]), (0.0, 0.0), 1e-12);
    for i in 1..3 {
        for j in 1..3 {
            close2(m1.layered.game.payoffs[i][j], g.payoffs[i][j], 1e-9);
        }
    }
    let standard = m2.layered.projection(ActionSpace::L2, ActionSpace::L2).unwrap();
    assert_eq!(standard.pure_equilibria(), [(1, 1)]);
    close2(tenths(standard.equilibrium().unwrap().payoff), (-1.92, -1.92), 1e-9);
}

#[test]
fn learn_cne_in_the_causal_prisoners_dilemma_table_9_15() {
    let game = CausalGame::from_model(&prisoners(false), [("X1", "Y1"), ("X2", "Y2")]).unwrap();
    let cne = game.causal_nash((ActionSpace::L3, ActionSpace::L3)).unwrap();
    // Table 9.15 (it rounds the standard equilibrium −1.92 to −2).
    let table = [
        [(-2.4, -2.4), (-8.9, 0.0), (-8.9, 0.0), (-8.9, 0.0)],
        [(0.0, -8.9), (-1.92, -1.92), (-1.92, -1.92), (-2.4, 0.0)],
        [(0.0, -8.9), (-1.92, -1.92), (-1.92, -1.92), (-2.4, 0.0)],
        [(0.0, -8.9), (0.0, -2.4), (0.0, -2.4), (0.0, 0.0)],
    ];
    assert_eq!(cne.selection.rows, ["A1", "A2", "A1∪A2", "A3"]);
    for (got, want) in cne.selection.payoffs.iter().flatten().zip(table.iter().flatten()) {
        close2(tenths(*got), *want, 0.051);
    }
    assert_eq!(cne.selection.pure_equilibria(), [(3, 3)]);
    close2(tenths(cne.equilibrium.payoff), (0.0, 0.0), 1e-9);
    assert_eq!(cne.equilibrium.support(), (vec![3], vec![3]));
}

/// Fig. 9.37(a), where considering L3 hurts player 1, and its layer
/// selection game, Fig. 9.37(b).
#[test]
fn learn_cne_on_figure_9_37() {
    let p = |a: f64, b: f64| (a, b);
    let game = Bimatrix::new(
        vec!["X1 = X1′".into(), "do(X1 = 0)".into(), "do(X1 = 1)".into(), "X1 = 1 − X1′".into()],
        vec!["X2 = X2′".into(), "do(X2 = 0)".into(), "do(X2 = 1)".into()],
        vec![
            vec![p(-2.0, 2.0), p(-2.0, -2.0), p(-2.0, -2.0)],
            vec![p(0.0, 0.0), p(-1.5, -1.5), p(-1.5, -1.5)],
            vec![p(0.0, 0.0), p(-1.5, -1.5), p(-1.5, -1.5)],
            vec![p(2.0, -2.0), p(-1.0, -1.0), p(-1.0, -1.0)],
        ],
    )
    .unwrap();
    let layered = LayeredGame::new(
        game,
        vec![Layer::Natural, Layer::Do, Layer::Do, Layer::Counterfactual],
        vec![Layer::Natural, Layer::Do, Layer::Do],
    )
    .unwrap();
    let cne = learn_cne(&layered, (ActionSpace::L3, ActionSpace::L1L2)).unwrap();
    let table = [
        [(-2.0, 2.0), (-2.0, -2.0), (-2.0, 2.0)],
        [(0.0, 0.0), (-1.5, -1.5), (0.0, 0.0)],
        [(0.0, 0.0), (-1.5, -1.5), (0.0, 0.0)],
        [(2.0, -2.0), (-1.0, -1.0), (-1.0, -1.0)],
    ];
    for (got, want) in cne.selection.payoffs.iter().flatten().zip(table.iter().flatten()) {
        close2(*got, *want, 1e-9);
    }
    // (A2, A1 ∪ A2) is the first pure equilibrium of L_Γ: the CNE pays (0, 0),
    // better for player 1 than the (−1, −1) of playing L3.
    assert_eq!(cne.selection_equilibrium.support(), (vec![1], vec![2]));
    close2(cne.equilibrium.payoff, (0.0, 0.0), 1e-9);
    assert_eq!(cne.strategies, (vec![1, 2], vec![0, 1, 2]));
    let all = layered.projection(ActionSpace::L3, ActionSpace::L1L2).unwrap();
    close2(all.equilibrium().unwrap().payoff, (-1.0, -1.0), 1e-9);
}

#[test]
fn mixed_equilibria_by_support_enumeration() {
    let p = |a: f64| (a, -a);
    let pennies = Bimatrix::new(
        vec!["H".into(), "T".into()],
        vec!["H".into(), "T".into()],
        vec![vec![p(1.0), p(-1.0)], vec![p(-1.0), p(1.0)]],
    )
    .unwrap();
    assert!(pennies.pure_equilibria().is_empty());
    let eq = pennies.equilibrium().unwrap();
    close(eq.row[0], 0.5, 1e-9);
    close(eq.col[0], 0.5, 1e-9);
    close2(eq.payoff, (0.0, 0.0), 1e-9);
    // Battle of the sexes: two pure equilibria and one mixed.
    let bos = Bimatrix::new(
        vec!["O".into(), "F".into()],
        vec!["O".into(), "F".into()],
        vec![vec![(2.0, 1.0), (0.0, 0.0)], vec![(0.0, 0.0), (1.0, 2.0)]],
    )
    .unwrap();
    let all = bos.nash_equilibria().unwrap();
    assert_eq!(all.len(), 3);
    close(all[2].row[0], 2.0 / 3.0, 1e-9);
}

#[test]
fn ctf_nash_learning_recovers_the_prisoners_payoffs_table_9_17() {
    let m = prisoners(false);
    let mut rng = SplitMix64::new(4);
    let dom = vec!["0".to_string(), "1".to_string()];
    let samples: Vec<CtfGameSample> = (0..100_000)
        .map(|_| {
            let u = super::cdm::sample_exogenous(&m, &mut || rng.unit());
            let natural = super::cdm::evaluate_world(&m, &u, &[]).unwrap();
            let (a1, a2) = (rng.below(2) == 1, rng.below(2) == 1);
            let w = super::cdm::evaluate_world(&m, &u, &[("X1", a1.into()), ("X2", a2.into())]).unwrap();
            CtfGameSample {
                intended: natural["X1"].to_string(),
                action1: u8::from(a1).to_string(),
                action2: u8::from(a2).to_string(),
                reward: (w["Y1"].as_int() as f64 / 10.0, w["Y2"].as_int() as f64 / 10.0),
            }
        })
        .collect();
    let learned = ctf_nash_learning(&samples, &dom, &dom).unwrap();
    // Components: weights ≈ P(X2′) = (0.6, 0.4); R(0, 0, 0) = (−1.4, −1.4)
    // and (−1, −1).
    let c = &learned.components[&("0".to_string(), "0".to_string(), "0".to_string())];
    close(c[0].0, 0.6, 0.03);
    close2(c[0].1, (-1.4, -1.4), 1e-12);
    close2(c[1].1, (-1.0, -1.0), 1e-12);
    // Columns are g = do(0), do(1), identity, flip; Fig. 9.36 orders them
    // natural, do(0), do(1), flip.
    let exact = CausalGame::from_model(&m, [("X1", "Y1"), ("X2", "Y2")]).unwrap();
    let reorder = [1, 2, 0, 3];
    for i in 0..4 {
        for (j, &k) in reorder.iter().enumerate() {
            close2(learned.game.game.payoffs[i][j], tenths(exact.layered.game.payoffs[i][k]), 0.08);
        }
    }
    close2(learned.cne.equilibrium.payoff, (0.0, 0.0), 0.05);
    assert_eq!(learned.cne.game.rows.len(), 4);
}

