use why_data::fairness::Dataset;
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::graph::dseparation::Admg;

use super::*;

fn graph(directed: &[(&str, &str)], bidirected: &[(&str, &str)]) -> Admg {
    let mut g = Admg::new();
    for (a, b) in directed {
        g.directed(a, b);
    }
    for (a, b) in bidirected {
        g.bidirected(a, b);
    }
    g
}

/// A weighted table over binary `header`, rows in binary counting order.
fn table(header: &[&str], weights: &[f64]) -> Dataset {
    let n = header.len();
    assert_eq!(weights.len(), 1 << n);
    let rows: Vec<Vec<String>> = (0..1usize << n)
        .map(|k| {
            (0..n)
                .map(|i| ((k >> (n - 1 - i)) & 1).to_string())
                .collect()
        })
        .collect();
    Dataset::from_records(header.iter().copied(), rows)
        .unwrap()
        .with_weights(weights.to_vec())
        .unwrap()
}

fn q(g: &Admg, s: &str) -> CtfQuery {
    CtfQuery::parse(g, s).unwrap()
}

/// Table 13.3.
fn diet() -> Dataset {
    table(&["D", "B"], &[34.0, 78.0, 90.0, 54.0])
}

#[test]
fn example_13_9_the_bow_graph_is_not_identifiable() {
    let g = graph(&[("D", "B")], &[("D", "B")]);
    let objective = Objective::new(q(&g, "P(b_d)"), &[("D", "1"), ("B", "1")]);
    let r = neural_id(
        &g,
        &objective,
        &[Experiment::observational(diet())],
        &NcmConfig::default(),
    )
    .unwrap();
    assert!(!r.identifiable);
    // Any model that fits P(V) stays within the natural bounds
    // [P(d, b), P(d, b) + P(d')] = [54/256, 166/256], which M̂_4 and M̂_3 of
    // Ex. 13.6 attain.
    assert!(r.min > 54.0 / 256.0 - 0.02 && r.max < 166.0 / 256.0 + 0.02);
    assert!(r.min < 0.32 && r.max > 0.55, "{} {}", r.min, r.max);
    assert!(r.misfit.iter().all(|k| *k < 0.01));
}

#[test]
fn example_13_10_the_front_door_effect_is_identified() {
    let g = graph(&[("D", "S"), ("S", "B")], &[("D", "B")]);
    let data = table(
        &["D", "S", "B"],
        &[13.0, 15.0, 21.0, 63.0, 81.0, 27.0, 9.0, 27.0],
    );
    let objective = Objective::new(q(&g, "P(b_d)"), &[("D", "1"), ("B", "1")]);
    let r = neural_id(
        &g,
        &objective,
        &[Experiment::observational(data)],
        &NcmConfig::default(),
    )
    .unwrap();
    assert!(r.identifiable);
    assert!((r.estimate.unwrap() - 0.46875).abs() < 0.02); // Eq. 13.79
}

/// Table 13.5: `P(V)`, `P(V_{R=0})` and `P(V_{R=1})` over `(R, E, J)`.
fn hiring() -> Vec<Experiment> {
    let h = ["R", "E", "J"];
    vec![
        Experiment::observational(table(&h, &[1.0, 3.0, 3.0, 9.0, 9.0, 3.0, 1.0, 3.0])),
        Experiment::interventional(
            &["R"],
            table(&h, &[4.0, 12.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        Experiment::interventional(
            &["R"],
            table(&h, &[0.0, 0.0, 0.0, 0.0, 18.0, 6.0, 2.0, 6.0]),
        ),
    ]
}

#[test]
fn example_13_15_the_natural_direct_effect_is_identified() {
    let g = graph(&[("R", "E"), ("R", "J"), ("E", "J")], &[("R", "E")]);
    let objective = Objective::difference(
        q(&g, "P(j_{r, E_{r'}})"),
        q(&g, "P(j_{r'})"),
        &[("R", "1"), ("R'", "0"), ("J", "1")],
    );
    let r = neural_id(&g, &objective, &hiring(), &NcmConfig::default()).unwrap();
    assert!(r.identifiable);
    assert!((r.estimate.unwrap() + 0.25).abs() < 0.03); // Eq. 13.137
}

#[test]
fn example_13_16_the_probability_of_sufficiency_is_not_identified() {
    let g = graph(&[("X", "Y")], &[]);
    let uniform = || table(&["X", "Y"], &[1.0; 4]);
    let data = [
        Experiment::observational(uniform()),
        Experiment::interventional(&["X"], uniform()),
    ];
    let objective = Objective::new(
        q(&g, "P(y_x | x', y')"),
        &[("X", "1"), ("X'", "0"), ("Y", "1"), ("Y'", "0")],
    );
    let r = neural_id(&g, &objective, &data, &NcmConfig::default()).unwrap();
    assert!(!r.identifiable);
    // Eqs. 13.138–13.139: M̂_1 and M̂_2 attain 0 and 1.
    assert!(r.min < 0.15 && r.max > 0.85, "{} {}", r.min, r.max);
}

#[test]
fn algorithm_45_decides_symbolically_and_fits_only_when_identified() {
    let config = NcmConfig::default();
    let bow = graph(&[("D", "B")], &[("D", "B")]);
    let objective = Objective::new(q(&bow, "P(b_d)"), &[("D", "1"), ("B", "1")]);
    let none = symbolic_then_neural(
        &bow,
        &objective,
        &[Experiment::observational(diet())],
        &config,
    )
    .unwrap();
    assert!(none.is_none());
    let g = graph(&[("D", "S"), ("S", "B")], &[("D", "B")]);
    let data = table(
        &["D", "S", "B"],
        &[13.0, 15.0, 21.0, 63.0, 81.0, 27.0, 9.0, 27.0],
    );
    let (value, model) = symbolic_then_neural(
        &g,
        &objective,
        &[Experiment::observational(data.clone())],
        &config,
    )
    .unwrap()
    .unwrap();
    assert!((value - 0.46875).abs() < 0.02, "{value}");
    // The fitted model reproduces P(V) (Thm 13.1.4's L1 half).
    let ll = model
        .log_likelihood(&Experiment::observational(data), 8192, 3)
        .unwrap();
    let entropy = -[13.0, 15.0, 21.0, 63.0, 81.0, 27.0, 9.0, 27.0]
        .iter()
        .map(|w: &f64| w / 256.0 * (w / 256.0).ln())
        .sum::<f64>();
    assert!(-ll - entropy < 0.01, "KL {}", -ll - entropy);
}

#[test]
fn worlds_share_each_variables_noise() {
    // An untrained model is still an SCM: consistency and the joint across
    // worlds hold for any parameters.
    let g = graph(&[("Z", "X"), ("X", "Y")], &[("X", "Y")]);
    let mut levels = std::collections::BTreeMap::new();
    for v in ["X", "Y", "Z"] {
        levels.insert(v.to_string(), vec!["0".to_string(), "1".to_string()]);
    }
    let m = Ncm::new(&g, &levels, &NcmConfig::default()).unwrap();
    let v = [("X", "1"), ("Y", "1"), ("Y'", "0"), ("Z", "1")];
    let p = |s: &str| m.estimate(&q(&g, s), &v, 4096, 11).unwrap();
    // Consistency: with X = x observed, Y_x = Y.
    assert!((p("P(y_x | x)") - p("P(y | x)")).abs() < 1e-5);
    // Z is not a parent of Y, so do(x) and do(x, z) give Y the same value.
    assert!(p("P(y_x, y'_{x, z})") < 1e-6);
    assert!((p("P(y_x)") + p("P(y'_x)") - 1.0).abs() < 1e-5);
    // Algorithm 48 agrees with the exact average.
    let draws = m.sample(&q(&g, "P(y_x | z)"), &v, 20_000, 5).unwrap();
    let freq = draws.iter().filter(|d| d[0] == "1").count() as f64 / draws.len() as f64;
    assert!((freq - p("P(y_x | z)")).abs() < 0.02, "{freq}");
}

#[test]
fn the_implicit_model_learns_a_markovian_effect() {
    let g = graph(&[("X", "Y")], &[]);
    // P(X = 1) = 0.4, P(Y = 1 | X = 0) = 0.2, P(Y = 1 | X = 1) = 0.7.
    let data = table(&["X", "Y"], &[0.48, 0.12, 0.12, 0.28]);
    let config = NcmConfig {
        kind: Kind::Implicit,
        ..NcmConfig::default()
    };
    let model = fit(&g, &[Experiment::observational(data)], &config).unwrap();
    let p = model
        .estimate(&q(&g, "P(y_x)"), &[("X", "1"), ("Y", "1")], 20_000, 1)
        .unwrap();
    let px = model
        .estimate(&q(&g, "P(x)"), &[("X", "1")], 20_000, 1)
        .unwrap();
    assert!((p - 0.7).abs() < 0.05, "{p}");
    assert!((px - 0.4).abs() < 0.05, "{px}");
}

#[test]
fn algorithm_49_agrees_on_a_markovian_effect() {
    // Every effect is identifiable without confounding (Corol. 13.1.10).
    let g = graph(&[("X", "Y")], &[]);
    let data = table(&["X", "Y"], &[0.48, 0.12, 0.12, 0.28]);
    let config = NcmConfig {
        kind: Kind::Implicit,
        ..NcmConfig::default()
    };
    let objective = Objective::new(q(&g, "P(y_x)"), &[("X", "1"), ("Y", "1")]);
    let r = neural_id(&g, &objective, &[Experiment::observational(data)], &config).unwrap();
    assert!(r.identifiable, "{} {}", r.min, r.max);
    assert!((r.estimate.unwrap() - 0.7).abs() < 0.05);
    let draws = r.models[0]
        .sample(&q(&g, "P(y | x)"), &[("X", "1"), ("Y", "1")], 4000, 2)
        .unwrap();
    let freq = draws.iter().filter(|d| d[0] == "1").count() as f64 / 4000.0;
    assert!((freq - 0.7).abs() < 0.05, "{freq}");
}
