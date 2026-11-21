use super::*;
use crate::graph::cinfer::counterfactual::CtfQuery;
use crate::graph::dseparation::Admg;

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

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

fn p(m: &StepNcm, g: &Admg, q: &str, values: &[(&str, &str)]) -> f64 {
    m.probability(&CtfQuery::parse(g, q).unwrap(), values)
        .unwrap()
}

fn bow() -> Admg {
    graph(&[("D", "B")], &[("D", "B")])
}

/// Table 13.3, the observational distribution every model of Exs. 13.5–13.6
/// must induce.
fn table_13_3(m: &StepNcm) {
    let g = bow();
    for (d, b, want) in [
        ("0", "0", 34.0),
        ("0", "1", 78.0),
        ("1", "0", 90.0),
        ("1", "1", 54.0),
    ] {
        close(p(m, &g, "P(d, b)", &[("D", d), ("B", b)]), want / 256.0);
    }
}

#[test]
fn c2_components_are_maximal_bidirected_cliques() {
    // A triangle plus a pendant edge: {A, B, C} and {C, D}.
    let g = graph(
        &[("A", "D")],
        &[("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")],
    );
    assert_eq!(c2_components(&g), vec![vec!["A", "B", "C"], vec!["C", "D"]]);
    let s = NcmStructure::from_graph(&g).unwrap();
    assert_eq!(s.noises_of("C"), [0, 1]);
    assert_eq!(s.noises_of("D"), [1]);
    assert!(!s.is_markovian());
    let markov = NcmStructure::from_graph(&graph(&[("X", "Y")], &[])).unwrap();
    assert!(markov.is_markovian());
    assert_eq!(markov.confounders, vec![vec!["X"], vec!["Y"]]);
}

#[test]
fn expansion_keeps_only_what_the_query_needs() {
    let g = graph(&[("X", "W"), ("W", "Y"), ("X", "Y"), ("Z", "X")], &[]);
    let s = NcmStructure::from_graph(&g).unwrap();
    // The counterfactual direct effect: Y_{x, W_{x'}} given X = x.
    let q = CtfQuery::parse(&g, "P(y_{x, W_{x'}} | x)").unwrap();
    let e = expand(&q, &[("X", "1"), ("X'", "0"), ("Y", "1")]).unwrap();
    assert_eq!(e.worlds.len(), 3);
    let rel = e.relevant(&s);
    // World 0 (do(X = 0)) needs W only through the copy; Z is cut off.
    assert_eq!(rel[0], ["X", "W"]);
    // World 1 (do(X = 1), W copied) needs Y and its parents.
    assert_eq!(rel[1], ["X", "W", "Y"]);
    // World 2 is the natural one, for the condition X = 1.
    assert_eq!(rel[2], ["Z", "X"]);
    assert!(matches!(expand(&q, &[("X", "1")]), Err(NcmError::MissingValue(v)) if v == "X'"));
}

#[test]
fn example_13_5_one_noise_matches_all_three_layers() {
    let u = || noise("U");
    let interval = |a: f64, b: f64| and([at_least(u(), a / 256.0), not(at_least(u(), b / 256.0))]);
    let cuts = [0.0, 16.0, 34.0, 64.0, 112.0, 166.0, 256.0];
    let i: Vec<Unit> = cuts.windows(2).map(|w| interval(w[0], w[1])).collect();
    // Eqs. 13.40–13.41: the one-hot code of Û mapped to R_D and R_B.
    let r_d = linear([(i[4].clone(), 1.0), (i[5].clone(), 1.0)], 0.0);
    let r_b = linear(
        [
            (i[1].clone(), 1.0),
            (i[2].clone(), 2.0),
            (i[3].clone(), 3.0),
            (i[4].clone(), 1.0),
            (i[5].clone(), 2.0),
        ],
        0.0,
    );
    // Eq. 13.43 for each value of R_B, then Eq. 13.44.
    let is = |k: f64| {
        and([
            at_least(r_b.clone(), k),
            not(at_least(r_b.clone(), k + 1.0)),
        ])
    };
    let d = || parent("D");
    let f_b = or([
        and([is(1.0), d()]),
        and([is(2.0), not(d())]),
        at_least(r_b.clone(), 3.0),
    ]);
    let m = StepNcm::new()
        .noise("U", &["D", "B"])
        .mechanism("D", r_d)
        .mechanism("B", f_b);
    table_13_3(&m);
    let g = bow();
    let v = [("D", "1"), ("D'", "0"), ("B", "1")];
    close(p(&m, &g, "P(b | d)", &v), 0.375); // Eq. 13.29
    close(p(&m, &g, "P(b_d)", &v), 0.46875); // Eq. 13.31
    close(p(&m, &g, "P(b_d | d', b)", &v), 48.0 / 78.0); // Eq. 13.33
    assert!(m.is_g_constrained(&g).unwrap());
}

/// The four NCMs of Ex. 13.6 (Eqs. 13.46–13.57).
fn example_13_6() -> [StepNcm; 4] {
    let (ud, ub, d, b) = (
        || noise("Ud"),
        || noise("Ub"),
        || parent("D"),
        || parent("B"),
    );
    let markov = |f_b: Unit| {
        StepNcm::new()
            .noise("Ud", &["D"])
            .noise("Ub", &["B"])
            .mechanism("D", at_least(ud(), 112.0 / 256.0))
            .mechanism("B", f_b)
    };
    let confounded = |f_b: Unit| {
        StepNcm::new()
            .noise("Ud", &["D", "B"])
            .noise("Ub", &["B"])
            .mechanism("D", at_least(ud(), 112.0 / 256.0))
            .mechanism("B", f_b)
    };
    let m1 = markov(or([
        and([not(d()), at_least(ub(), 34.0 / 112.0)]),
        and([d(), at_least(ub(), 90.0 / 144.0)]),
    ]));
    let m2 = StepNcm::new()
        .noise("Ud", &["D"])
        .noise("Ub", &["B"])
        .mechanism(
            "D",
            or([
                and([not(b()), at_least(ud(), 34.0 / 124.0)]),
                and([b(), at_least(ud(), 78.0 / 132.0)]),
            ]),
        )
        .mechanism("B", at_least(ub(), 124.0 / 256.0));
    let m3 = confounded(or([
        and([d(), below(ub(), 34.0 / 112.0), below(ud(), 112.0 / 256.0)]),
        and([at_least(ub(), 34.0 / 112.0), below(ud(), 112.0 / 256.0)]),
        and([
            d(),
            below(ub(), 54.0 / 144.0),
            at_least(ud(), 112.0 / 256.0),
        ]),
    ]));
    let m4 = confounded(or([
        and([
            not(d()),
            at_least(ub(), 34.0 / 112.0),
            below(ud(), 112.0 / 256.0),
        ]),
        and([
            d(),
            below(ub(), 54.0 / 144.0),
            at_least(ud(), 112.0 / 256.0),
        ]),
    ]));
    [m1, m2, m3, m4]
}

#[test]
fn example_13_6_four_fits_of_p_v_disagree_on_the_effect() {
    let g = bow();
    let v = [("D", "1"), ("B", "1")];
    let want = [54.0 / 144.0, 132.0 / 256.0, 166.0 / 256.0, 54.0 / 256.0]; // Eqs. 13.58–13.61
    for (m, want) in example_13_6().iter().zip(want) {
        table_13_3(m);
        let mg = m.graph().unwrap();
        close(p(m, &mg, "P(b_d)", &v), want);
    }
    // Ex. 13.9: M̂_2 reverses the arrow, so only it is ruled out by G.
    let constrained: Vec<bool> = example_13_6()
        .iter()
        .map(|m| m.is_g_constrained(&g).unwrap())
        .collect();
    assert_eq!(constrained, [true, false, true, true]);
}

#[test]
fn example_13_10_any_fit_of_the_front_door_graph_gets_the_effect() {
    let (udb, us, d, s) = (
        || noise("Udb"),
        || noise("Us"),
        || parent("D"),
        || parent("S"),
    );
    let m = StepNcm::new()
        .noise("Udb", &["D", "B"])
        .noise("Us", &["S"])
        .mechanism("D", at_least(udb(), 112.0 / 256.0))
        .mechanism(
            "S",
            or([
                and([not(d()), at_least(us(), 0.25)]),
                and([d(), at_least(us(), 0.75)]),
            ]),
        )
        .mechanism(
            "B",
            or([
                and([
                    not(s()),
                    at_least(udb(), 52.0 / 256.0),
                    below(udb(), 112.0 / 256.0),
                ]),
                and([not(s()), at_least(udb(), 220.0 / 256.0)]),
                and([
                    s(),
                    at_least(udb(), 28.0 / 256.0),
                    below(udb(), 112.0 / 256.0),
                ]),
                and([s(), at_least(udb(), 148.0 / 256.0)]),
            ]),
        );
    let g = graph(&[("D", "S"), ("S", "B")], &[("D", "B")]);
    assert!(m.is_g_constrained(&g).unwrap());
    // Table 13.14(a).
    let table = [13.0, 15.0, 21.0, 63.0, 81.0, 27.0, 9.0, 27.0];
    for (k, want) in table.iter().enumerate() {
        let bit = |i: usize| if k >> (2 - i) & 1 == 1 { "1" } else { "0" };
        let v = [("D", bit(0)), ("S", bit(1)), ("B", bit(2))];
        close(p(&m, &g, "P(d, s, b)", &v), want / 256.0);
    }
    close(p(&m, &g, "P(b_d)", &[("D", "1"), ("B", "1")]), 0.46875); // Eq. 13.79
}

/// The G-NCM of Ex. 13.15, fitted to Table 13.5.
fn example_13_15() -> StepNcm {
    let (ure, uj, r, e) = (
        || noise("Ure"),
        || noise("Uj"),
        || parent("R"),
        || parent("E"),
    );
    StepNcm::new()
        .noise("Ure", &["R", "E"])
        .noise("Uj", &["J"])
        .mechanism("R", at_least(ure(), 0.5))
        .mechanism(
            "E",
            or([
                and([below(ure(), 0.375), not(r())]),
                and([below(ure(), 0.125), r()]),
                at_least(ure(), 0.875),
            ]),
        )
        .mechanism(
            "J",
            or([
                and([not(r()), below(uj(), 0.75)]),
                and([r(), not(e()), below(uj(), 0.25)]),
                and([r(), e(), below(uj(), 0.75)]),
            ]),
        )
}

#[test]
fn example_13_15_the_natural_direct_effect_through_a_fitted_ncm() {
    let m = example_13_15();
    let g = graph(&[("R", "E"), ("R", "J"), ("E", "J")], &[("R", "E")]);
    assert!(m.is_g_constrained(&g).unwrap());
    // Table 13.5: the three input distributions.
    let obs = [1.0, 3.0, 3.0, 9.0, 9.0, 3.0, 1.0, 3.0];
    let bits = |k: usize, i: usize| if k >> (2 - i) & 1 == 1 { "1" } else { "0" };
    for (k, want) in obs.iter().enumerate() {
        let v = [("R", bits(k, 0)), ("E", bits(k, 1)), ("J", bits(k, 2))];
        close(p(&m, &g, "P(r, e, j)", &v), want / 32.0);
    }
    let exp = [[4.0, 12.0, 4.0, 12.0], [18.0, 6.0, 2.0, 6.0]];
    for (r, row) in exp.iter().enumerate() {
        for (k, want) in row.iter().enumerate() {
            let v = [
                ("R", if r == 1 { "1" } else { "0" }),
                ("E", bits(k, 1)),
                ("J", bits(k, 2)),
            ];
            close(p(&m, &g, "P(e_r, j_r)", &v), want / 32.0);
        }
    }
    let v = [("R", "1"), ("R'", "0"), ("J", "1")];
    let nested = p(&m, &g, "P(j_{r, E_{r'}})", &v);
    close(nested, 0.5);
    close(nested - p(&m, &g, "P(j_{r'})", &v), -0.25); // Eq. 13.137
}

#[test]
fn example_13_16_two_fits_of_l1_and_l2_give_ps_zero_and_one() {
    let (ux, uy, x) = (|| noise("Ux"), || noise("Uy"), || parent("X"));
    let base = || {
        StepNcm::new()
            .noise("Ux", &["X"])
            .noise("Uy", &["Y"])
            .mechanism("X", below(ux(), 0.5))
    };
    let m1 = base().mechanism("Y", below(uy(), 0.5));
    let m2 = base().mechanism(
        "Y",
        or([
            and([x(), below(uy(), 0.5)]),
            and([not(x()), at_least(uy(), 0.5)]),
        ]),
    );
    let g = graph(&[("X", "Y")], &[]);
    let v = [("X", "1"), ("X'", "0"), ("Y", "1"), ("Y'", "0")];
    for m in [&m1, &m2] {
        // Table 13.27(b): both match P(X, Y) and P(Y_x).
        for (xv, yv) in [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")] {
            let w = [("X", xv), ("Y", yv)];
            close(p(m, &g, "P(x, y)", &w), 0.25);
            close(p(m, &g, "P(y_x)", &w), 0.5);
        }
    }
    close(p(&m1, &g, "P(y_x | x', y')", &v), 0.0); // Eq. 13.138
    close(p(&m2, &g, "P(y_x | x', y')", &v), 1.0); // Eq. 13.139
}

#[test]
fn algorithm_48_samples_the_conditional_counterfactual() {
    let m = example_13_15();
    let g = m.graph().unwrap();
    let q = CtfQuery::parse(&g, "P(j_{r'} | r, e')").unwrap();
    let v = [("R", "1"), ("R'", "0"), ("J", "1"), ("E'", "0")];
    let exact = m.probability(&q, &v).unwrap();
    let est = m.estimate(&q, &v, 40_000, 7).unwrap();
    assert!((est - exact).abs() < 0.01, "{est} vs {exact}");
    let draws = m.sample(&q, &v, 5, 1).unwrap();
    assert!(draws.iter().all(|d| d.len() == 1));
    let impossible = CtfQuery::parse(&g, "P(j | r, r')").unwrap();
    assert_eq!(
        m.probability(&impossible, &v),
        Err(NcmError::ImpossibleCondition)
    );
}

#[test]
fn ill_formed_models_are_rejected() {
    let g = graph(&[("X", "Y")], &[]);
    let q = CtfQuery::parse(&g, "P(y)").unwrap();
    let v = [("Y", "1")];
    let wrong_scope = StepNcm::new()
        .noise("U", &["X"])
        .mechanism("X", at_least(noise("U"), 0.5))
        .mechanism("Y", at_least(noise("U"), 0.5));
    assert!(matches!(
        wrong_scope.probability(&q, &v),
        Err(NcmError::InvalidInput(_))
    ));
    let mixed = StepNcm::new()
        .noise("U", &["X", "Y"])
        .mechanism("X", at_least(noise("U"), 0.5))
        .mechanism(
            "Y",
            linear([(noise("U"), 1.0), (Input::from(parent("X")), 1.0)], 0.0),
        );
    assert_eq!(
        mixed.probability(&q, &v),
        Err(NcmError::NotAxisAligned("Y".into()))
    );
    let cyclic = StepNcm::new()
        .mechanism("X", and([parent("Y")]))
        .mechanism("Y", and([parent("X")]));
    assert!(matches!(
        cyclic.probability(&q, &v),
        Err(NcmError::Cyclic(_))
    ));
}
