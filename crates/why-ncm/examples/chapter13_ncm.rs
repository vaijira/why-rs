//! Chapter 13: causal generative models (neural causal models).
//!
//! Walks through the chapter's worked examples, asserting each result:
//!
//! 1. NCMs built by hand from step units (Sec. 13.1.2.5), evaluated
//!    exactly. They show that fitting `P(V)` does not pin down `P(b_d)`
//!    (Ex. 13.6, Corol. 13.1.5), and that every G-NCM fitting `P(V)` agrees
//!    once the effect is identifiable (Ex. 13.10). The same holds one layer
//!    up (Exs. 13.15, 13.16).
//! 2. Neural identification by training two G-NCMs to push the query up
//!    and down while fitting the data: Algorithm 44 on the bow and
//!    front-door graphs, and Algorithm 46 (NeuralID) for the natural direct
//!    effect and the probability of sufficiency.
//! 3. Algorithm 45: identify symbolically with C-INFER, then estimate with
//!    a fitted G-NCM.
//! 4. The implicit NCM (Algorithms 48–49) on a Markovian model.
//!
//! The book's companion Python has no chapter 13 code; the numbers are the
//! book's.
//!
//! Run with `cargo run -p why-ncm --example chapter13_ncm`. Training takes
//! a minute or two.

use why_data::fairness::Dataset;
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::graph::dseparation::Admg;
use why_data::ncm::{StepNcm, Unit, and, at_least, below, noise, not, or, parent};
use why_ncm::{Experiment, Kind, NcmConfig, Objective, fit, neural_id, symbolic_then_neural};

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

fn exact(m: &StepNcm, g: &Admg, s: &str, values: &[(&str, &str)]) -> f64 {
    m.probability(&q(g, s), values).unwrap()
}

fn hand_built() {
    println!("1. NCMs built from step units");
    let bow = graph(&[("D", "B")], &[("D", "B")]);
    let (ud, ub, d) = (|| noise("Ud"), || noise("Ub"), || parent("D"));
    let model = |confounded: bool, f_b: Unit| {
        let scope: &[&str] = if confounded { &["D", "B"] } else { &["D"] };
        StepNcm::new()
            .noise("Ud", scope)
            .noise("Ub", &["B"])
            .mechanism("D", at_least(ud(), 112.0 / 256.0))
            .mechanism("B", f_b)
    };
    // Ex. 13.6: three G-constrained NCMs of the diet study, Eqs. 13.46,
    // 13.52 and 13.55. All induce Table 13.3.
    let fits = [
        (
            "M̂1 (no confounding)",
            model(
                false,
                or([
                    and([not(d()), at_least(ub(), 34.0 / 112.0)]),
                    and([d(), at_least(ub(), 90.0 / 144.0)]),
                ]),
            ),
            54.0 / 144.0,
        ),
        (
            "M̂3 (maximizes the effect)",
            model(
                true,
                or([
                    and([d(), below(ub(), 34.0 / 112.0), below(ud(), 112.0 / 256.0)]),
                    and([at_least(ub(), 34.0 / 112.0), below(ud(), 112.0 / 256.0)]),
                    and([
                        d(),
                        below(ub(), 54.0 / 144.0),
                        at_least(ud(), 112.0 / 256.0),
                    ]),
                ]),
            ),
            166.0 / 256.0,
        ),
        (
            "M̂4 (minimizes the effect)",
            model(
                true,
                or([
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
                ]),
            ),
            54.0 / 256.0,
        ),
    ];
    for (name, m, want) in &fits {
        let fit = exact(m, &bow, "P(d, b)", &[("D", "1"), ("B", "1")]);
        let effect = exact(m, &bow, "P(b_d)", &[("D", "1"), ("B", "1")]);
        println!("  Ex. 13.6 {name}: P(d, b) = {fit:.4}, P(b_d) = {effect:.4}");
        assert!((fit - 54.0 / 256.0).abs() < 1e-12);
        assert!((effect - want).abs() < 1e-12);
        assert!(m.is_g_constrained(&bow).unwrap());
    }

    // Ex. 13.10: sodium S mediates, and the effect becomes identifiable;
    // the NCM of Eq. 13.80 gets it right without knowing M*.
    let (udb, us, s) = (|| noise("Udb"), || noise("Us"), || parent("S"));
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
    let front = graph(&[("D", "S"), ("S", "B")], &[("D", "B")]);
    let effect = exact(&m, &front, "P(b_d)", &[("D", "1"), ("B", "1")]);
    println!("  Ex. 13.10: P(b_d) = {effect:.5} (Eq. 13.79: 15/32)");
    assert!((effect - 0.46875).abs() < 1e-12);

    // Ex. 13.16: two fits of P(V) and P(V_x) with opposite probabilities of
    // sufficiency.
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
    let xy = graph(&[("X", "Y")], &[]);
    let v = [("X", "1"), ("X'", "0"), ("Y", "1"), ("Y'", "0")];
    let (ps1, ps2) = (
        exact(&m1, &xy, "P(y_x | x', y')", &v),
        exact(&m2, &xy, "P(y_x | x', y')", &v),
    );
    println!("  Ex. 13.16: PS = {ps1} under M̂1, {ps2} under M̂2");
    assert_eq!((ps1, ps2), (0.0, 1.0));

    // Algorithm 48: rejection sampling agrees with the exact value.
    let est = m2
        .estimate(&q(&xy, "P(y_x | x', y')"), &v, 20_000, 1)
        .unwrap();
    println!("  Algorithm 48 on M̂2: {est:.3}");
    assert!((est - 1.0).abs() < 1e-12);
}

fn neural() {
    println!("2. Neural identification (Algorithms 44 and 46)");
    let config = NcmConfig::default();
    let bow = graph(&[("D", "B")], &[("D", "B")]);
    let diet = Experiment::observational(table(&["D", "B"], &[34.0, 78.0, 90.0, 54.0]));
    let effect = Objective::new(q(&bow, "P(b_d)"), &[("D", "1"), ("B", "1")]);
    let r = neural_id(&bow, &effect, std::slice::from_ref(&diet), &config).unwrap();
    println!(
        "  Ex. 13.9 (bow): min {:.3}, max {:.3}: {}",
        r.min,
        r.max,
        if r.identifiable { "ID" } else { "FAIL" }
    );
    println!("    (any fit lies in the natural bounds [0.211, 0.648])");
    assert!(!r.identifiable);

    let front = graph(&[("D", "S"), ("S", "B")], &[("D", "B")]);
    let sodium = Experiment::observational(table(
        &["D", "S", "B"],
        &[13.0, 15.0, 21.0, 63.0, 81.0, 27.0, 9.0, 27.0],
    ));
    let r = neural_id(&front, &effect, std::slice::from_ref(&sodium), &config).unwrap();
    println!(
        "  Ex. 13.10 (front door): min {:.3}, max {:.3}, estimate {:.3} (truth 0.469)",
        r.min,
        r.max,
        r.estimate.unwrap()
    );
    assert!((r.estimate.unwrap() - 0.46875).abs() < 0.03);

    // Ex. 13.15: the natural direct effect of race on hiring, from Table 13.5.
    let g = graph(&[("R", "E"), ("R", "J"), ("E", "J")], &[("R", "E")]);
    let h = ["R", "E", "J"];
    let data = [
        Experiment::observational(table(&h, &[1.0, 3.0, 3.0, 9.0, 9.0, 3.0, 1.0, 3.0])),
        Experiment::interventional(
            &["R"],
            table(&h, &[4.0, 12.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        Experiment::interventional(
            &["R"],
            table(&h, &[0.0, 0.0, 0.0, 0.0, 18.0, 6.0, 2.0, 6.0]),
        ),
    ];
    let nde = Objective::difference(
        q(&g, "P(j_{r, E_{r'}})"),
        q(&g, "P(j_{r'})"),
        &[("R", "1"), ("R'", "0"), ("J", "1")],
    );
    let r = neural_id(&g, &nde, &data, &config).unwrap();
    println!(
        "  Ex. 13.15: NDE min {:.3}, max {:.3}, estimate {:.3} (Eq. 13.137: -0.25)",
        r.min,
        r.max,
        r.estimate.unwrap()
    );
    assert!((r.estimate.unwrap() + 0.25).abs() < 0.03);

    // Ex. 13.16: the probability of sufficiency from P(V) and P(V_x).
    let xy = graph(&[("X", "Y")], &[]);
    let uniform = || table(&["X", "Y"], &[1.0; 4]);
    let data = [
        Experiment::observational(uniform()),
        Experiment::interventional(&["X"], uniform()),
    ];
    let ps = Objective::new(
        q(&xy, "P(y_x | x', y')"),
        &[("X", "1"), ("X'", "0"), ("Y", "1"), ("Y'", "0")],
    );
    let r = neural_id(&xy, &ps, &data, &config).unwrap();
    println!("  Ex. 13.16: PS min {:.3}, max {:.3}: FAIL", r.min, r.max);
    assert!(!r.identifiable);

    println!("3. Symbolic identification, neural estimation (Algorithm 45)");
    assert!(
        symbolic_then_neural(&bow, &effect, &[diet], &config)
            .unwrap()
            .is_none()
    );
    println!("  bow: FAIL before any training");
    let (value, _) = symbolic_then_neural(&front, &effect, &[sodium], &config)
        .unwrap()
        .unwrap();
    println!("  front door: {value:.3}");
    assert!((value - 0.46875).abs() < 0.03);
}

fn implicit() {
    println!("4. The implicit NCM (Algorithms 48 and 49)");
    let g = graph(&[("X", "Y")], &[]);
    let data = Experiment::observational(table(&["X", "Y"], &[0.48, 0.12, 0.12, 0.28]));
    let config = NcmConfig {
        kind: Kind::Implicit,
        ..NcmConfig::default()
    };
    let model = fit(&g, &[data], &config).unwrap();
    let p = model
        .estimate(&q(&g, "P(y_x)"), &[("X", "1"), ("Y", "1")], 20_000, 1)
        .unwrap();
    println!("  P(y_x) = {p:.3} (truth 0.7, = P(y | x) without confounding)");
    assert!((p - 0.7).abs() < 0.05);
}

fn main() {
    hand_built();
    neural();
    implicit();
}
