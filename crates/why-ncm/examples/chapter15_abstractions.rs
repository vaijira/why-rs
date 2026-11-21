//! Chapter 15: causal abstractions.
//!
//! Walks through the chapter, asserting each result:
//!
//! 1. Q-τ consistency (Ex. 15.5): a high-level model can match `M_L` on
//!    layer 1 and miss on layer 2. The abstract invariance condition decides
//!    whether a clustering can work at all (Exs. 15.6-15.7), and Algorithm 51
//!    builds `M_H` when it holds (Ex. 15.8).
//! 2. Abstract identification (Exs. 15.9-15.10): the same query is τ-ID or
//!    not depending on the clusters, decided by C-INFER on the C-DAG
//!    (Thm 15.3.2) and by Algorithm 52 with NCMs (Corol. 15.3.3).
//! 3. The nutrition study (Sec. 15.5.1): the effect of an unhealthy dish on
//!    being overweight, identified by the front door of the C-DAG and
//!    estimated from sampled low-level data.
//!
//! The book's companion Python has no chapter 15 code; the numbers are the
//! book's. Run with `cargo run -p why-ncm --example chapter15_abstractions`.

use why_data::abstraction::{
    Abstraction, HighLevel, Term, abstract_estimate, abstract_identify, abstract_invariance,
    construct, layer_consistency, low_level, q_consistency,
};
use why_data::fairness::Dataset;
use why_data::graph::cinfer::Input;
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::graph::dseparation::Admg;
use why_data::scm::{Dist, Model, expr};
use why_ncm::{Experiment, NcmConfig, Objective, neural_abstract_id};

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

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// `M_L` of Ex. 15.5.
fn drug() -> Model {
    let v = expr::var;
    Model::new(
        [
            ("R".to_string(), v("URY")),
            ("A".to_string(), expr::xor(v("R"), v("UA"))),
            (
                "B".to_string(),
                expr::xor(expr::and([v("R"), v("A")]), v("UB")),
            ),
            (
                "Y".to_string(),
                expr::xor(expr::and([v("A"), v("B"), v("URY")]), v("UY")),
            ),
        ],
        [
            ("URY".to_string(), Dist::Bernoulli(0.5)),
            ("UA".to_string(), Dist::Bernoulli(0.2)),
            ("UB".to_string(), Dist::Bernoulli(0.2)),
            ("UY".to_string(), Dist::Bernoulli(0.1)),
        ],
    )
    .unwrap()
}

fn drug_tau(with_r: bool) -> Abstraction {
    let both = |v: &[&str]| if v == ["1", "1"] { "1" } else { "0" }.to_string();
    let t = Abstraction::new()
        .cluster("X", &["A", "B"], both)
        .unwrap()
        .keep("Y")
        .unwrap();
    if with_r { t.keep("R").unwrap() } else { t }
}

fn drug_graph() -> Admg {
    graph(
        &[("R", "A"), ("R", "B"), ("A", "B"), ("A", "Y"), ("B", "Y")],
        &[("R", "Y")],
    )
}

/// `P(V)` of a structural model as a weighted data set.
fn observational(m: &Model, vars: &[&str]) -> Dataset {
    let t = m
        .world(m.observational())
        .probability_table(Some(vars))
        .unwrap();
    let rows: Vec<Vec<String>> = t
        .rows
        .iter()
        .map(|(r, _)| r.iter().map(|x| x.as_int().to_string()).collect())
        .collect();
    let w = t.rows.iter().map(|(_, p)| *p).collect();
    Dataset::from_records(vars.iter().copied(), rows)
        .unwrap()
        .with_weights(w)
        .unwrap()
}

fn part1() {
    println!("== 1. Q-τ consistency and Algorithm 51 (Exs. 15.5-15.8) ==");
    let v = expr::var;
    let tau = drug_tau(false);
    let mut low = drug();
    // M_H of Eq. 15.33 equates correlation with causation.
    let mut naive = Model::new(
        [
            ("X".to_string(), v("UX")),
            (
                "Y".to_string(),
                expr::or([
                    expr::and([expr::not(v("X")), v("UY0")]),
                    expr::and([v("X"), v("UY1")]),
                ]),
            ),
        ],
        [
            ("UX".to_string(), Dist::Bernoulli(0.34)),
            ("UY0".to_string(), Dist::Bernoulli(0.1)),
            ("UY1".to_string(), Dist::Bernoulli(0.852941)),
        ],
    )
    .unwrap();
    let l1 = layer_consistency(&mut low, &mut naive, &tau, 1, 1e-6).unwrap();
    let l2 = layer_consistency(&mut low, &mut naive, &tau, 2, 1e-6).unwrap();
    let effect = [Term::new(&[("A", "1"), ("B", "1")], &[("Y", "1")])];
    let (q, tq) = q_consistency(&mut low, &mut naive, &tau, &effect).unwrap();
    println!(
        "Eq. 15.33's M_H: L1-τ consistent {}, L2-τ consistent {}; P(Y_(A=1,B=1) = 1) = {q:.3} vs P(Y_(X=1) = 1) = {tq:.3}",
        l1.holds, l2.holds
    );
    assert!(l1.holds && !l2.holds && close(q, 0.5));

    // Exs. 15.6-15.7: total cholesterol breaks the AIC, the difference not.
    let mut chol = Model::new(
        [
            ("X".to_string(), v("UX")),
            ("HDL".to_string(), expr::xor(v("X"), v("UC1"))),
            ("LDL".to_string(), expr::xor(v("X"), v("UC2"))),
            (
                "Y".to_string(),
                expr::xor(expr::and([v("LDL"), expr::not(v("HDL"))]), v("UY")),
            ),
        ],
        [
            ("UX".to_string(), Dist::Bernoulli(0.5)),
            ("UC1".to_string(), Dist::Bernoulli(0.1)),
            ("UC2".to_string(), Dist::Bernoulli(0.1)),
            ("UY".to_string(), Dist::Bernoulli(0.1)),
        ],
    )
    .unwrap();
    for (name, f) in [
        (
            "TC = HDL + LDL",
            (|a: i32, b: i32| a + b) as fn(i32, i32) -> i32,
        ),
        ("Z = LDL - HDL", |a, b| b - a),
    ] {
        let t = Abstraction::new()
            .keep("X")
            .unwrap()
            .cluster("C", &["HDL", "LDL"], move |x| {
                f(x[0].parse().unwrap(), x[1].parse().unwrap()).to_string()
            })
            .unwrap()
            .keep("Y")
            .unwrap();
        match abstract_invariance(&mut chol, &t).unwrap() {
            Some(violation) => println!("{name}: the AIC fails ({violation})"),
            None => {
                let mut high = construct(chol.clone(), t.clone()).unwrap();
                let c = layer_consistency(&mut chol, &mut high, &t, 3, 1e-12).unwrap();
                println!(
                    "{name}: the AIC holds; Algorithm 51's M_H is L3-τ consistent: {}",
                    c.holds
                );
                assert!(c.holds);
            }
        }
    }

    // Ex. 15.8.
    let mut built = construct(drug(), tau.clone()).unwrap();
    let joint = built
        .joint(&tau, &[vec![("X".into(), "1".into())]])
        .unwrap();
    let p: f64 = joint
        .iter()
        .filter(|(k, _)| k[1] == "1")
        .map(|(_, p)| p)
        .sum();
    println!("Algorithm 51 on Ex. 15.5: P(Y_(X=1) = 1) = {p:.3} (Eq. 15.65)");
    assert!(close(p, 0.5));
}

fn part2() {
    println!("\n== 2. Abstract identification (Exs. 15.9-15.10) ==");
    let low_data = observational(&drug(), &["R", "A", "B", "Y"]);
    let g = drug_graph();
    for with_r in [false, true] {
        let tau = drug_tau(with_r);
        let gc = tau.cdag(&g).unwrap();
        let q = CtfQuery::parse(&gc, "P(y_x)").unwrap();
        let name = if with_r {
            "C = {AB, Y, R}"
        } else {
            "C = {AB, Y}"
        };
        let symbolic = abstract_identify(&g, &tau, &q, &[Input::observational()]);
        let objective = Objective::new(q.clone(), &[("X", "1"), ("Y", "1")]);
        let config = NcmConfig {
            epochs: if with_r { 3000 } else { 1000 },
            ..NcmConfig::default()
        };
        let neural = neural_abstract_id(
            &g,
            &tau,
            &objective,
            &[Experiment::observational(low_data.clone())],
            &config,
        )
        .unwrap();
        match symbolic {
            Ok(d) => {
                let p =
                    abstract_estimate(&g, &tau, &q, &[("X", "1"), ("Y", "1")], &low_data).unwrap();
                println!(
                    "{name}: τ-ID, P(Y_(X=1) = 1) = {} = {p:.3}; Algorithm 52: [{:.3}, {:.3}]",
                    d.estimand, neural.min, neural.max
                );
                assert!(close(p, 0.5) && neural.identifiable);
            }
            Err(_) => {
                println!(
                    "{name}: not τ-ID (C-INFER); Algorithm 52's models span [{:.3}, {:.3}], where M1 and M2 give 0.95 and 0.29",
                    neural.min, neural.max
                );
                assert!(!neural.identifiable);
            }
        }
    }
}

/// Draws from a uniform on [0, 1), SplitMix64.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    }
}

fn part3() {
    println!("\n== 3. The nutrition study (Sec. 15.5.1) ==");
    let v = expr::var;
    let int = expr::int;
    let uniform = |n: usize| Dist::Categorical(vec![1.0 / n as f64; n]);
    let unhealthy = || expr::ge(v("D"), int(4));
    let calories = || {
        expr::add(
            expr::add(expr::mul(int(4), v("C")), expr::mul(int(9), v("F"))),
            expr::mul(int(4), v("P")),
        )
    };
    // Four restaurants (R) shape both the dish and the patron's BMI. Dishes
    // 4-7 are unhealthy and richer in carbohydrates and fat; BMI reads the
    // calories only through whether they reach 58.
    let mut low = Model::new(
        [
            ("R".to_string(), v("UR")),
            (
                "D".to_string(),
                expr::add(
                    v("UD"),
                    expr::mul(
                        int(4),
                        expr::lt(v("UH"), expr::add(int(2), expr::mul(int(2), v("R")))),
                    ),
                ),
            ),
            (
                "C".to_string(),
                expr::add(expr::add(int(2), expr::mul(int(3), unhealthy())), v("UC")),
            ),
            (
                "F".to_string(),
                expr::add(expr::add(int(1), expr::mul(int(2), unhealthy())), v("UF")),
            ),
            ("P".to_string(), expr::add(int(2), v("UP"))),
            (
                "B".to_string(),
                expr::add(
                    expr::add(int(20), expr::mul(int(3), expr::ge(calories(), int(58)))),
                    expr::add(v("R"), v("UB")),
                ),
            ),
        ],
        [
            ("UR".to_string(), uniform(4)),
            ("UH".to_string(), uniform(10)),
            ("UD".to_string(), uniform(4)),
            ("UC".to_string(), uniform(4)),
            ("UF".to_string(), uniform(2)),
            ("UP".to_string(), uniform(5)),
            ("UB".to_string(), uniform(3)),
        ],
    )
    .unwrap();
    let g = graph(
        &[
            ("R", "D"),
            ("R", "B"),
            ("D", "C"),
            ("D", "F"),
            ("D", "P"),
            ("C", "B"),
            ("F", "B"),
            ("P", "B"),
        ],
        &[],
    );
    let tau = Abstraction::new()
        .cluster("DH", &["D"], |x| {
            u8::from(x[0].parse::<i64>().unwrap() >= 4).to_string()
        })
        .unwrap()
        .cluster("Z", &["C", "F", "P"], |x| {
            let n: Vec<i64> = x.iter().map(|s| s.parse().unwrap()).collect();
            u8::from(4 * n[0] + 9 * n[1] + 4 * n[2] >= 58).to_string()
        })
        .unwrap()
        .cluster("BH", &["B"], |x| {
            u8::from(x[0].parse::<i64>().unwrap() >= 25).to_string()
        })
        .unwrap();
    let gc = tau.cdag(&g).unwrap();

    // The truth, P(BH_(D=4) = 1), from the low-level model.
    let truth = low_level(&mut low, &tau, &[Term::new(&[("D", "4")], &[("BH", "1")])]).unwrap();
    // 5000 sampled patrons.
    let mut rng = Rng(15);
    let vars = ["R", "D", "C", "F", "P", "B"];
    let rows: Vec<Vec<String>> = low
        .world(low.observational())
        .sample(5000, || rng.unit())
        .unwrap()
        .into_iter()
        .map(|r| r.iter().map(|x| x.as_int().to_string()).collect())
        .collect();
    let all = low.world(low.observational()).world().variables().to_vec();
    let at: Vec<usize> = vars
        .iter()
        .map(|v| all.iter().position(|a| a == v).unwrap())
        .collect();
    let rows: Vec<Vec<String>> = rows
        .into_iter()
        .map(|r| at.iter().map(|&i| r[i].clone()).collect())
        .collect();
    // The restaurant is not recorded in the study's data.
    let data = Dataset::from_records(vars, rows).unwrap();

    let q = CtfQuery::parse(&gc, "P(bh_{dh})").unwrap();
    let values = [("DH", "1"), ("BH", "1")];
    let front_door = abstract_estimate(&g, &tau, &q, &values, &data).unwrap();
    let high = tau.transform(&data).unwrap();
    let count = |f: &dyn Fn(usize) -> bool| (0..high.len()).filter(|&r| f(r)).count() as f64;
    let naive = count(&|r| high.value("DH", r) == Some("1") && high.value("BH", r) == Some("1"))
        / count(&|r| high.value("DH", r) == Some("1"));
    let objective = Objective::new(q, &values);
    let neural = neural_abstract_id(
        &g,
        &tau,
        &objective,
        &[Experiment::observational(data)],
        &NcmConfig {
            epochs: 3000,
            ..NcmConfig::default()
        },
    )
    .unwrap();
    println!("C-DAG over DH, Z, BH: the front door (R is projected into DH <-> BH)");
    println!(
        "P(BH_(DH=1) = 1): truth {truth:.3}; front door on τ(data) {front_door:.3}; Algorithm 52 [{:.3}, {:.3}] (τ-ID: {}); naive P(BH = 1 | DH = 1) = {naive:.3}",
        neural.min, neural.max, neural.identifiable
    );
    assert!((front_door - truth).abs() < 0.03);
    assert!(neural.identifiable && (neural.estimate.unwrap() - truth).abs() < 0.05);
    assert!((naive - truth).abs() > 0.05);
}

fn main() {
    part1();
    part2();
    part3();
}
