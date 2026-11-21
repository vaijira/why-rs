//! Tests for chapter 15: the book's examples with exact numbers.

use super::*;
use crate::fairness::Dataset;
use crate::graph::cinfer::Input;
use crate::graph::cinfer::counterfactual::{CtfError, CtfQuery};
use crate::graph::dseparation::Admg;
use crate::scm::{Dist, Model, expr};

type E = rssn::symbolic::core::Expr;

fn eq(name: &str, e: E) -> (String, E) {
    (name.to_string(), e)
}

fn bern(name: &str, p: f64) -> (String, Dist) {
    (name.to_string(), Dist::Bernoulli(p))
}

fn v(name: &str) -> E {
    expr::var(name)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// `M_L` of Ex. 15.5 (Eqs. 15.22-15.25).
fn drug() -> Model {
    Model::new(
        [
            eq("R", v("URY")),
            eq("A", expr::xor(v("R"), v("UA"))),
            eq("B", expr::xor(expr::and([v("R"), v("A")]), v("UB"))),
            eq(
                "Y",
                expr::xor(expr::and([v("A"), v("B"), v("URY")]), v("UY")),
            ),
        ],
        [
            bern("URY", 0.5),
            bern("UA", 0.2),
            bern("UB", 0.2),
            bern("UY", 0.1),
        ],
    )
    .unwrap()
}

/// Both doses make a completed treatment (Eq. 15.28).
fn both(v: &[&str]) -> String {
    if v == ["1", "1"] { "1" } else { "0" }.to_string()
}

fn drug_tau(with_r: bool) -> Abstraction {
    let t = Abstraction::new()
        .cluster("X", &["A", "B"], both)
        .unwrap()
        .keep("Y")
        .unwrap();
    if with_r { t.keep("R").unwrap() } else { t }
}

/// The diagram of Ex. 15.5's `M_L`: `R` and `Y` share `U_RY`.
fn drug_graph() -> Admg {
    let mut g = Admg::new();
    for (a, b) in [("R", "A"), ("R", "B"), ("A", "B"), ("A", "Y"), ("B", "Y")] {
        g.directed(a, b);
    }
    g.bidirected("R", "Y");
    g
}

/// The data set `P(V)` of a structural model, weighted by probability.
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

/// The edges of `g` as `A->B` and `A<->B` (endpoints sorted).
fn edges(g: &Admg) -> Vec<String> {
    use crate::graph::dseparation::EdgeKind;
    let graph = g.graph();
    let mut out: Vec<String> = graph
        .edge_indices()
        .map(|e| {
            let (a, b) = graph.edge_endpoints(e).unwrap();
            let (a, b) = (graph[a].clone(), graph[b].clone());
            match graph[e] {
                EdgeKind::Directed => format!("{a}->{b}"),
                EdgeKind::Bidirected => {
                    let (x, y) = if a < b { (a, b) } else { (b, a) };
                    format!("{x}<->{y}")
                }
            }
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

fn probability(d: &Dataset, row: &[(&str, &str)]) -> f64 {
    (0..d.len())
        .filter(|&r| row.iter().all(|(n, x)| d.value(n, r) == Some(*x)))
        .map(|r| d.weights().unwrap()[r])
        .sum()
}

#[test]
fn example_15_1_the_total_vote() {
    let low = Model::new(
        [
            eq("X1", v("U1")),
            eq("X2", v("U2")),
            eq("X3", v("U3")),
            eq(
                "Y",
                expr::ge(
                    expr::add(expr::add(v("X1"), v("X2")), v("X3")),
                    expr::int(2),
                ),
            ),
        ],
        [bern("U1", 0.5), bern("U2", 0.5), bern("U3", 0.5)],
    )
    .unwrap();
    // M_H of Eq. 15.3.
    let mut high = Model::new(
        [
            eq("XT", expr::add(expr::add(v("U1"), v("U2")), v("U3"))),
            eq("Y", expr::ge(v("XT"), expr::int(2))),
        ],
        [bern("U1", 0.5), bern("U2", 0.5), bern("U3", 0.5)],
    )
    .unwrap();
    let tau = Abstraction::new()
        .cluster("XT", &["X1", "X2", "X3"], |x| {
            x.iter()
                .map(|b| b.parse::<u32>().unwrap())
                .sum::<u32>()
                .to_string()
        })
        .unwrap()
        .keep("Y")
        .unwrap();
    let mut low2 = low.clone();
    for layer in 1..=3 {
        let c = layer_consistency(&mut low2, &mut high, &tau, layer, 1e-12).unwrap();
        assert!(c.holds, "layer {layer}: {:?}", c.worst);
    }
    // Two forced votes for Alice, or a forced total of two, elect her.
    let q = [Term::new(
        &[("X1", "1"), ("X2", "1"), ("X3", "0")],
        &[("Y", "1")],
    )];
    let (l, h) = q_consistency(&mut low2, &mut high, &tau, &q).unwrap();
    assert!(close(l, 1.0) && close(h, 1.0));
    // Alg. 51 recovers an equivalent M_H.
    let mut built = construct(low, tau.clone()).unwrap();
    assert!(
        layer_consistency(&mut low2, &mut built, &tau, 3, 1e-12)
            .unwrap()
            .holds
    );
}

#[test]
fn example_15_5_matches_layer_1_but_not_layer_2() {
    let mut low = drug();
    // Eqs. 15.26 and 15.27.
    let tau = drug_tau(false);
    let p11 = low_level(&mut low, &tau, &[Term::new(&[], &[("X", "1"), ("Y", "1")])]).unwrap();
    let p1 = low_level(&mut low, &tau, &[Term::new(&[], &[("X", "1")])]).unwrap();
    assert!((p11 / p1 - 0.853).abs() < 5e-4);
    let effect = [Term::new(&[("A", "1"), ("B", "1")], &[("Y", "1")])];
    assert!(close(low_level(&mut low, &tau, &effect).unwrap(), 0.5));

    // M_H of Eqs. 15.31-15.34.
    let mut high = Model::new(
        [
            eq("X", v("UX")),
            eq(
                "Y",
                expr::or([
                    expr::and([expr::not(v("X")), v("UY0")]),
                    expr::and([v("X"), v("UY1")]),
                ]),
            ),
        ],
        [bern("UX", 0.34), bern("UY0", 0.1), bern("UY1", 0.852941)],
    )
    .unwrap();
    let l1 = layer_consistency(&mut low, &mut high, &tau, 1, 1e-6).unwrap();
    assert!(l1.holds, "{:?}", l1.worst);
    let l2 = layer_consistency(&mut low, &mut high, &tau, 2, 1e-6).unwrap();
    assert!(!l2.holds);
    // Q = P(Y = 1, A = 1, B = 1) agrees; Q' = P(Y_{A=1,B=1} = 1) does not.
    let q = [Term::new(&[], &[("X", "1"), ("Y", "1")])];
    let (l, h) = q_consistency(&mut low, &mut high, &tau, &q).unwrap();
    assert!((l - h).abs() < 1e-6 && (l - 0.29).abs() < 1e-12);
    let (l, h) = q_consistency(&mut low, &mut high, &tau, &effect).unwrap();
    assert!(close(l, 0.5) && (h - 0.852941).abs() < 1e-9);
    // A low-level intervention on part of a cluster has no counterpart.
    assert_eq!(
        low_level(&mut low, &tau, &[Term::new(&[("A", "1")], &[("Y", "1")])]),
        Err(AbstractionError::Partial("X".into()))
    );
}

#[test]
fn example_15_8_algorithm_51() {
    let tau = drug_tau(false);
    let mut low = drug();
    assert_eq!(abstract_invariance(&mut low, &tau).unwrap(), None);
    let mut high = construct(drug(), tau.clone()).unwrap();
    // Eq. 15.65.
    let j = high.joint(&tau, &[vec![("X".into(), "1".into())]]).unwrap();
    let p: f64 = j.iter().filter(|(k, _)| k[1] == "1").map(|(_, p)| p).sum();
    assert!(close(p, 0.5));
    // Prop. 15.2.4.
    for layer in 1..=3 {
        let c = layer_consistency(&mut low, &mut high, &tau, layer, 1e-12).unwrap();
        assert!(c.holds, "layer {layer}: {:?}", c.worst);
    }
}

/// `M_L` of Ex. 15.6 (Eqs. 15.39-15.42).
fn cholesterol() -> Model {
    Model::new(
        [
            eq("X", v("UX")),
            eq("HDL", expr::xor(v("X"), v("UC1"))),
            eq("LDL", expr::xor(v("X"), v("UC2"))),
            eq(
                "Y",
                expr::xor(expr::and([v("LDL"), expr::not(v("HDL"))]), v("UY")),
            ),
        ],
        [
            bern("UX", 0.5),
            bern("UC1", 0.1),
            bern("UC2", 0.1),
            bern("UY", 0.1),
        ],
    )
    .unwrap()
}

fn cholesterol_graph() -> Admg {
    let mut g = Admg::new();
    for (a, b) in [("X", "HDL"), ("X", "LDL"), ("HDL", "Y"), ("LDL", "Y")] {
        g.directed(a, b);
    }
    g
}

fn cholesterol_tau(total: bool) -> Abstraction {
    let map: fn(&[&str]) -> String = if total {
        |v| (v[0].parse::<i32>().unwrap() + v[1].parse::<i32>().unwrap()).to_string()
    } else {
        |v| (v[1].parse::<i32>().unwrap() - v[0].parse::<i32>().unwrap()).to_string()
    };
    Abstraction::new()
        .keep("X")
        .unwrap()
        .cluster("C", &["HDL", "LDL"], map)
        .unwrap()
        .keep("Y")
        .unwrap()
}

#[test]
fn examples_15_6_and_15_7_the_invariance_condition() {
    let mut low = cholesterol();
    // Eqs. 15.48-15.49: one high-level query, two low-level values.
    let tc = cholesterol_tau(true);
    let q1 = [Term::new(&[("HDL", "0"), ("LDL", "1")], &[("Y", "1")])];
    let q2 = [Term::new(&[("HDL", "1"), ("LDL", "0")], &[("Y", "1")])];
    assert!(close(low_level(&mut low, &tc, &q1).unwrap(), 0.9));
    assert!(close(low_level(&mut low, &tc, &q2).unwrap(), 0.1));
    fn pairs(t: &[Term]) -> Vec<(&str, &str)> {
        t[0].intervention
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect()
    }
    assert_eq!(tc.tau(&pairs(&q1)).unwrap(), tc.tau(&pairs(&q2)).unwrap());

    // The sum breaks the AIC at Y; Alg. 51 refuses it.
    let v = abstract_invariance(&mut low, &tc)
        .unwrap()
        .expect("a violation");
    assert_eq!(v.cluster, "Y");
    assert!(matches!(
        construct(cholesterol(), tc.clone()),
        Err(AbstractionError::Invariance(_))
    ));
    // So does its observational shadow (Def. 15.2.5).
    let data = observational(&cholesterol(), &["X", "HDL", "LDL", "Y"]);
    assert!(
        conditional_invariance(&data, &cholesterol_graph(), &tc, 1e-9)
            .unwrap()
            .is_some()
    );

    // The difference satisfies it (Ex. 15.7), and Alg. 51 then gives an
    // L3-τ consistent model (Prop. 15.2.4).
    let diff = cholesterol_tau(false);
    assert_eq!(abstract_invariance(&mut low, &diff).unwrap(), None);
    assert!(
        conditional_invariance(&data, &cholesterol_graph(), &diff, 1e-9)
            .unwrap()
            .is_none()
    );
    let mut high = construct(cholesterol(), diff.clone()).unwrap();
    assert!(
        layer_consistency(&mut low, &mut high, &diff, 3, 1e-12)
            .unwrap()
            .holds
    );
}

#[test]
fn example_15_9_is_not_tau_identifiable() {
    let tau = drug_tau(false);
    let low = drug();
    // Fig. 15.11(d).
    let high_data = tau
        .transform(&observational(&low, &["R", "A", "B", "Y"]))
        .unwrap();
    for (x, y, p) in [
        ("0", "0", 0.594),
        ("0", "1", 0.066),
        ("1", "0", 0.05),
        ("1", "1", 0.29),
    ] {
        assert!(
            close(probability(&high_data, &[("X", x), ("Y", y)]), p),
            "{x}{y}"
        );
    }
    // Fig. 15.11(c): projecting R leaves X -> Y and X <-> Y.
    let gc = tau.cdag(&drug_graph()).unwrap();
    assert_eq!(edges(&gc), ["X->Y", "X<->Y"]);
    let q = CtfQuery::parse(&gc, "P(y_x)").unwrap();
    assert!(matches!(
        abstract_identify(&drug_graph(), &tau, &q, &[Input::observational()]),
        Err(AbstractionError::Identification(
            CtfError::NotIdentifiable { .. }
        ))
    ));

    // M1 and M2 (Eqs. 15.71-15.72) both fit τ(P(V_L)) and disagree.
    let model = |first: bool| {
        let branch = if first {
            expr::or([v("X"), v("UY0")])
        } else {
            expr::and([expr::not(v("X")), v("UY0")])
        };
        Model::new(
            [
                eq("X", v("UXY")),
                eq(
                    "Y",
                    expr::or([
                        expr::and([expr::not(v("UXY")), branch]),
                        expr::and([v("UXY"), v("UY1")]),
                    ]),
                ),
            ],
            [bern("UXY", 0.34), bern("UY0", 0.1), bern("UY1", 0.852941)],
        )
        .unwrap()
    };
    let mut low = drug();
    let effect = [Term::new(&[("A", "1"), ("B", "1")], &[("Y", "1")])];
    let mut values = Vec::new();
    for first in [true, false] {
        let mut m = model(first);
        assert!(
            layer_consistency(&mut low, &mut m, &tau, 1, 1e-6)
                .unwrap()
                .holds
        );
        values.push(q_consistency(&mut low, &mut m, &tau, &effect).unwrap());
    }
    // Eqs. 15.73-15.74; the truth is 0.5.
    assert!(close(values[0].0, 0.5));
    assert!(close(values[0].1, 0.66 + 0.34 * 0.852941));
    assert!(close(values[1].1, 0.34 * 0.852941));
    assert!((values[0].1 - 0.95).abs() < 5e-3 && (values[1].1 - 0.29).abs() < 5e-3);
}

#[test]
fn example_15_10_is_tau_identifiable() {
    let tau = drug_tau(true);
    let low_data = observational(&drug(), &["R", "A", "B", "Y"]);
    // Fig. 15.12(b).
    let high = tau.transform(&low_data).unwrap();
    for (r, x, y, p) in [
        ("0", "0", "0", 0.432),
        ("0", "0", "1", 0.048),
        ("0", "1", "0", 0.018),
        ("0", "1", "1", 0.002),
        ("1", "0", "0", 0.162),
        ("1", "0", "1", 0.018),
        ("1", "1", "0", 0.032),
        ("1", "1", "1", 0.288),
    ] {
        assert!(close(
            probability(&high, &[("R", r), ("X", x), ("Y", y)]),
            p
        ));
    }
    // Fig. 15.12(a): R -> X -> Y with R <-> Y.
    let gc = tau.cdag(&drug_graph()).unwrap();
    assert_eq!(edges(&gc), ["R->X", "R<->Y", "X->Y"]);
    // Eq. 15.75 gives the true 0.5.
    let q = CtfQuery::parse(&gc, "P(y_x)").unwrap();
    let p = abstract_estimate(
        &drug_graph(),
        &tau,
        &q,
        &[("X", "1"), ("Y", "1")],
        &low_data,
    )
    .unwrap();
    assert!(close(p, 0.5));

    // M_H of Eqs. 15.76-15.79 fits τ(P(V_L)) and gets 0.5 (Eq. 15.80).
    let mut mh = Model::new(
        [
            eq("R", v("URY")),
            eq(
                "X",
                expr::or([
                    expr::and([expr::not(v("R")), v("UX0")]),
                    expr::and([v("R"), v("UX1")]),
                ]),
            ),
            eq(
                "Y",
                expr::or([
                    expr::and([expr::not(v("X")), expr::not(v("URY")), v("UY0")]),
                    expr::and([expr::not(v("X")), v("URY"), v("UY1")]),
                    expr::and([v("X"), expr::not(v("URY")), v("UY2")]),
                    expr::and([v("X"), v("URY"), v("UY3")]),
                ]),
            ),
        ],
        [
            bern("URY", 0.5),
            bern("UX0", 0.04),
            bern("UX1", 0.64),
            bern("UY0", 0.1),
            bern("UY1", 0.1),
            bern("UY2", 0.1),
            bern("UY3", 0.9),
        ],
    )
    .unwrap();
    let mut low = drug();
    assert!(
        layer_consistency(&mut low, &mut mh, &tau, 1, 1e-12)
            .unwrap()
            .holds
    );
    let effect = [Term::new(&[("A", "1"), ("B", "1")], &[("Y", "1")])];
    let (l, h) = q_consistency(&mut low, &mut mh, &tau, &effect).unwrap();
    assert!(close(l, 0.5) && close(h, 0.5));
}

#[test]
fn admissibility_and_cdags() {
    // A -> B -> C with {A, C} clustered: B is a descendant and an ancestor.
    let mut g = Admg::new();
    g.directed("A", "B");
    g.directed("B", "C");
    let bad = Abstraction::new()
        .cluster("AC", &["A", "C"], |v| v.join(""))
        .unwrap()
        .keep("B")
        .unwrap();
    assert!(!bad.is_admissible(&g).unwrap());
    assert_eq!(
        bad.cdag(&g).unwrap_err(),
        AbstractionError::NotAdmissible("AC".into())
    );
    // Clusters cannot overlap.
    assert_eq!(
        Abstraction::new()
            .keep("A")
            .unwrap()
            .cluster("AB", &["A", "B"], |v| v.join(""))
            .unwrap_err(),
        AbstractionError::Overlap("A".into())
    );
    // Fig. 15.9: the nutrition study's C-DAG.
    let mut g = Admg::new();
    for (a, b) in [
        ("R", "D"),
        ("D", "C"),
        ("D", "F"),
        ("D", "P"),
        ("C", "B"),
        ("F", "B"),
        ("P", "B"),
    ] {
        g.directed(a, b);
    }
    g.directed("R", "B");
    let tau = Abstraction::new()
        .keep("D")
        .unwrap()
        .cluster("Z", &["C", "F", "P"], |v| {
            let n: Vec<i64> = v.iter().map(|x| x.parse().unwrap()).collect();
            (4 * n[0] + 9 * n[1] + 4 * n[2]).to_string()
        })
        .unwrap()
        .keep("B")
        .unwrap();
    let gc = tau.cdag(&g).unwrap();
    // D -> Z -> B, and R projected into D <-> B: the front door.
    assert_eq!(edges(&gc), ["B<->D", "D->Z", "Z->B"]);
    let q = CtfQuery::parse(&gc, "P(b_d)").unwrap();
    assert!(abstract_identify(&g, &tau, &q, &[Input::observational()]).is_ok());
}

#[test]
fn the_interventional_partition_coarsens_the_observational() {
    // I = 2 UA + UB; T reads I's parity, flipped by U ∧ UA: UA confounds.
    let parity = expr::or([
        expr::eq(v("I"), expr::int(1)),
        expr::eq(v("I"), expr::int(3)),
    ]);
    let mut m = Model::new(
        [
            eq("I", expr::add(expr::mul(expr::int(2), v("UA")), v("UB"))),
            eq(
                "T",
                expr::xor(expr::xor(parity, expr::and([v("U"), v("UA")])), v("UT")),
            ),
        ],
        [
            bern("UA", 0.5),
            bern("UB", 0.5),
            bern("U", 0.3),
            bern("UT", 0.2),
        ],
    )
    .unwrap();
    let data = observational(&m, &["I", "T"]);
    let obs = observational_partition(&data, "I", "T", 1e-9).unwrap();
    let int = interventional_partition(&mut m, "I", "T", 1e-9).unwrap();
    assert_eq!(obs.len(), 4);
    assert_eq!(int, vec![vec!["0", "2"], vec!["1", "3"]]);
    // Thm 15.2.5: every observational block sits inside an interventional one.
    for b in &obs {
        assert!(int.iter().any(|c| b.iter().all(|x| c.contains(x))));
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A random binary SCM over `V0 … V3` (topological order): each variable
/// xors its own noise into an AND or OR of a random subset of earlier
/// variables, and one noise is shared by two variables.
fn random_model(seed: u64) -> Model {
    let mut r = seed;
    let mut next = || {
        r = splitmix(r);
        r
    };
    let mut f = Vec::new();
    let shared = (next() % 4) as usize;
    for i in 0..4 {
        let parents: Vec<E> = (0..i)
            .filter(|_| next() % 2 == 0)
            .map(|j| v(&format!("V{j}")))
            .collect();
        let noise = if i == shared || i == (shared + 1) % 4 {
            v("US")
        } else {
            v(&format!("U{i}"))
        };
        let body = if parents.is_empty() {
            noise
        } else if next() % 2 == 0 {
            expr::xor(expr::and(parents), noise)
        } else {
            expr::xor(expr::or(parents), noise)
        };
        f.push(eq(&format!("V{i}"), body));
    }
    let mut u: Vec<(String, Dist)> = (0..4)
        .map(|i| bern(&format!("U{i}"), 0.15 + 0.7 * (next() % 100) as f64 / 100.0))
        .collect();
    u.push(bern("US", 0.4));
    Model::new(f, u).unwrap()
}

/// Consecutive blocks of the topological order, which are always
/// admissible, each with a random map to `levels` levels (injective when
/// `levels` is 0).
fn random_tau(seed: u64, levels: u64) -> Abstraction {
    let mut r = splitmix(seed ^ 0x7a);
    let mut tau = Abstraction::new();
    let mut at = 0;
    let mut k = 0;
    while at < 4 {
        r = splitmix(r);
        let size = 1 + (r % 2) as usize;
        let members: Vec<String> = (at..(at + size).min(4)).map(|i| format!("V{i}")).collect();
        let refs: Vec<&str> = members.iter().map(String::as_str).collect();
        let salt = splitmix(r);
        tau = tau
            .cluster(&format!("C{k}"), &refs, move |x| {
                let joined = x.join("");
                if levels == 0 {
                    joined
                } else {
                    let h = joined.bytes().fold(salt, |h, b| splitmix(h ^ u64::from(b)));
                    (h % levels).to_string()
                }
            })
            .unwrap();
        at += size;
        k += 1;
    }
    tau
}

/// Prop. 15.4.1: the finest intravariable clustering always satisfies the
/// AIC. Prop. 15.2.4: whenever the AIC holds, Alg. 51's model is τ-consistent
/// with M_L, so the check never accepts a clustering it should not.
#[test]
fn the_aic_check_is_sound_on_random_models() {
    let (mut accepted, mut rejected) = (0, 0);
    for seed in 0..40u64 {
        let mut low = random_model(splitmix(seed));
        let fine = random_tau(seed, 0);
        assert_eq!(
            abstract_invariance(&mut low, &fine).unwrap(),
            None,
            "seed {seed}"
        );
        let coarse = random_tau(seed, 2);
        match abstract_invariance(&mut low, &coarse).unwrap() {
            None => {
                accepted += 1;
                let mut high = construct(low.clone(), coarse.clone()).unwrap();
                let layer = if seed % 8 == 0 { 3 } else { 2 };
                let c = layer_consistency(&mut low, &mut high, &coarse, layer, 1e-12).unwrap();
                assert!(c.holds, "seed {seed}: {:?}", c.worst);
            }
            Some(_) => rejected += 1,
        }
    }
    assert!(
        accepted >= 5 && rejected >= 5,
        "{accepted} accepted, {rejected} rejected"
    );
}
