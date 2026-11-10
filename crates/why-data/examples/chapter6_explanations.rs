//! Chapter 6, section 6.2: explaining events with counterfactual (L3)
//! Shapley values, against SHAP (L1) and do-Shapley (L2).
//!
//! Reproduces the book's examples and asserts the values it reports:
//! rain & sprinkler (Ex. 6.38-6.39), watering plants (Ex. 6.37), shark
//! attacks (Ex. 6.40), the flu shot (Ex. 6.41) and lightning in two regions
//! (Ex. 6.43). There is no notebook for this section; the numbers come from
//! the text.
//!
//! Run with `cargo run -p why-data --example chapter6_explanations`.

use rssn::symbolic::core::Expr;
use why_data::explain::Explainer;
use why_data::scm::{Dist, Model, Value, expr};

const T: Value = Value::Bool(true);
const F: Value = Value::Bool(false);

fn model(f: Vec<(&str, Expr)>, u: Vec<(&str, Dist)>) -> Model {
    Model::new(
        f.into_iter().map(|(k, e)| (k.to_string(), e)),
        u.into_iter().map(|(k, d)| (k.to_string(), d)),
    )
    .unwrap()
}

fn compare(title: &str, ex: &mut Explainer) -> [Vec<f64>; 3] {
    let l1 = ex.shap_l1().unwrap();
    let l2 = ex.do_shapley_l2().unwrap();
    let l3 = ex.l3_shapley().unwrap();
    println!("{title}");
    println!("  {:<10} {:>9} {:>9} {:>9}", "", "SHAP L1", "do-SV L2", "L3 SV");
    for (i, v) in ex.explanatory().iter().enumerate() {
        println!("  {v:<10} {:>+9.4} {:>+9.4} {:>+9.4}", l1.values[i], l2.values[i], l3.values[i]);
    }
    println!("  {:<10} {:>+9.4} {:>+9.4} {:>+9.4}   (L3 total = NTE(X, Y | e))\n", "total", l1.total, l2.total, l3.total);
    [l1.values, l2.values, l3.values]
}

fn main() {
    let v = expr::var;

    // Ex. 6.36-6.39: neither rain nor sprinkler is a but-for cause of the wet
    // grass, yet together they are; L3 gives each 1/8.
    let rain = model(
        vec![("rain", v("u1")), ("sprinkler", v("u2")), ("wet", expr::or([v("rain"), v("sprinkler")]))],
        vec![("u1", Dist::Bernoulli(0.5)), ("u2", Dist::Bernoulli(0.5))],
    );
    let mut ex = Explainer::new(
        &rain,
        "wet",
        &[("rain", T), ("sprinkler", T), ("wet", T)],
        &["rain", "sprinkler"],
    )
    .unwrap();
    println!(
        "Rain & sprinkler GDEs: GDE_∅(rain) = {:.4}, GDE_sprinkler(rain) = {:.4}  (book: 0, 1/4)",
        ex.gde("rain", &[]).unwrap(),
        ex.gde("rain", &["sprinkler"]).unwrap()
    );
    let [_, _, l3] = compare("Ex. 6.39  Why is the grass wet?", &mut ex);
    assert!((l3[0] - 0.125).abs() < 1e-12 && (l3[1] - 0.125).abs() < 1e-12);

    // Ex. 6.37: two units only. Univariate and saturated NTEs are all zero;
    // the pair (Alice, Bob) is what mattered.
    let plants = model(
        vec![
            ("alice", v("u")),
            ("bob", v("u")),
            ("charlie", expr::not(v("u"))),
            ("alive", expr::or([v("alice"), v("bob"), v("charlie")])),
        ],
        vec![("u", Dist::Bernoulli(0.5))],
    );
    let mut ex = Explainer::new(
        &plants,
        "alive",
        &[("alice", T), ("bob", T), ("charlie", F), ("alive", T)],
        &["alice", "bob", "charlie"],
    )
    .unwrap();
    let pair = ex.nte(&["alice", "bob"]).unwrap();
    let all = ex.nte(&["alice", "bob", "charlie"]).unwrap();
    println!(
        "Ex. 6.37  Watering plants: NTE(alice) = {:.2}, NTE(all) = {all:.2}, NTE(alice, bob) = {pair:.2}  (book: 0, 0, 1/2)\n",
        ex.nte(&["alice"]).unwrap()
    );
    assert!((pair - 0.5).abs() < 1e-12 && all.abs() < 1e-12);

    // Ex. 6.40: shark attacks share a cause with ice-cream sales but cannot
    // affect profits. SHAP credits them; L3 does not (admissibility).
    let sharks = model(
        vec![("sharks", v("u")), ("sales", v("u")), ("profit", v("sales"))],
        vec![("u", Dist::Bernoulli(0.5))],
    );
    let mut ex = Explainer::new(
        &sharks,
        "profit",
        &[("sharks", T), ("sales", T), ("profit", T)],
        &["sharks", "sales"],
    )
    .unwrap();
    let [l1, _, l3] = compare("Ex. 6.40  Why were profits high?", &mut ex);
    assert!((l1[0] - 0.25).abs() < 1e-12 && l3[0].abs() < 1e-12);

    // Ex. 6.41: Alice had the shot and still got the flu. The shot never
    // harms, so it cannot explain her illness, but L1 and L2 say 0.81.
    let t = |k: i64| expr::eq(v("type"), expr::int(k));
    let flu = model(
        vec![
            ("shot", v("ushot")),
            (
                "healthy",
                expr::or([
                    expr::and([v("shot"), expr::or([t(1), t(3)])]),
                    expr::and([expr::not(v("shot")), expr::or([t(2), t(3)])]),
                ]),
            ),
        ],
        vec![
            ("ushot", Dist::Bernoulli(0.1)),
            // (Y0, Y1) = (0,0), (0,1), (1,0), (1,1)
            ("type", Dist::Categorical(vec![0.09, 0.90, 0.0, 0.01])),
        ],
    );
    let mut ex = Explainer::new(&flu, "healthy", &[("shot", T), ("healthy", F)], &["shot"]).unwrap();
    let [l1, l2, l3] = compare("Ex. 6.41  Why did Alice get the flu despite the shot?", &mut ex);
    assert!((l1[0] - 0.81).abs() < 1e-9 && (l2[0] - 0.81).abs() < 1e-9 && l3[0].abs() < 1e-12);

    // Ex. 6.43: the same fire, explained in two regions. Dry ground is rare
    // in the rainforest, so there it carries more of the explanation.
    for (region, p_dry) in [("rainforest", 0.05), ("desert", 0.95)] {
        let fire = model(
            vec![("lightning", v("u1")), ("dry", v("u2")), ("fire", expr::and([v("lightning"), v("dry")]))],
            vec![("u1", Dist::Bernoulli(0.01)), ("u2", Dist::Bernoulli(p_dry))],
        );
        let mut ex = Explainer::new(
            &fire,
            "fire",
            &[("lightning", T), ("dry", T), ("fire", T)],
            &["lightning", "dry"],
        )
        .unwrap();
        let gde = ex.gde("dry", &[]).unwrap();
        println!("Ex. 6.43  {region}: GDE_∅(dry) = {gde:.4}  (book: ≈ {:.2})", 1.0 - p_dry);
        assert!((gde - (1.0 - p_dry)).abs() < 1e-9);
        compare(&format!("          Why was there a fire in the {region}?"), &mut ex);
    }
}
