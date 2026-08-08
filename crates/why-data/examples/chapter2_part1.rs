//! Causal AI book, chapter 2 part 1 — "SCM Introduction".
//!
//! A migration of `chapter2_part1.ipynb` to [`why_data::scm`]. Every printed
//! number is asserted against the value the Python notebook produces, so this
//! doubles as a conformance test of the port.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p why-data --example chapter2_part1
//! ```

use why_data::scm::{Dist, Model, Value, expr};

/// The notebook's `sample` draws from a `DataFrame`; here the caller supplies
/// the randomness, so the example carries its own generator and stays
/// reproducible. xorshift64*, seeded fixed.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        // 53 bits of mantissa, scaled into [0, 1).
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Asserts a computed probability against the notebook's value and prints it.
fn show(label: &str, got: f64, expected: f64) {
    assert!(
        approx(got, expected),
        "{label}: got {got}, notebook says {expected}"
    );
    println!("  {label} = {got:.4}");
}

fn main() {
    example_2_1();
    example_2_2();
    example_2_4();
    example_2_5();
    example_2_6();
}

/// Example 2.1 — a game of chance. `X := U1 + U2`, `Y := U1 - U2`, with each
/// die uniform on 1..=6. The leading `0.0` gives value `0` zero probability,
/// exactly as the notebook's comment describes.
fn example_2_1() {
    println!("\n=== Example 2.1 — game of chance (M1) ===");

    let die = Dist::Categorical(vec![0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]);
    let m1 = Model::new(
        [
            ("x".to_string(), expr::add(expr::var("u1"), expr::var("u2"))),
            ("y".to_string(), expr::sub(expr::var("u1"), expr::var("u2"))),
        ],
        [("u1".to_string(), die.clone()), ("u2".to_string(), die)],
    )
    .expect("M1 is well formed");

    println!("{m1}");

    // `get_probability_table(u=True).head()` — exogenous columns included.
    let obs = m1.observational();
    let full = m1
        .world(obs)
        .probability_table(Some(&["u1", "u2", "x", "y"]))
        .expect("table over U and V");
    println!("\nP(U, V), first 5 of {} rows:", full.rows.len());
    print!("{}", head(&full.to_string(), 6));

    // `get_probability_table().head()` — the joint over V alone.
    let joint = m1.world(obs).probability_table(None).expect("table over V");
    println!("\nP(V), first 5 of {} rows:", joint.rows.len());
    print!("{}", head(&joint.to_string(), 6));

    let total: f64 = joint.rows.iter().map(|(_, p)| p).sum();
    assert!(approx(total, 1.0), "the joint must be a distribution");

    // The notebook plots a histogram of 10^6 draws. Without a plotting stack the
    // equivalent check is that the empirical marginal converges on the exact one.
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let draws = m1
        .world(obs)
        .sample(100_000, || rng.next_f64())
        .expect("sampling the joint");
    let x_col = joint.columns.iter().position(|c| c == "x").expect("x is a column");
    let sevens = draws.iter().filter(|row| row[x_col] == Value::Int(7)).count();
    let empirical = sevens as f64 / draws.len() as f64;
    let exact = m1
        .world(obs)
        .query(&[("x", Value::Int(7))], &[])
        .expect("P(x = 7)");
    println!("\n  P(x = 7) exact = {exact:.4}, empirical over 100k draws = {empirical:.4}");
    assert!(approx(exact, 6.0 / 36.0));
    assert!(
        (empirical - exact).abs() < 0.01,
        "100k draws should land within a point of the exact marginal"
    );
}

/// Example 2.2 — treatment `X`, outcome `Y`, symptom `Z`, natural resistance
/// `U_r`. Note the operator precedence: Python's `&` binds tighter than `|`,
/// so `z & ux | ~z & ~ux` is `(z & ux) | (!z & !ux)`.
fn example_2_2() {
    println!("\n=== Examples 2.2 / 2.3 — treatment and outcome (M2) ===");

    let m2 = build_m2();
    println!("{m2}");

    let obs = m2.observational();
    let table = m2
        .world(obs)
        .probability_table(None)
        .expect("table over V");
    println!("\nP(V) over {} rows:", table.rows.len());
    print!("{table}");

    // Reproduces Table 2.2 of the textbook.
    let with_u = m2
        .world(obs)
        .probability_table(Some(&["ur", "uz", "ux", "uy", "z", "x", "y"]))
        .expect("table over U and V");
    println!("\nTable 2.2 — P(U, V) over {} rows:", with_u.rows.len());
    print!("{with_u}");

    // The notebook first does this by hand from the DataFrame, then via `query`.
    // Here the manual form is the same summation, spelled out.
    let joint = m2
        .world(obs)
        .query(&[("y", Value::Bool(true)), ("x", Value::Bool(true))], &[])
        .expect("P(y = 1, x = 1)");
    let marginal = m2
        .world(obs)
        .query(&[("x", Value::Bool(true))], &[])
        .expect("P(x = 1)");
    println!("\nBy hand, P(y=1, x=1) / P(x=1):");
    show("P(y = 1 | x = 1)", joint / marginal, 0.741_379_310_344_827_6);

    println!("And via query():");
    show(
        "P(y = 1 | x = 1)",
        m2.world(obs)
            .query(&[("y", Value::Bool(true))], &[("x", Value::Bool(true))])
            .expect("conditional"),
        0.741_379_310_344_827_6,
    );
    show(
        "P(y = 1 | x = 0)",
        m2.world(obs)
            .query(&[("y", Value::Bool(true))], &[("x", Value::Bool(false))])
            .expect("conditional"),
        0.319_718_309_859_155,
    );
    show(
        "P(z = 1)",
        m2.world(obs)
            .query(&[("z", Value::Bool(true))], &[])
            .expect("marginal"),
        0.2375,
    );
}

/// Example 2.4 — intervening on a variable the outcome does not depend on.
fn example_2_4() {
    println!("\n=== Example 2.4 — do(x = 2) on the game of chance ===");

    let die = Dist::Categorical(vec![0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]);
    let mut m1 = Model::new(
        [
            ("x".to_string(), expr::add(expr::var("u1"), expr::var("u2"))),
            ("y".to_string(), expr::sub(expr::var("u1"), expr::var("u2"))),
        ],
        [("u1".to_string(), die.clone()), ("u2".to_string(), die)],
    )
    .expect("M1 is well formed");

    let obs = m1.observational();
    show(
        "P(y = 0)",
        m1.world(obs).query(&[("y", Value::Int(0))], &[]).expect("marginal"),
        0.166_666_666_666_666_69,
    );

    // The submodel M_{X=2}: X's equation is replaced by the constant, and every
    // endogenous variable is renamed. Exogenous variables are untouched.
    let x2 = m1.intervene(&[("x", Value::Int(2))]).expect("x is endogenous");
    println!("\nM_(x=2) — note the renamed endogenous variables:");
    print!("{}", m1.world(x2).world());
    assert_eq!(
        m1.world(x2).world().variables(),
        ["{x}_{x=2}", "{y}_{x=2}"],
        "renaming must match the notebook's convention"
    );

    // Three spellings of the same query, mirroring the notebook's build-up.
    // 1. Loose: "y" inside the submodel, resolved through its renaming.
    show(
        "P(y = 0 | do(x = 2))",
        m1.world(x2).query(&[("y", Value::Int(0))], &[]).expect("query"),
        0.166_666_666_666_666_69,
    );
    // 2. Precise: the counterfactual variable named in full.
    show(
        "P(Y_(x=2) = 0), named in full",
        m1.world(x2)
            .query(&[("{y}_{x=2}", Value::Int(0))], &[])
            .expect("query"),
        0.166_666_666_666_666_69,
    );
    // 3. From the parent model, which remembers the variables its interventions
    //    created — the convenience mapping the notebook describes.
    show(
        "P(Y_(x=2) = 0), asked of M1",
        m1.world(obs)
            .query(&[("{y}_{x=2}", Value::Int(0))], &[])
            .expect("query"),
        0.166_666_666_666_666_69,
    );
}

/// Example 2.5 — total variation versus average treatment effect. The two have
/// opposite signs here, which is the point of the example.
fn example_2_5() {
    println!("\n=== Example 2.5 — treatment effects ===");

    let mut m2 = build_m2();
    let obs = m2.observational();
    let do1 = m2.intervene(&[("x", Value::Bool(true))]).expect("x is endogenous");
    let do0 = m2.intervene(&[("x", Value::Bool(false))]).expect("x is endogenous");

    let t = Value::Bool(true);
    let f = Value::Bool(false);

    let y_do1 = m2.world(do1).query(&[("y", t)], &[]).expect("query");
    let y_do0 = m2.world(do0).query(&[("y", t)], &[]).expect("query");
    show("P(y = 1 | do(x = 1))", y_do1, 0.25);
    show("P(y = 1 | do(x = 0))", y_do0, 0.4);

    let tv = m2.world(obs).query(&[("y", t)], &[("x", t)]).expect("query")
        - m2.world(obs).query(&[("y", t)], &[("x", f)]).expect("query");
    show("TV  = P(y=1|x=1) - P(y=1|x=0)", tv, 0.421_661_000_485_672_6);
    show("ATE = P(y=1|do(x=1)) - P(y=1|do(x=0))", y_do1 - y_do0, -0.15);

    assert!(tv > 0.0 && y_do1 - y_do0 < 0.0);
    println!(
        "\n  The total variation is positive but the causal effect is negative:\n  \
         enforcing the treatment would harm the population."
    );
}

/// Example 2.6 — a layer-3 query mixing the factual world with a counterfactual
/// one: "would the treatment have saved patients who went untreated and died?"
fn example_2_6() {
    println!("\n=== Example 2.6 — counterfactual ===");

    let mut m2 = build_m2();
    let obs = m2.observational();
    let do1 = m2.intervene(&[("x", Value::Bool(true))]).expect("x is endogenous");

    assert_eq!(
        m2.world(do1).world().variables(),
        ["{z}_{x=1}", "{x}_{x=1}", "{y}_{x=1}"]
    );
    println!("  variable of interest: {{y}}_{{x=1}}");

    // P(Y_{X=1} = 1 | X = 0, Y = 0). The conditioning event is factual and the
    // queried event is counterfactual, so this spans two worlds — both are
    // evaluated under the same draw of U, which is what makes it well defined.
    show(
        "P(Y_(x=1) = 1 | x = 0, y = 0)",
        m2.world(obs)
            .query(
                &[("{y}_{x=1}", Value::Bool(true))],
                &[("x", Value::Bool(false)), ("y", Value::Bool(false))],
            )
            .expect("counterfactual query"),
        0.021_739_130_434_782_615,
    );
}

/// `M2`, shared by examples 2.2, 2.5 and 2.6.
fn build_m2() -> Model {
    let (z, x, ur, ux, uy) = ("z", "x", "ur", "ux", "uy");
    Model::new(
        [
            // Z := U_r & U_z
            ("z".to_string(), expr::and([expr::var(ur), expr::var("uz")])),
            // X := (Z & U_x) | (!Z & !U_x)
            (
                "x".to_string(),
                expr::or([
                    expr::and([expr::var(z), expr::var(ux)]),
                    expr::and([expr::not(expr::var(z)), expr::not(expr::var(ux))]),
                ]),
            ),
            // Y := (X & U_r) | (!X & U_r & U_y) | (!X & !U_r & !U_y)
            (
                "y".to_string(),
                expr::or([
                    expr::and([expr::var(x), expr::var(ur)]),
                    expr::and([
                        expr::not(expr::var(x)),
                        expr::var(ur),
                        expr::var(uy),
                    ]),
                    expr::and([
                        expr::not(expr::var(x)),
                        expr::not(expr::var(ur)),
                        expr::not(expr::var(uy)),
                    ]),
                ]),
            ),
        ],
        [
            ("ur".to_string(), Dist::Bernoulli(0.25)),
            ("uz".to_string(), Dist::Bernoulli(0.95)),
            ("ux".to_string(), Dist::Bernoulli(0.9)),
            ("uy".to_string(), Dist::Bernoulli(0.7)),
        ],
    )
    .expect("M2 is well formed")
}

/// The first `n` lines of a rendered table, standing in for `DataFrame.head()`.
fn head(s: &str, n: usize) -> String {
    s.lines().take(n).map(|l| format!("{l}\n")).collect()
}
