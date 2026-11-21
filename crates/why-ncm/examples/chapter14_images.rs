//! Chapter 14: image-based representations and generation.
//!
//! Walks through the chapter's worked examples, asserting each result:
//!
//! 1. Augmented SCMs over a 4-pixel image (Exs. 14.2-14.9), computed
//!    exactly. Models that fit `P(V, I)` disagree on how an image edits
//!    (Corol. 14.2.1, Thm 14.2.2). F-ctf queries reduce to the generative
//!    level (Def. 14.3.1, Lemma 14.3.1), and their optimal bounds pick out
//!    the ctf-consistent models (Def. 14.3.2).
//! 2. The optimal bounds of the colored-MNIST tasks (Sec. 14.4, Figs.
//!    14.8-14.12), with the true values inside.
//! 3. Algorithm 50 (ANCM) on rendered face images: it fits `P(V)`, its
//!    F-ctf query lies within the bounds, and its edits keep the gender.
//!    A Markovian proxy (the diagram without `F <-> Y`, as DEAR assumes)
//!    and a ceteris-paribus edit both fail Def. 14.3.2.
//! 4. Algorithm 50 on rendered digits from the MNIST backdoor model, on
//!    Task 1: editing a red 3 with a bar into a 6.
//!
//! The book's companion Python has no chapter 14 code; the numbers are the
//! book's. Run with `cargo run -p why-ncm --example chapter14_images`.

use why_data::ascm::{Ascm, Interval, ctf_consistency, optimal_bounds, total_variation};
use why_data::fairness::Dataset;
use why_data::graph::cinfer::counterfactual::CtfQuery;
use why_data::graph::dseparation::Admg;
use why_data::scm::{Dist, Model, Value, expr};
use why_ncm::{AncmConfig, ImageData, ancm};

const T: Value = Value::Bool(true);
const F: Value = Value::Bool(false);

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

fn q(g: &Admg, s: &str) -> CtfQuery {
    CtfQuery::parse(g, s).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn show(b: Interval) -> String {
    format!("[{:.4}, {:.4}]", b.lower, b.upper)
}

fn bits(s: &str) -> Vec<Value> {
    s.chars().map(|c| Value::Bool(c == '1')).collect()
}

/// Ex. 14.2's faces with Ex. 14.6's 4-pixel image (Eq. 14.12). `variant`
/// picks `H`'s mechanism: `*` (Eq. 14.3), `'` (Eq. 14.5) or `3` (Eq. 14.9).
fn faces(variant: char) -> Ascm {
    let v = expr::var;
    let h = match variant {
        '*' => expr::xor(
            expr::and([expr::not(v("Y")), v("UH1")]),
            expr::and([v("Y"), v("UH2")]),
        ),
        '\'' => expr::xor(
            expr::and([expr::not(v("UY")), v("UH1")]),
            expr::and([v("UY"), v("UH2")]),
        ),
        _ => expr::or([expr::and([expr::not(v("Y")), v("UH1")]), v("UH2")]),
    };
    let p_h1 = if variant == '3' { 0.25 } else { 0.4 };
    let m = Model::new(
        [
            ("F".to_string(), expr::xor(v("UF"), v("UY"))),
            ("Y".to_string(), v("UY")),
            ("H".to_string(), h),
            (
                "I1".to_string(),
                expr::xor(expr::xor(v("Y"), v("H")), v("UI")),
            ),
            ("I2".to_string(), expr::xor(v("Y"), v("UI"))),
            ("I3".to_string(), expr::xor(v("F"), v("H"))),
            ("I4".to_string(), v("UI")),
        ],
        [
            ("UF".to_string(), Dist::Bernoulli(0.4)),
            ("UY".to_string(), Dist::Bernoulli(0.4)),
            ("UH1".to_string(), Dist::Bernoulli(p_h1)),
            ("UH2".to_string(), Dist::Bernoulli(0.2)),
            ("UI".to_string(), Dist::Bernoulli(0.3)),
        ],
    )
    .unwrap();
    Ascm::new(m, &["F", "Y", "H"], &["I1", "I2", "I3", "I4"]).unwrap()
}

/// The MNIST models of Eqs. 14.30-14.38, at the generative level plus a
/// one-pixel-per-factor image. `C` draws its Bernoulli from `U_C` on 0..100.
fn mnist(frontdoor: bool) -> Ascm {
    let v = expr::var;
    let (p1, p2, p3) = if frontdoor {
        (0.9, 0.8, 0.7)
    } else {
        (0.8, 0.9, 0.75)
    };
    let (c, big) = if frontdoor {
        (
            expr::lt(
                v("UC"),
                expr::sub(expr::int(95), expr::mul(expr::int(10), v("D"))),
            ),
            expr::lt(v("UD"), expr::int(4)),
        )
    } else {
        (
            expr::lt(
                v("UC"),
                expr::add(expr::int(5), expr::mul(expr::int(10), v("UD"))),
            ),
            expr::ge(v("D"), expr::int(5)),
        )
    };
    let b = expr::and([
        expr::or([expr::xor(big, v("U1")), expr::xor(v("C"), v("U2"))]),
        v("U3"),
    ]);
    let m = Model::new(
        [
            ("D".to_string(), v("UD")),
            ("C".to_string(), c),
            ("B".to_string(), b),
            ("P1".to_string(), v("D")),
            ("P2".to_string(), v("C")),
            ("P3".to_string(), v("B")),
        ],
        [
            ("UD".to_string(), Dist::Categorical(vec![0.1; 10])),
            ("UC".to_string(), Dist::Categorical(vec![0.01; 100])),
            ("U1".to_string(), Dist::Bernoulli(p1)),
            ("U2".to_string(), Dist::Bernoulli(p2)),
            ("U3".to_string(), Dist::Bernoulli(p3)),
        ],
    )
    .unwrap();
    Ascm::new(m, &["D", "C", "B"], &["P1", "P2", "P3"]).unwrap()
}

fn part1() {
    println!("== 1. Augmented SCMs (Exs. 14.2-14.9) ==");
    let mut star = faces('*');
    let p = star.observational().unwrap();
    println!("Table 14.1, P(F, Y, H):");
    for r in 0..p.len() {
        let row: Vec<&str> = ["F", "Y", "H"]
            .iter()
            .map(|n| p.value(n, r).unwrap())
            .collect();
        println!("  {} {:.3}", row.join(" "), p.weights().unwrap()[r]);
    }

    // Exs. 14.3 and 14.5: same P(V), different edits.
    let query = |m: &mut Ascm| {
        m.counterfactual(
            &[("Y", F)],
            &[("F", F), ("H", T)],
            &[("F", F), ("Y", T), ("H", F)],
        )
        .unwrap()
    };
    let (mut prime, mut three) = (faces('\''), faces('3'));
    for m in [&prime, &three] {
        let tv = total_variation(&p, &m.observational().unwrap(), &["F", "Y", "H"]).unwrap();
        assert!(tv < 1e-12);
    }
    let (a, b, c) = (query(&mut star), query(&mut prime), query(&mut three));
    println!(
        "P(F_(Y=0) = 0, H_(Y=0) = 1 | F = 0, Y = 1, H = 0): M* {a:.2}, M' {b:.2}, M(3) {c:.2}"
    );
    assert!(close(a, 0.4) && close(b, 0.0) && close(c, 0.25));

    // Fig. 14.5 and Eqs. 14.14, 14.16, 14.20.
    let ctf = star.image_counterfactual(&[("Y", F)]).unwrap();
    println!("P(I, I_(Y=0)) has {} image pairs, e.g.", ctf.rows.len());
    for (i, j) in [("0000", "0000"), ("0001", "1101"), ("1110", "0010")] {
        println!("  {i} -> {j}: {:.5}", ctf.probability(&bits(i), &bits(j)));
    }
    let (i, j) = (bits("0001"), bits("1101"));
    let pixel = ctf.feature_of(|im| im[0].as_bool(), &i, &j);
    let parity = ctf.feature_of(
        |im| im[0].as_bool() ^ im[1].as_bool() ^ im[2].as_bool(),
        &i,
        &j,
    );
    let lemma = star
        .joint_counterfactual(&[("Y", F)], &[("F", F)], &[("F", F)])
        .unwrap();
    println!(
        "F-ctf with W = I1: {pixel:.2}; W = I1 xor I2 xor I3 = F: {parity:.2}; P(F = 0, F_(Y=0) = 0): {lemma:.2}"
    );
    assert!(close(pixel, 0.08) && close(parity, 0.52) && close(lemma, 0.52));

    // Ex. 14.9.
    let g = graph(&[("Y", "H")], &[("F", "Y")]);
    let values = [("F", "0"), ("Y", "1"), ("Y'", "0"), ("H", "0"), ("H'", "1")];
    let qq = q(&g, "P(f_{y'}, h'_{y'} | f, y, h)");
    let bounds = optimal_bounds(&g, &p, &qq, &values).unwrap();
    println!(
        "Ex. 14.9 optimal bounds {}: M* (0.4) and M(3) (0.25) are ctf-consistent, M' (0) is not",
        show(bounds)
    );
    assert!(close(bounds.lower, 0.25) && close(bounds.upper, 0.5));
    let b = optimal_bounds(
        &g,
        &p,
        &q(&g, "P(f_{y'} | f, y)"),
        &[("F", "0"), ("Y", "1"), ("Y'", "0")],
    )
    .unwrap();
    println!(
        "with W = {{F, Y}}, P(F_(Y=0) = 0 | F = 0, Y = 1) is bounded by {}",
        show(b)
    );
    assert!(close(b.lower, 1.0));
}

fn part2() {
    println!("\n== 2. Colored MNIST bounds (Sec. 14.4) ==");
    let int = Value::Int;
    let mut bd = mnist(false);
    let p = bd.observational().unwrap();
    let g = graph(&[("D", "B"), ("C", "B")], &[("D", "C")]);
    let base = [("D", "6"), ("D'", "3"), ("C", "1"), ("B", "1")];
    let keep = optimal_bounds(&g, &p, &q(&g, "P(c_{d}, b_{d} | d', c, b)"), &base).unwrap();
    let mut v = base.to_vec();
    v.push(("B'", "0"));
    let drop = optimal_bounds(&g, &p, &q(&g, "P(c_{d}, b'_{d} | d', c, b)"), &v).unwrap();
    let truth = bd
        .counterfactual(
            &[("D", int(6))],
            &[("C", T), ("B", T)],
            &[("D", int(3)), ("C", T), ("B", T)],
        )
        .unwrap();
    println!(
        "Task 1, red 3 with a bar -> 6: keeps the bar {} (truth {truth:.3}), loses it {}",
        show(keep),
        show(drop)
    );
    assert!((keep.upper - 0.34).abs() < 0.005 && (drop.lower - 0.66).abs() < 0.005);
    let t2 = optimal_bounds(
        &g,
        &p,
        &q(&g, "P(d_{b'}, c_{b'} | d, c, b)"),
        &[("D", "1"), ("C", "0"), ("B", "1"), ("B'", "0")],
    )
    .unwrap();
    println!(
        "Task 2, green 1 loses its bar, digit and color kept: {}",
        show(t2)
    );
    assert!(close(t2.lower, 1.0));

    let mut fd = mnist(true);
    let p = fd.observational().unwrap();
    let g = graph(&[("D", "C"), ("C", "B")], &[("D", "B")]);
    let base = [("D", "2"), ("D'", "7"), ("C", "0"), ("B", "1")];
    let green_bar = optimal_bounds(&g, &p, &q(&g, "P(c_{d}, b_{d} | d', c, b)"), &base).unwrap();
    let mut v = base.to_vec();
    v.push(("B'", "0"));
    let green_none = optimal_bounds(&g, &p, &q(&g, "P(c_{d}, b'_{d} | d', c, b)"), &v).unwrap();
    let mut v = base.to_vec();
    v.push(("C'", "1"));
    let red = optimal_bounds(&g, &p, &q(&g, "P(c'_{d} | d', c, b)"), &v).unwrap();
    let truth = fd
        .counterfactual(
            &[("D", int(2))],
            &[("C", F), ("B", T)],
            &[("D", int(7)), ("C", F), ("B", T)],
        )
        .unwrap();
    println!(
        "Task 3, green 7 with a bar -> 2: green with bar {} (truth {truth:.3}), green without {}, red {}",
        show(green_bar),
        show(green_none),
        show(red)
    );
    assert!(
        close(green_bar.upper, 1.0 / 3.0)
            && close(green_none.upper, 0.0)
            && close(red.lower, 2.0 / 3.0)
    );
    let v = [("D", "4"), ("C", "1"), ("C'", "0"), ("B", "0"), ("B'", "1")];
    let bar = optimal_bounds(&g, &p, &q(&g, "P(d_{c'}, b'_{c'} | d, c, b)"), &v).unwrap();
    let truth = fd
        .counterfactual(
            &[("C", F)],
            &[("D", int(4)), ("B", T)],
            &[("D", int(4)), ("C", T), ("B", F)],
        )
        .unwrap();
    println!(
        "Task 4, red 4 without a bar -> green, gains a bar: {} (truth {truth:.3})",
        show(bar)
    );
    assert!((bar.lower - 0.12).abs() < 0.005 && truth >= bar.lower);
}

/// Draws from a uniform on [0, 1), SplitMix64.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let (a, b) = (f64::from(self.unit()).max(1e-12), f64::from(self.unit()));
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

/// Renders labelled factors as pixels: each factor lights one pixel per
/// level, the unlabelled `extra` bits light more, and a brightness
/// `t ∈ [0.6, 1]` scales everything and gets a pixel of its own.
fn render(levels: &[(usize, usize)], extra: &[bool], rng: &mut Rng) -> Vec<f32> {
    let t = 0.6 + 0.4 * rng.unit();
    let mut clean = Vec::new();
    for &(l, d) in levels {
        clean.extend((0..d).map(|i| f32::from(u8::from(i == l))));
    }
    clean.extend(extra.iter().map(|b| f32::from(u8::from(*b))));
    clean.push(1.0);
    clean.iter().map(|x| x * t + 0.03 * rng.normal()).collect()
}

/// The labels a rendered image shows: the brightest pixel of each block.
fn read(image: &[f32], sizes: &[usize]) -> Vec<String> {
    let mut at = 0;
    sizes
        .iter()
        .map(|&d| {
            let block = &image[at..at + d];
            at += d;
            let best = (0..d)
                .max_by(|a, b| block[*a].total_cmp(&block[*b]))
                .unwrap();
            best.to_string()
        })
        .collect()
}

fn part3() {
    println!("\n== 3. Algorithm 50 (ANCM) on face images ==");
    let mut rng = Rng(7);
    let (mut rows, mut images) = (Vec::new(), Vec::new());
    for _ in 0..2000 {
        let mut bern = |p: f32| rng.unit() < p;
        let (uf, uy, uh1, uh2, smile) = (bern(0.4), bern(0.4), bern(0.4), bern(0.2), bern(0.5));
        let (f, y) = (uf ^ uy, uy);
        let h = if y { uh2 } else { uh1 };
        let lv = [f, y, h].map(usize::from);
        rows.push(lv.map(|l| l.to_string()).to_vec());
        images.push(render(
            &[(lv[0], 2), (lv[1], 2), (lv[2], 2)],
            &[smile],
            &mut rng,
        ));
    }
    let data = ImageData::new(
        Dataset::from_records(["F", "Y", "H"], rows).unwrap(),
        images,
    )
    .unwrap();
    let truth = faces('*').observational().unwrap();
    let qq_values = [("F", "0"), ("Y", "1"), ("Y'", "0"), ("H", "0"), ("H'", "1")];

    for (name, g) in [
        ("G of Fig. 14.3", graph(&[("Y", "H")], &[("F", "Y")])),
        ("Markovian G (no F <-> Y)", {
            let mut g = graph(&[("Y", "H")], &[]);
            g.add_node("F");
            g
        }),
    ] {
        let model = ancm(&g, &data, &AncmConfig::default()).unwrap();
        let fitted = model.ncm().distribution(8192, 1).unwrap();
        // The bounds come from the true diagram either way.
        let g_true = graph(&[("Y", "H")], &[("F", "Y")]);
        let qq = q(&g_true, "P(f_{y'}, h'_{y'} | f, y, h)");
        let value = model
            .ncm()
            .estimate(&q(&g, "P(f_{y'}, h'_{y'} | f, y, h)"), &qq_values, 8192, 2)
            .unwrap();
        let c = ctf_consistency(&g_true, &truth, &fitted, &qq, &qq_values, value, 0.05).unwrap();
        println!(
            "{name}: TV(P, P^) = {:.3}, F-ctf {:.3} vs bounds {} -> ctf-consistent: {}",
            c.fit,
            c.value,
            show(c.bounds),
            c.consistent
        );
        if name.starts_with("G of") {
            assert!(c.consistent);
            let (mut kept, mut aged) = (0, 0);
            for (k, image) in data.images().iter().take(200).enumerate() {
                let e = model.edit(image, None, &[("Y", "0")], k as u64).unwrap();
                let shown = read(&e.edited_image[..6], &[2, 2, 2]);
                kept += usize::from(shown[0] == data.labels().value("F", k).unwrap());
                aged += usize::from(shown[1] == "0");
            }
            println!("  editing 200 images to look older: gender kept in {kept}, older in {aged}");
            assert!(kept >= 190 && aged >= 190);
        } else {
            assert!(!c.consistent);
        }
    }
    // A ceteris-paribus edit changes Y and nothing else, so H never turns
    // gray: its F-ctf value is 0 (Ex. 14.9).
    let g = graph(&[("Y", "H")], &[("F", "Y")]);
    let qq = q(&g, "P(f_{y'}, h'_{y'} | f, y, h)");
    let c = ctf_consistency(&g, &truth, &truth, &qq, &qq_values, 0.0, 0.05).unwrap();
    println!(
        "ceteris paribus: F-ctf 0 -> ctf-consistent: {}",
        c.consistent
    );
    assert!(!c.consistent);
}

fn part4() {
    println!("\n== 4. Algorithm 50 on MNIST-backdoor digits, Task 1 ==");
    let mut rng = Rng(11);
    let (mut rows, mut images) = (Vec::new(), Vec::new());
    for _ in 0..6000 {
        let d = (rng.unit() * 10.0) as usize % 10;
        let c = rng.unit() < 0.05 + 0.1 * d as f32;
        let (u1, u2, u3) = (rng.unit() < 0.8, rng.unit() < 0.9, rng.unit() < 0.75);
        let b = (((d >= 5) ^ u1) || (c ^ u2)) && u3;
        let thick = rng.unit() < 0.5;
        rows.push(vec![
            d.to_string(),
            u8::from(c).to_string(),
            u8::from(b).to_string(),
        ]);
        images.push(render(
            &[(d, 10), (usize::from(c), 2), (usize::from(b), 2)],
            &[thick],
            &mut rng,
        ));
    }
    let data = ImageData::new(
        Dataset::from_records(["D", "C", "B"], rows).unwrap(),
        images,
    )
    .unwrap();
    let g = graph(&[("D", "B"), ("C", "B")], &[("D", "C")]);
    let model = ancm(
        &g,
        &data,
        &AncmConfig {
            epochs: 30,
            ..AncmConfig::default()
        },
    )
    .unwrap();
    let truth = mnist(false).observational().unwrap();
    let fitted = model.ncm().distribution(8192, 1).unwrap();
    let qq = q(&g, "P(c_{d}, b_{d} | d', c, b)");
    let values = [("D", "6"), ("D'", "3"), ("C", "1"), ("B", "1")];
    let value = model.ncm().estimate(&qq, &values, 8192, 2).unwrap();
    let c = ctf_consistency(&g, &truth, &fitted, &qq, &values, value, 0.05).unwrap();
    println!(
        "TV(P, P^) = {:.3}; P^(C_(D=6) = 1, B_(D=6) = 1 | D = 3, C = 1, B = 1) = {:.3} vs bounds {} -> ctf-consistent: {}",
        c.fit,
        c.value,
        show(c.bounds),
        c.consistent
    );
    assert!(c.consistent);
    // Edit every red 3 with a bar in the data into a 6.
    let (mut n, mut red, mut lost) = (0, 0, 0);
    for (k, image) in data.images().iter().enumerate() {
        let shown_now: Vec<&str> = ["D", "C", "B"]
            .iter()
            .map(|x| data.labels().value(x, k).unwrap())
            .collect();
        if shown_now != ["3", "1", "1"] {
            continue;
        }
        // Labels go in the model's (topological) order.
        let labels: Vec<&str> = model
            .variables()
            .iter()
            .map(|x| data.labels().value(x, k).unwrap())
            .collect();
        // Ten abductions per image.
        for r in 0..10u64 {
            let e = model
                .edit(image, Some(&labels), &[("D", "6")], k as u64 * 10 + r)
                .unwrap();
            let shown = read(&e.edited_image[..14], &[10, 2, 2]);
            assert_eq!(shown[0], "6");
            n += 1;
            red += usize::from(shown[1] == "1");
            lost += usize::from(shown[2] == "0");
        }
    }
    println!(
        "{n} edits of red 3s with a bar into 6s: {red} stay red, {lost} lose the bar ({:.3}; the model's own value is {:.3})",
        lost as f64 / n as f64,
        1.0 - value
    );
    assert_eq!(red, n);
    // Abduction reproduces the generative level's counterfactual
    // (Lemma 14.3.1). The value itself can sit a little past the exact
    // bound: about 210 rows show a red 3, so P(B | D, C) is fitted only to
    // a few hundredths, and Def. 14.3.2 is checked within 0.05.
    assert!((lost as f64 / n as f64 - (1.0 - value)).abs() < 0.03);
}

fn main() {
    part1();
    part2();
    part3();
    part4();
}
