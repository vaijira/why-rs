//! Tests for chapter 16: the book's examples on reconstructed diagrams,
//! random diagrams checking soundness against the true (augmented) MAG,
//! and the Markov-equivalence test against brute-force separation.

use super::*;
use crate::fairness::Dataset;
use crate::graph::dseparation::Admg;
use crate::scm::{Dist, Model, expr};

type E = rssn::symbolic::core::Expr;

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

/// Fig. 16.5(a), reconstructed: A -> B <- D, B -> C, D <-> C. It implies
/// exactly Eq. 16.2 and makes ⟨A, B, D, C⟩ discriminating for D.
fn fig_16_5() -> Admg {
    graph(&[("A", "B"), ("D", "B"), ("B", "C")], &[("D", "C")])
}

#[test]
fn figure_16_5_mags_and_the_pag() {
    let m1 = mag(&fig_16_5(), &["A", "B", "C", "D"]).unwrap();
    assert_eq!(m1.edges(), ["A --> B", "B --> C", "B <-- D", "C <-- D"]);
    // M2-M4 differ in the marks at A and D on the edges into B.
    let names = ["A", "B", "C", "D"];
    let variant = |a_ab: Mark, d_bd: Mark| {
        let mut p = Pag::new(&names);
        p.set(0, 1, a_ab, Mark::Arrow);
        p.set(3, 1, d_bd, Mark::Arrow);
        p.set(1, 2, Mark::Tail, Mark::Arrow);
        p.set(3, 2, Mark::Tail, Mark::Arrow);
        p
    };
    for (a, d) in [
        (Mark::Arrow, Mark::Tail),
        (Mark::Tail, Mark::Arrow),
        (Mark::Arrow, Mark::Arrow),
    ] {
        assert!(markov_equivalent(&m1, &variant(a, d)).unwrap());
    }
    // Reversing B -> C into C -> B makes ⟨A, B, C⟩ an unshielded collider,
    // so the class changes.
    let mut other = variant(Mark::Tail, Mark::Tail);
    other.set(1, 2, Mark::Arrow, Mark::Tail);
    assert!(!markov_equivalent(&m1, &other).unwrap());

    // Ex. 16.7 and the visibility of Lemma 16.1.4.
    let pag = fci(&names, &mut GraphOracle::new(fig_16_5())).unwrap();
    assert_eq!(pag.edges(), ["A o-> B", "B --> C", "B <-o D", "C <-- D"]);
    assert!(pag.is_visible("B", "C").unwrap());
    assert!(!pag.is_visible("D", "C").unwrap());
    assert!(!m1.is_visible("D", "C").unwrap());
    // In M3 (Fig. 16.5(e)) the collider path A *-> B <-> D makes D -> C
    // visible.
    let m3 = variant(Mark::Tail, Mark::Arrow);
    assert!(m3.is_visible("D", "C").unwrap());
}

/// Fig. 16.12(a), reconstructed from Ex. 16.14: X -> Y with X <-> Y,
/// Z -> Y, and X -> W.
fn fig_16_12() -> Admg {
    graph(&[("X", "Y"), ("Z", "Y"), ("X", "W")], &[("X", "Y")])
}

#[test]
fn example_16_14_i_fci() {
    let names = ["W", "X", "Y", "Z"];
    let targets: [&[&str]; 2] = [&[], &["X"]];
    let mut oracle = AugmentedOracle::new(fig_16_12(), &targets).unwrap();
    let pag = i_fci(&names, &targets, &mut oracle).unwrap();
    // X causes W and Y (the tail at X by R12), not Z, and F1 -> Y records
    // the confounding of X and Y.
    assert_eq!(
        pag.edges(),
        ["F1 --> X", "F1 --> Y", "W <-- X", "X --> Y", "Y <-o Z"]
    );
    // Fig. 16.12(f): from P(V) alone.
    let obs = fci(&names, &mut GraphOracle::new(fig_16_12())).unwrap();
    assert_eq!(obs.edges(), ["W o-o X", "X o-> Y", "Y <-o Z"]);
    // The invariances behind it: P(Z) = P_X(Z), P(W | X) = P_X(W | X).
    assert!(oracle.invariant(0, 1, "Z", &[]).unwrap());
    assert!(oracle.invariant(0, 1, "W", &["X"]).unwrap());
    assert!(!oracle.invariant(0, 1, "Y", &["X"]).unwrap());
}

#[test]
fn figure_16_16_known_versus_unknown_targets() {
    let g = graph(&[("X", "Y")], &[("X", "Y")]);
    let names = ["X", "Y"];
    let targets: [&[&str]; 2] = [&[], &["X"]];
    let mut oracle = AugmentedOracle::new(g.clone(), &targets).unwrap();
    assert_eq!(
        psi_fci(&names, &mut oracle).unwrap().edges(),
        ["F1 --> X", "F1 --> Y", "X o-o Y"]
    );
    assert_eq!(
        i_fci(&names, &targets, &mut oracle).unwrap().edges(),
        ["F1 --> X", "F1 --> Y", "X --> Y"]
    );
    assert_eq!(
        fci(&names, &mut GraphOracle::new(g)).unwrap().edges(),
        ["X o-o Y"]
    );
}

/// Fig. 16.14(a), reconstructed from Ex. 16.24: Z -> X -> Y, Z -> Y,
/// X <-> W and Y -> W, with targets ⟨∅, {X}⟩.
#[test]
fn example_16_24_psi_fci() {
    let g = graph(
        &[("Z", "X"), ("X", "Y"), ("Z", "Y"), ("Y", "W")],
        &[("X", "W")],
    );
    let names = ["W", "X", "Y", "Z"];
    let mut oracle = AugmentedOracle::new(g.clone(), &[&[], &["X"]]).unwrap();
    // F1 and Y are separated by {X, Z}: P1(Y | X, Z) = P2(Y | X, Z).
    assert!(oracle.invariant(0, 1, "Y", &["X", "Z"]).unwrap());
    let pag = psi_fci(&names, &mut oracle).unwrap();
    // Fig. 16.15(c): edges out of F1, X -> Y by R1, the arrowhead of
    // X o-> W, and Z -> Y by R4. Z o-> W is there too: X is an ancestor of
    // W (which F1's adjacency to W needs), so ⟨Z, X, U, W⟩ is an inducing
    // path. The book's narration does not mention that edge.
    assert_eq!(
        pag.edges(),
        [
            "F1 --> W", "F1 --> X", "W <-o X", "W <-o Y", "W <-o Z", "X --> Y", "X <-o Z",
            "Y <-- Z"
        ]
    );
    // The Ψ-PAG is sound for the true Ψ-MAG.
    let (aug, _) = augment(&g, &[&[], &["X"]], false).unwrap();
    assert_sound(&pag, &mag(&aug, &aug.variables()).unwrap());
}

#[test]
fn the_invariances_of_examples_16_9_and_16_11() {
    // Ex. 16.9: mechanisms matter in the symmetric difference.
    let g = graph(&[("X", "Y"), ("Y", "Z")], &[]);
    let (_, f) = augment(&g, &[&["X:1", "Y", "Z"], &["X:2", "Y"]], true).unwrap();
    assert_eq!(f[0].targets, ["X", "Z"]);
    // Ex. 16.11: G = {C <-> A, A -> B, C <-> B}, P_{A,B}(B | A) = P_{B,C}(B | A).
    let g = graph(&[("A", "B")], &[("C", "A"), ("C", "B")]);
    let mut oracle = AugmentedOracle::new(g, &[&["A", "B"], &["B", "C"]]).unwrap();
    assert!(oracle.invariant(0, 1, "B", &["A"]).unwrap());
    // Ex. 16.17: Y -> X under ⟨∅, {X, Y}⟩ and ⟨{X}, {Y}⟩.
    let g = graph(&[("Y", "X")], &[]);
    assert!(psi_markov_equivalent(&g, &[&[], &["X", "Y"]], &g, &[&["X"], &["Y"]]).unwrap());
}

// ---------------------------------------------------------------------------
// Data.

fn eq(name: &str, e: E) -> (String, E) {
    (name.to_string(), e)
}

fn bern(name: &str, p: f64) -> (String, Dist) {
    (name.to_string(), Dist::Bernoulli(p))
}

fn v(name: &str) -> E {
    expr::var(name)
}

/// The exact table of `vars` in world `w` of `m`.
fn table(m: &Model, w: crate::scm::WorldId, vars: &[&str]) -> Dataset {
    let t = m.world(w).probability_table(Some(vars)).unwrap();
    let rows: Vec<Vec<String>> = t
        .rows
        .iter()
        .map(|(r, _)| r.iter().map(|x| x.as_int().to_string()).collect())
        .collect();
    let weights = t.rows.iter().map(|(_, p)| *p).collect();
    Dataset::from_records(vars.iter().copied(), rows)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

#[test]
fn fci_and_i_fci_from_exact_tables() {
    // A faithful model of Fig. 16.5(a); L is the latent behind D <-> C.
    let m = Model::new(
        [
            eq("A", v("UA")),
            eq("D", expr::or([v("L"), v("UD")])),
            eq("B", expr::or([expr::and([v("A"), v("D")]), v("UB")])),
            eq(
                "C",
                expr::or([expr::and([v("B"), v("UC1")]), expr::and([v("L"), v("UC2")])]),
            ),
        ],
        [
            bern("UA", 0.3),
            bern("L", 0.4),
            bern("UD", 0.2),
            bern("UB", 0.15),
            bern("UC1", 0.8),
            bern("UC2", 0.5),
        ],
    )
    .unwrap();
    let names = ["A", "B", "C", "D"];
    let p = table(&m, m.observational(), &names);
    let pag = fci(&names, &mut DataTests::exact(&[&p], 1e-9)).unwrap();
    assert_eq!(pag, fci(&names, &mut GraphOracle::new(fig_16_5())).unwrap());

    // Fig. 16.12's model, observed and under a soft intervention on X that
    // keeps its parent L.
    let mut m = Model::new(
        [
            eq("X", expr::or([v("L"), v("UX")])),
            eq("Z", v("UZ")),
            eq("W", expr::xor(v("X"), v("UW"))),
            eq(
                "Y",
                expr::or([
                    expr::and([v("X"), v("UY1")]),
                    expr::and([v("L"), v("UY2")]),
                    expr::and([v("Z"), v("UY3")]),
                ]),
            ),
        ],
        [
            bern("L", 0.5),
            bern("UX", 0.3),
            bern("UZ", 0.4),
            bern("UW", 0.2),
            bern("UY1", 0.7),
            bern("UY2", 0.4),
            bern("UY3", 0.6),
        ],
    )
    .unwrap();
    let names = ["W", "X", "Y", "Z"];
    let p0 = table(&m, m.observational(), &names);
    let soft = m
        .intervene_with(&[("X", expr::and([v("L"), v("UX")]))])
        .unwrap();
    let p1 = table(&m, soft, &names);
    let targets: [&[&str]; 2] = [&[], &["X"]];
    let mut tests = DataTests::exact(&[&p0, &p1], 1e-9);
    let learned = i_fci(&names, &targets, &mut tests).unwrap();
    let mut oracle = AugmentedOracle::new(fig_16_12(), &targets).unwrap();
    assert_eq!(learned, i_fci(&names, &targets, &mut oracle).unwrap());
}

// ---------------------------------------------------------------------------
// Random diagrams.

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const NAMES: [&str; 5] = ["A", "B", "C", "D", "E"];

/// A random acyclic diagram over A-E (topological order), with bidirected
/// edges unless `markovian`.
fn random_graph(seed: u64, markovian: bool) -> Admg {
    let mut r = seed;
    let mut next = || {
        r = splitmix(r);
        r % 100
    };
    let mut g = Admg::new();
    for n in NAMES {
        g.add_node(n);
    }
    for (j, b) in NAMES.iter().enumerate() {
        for a in &NAMES[..j] {
            if next() < 35 {
                g.directed(a, b);
            }
            if !markovian && next() < 15 {
                g.bidirected(a, b);
            }
        }
    }
    g
}

fn random_targets(seed: u64) -> Vec<Vec<&'static str>> {
    let mut r = splitmix(seed ^ 0x7a6);
    let mut out = vec![Vec::new()];
    for _ in 0..(1 + seed % 2) {
        r = splitmix(r);
        let t: Vec<&str> = (0..5)
            .filter(|i| (r >> (i * 3)).is_multiple_of(4))
            .map(|i| NAMES[i])
            .collect();
        out.push(if t.is_empty() {
            vec![NAMES[(r % 5) as usize]]
        } else {
            t
        });
    }
    out
}

/// The learned graph has the true adjacencies, and each of its non-circle
/// marks is the true one.
fn assert_sound(learned: &Pag, truth: &Pag) {
    let names: Vec<&str> = truth.names().iter().map(String::as_str).collect();
    for a in &names {
        for b in &names {
            if a >= b {
                continue;
            }
            let (l, t) = (learned.edge(a, b).unwrap(), truth.edge(a, b).unwrap());
            assert_eq!(l.is_some(), t.is_some(), "{a}-{b}: {learned} vs {truth}");
            if let (Some((la, lb)), Some((ta, tb))) = (l, t) {
                assert!(
                    la == Mark::Circle || la == ta,
                    "{a}-{b}: {learned} vs {truth}"
                );
                assert!(
                    lb == Mark::Circle || lb == tb,
                    "{a}-{b}: {learned} vs {truth}"
                );
            }
        }
    }
}

#[test]
fn fci_i_fci_and_psi_fci_are_sound_on_random_diagrams() {
    let mut circles = 0;
    for seed in 0..40u64 {
        let g = random_graph(splitmix(seed), false);
        let truth = mag(&g, &NAMES).unwrap();
        let pag = fci(&NAMES, &mut GraphOracle::new(g.clone())).unwrap();
        assert_sound(&pag, &truth);
        circles += pag.edges().iter().filter(|e| e.contains('o')).count();

        let targets = random_targets(seed);
        let t: Vec<&[&str]> = targets.iter().map(Vec::as_slice).collect();
        let mut oracle = AugmentedOracle::new(g.clone(), &t).unwrap();
        let (aug, _) = augment(&g, &t, true).unwrap();
        assert_sound(
            &i_fci(&NAMES, &t, &mut oracle).unwrap(),
            &mag(&aug, &aug.variables()).unwrap(),
        );
        let (aug, _) = augment(&g, &t, false).unwrap();
        assert_sound(
            &psi_fci(&NAMES, &mut oracle).unwrap(),
            &mag(&aug, &aug.variables()).unwrap(),
        );
    }
    assert!(circles > 20, "the class should leave some marks open");
}

#[test]
fn psi_pc_is_sound_on_markovian_diagrams() {
    for seed in 0..40u64 {
        let g = random_graph(splitmix(seed ^ 0x9c), true);
        let targets = random_targets(seed);
        let t: Vec<&[&str]> = targets.iter().map(Vec::as_slice).collect();
        let mut oracle = AugmentedOracle::new(g.clone(), &t).unwrap();
        // Corol. 16.2.13: without latents the Ψ-MAG is the augmented diagram.
        let (aug, _) = augment(&g, &t, false).unwrap();
        let truth = Pag::from_admg(&aug);
        assert_eq!(truth, mag(&aug, &aug.variables()).unwrap());
        assert_sound(&psi_pc(&NAMES, &mut oracle).unwrap(), &truth);
    }
}

/// Prop. 16.1.1 against its definition: two MAGs are Markov equivalent
/// exactly when they imply the same m-separations.
#[test]
fn markov_equivalence_matches_m_separation() {
    let as_admg = |m: &Pag| {
        let mut g = Admg::new();
        for n in m.names() {
            g.add_node(n);
        }
        for e in m.edges() {
            let parts: Vec<&str> = e.split(' ').collect();
            match parts[1] {
                "-->" => g.directed(parts[0], parts[2]),
                "<--" => g.directed(parts[2], parts[0]),
                _ => g.bidirected(parts[0], parts[2]),
            }
        }
        g
    };
    let separations = |g: &Admg| {
        let mut out = Vec::new();
        for (i, a) in NAMES.iter().enumerate() {
            for b in &NAMES[i + 1..] {
                let rest: Vec<usize> = (0..5)
                    .filter(|k| NAMES[*k] != *a && NAMES[*k] != *b)
                    .collect();
                for z in fci::subsets(&rest) {
                    let zs: Vec<&str> = z.iter().map(|&k| NAMES[k]).collect();
                    out.push(g.is_d_separated(&[a], &[b], &zs).unwrap());
                }
            }
        }
        out
    };
    let (mut same, mut different) = (0, 0);
    for seed in 0..150u64 {
        let g1 = random_graph(splitmix(seed), false);
        // A second diagram from a nearby seed often shares the skeleton.
        let m1 = mag(&g1, &NAMES).unwrap();
        let pag = fci(&NAMES, &mut GraphOracle::new(g1.clone())).unwrap();
        // Another member of the class or a near miss: flip one open mark.
        let mut m2 = m1.clone();
        let mut flipped = false;
        for e in pag.edges() {
            if flipped {
                break;
            }
            let parts: Vec<&str> = e.split(' ').collect();
            let (i, j) = (m2.index(parts[0]).unwrap(), m2.index(parts[2]).unwrap());
            if parts[1].starts_with('o') {
                let new = if m2.at(j, i) == Some(Mark::Tail) {
                    Mark::Arrow
                } else {
                    Mark::Tail
                };
                if seed % 3 != 0 || m2.at(i, j) == Some(Mark::Arrow) {
                    m2.set_at(j, i, new);
                    flipped = true;
                }
            }
        }
        if !flipped {
            continue;
        }
        let expected = separations(&as_admg(&m1)) == separations(&as_admg(&m2));
        assert_eq!(
            markov_equivalent(&m1, &m2).unwrap(),
            expected,
            "seed {seed}: {m1} vs {m2}"
        );
        if expected { same += 1 } else { different += 1 }
    }
    assert!(
        same >= 10 && different >= 5,
        "{same} equivalent, {different} not"
    );
}

/// FCI with G-tests on 20 000 samples of the Fig. 16.5 model recovers the
/// oracle's PAG.
#[test]
fn fci_from_samples_with_g_tests() {
    let m = Model::new(
        [
            eq("A", v("UA")),
            eq("D", expr::or([v("L"), v("UD")])),
            eq("B", expr::or([expr::and([v("A"), v("D")]), v("UB")])),
            eq(
                "C",
                expr::or([expr::and([v("B"), v("UC1")]), expr::and([v("L"), v("UC2")])]),
            ),
        ],
        [
            bern("UA", 0.3),
            bern("L", 0.4),
            bern("UD", 0.2),
            bern("UB", 0.15),
            bern("UC1", 0.8),
            bern("UC2", 0.5),
        ],
    )
    .unwrap();
    let mut state = 7u64;
    let names = ["A", "B", "C", "D"];
    let all = m.world(m.observational()).world().variables().to_vec();
    let at: Vec<usize> = names
        .iter()
        .map(|n| all.iter().position(|a| a == n).unwrap())
        .collect();
    let rows: Vec<Vec<String>> = m
        .world(m.observational())
        .sample(20_000, || {
            state = splitmix(state);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .unwrap()
        .into_iter()
        .map(|r| at.iter().map(|&i| r[i].as_int().to_string()).collect())
        .collect();
    let data = Dataset::from_records(names, rows).unwrap();
    let pag = fci(&names, &mut DataTests::g_test(&[&data], 0.01)).unwrap();
    assert_eq!(pag, fci(&names, &mut GraphOracle::new(fig_16_5())).unwrap());
}

// ---------------------------------------------------------------------------
// Chapter 17: multiple domains.

/// Fig. 17.4(a): X -> Y -> Z, the domains differing at X.
fn fig_17_4() -> Admg {
    graph(&[("X", "Y"), ("Y", "Z")], &[])
}

const S_X: [(usize, usize, &[&str]); 1] = [(0, 1, &["X"])];

#[test]
fn examples_17_12_to_17_15_augmented_selection_diagrams() {
    let children = |f: &[FNode]| -> Vec<String> { f.iter().map(|n| n.targets.join("")).collect() };
    // Ex. 17.12: observations in both domains add nothing.
    let (_, f, s) =
        augment_selection(&fig_17_4(), &S_X, &[(0, &[], true), (1, &[], true)]).unwrap();
    assert!(f.is_empty());
    assert_eq!(s[0].targets, ["X"]);
    // Ex. 17.13: ⟨{}¹, {Y}²⟩.
    let (_, f, s) =
        augment_selection(&fig_17_4(), &S_X, &[(0, &[], true), (1, &["Y"], true)]).unwrap();
    assert_eq!(children(&f), ["Y"]);
    assert_eq!(s[0].targets, ["X", "Y"]);
    // Ex. 17.14: ⟨{Z}¹, {Y}²⟩.
    let (_, f, s) =
        augment_selection(&fig_17_4(), &S_X, &[(0, &["Z"], true), (1, &["Y"], true)]).unwrap();
    assert_eq!(children(&f), ["YZ"]);
    assert_eq!(s[0].targets, ["X", "Y", "Z"]);
    // Ex. 17.15: ⟨{}¹, {Y}², {Z}²⟩: three F-nodes, no extra S-node edges.
    let setup: [Regime<'_>; 3] = [(0, &[], true), (1, &["Y"], true), (1, &["Z"], true)];
    let (_, f, s) = augment_selection(&fig_17_4(), &S_X, &setup).unwrap();
    assert_eq!(children(&f), ["Y", "Z", "YZ"]);
    assert_eq!(s[0].targets, ["X"]);
    // A domain without data is rejected.
    assert!(augment_selection(&fig_17_4(), &S_X, &[(1, &[], true)]).is_err());
}

#[test]
fn examples_17_16_17_and_17_20_invariances_and_equivalence() {
    // Fig. 17.5(a), reconstructed: X -> Y -> Z, the domains differing at Y.
    let g = graph(&[("X", "Y"), ("Y", "Z")], &[]);
    let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["Y"])];
    let setup: [Regime<'_>; 3] = [(0, &[], true), (0, &["Z"], true), (1, &[], true)];
    let mut oracle = SelectionOracle::new(g.clone(), &s, &setup).unwrap();
    // Ex. 17.16: P¹(Y) = P¹_Z(Y).
    assert!(oracle.invariant(0, 1, "Y", &[]).unwrap());
    // Ex. 17.17: P¹_Z(Y | X, Z) = P²(Y | X, Z) is not implied.
    assert!(!oracle.invariant(1, 2, "Y", &["X", "Z"]).unwrap());
    // Ex. 17.20: dropping S12 -> Y changes the class.
    assert!(md_markov_equivalent(&g, &s, &setup, &g, &s, &setup).unwrap());
    let none: [(usize, usize, &[&str]); 0] = [];
    assert!(!md_markov_equivalent(&g, &s, &setup, &g, &none, &setup).unwrap());
}

/// Thm 17.4.1: with observations alone, the MD-Markov invariances of two
/// domains are the Ψ-Markov invariances of interventions on the S-node's
/// children.
#[test]
fn theorem_17_4_1_domains_as_interventions() {
    for seed in 0..30u64 {
        let g = random_graph(splitmix(seed ^ 0x17), false);
        let children: Vec<&str> = NAMES
            .iter()
            .copied()
            .filter(|_| false)
            .chain(random_targets(seed)[1].iter().copied())
            .collect();
        let s: [(usize, usize, &[&str]); 1] = [(0, 1, &children)];
        let mut md =
            SelectionOracle::new(g.clone(), &s, &[(0, &[], true), (1, &[], true)]).unwrap();
        let mut psi = AugmentedOracle::new(g.clone(), &[&[], &children]).unwrap();
        for y in NAMES {
            let rest: Vec<usize> = (0..5).filter(|k| NAMES[*k] != y).collect();
            for w in fci::subsets(&rest) {
                let ws: Vec<&str> = w.iter().map(|&k| NAMES[k]).collect();
                assert_eq!(
                    md.invariant(0, 1, y, &ws).unwrap(),
                    psi.invariant(0, 1, y, &ws).unwrap(),
                    "seed {seed}: {y} | {ws:?}"
                );
            }
        }
    }
}

/// Ex. 17.27 (Fig. 17.11): pooling the domains' observations (MDIX) makes
/// I-FCI orient Y -> X, which is false; MD-FCI learns X -> Y.
#[test]
fn example_17_27_pooling_domains_is_unsound() {
    let model = |p_x: f64| {
        Model::new(
            [
                eq("X", v("UX")),
                eq("Y", expr::or([expr::and([v("X"), v("UY1")]), v("UY2")])),
            ],
            [bern("UX", p_x), bern("UY1", 0.7), bern("UY2", 0.2)],
        )
        .unwrap()
    };
    let names = ["X", "Y"];
    let m1 = model(0.3);
    let mut m1y = model(0.3);
    let soft = m1y
        .intervene_with(&[(
            "Y",
            expr::or([expr::and([v("X"), v("UY1")]), expr::not(v("UY2"))]),
        )])
        .unwrap();
    let m2 = model(0.7);
    let p1 = table(&m1, m1.observational(), &names);
    let p1y = table(&m1y, soft, &names);
    let p2 = table(&m2, m2.observational(), &names);
    // The pooled observational data: half from each domain.
    let rows: Vec<Vec<String>> = [&p1, &p2]
        .iter()
        .flat_map(|d| {
            (0..d.len()).map(|r| {
                names
                    .iter()
                    .map(|n| d.value(n, r).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let weights: Vec<f64> = [&p1, &p2]
        .iter()
        .flat_map(|d| d.weights().unwrap().iter().map(|w| w / 2.0))
        .collect();
    let pooled = Dataset::from_records(names, rows)
        .unwrap()
        .with_weights(weights)
        .unwrap();

    let wrong = i_fci(
        &names,
        &[&[], &["Y"]],
        &mut DataTests::exact(&[&pooled, &p1y], 1e-9),
    )
    .unwrap();
    assert_eq!(
        wrong.edge("X", "Y").unwrap(),
        Some((Mark::Arrow, Mark::Tail))
    );

    let setup: [Regime<'_>; 3] = [(0, &[], true), (0, &["Y"], true), (1, &[], true)];
    let right = md_fci(
        &names,
        &setup,
        &mut DataTests::exact(&[&p1, &p1y, &p2], 1e-9),
    )
    .unwrap();
    assert_eq!(
        right.edge("X", "Y").unwrap(),
        Some((Mark::Tail, Mark::Arrow))
    );
    assert_eq!(
        right.edge("S12", "X").unwrap(),
        Some((Mark::Tail, Mark::Arrow))
    );
    assert_eq!(right.edge("S12", "Y").unwrap(), None);
}

/// Random selection diagrams and regimes: MD-FCI and MD-PC are sound for
/// the graphs they learn (see `md_fci`, `md_pc`).
#[test]
fn md_fci_and_md_pc_are_sound_on_random_selection_diagrams() {
    for seed in 0..120u64 {
        for markovian in [false, true] {
            let g = random_graph(splitmix(seed ^ 0x170 ^ u64::from(markovian)), markovian);
            let targets = random_targets(seed);
            let s_children = random_targets(seed ^ 0x55)[1].clone();
            let s: [(usize, usize, &[&str]); 1] = [(0, 1, &s_children)];
            let second: &[&str] = if seed % 3 == 0 { &[] } else { &targets[1] };
            let setup: Vec<Regime<'_>> = vec![
                (0, &[], true),
                (0, &targets[targets.len() - 1], seed % 4 != 1),
                (1, second, true),
            ];
            let mut oracle = SelectionOracle::new(g.clone(), &s, &setup).unwrap();
            let truth = multidomain::learning_target(&g, &s, &setup, markovian).unwrap();
            let learned = if markovian {
                md_pc(&NAMES, &setup, &mut oracle).unwrap()
            } else {
                md_fci(&NAMES, &setup, &mut oracle).unwrap()
            };
            assert_sound(&learned, &mag(&truth, &truth.variables()).unwrap());
        }
    }
}
