//! Tests for chapter 18: the book's ΔV/ΔQ sets and CRID outputs.

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

/// Fig. 18.12(a), reconstructed: V1 -> V2 -> V3 with V1 <-> V2 (V2 shares
/// V1's c-component, Ex. 18.22), the domains differing at V3.
/// S-node children per pair of domains.
type SEdges = [(usize, usize, &'static [&'static str]); 1];

fn fig_18_12a() -> (Admg, SEdges) {
    (
        graph(&[("V1", "V2"), ("V2", "V3")], &[("V1", "V2")]),
        [(0, 1, &["V3"])],
    )
}

/// Fig. 18.12(c), reconstructed: V1 -> V2 -> V3 -> V4 with V2 <-> V3.
fn fig_18_12c() -> Admg {
    graph(&[("V1", "V2"), ("V2", "V3"), ("V3", "V4")], &[("V2", "V3")])
}

#[test]
fn the_changed_and_delta_q_sets_of_examples_18_14_to_18_23() {
    // Ex. 18.14: V2 -> V1, V2 -> V3; a soft intervention on V3.
    let g = graph(&[("V2", "V1"), ("V2", "V3")], &[]);
    assert_eq!(delta_v(&g, &[], (0, &[]), (0, &["V3"])).unwrap(), ["V3"]);
    assert_eq!(
        delta_q(&g, &[], (0, &[]), (0, &["V3"])).unwrap(),
        ["V2", "V3"]
    );
    // Ex. 18.16: both comparisons with the baseline change V3 only.
    let (g, s) = fig_18_12a();
    assert_eq!(delta_v(&g, &s, (1, &[]), (0, &[])).unwrap(), ["V3"]);
    assert_eq!(
        delta_q(&g, &s, (0, &[]), (1, &["V3"])).unwrap(),
        ["V2", "V3"]
    );
    // A shared mechanism overrides the domain change.
    assert!(
        delta_v(&g, &s, (0, &["V3:1"]), (1, &["V3:1"]))
            .unwrap()
            .is_empty()
    );
    assert_eq!(delta_v(&g, &s, (0, &["V3"]), (1, &["V3"])).unwrap(), ["V3"]);
    // Ex. 18.17: the collider with S-nodes at V1 and V2.
    let col = graph(&[("V1", "V2"), ("V3", "V2")], &[]);
    let s2: [(usize, usize, &[&str]); 1] = [(0, 1, &["V1", "V2"])];
    assert_eq!(
        delta_v(&col, &s2, (1, &[]), (0, &[])).unwrap(),
        ["V1", "V2"]
    );
    // Exs. 18.18 and 18.23 on Fig. 18.12(c).
    let g = fig_18_12c();
    assert_eq!(
        delta_q(&g, &[], (0, &[]), (0, &["V2"])).unwrap(),
        ["V1", "V2", "V3"]
    );
    assert_eq!(
        delta_q(&g, &[], (0, &[]), (0, &["V3"])).unwrap(),
        ["V1", "V2", "V3"]
    );
    assert_eq!(delta_q(&g, &[], (0, &[]), (0, &["V1"])).unwrap(), ["V1"]);
    // Ex. 18.19: two homogeneous domains change nothing.
    assert!(delta_q(&g, &[], (0, &[]), (1, &[])).unwrap().is_empty());
}

#[test]
fn example_18_38_crid_step_by_step() {
    let (g, s) = fig_18_12a();
    // Without the marginal intervention on V2, only Prop. 18.3.3 applies:
    // {V2, V3} is ID w.r.t. V1 (Ex. 18.22).
    let cdm = crid(&g, &s, &[(0, &[]), (1, &[]), (1, &["V3"])]).unwrap();
    assert!(cdm.identifiable(&["V2", "V3"], &["V1"]).unwrap());
    assert!(!cdm.identifiable(&["V1"], &["V2"]).unwrap());
    // Ex. 18.26/18.38: the marginal intervention on V2 makes V1 ID w.r.t.
    // {V2, V3} (Prop. 18.3.6), and then one comparison removes V3 -> V̂2.
    let cdm = crid(&g, &s, &[(0, &[]), (1, &[]), (1, &["V3"]), (0, &["V2!"])]).unwrap();
    assert_eq!(
        cdm.edges(),
        ["V1^ <- {V1}", "V2^ <- {V2}", "V3^ <- {V2, V3}"]
    );
}

#[test]
fn example_18_24_within_delta_q_sets() {
    let (g, _) = fig_18_12a();
    let v1: &[&str] = &["V1"];
    let v23: &[&str] = &["V2", "V3"];
    // σ(1) = {}, σ(2-5) = {V1} × 4, σ(6-9) = {V2, V3} × 4: V1 and V3 are ID
    // w.r.t. each other.
    let mut regimes: Vec<(usize, &[&str])> = vec![(0, &[])];
    regimes.extend(std::iter::repeat_n((0, v1), 4));
    regimes.extend(std::iter::repeat_n((0, v23), 4));
    let cdm = crid(&g, &[], &regimes).unwrap();
    assert!(cdm.identifiable(&["V1"], &["V3"]).unwrap());
    assert!(cdm.identifiable(&["V3"], &["V1"]).unwrap());
    // The book's second set, {V1} × 6 and {V2, V3} × 2, fails (3′) with the
    // baseline σ(1); but with a {V2, V3} distribution as the baseline all
    // eight ΔQ sets are V, and CRID tries every baseline.
    let mut regimes: Vec<(usize, &[&str])> = vec![(0, &[])];
    regimes.extend(std::iter::repeat_n((0, v1), 6));
    regimes.extend(std::iter::repeat_n((0, v23), 2));
    let cdm = crid(&g, &[], &regimes).unwrap();
    assert!(cdm.identifiable(&["V3"], &["V1"]).unwrap());
    // With one {V2, V3} distribution no baseline works: only V1 is ID
    // w.r.t. V3 (Prop. 18.3.3).
    regimes.pop();
    let cdm = crid(&g, &[], &regimes).unwrap();
    assert!(cdm.identifiable(&["V1"], &["V3"]).unwrap());
    assert!(!cdm.identifiable(&["V3"], &["V1"]).unwrap());
}

#[test]
fn example_18_25_five_heterogeneous_domains() {
    // The collider V1 -> V2 <- V3; every pair of five domains differs at V1
    // and V3; observations in each.
    let g = graph(&[("V1", "V2"), ("V3", "V2")], &[]);
    let mut s: Vec<(usize, usize, &[&str])> = Vec::new();
    for i in 0..5 {
        for j in i + 1..5 {
            s.push((i, j, &["V1", "V3"]));
        }
    }
    let regimes: Vec<(usize, &[&str])> = (0..5).map(|d| (d, &[] as &[&str])).collect();
    let cdm = crid(&g, &s, &regimes).unwrap();
    assert!(cdm.identifiable(&["V1"], &["V3"]).unwrap());
    assert!(cdm.identifiable(&["V3"], &["V1"]).unwrap());
    // Three domains are not enough for Corol. 18.3.5 (L = 2 < 4).
    let cdm = crid(&g, &s, &regimes[..3]).unwrap();
    assert!(!cdm.identifiable(&["V1"], &["V3"]).unwrap());
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Two properties any sound CRID has: a factor's own representation keeps
/// its edge, and more distributions never put an edge back.
#[test]
fn crid_is_monotone_and_keeps_the_diagonal() {
    const NAMES: [&str; 4] = ["V1", "V2", "V3", "V4"];
    let targets: [&[&str]; 8] = [
        &[],
        &["V1"],
        &["V2"],
        &["V3!"],
        &["V4"],
        &["V1", "V2"],
        &["V3"],
        &["V2!"],
    ];
    for seed in 0..25u64 {
        let mut r = splitmix(seed);
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
                if next() < 40 {
                    g.directed(a, b);
                }
                if next() < 15 {
                    g.bidirected(a, b);
                }
            }
        }
        let regimes: Vec<(usize, &[&str])> = (0..7)
            .map(|_| ((next() % 2) as usize, targets[(next() % 8) as usize]))
            .collect();
        let s: [(usize, usize, &[&str]); 1] = [(0, 1, &["V2"])];
        let few = crid(&g, &s, &regimes[..4]).unwrap();
        let many = crid(&g, &s, &regimes).unwrap();
        for a in NAMES {
            assert!(many.depends(a, a).unwrap(), "seed {seed}: {many}");
            for b in NAMES {
                if many.depends(a, b).unwrap() {
                    assert!(few.depends(a, b).unwrap(), "seed {seed}: {few} vs {many}");
                }
            }
        }
    }
}
