use super::*;

const EPS: f64 = 1e-9;

fn shuffled(rng: &mut SplitMix64, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..k).collect();
    for i in (1..k).rev() {
        v.swap(i, rng.below(i + 1));
    }
    v
}

fn probs(rng: &mut SplitMix64, k: usize) -> Vec<f64> {
    let v: Vec<f64> = (0..k).map(|_| 0.2 + rng.unit()).collect();
    let s: f64 = v.iter().sum();
    v.into_iter().map(|p| p / s).collect()
}

/// A random Markovian SFM with two confounders, two mediators (with an edge
/// between them) and a real-valued outcome, each with its own finite noise.
///
/// `X` and both mediators are random *permutations* of their noise in every
/// parent configuration, so every level stays reachable under every parent
/// value. Positivity then holds exactly while the distributions still differ.
struct RandomSfm {
    p_z1: Vec<f64>,
    z1: Vec<usize>,
    p_z2: Vec<f64>,
    z2: Vec<Vec<usize>>,
    p_x: Vec<f64>,
    x: Vec<Vec<usize>>,
    p_w1: Vec<f64>,
    w1: Vec<Vec<usize>>,
    p_w2: Vec<f64>,
    w2: Vec<Vec<usize>>,
    p_y: Vec<f64>,
    y: Vec<f64>,
}

impl RandomSfm {
    fn new(rng: &mut SplitMix64) -> Self {
        let z1 = (0..3).map(|_| rng.below(2)).collect();
        let z2 = (0..2).map(|_| (0..3).map(|_| rng.below(2)).collect()).collect();
        let x = (0..4).map(|_| shuffled(rng, 2)).collect();
        let w1 = (0..8).map(|_| shuffled(rng, 3)).collect();
        let w2 = (0..24).map(|_| shuffled(rng, 2)).collect();
        let y = (0..2 * 4 * 3 * 2 * 2).map(|_| rng.unit()).collect();
        Self {
            p_z1: probs(rng, 3),
            z1,
            p_z2: probs(rng, 3),
            z2,
            p_x: probs(rng, 2),
            x,
            p_w1: probs(rng, 3),
            w1,
            p_w2: probs(rng, 2),
            w2,
            p_y: probs(rng, 2),
            y,
        }
    }

    /// Mediators under `do(X = x)` for confounder index `zi`.
    fn w(&self, x: usize, zi: usize, uw1: usize, uw2: usize) -> (usize, usize) {
        let w1 = self.w1[x * 4 + zi][uw1];
        let w2 = self.w2[(x * 4 + zi) * 3 + w1][uw2];
        (w1, w2)
    }

    fn y(&self, x: usize, zi: usize, (w1, w2): (usize, usize), uy: usize) -> f64 {
        self.y[(((x * 4 + zi) * 3 + w1) * 2 + w2) * 2 + uy]
    }

    /// Exact population as a weighted dataset (one row per noise
    /// configuration), together with the four counterfactual means computed
    /// by brute force over the noise, for `x0 = 0`, `x1 = 1`.
    fn population(&self) -> (Dataset, [f64; 5]) {
        let mut records = Vec::new();
        let mut weights = Vec::new();
        let (mut p, mut nested, mut cross, mut fact) = ([0.0; 2], 0.0, 0.0, [0.0; 2]);
        let mut reverse = 0.0;
        for (uz1, &pz1) in self.p_z1.iter().enumerate() {
            for (uz2, &pz2) in self.p_z2.iter().enumerate() {
                for (ux, &px) in self.p_x.iter().enumerate() {
                    for (uw1, &pw1) in self.p_w1.iter().enumerate() {
                        for (uw2, &pw2) in self.p_w2.iter().enumerate() {
                            for (uy, &py) in self.p_y.iter().enumerate() {
                                let prob = pz1 * pz2 * px * pw1 * pw2 * py;
                                let z1 = self.z1[uz1];
                                let z2 = self.z2[z1][uz2];
                                let zi = z1 * 2 + z2;
                                let x = self.x[zi][ux];
                                let w = self.w(x, zi, uw1, uw2);
                                let y = self.y(x, zi, w, uy);
                                p[x] += prob;
                                fact[x] += prob * y;
                                if x == 0 {
                                    let w_x0 = w;
                                    let w_x1 = self.w(1, zi, uw1, uw2);
                                    nested += prob * self.y(1, zi, w_x0, uy);
                                    cross += prob * self.y(1, zi, w_x1, uy);
                                    reverse += prob * self.y(0, zi, w_x1, uy);
                                }
                                records.push(vec![
                                    z1.to_string(),
                                    z2.to_string(),
                                    x.to_string(),
                                    w.0.to_string(),
                                    w.1.to_string(),
                                    format!("{y}"),
                                ]);
                                weights.push(prob);
                            }
                        }
                    }
                }
            }
        }
        let data = Dataset::from_records(["z1", "z2", "x", "w1", "w2", "y"], records)
            .unwrap()
            .with_weights(weights)
            .unwrap();
        let truth = [
            nested / p[0],
            fact[0] / p[0],
            cross / p[0],
            fact[1] / p[1],
            reverse / p[0],
        ];
        (data, truth)
    }

    /// Samples `n` units from the model.
    fn sample(&self, rng: &mut SplitMix64, n: usize) -> Dataset {
        let draw = |rng: &mut SplitMix64, p: &[f64]| {
            let u = rng.unit();
            let mut acc = 0.0;
            for (i, pi) in p.iter().enumerate() {
                acc += pi;
                if u < acc {
                    return i;
                }
            }
            p.len() - 1
        };
        let records: Vec<Vec<String>> = (0..n)
            .map(|_| {
                let z1 = self.z1[draw(rng, &self.p_z1)];
                let z2 = self.z2[z1][draw(rng, &self.p_z2)];
                let zi = z1 * 2 + z2;
                let x = self.x[zi][draw(rng, &self.p_x)];
                let (uw1, uw2) = (draw(rng, &self.p_w1), draw(rng, &self.p_w2));
                let w = self.w(x, zi, uw1, uw2);
                let y = self.y(x, zi, w, draw(rng, &self.p_y));
                vec![
                    z1.to_string(),
                    z2.to_string(),
                    x.to_string(),
                    w.0.to_string(),
                    w.1.to_string(),
                    format!("{y}"),
                ]
            })
            .collect();
        Dataset::from_records(["z1", "z2", "x", "w1", "w2", "y"], records).unwrap()
    }
}

fn sfm() -> Sfm {
    Sfm::new("x", ["z1", "z2"], ["w1", "w2"], "y")
}

/// The estimands equal the counterfactual measures they target, computed by
/// brute force over the exogenous noise of random SCMs.
#[test]
fn estimands_match_brute_force_counterfactuals() {
    let mut rng = SplitMix64::new(7);
    for _ in 0..50 {
        let scm = RandomSfm::new(&mut rng);
        let (data, [nested, f0, cross, f1, reverse]) = scm.population();
        let m = estimate(&data, &sfm(), "0", "1").unwrap();
        assert!((m.nested - nested).abs() < EPS, "{m:?} vs {nested}");
        assert!((m.factual_x0 - f0).abs() < EPS);
        assert!((m.cross - cross).abs() < EPS);
        assert!((m.factual_x1 - f1).abs() < EPS);
        assert!((m.de - (nested - f0)).abs() < EPS);
        assert!((m.ie - (nested - cross)).abs() < EPS);
        assert!((m.se - (cross - f1)).abs() < EPS);
        assert!((m.reverse_nested - reverse).abs() < EPS);
        assert!((m.de_reverse - (reverse - cross)).abs() < EPS);
        assert!((m.ie_reverse - (reverse - f0)).abs() < EPS);
    }
}

/// `TV = DE − IE − SE` holds exactly on finite samples too, not just in the
/// population.
#[test]
fn decomposition_is_exact_on_samples() {
    let mut rng = SplitMix64::new(11);
    for _ in 0..20 {
        let scm = RandomSfm::new(&mut rng);
        let data = scm.sample(&mut rng, 3000);
        let m = estimate(&data, &sfm(), "0", "1").unwrap();
        assert!(m.residual().abs() < 1e-12, "{m:?}");
        assert!(m.residual_sym().abs() < 1e-12, "{m:?}");
        // And the raw TV agrees with a direct computation.
        let (mut s, mut n) = ([0.0; 2], [0.0; 2]);
        let ys = data.levels("y").unwrap().to_vec();
        let xs = data.levels("x").unwrap().to_vec();
        let (xc, yc) = (data.column("x").unwrap(), data.column("y").unwrap());
        for i in 0..data.len() {
            let g: usize = xs[xc.codes[i] as usize].parse().unwrap();
            n[g] += 1.0;
            s[g] += ys[yc.codes[i] as usize].parse::<f64>().unwrap();
        }
        assert!((m.tv - (s[1] / n[1] - s[0] / n[0])).abs() < 1e-12);
    }
}

#[test]
fn sample_estimates_converge() {
    let mut rng = SplitMix64::new(3);
    let scm = RandomSfm::new(&mut rng);
    let (pop, _) = scm.population();
    let truth = estimate(&pop, &sfm(), "0", "1").unwrap();
    let m = estimate(&scm.sample(&mut rng, 60_000), &sfm(), "0", "1").unwrap();
    for (a, b) in [(m.de, truth.de), (m.ie, truth.ie), (m.se, truth.se)] {
        assert!((a - b).abs() < 0.03, "{m:?} vs {truth:?}");
    }
}

#[test]
fn empty_z_and_w_reduce_to_tv() {
    let data = Dataset::from_csv("x,y\na,0\na,1\nb,1\nb,1\n").unwrap();
    let m = estimate(&data, &Sfm::new("x", Vec::<String>::new(), Vec::<String>::new(), "y"), "a", "b")
        .unwrap();
    assert_eq!((m.tv, m.de, m.ie, m.se), (0.5, 0.5, 0.0, 0.0));
}

#[test]
fn positivity_violation_names_the_stratum() {
    let data = Dataset::from_csv("x,z,y\na,u,0\na,v,1\nb,u,1\n").unwrap();
    let err = estimate(&data, &Sfm::new("x", ["z"], Vec::<String>::new(), "y"), "a", "b")
        .unwrap_err();
    assert_eq!(
        err,
        FairnessError::Positivity {
            level: "b".into(),
            stratum: "z=v".into()
        }
    );
}

#[test]
fn input_errors() {
    let data = Dataset::from_csv("x,z,y\na,u,0\nb,u,yes\n").unwrap();
    let s = |z: &str, y: &str| Sfm::new("x", [z], Vec::<String>::new(), y);
    assert_eq!(
        estimate(&data, &s("q", "y"), "a", "b").unwrap_err(),
        FairnessError::UnknownColumn("q".into())
    );
    assert_eq!(
        estimate(&data, &s("y", "y"), "a", "b").unwrap_err(),
        FairnessError::OverlappingRoles("y".into())
    );
    assert_eq!(
        estimate(&data, &s("z", "y"), "a", "c").unwrap_err(),
        FairnessError::UnknownLevel {
            column: "x".into(),
            level: "c".into()
        }
    );
    assert_eq!(
        estimate(&data, &s("z", "y"), "a", "a").unwrap_err(),
        FairnessError::IdenticalContrast("a".into())
    );
    assert!(matches!(
        estimate(&data, &s("z", "y"), "a", "b").unwrap_err(),
        FairnessError::NonNumericOutcome { .. }
    ));
}

#[test]
fn rows_outside_the_contrast_are_ignored() {
    let a = Dataset::from_csv("x,y\na,0\nb,1\n").unwrap();
    let b = Dataset::from_csv("x,y\na,0\nc,7\nb,1\n").unwrap();
    let s = Sfm::new("x", Vec::<String>::new(), Vec::<String>::new(), "y");
    assert_eq!(estimate(&a, &s, "a", "b"), estimate(&b, &s, "a", "b"));
}

#[test]
fn weights_equal_duplicated_rows() {
    let csv = "x,z,y\na,u,0\na,v,1\nb,u,1\nb,v,0\nb,v,1\n";
    let weighted = Dataset::from_csv(csv)
        .unwrap()
        .with_weights(vec![2.0, 1.0, 1.0, 3.0, 1.0])
        .unwrap();
    let dup = Dataset::from_csv(
        "x,z,y\na,u,0\na,u,0\na,v,1\nb,u,1\nb,v,0\nb,v,0\nb,v,0\nb,v,1\n",
    )
    .unwrap();
    let s = Sfm::new("x", ["z"], Vec::<String>::new(), "y");
    let (m1, m2) = (
        estimate(&weighted, &s, "a", "b").unwrap(),
        estimate(&dup, &s, "a", "b").unwrap(),
    );
    assert!((m1.se - m2.se).abs() < 1e-12 && (m1.tv - m2.tv).abs() < 1e-12);
    assert!(Dataset::from_csv(csv).unwrap().with_weights(vec![1.0]).is_err());
}

#[test]
fn csv_quoting_and_layout() {
    let data =
        Dataset::from_csv("name , \"a,b\"\r\n\n  x , \"he said \"\"hi\"\"\" \r\n\"multi\nline\",2").unwrap();
    assert_eq!(data.column_names(), ["name", "a,b"]);
    assert_eq!(data.len(), 2);
    assert_eq!(data.levels("name").unwrap(), ["x", "multi\nline"]);
    assert_eq!(data.levels("a,b").unwrap(), ["he said \"hi\"", "2"]);

    assert_eq!(
        Dataset::from_csv("a,b\n1\n").unwrap_err(),
        FairnessError::RaggedRow {
            row: 1,
            expected: 2,
            found: 1
        }
    );
    assert!(matches!(
        Dataset::from_csv("a\n\"open").unwrap_err(),
        FairnessError::Csv { line: 2, .. }
    ));
    assert_eq!(
        Dataset::from_csv("a,a\n").unwrap_err(),
        FairnessError::DuplicateColumn("a".into())
    );
}

/// Purely indirect discrimination: `Z` is independent of `X` and `Y` reads
/// only `W`. The cookbook should flag IE and nothing else, and stop flagging
/// it once `W` is in the business-necessity set.
#[test]
fn cookbook_isolates_the_indirect_pathway() {
    let mut rng = SplitMix64::new(42);
    let mut records = Vec::new();
    for _ in 0..8000 {
        let z = usize::from(rng.unit() < 0.5);
        let x = usize::from(rng.unit() < 0.5);
        let w = usize::from(rng.unit() < 0.2 + 0.4 * x as f64 + 0.2 * z as f64);
        let y = usize::from(rng.unit() < 0.2 + 0.5 * w as f64 + 0.1 * z as f64);
        records.push([z, x, w, y].map(|v| v.to_string()));
    }
    let data = Dataset::from_records(["z", "x", "w", "y"], records).unwrap();
    let base = Cookbook::new(Sfm::new("x", ["z"], ["w"], "y"), "0", "1").with_bootstrap(
        BootstrapConfig {
            replicates: 200,
            seed: 1,
        },
    );

    let r = base.run(&data).unwrap();
    assert_eq!(r.disparate_treatment, Verdict::NoEvidence, "{r}");
    assert_eq!(r.indirect, Verdict::Discrimination, "{r}");
    assert_eq!(r.spurious, Verdict::NoEvidence, "{r}");
    // x-IE^sym has the sign of the x0 → x1 transition: positive when x1
    // raises W and W raises Y. The asymmetric x-IE_{x1,x0} is its negative.
    assert!((r.ie.estimate - 0.2).abs() < 0.03, "{r}");
    assert!((r.measures.ie + 0.2).abs() < 0.03, "{r}");
    assert!(r.ie.ci_low < r.ie.estimate && r.ie.estimate < r.ie.ci_high);
    assert_eq!(r.replicates_used + r.replicates_failed, 200);

    let r = base
        .clone()
        .with_business_necessity(BusinessNecessity::W)
        .run(&data)
        .unwrap();
    assert_eq!(r.indirect, Verdict::Permitted);
    assert!(r.to_string().contains("BN-set {W}"));
}

/// A predictor equal to the true outcome has no excess effect on any
/// permitted pathway; one that ignores `W` has a large one.
#[test]
fn cookbook_audits_a_predictor_against_the_reference() {
    let mut rng = SplitMix64::new(5);
    let mut records = Vec::new();
    for _ in 0..6000 {
        let z = usize::from(rng.unit() < 0.5);
        let x = usize::from(rng.unit() < 0.3 + 0.4 * z as f64);
        let w = usize::from(rng.unit() < 0.2 + 0.5 * x as f64);
        let y = usize::from(rng.unit() < 0.1 + 0.6 * w as f64 + 0.2 * z as f64);
        // A predictor that never looks at W, so it carries no indirect effect.
        let blind = 0.1 + 0.2 * z as f64;
        records.push(vec![
            z.to_string(),
            x.to_string(),
            w.to_string(),
            y.to_string(),
            y.to_string(),
            blind.to_string(),
        ]);
    }
    let data =
        Dataset::from_records(["z", "x", "w", "y", "copy", "blind"], records).unwrap();
    let cfg = BootstrapConfig {
        replicates: 200,
        seed: 9,
    };

    let r = Cookbook::new(Sfm::new("x", ["z"], ["w"], "copy"), "0", "1")
        .with_reference("y")
        .with_business_necessity(BusinessNecessity::ZW)
        .with_bootstrap(cfg)
        .run(&data)
        .unwrap();
    let ex = r.ie_excess.unwrap();
    assert_eq!(ex.estimate, 0.0);
    assert_eq!(r.indirect, Verdict::NoEvidence);
    assert_eq!(r.spurious, Verdict::NoEvidence);

    let r = Cookbook::new(Sfm::new("x", ["z"], ["w"], "blind"), "0", "1")
        .with_reference("y")
        .with_business_necessity(BusinessNecessity::W)
        .with_bootstrap(cfg)
        .run(&data)
        .unwrap();
    assert!(r.ie.estimate.abs() < 1e-12, "{r}");
    assert!(r.de.estimate.abs() < 1e-12, "{r}");
    assert_eq!(r.indirect, Verdict::Discrimination, "{r}");
    assert_eq!(r.disparate_treatment, Verdict::NoEvidence, "{r}");
    // Z is not in the BN-set here, so SE is tested against zero; `blind`
    // depends on Z, and Z is associated with X.
    assert_eq!(r.spurious, Verdict::Discrimination, "{r}");
}

#[test]
fn bootstrap_is_reproducible() {
    let mut rng = SplitMix64::new(1);
    let scm = RandomSfm::new(&mut rng);
    let data = scm.sample(&mut rng, 2000);
    let c = Cookbook::new(sfm(), "0", "1").with_bootstrap(BootstrapConfig {
        replicates: 50,
        seed: 3,
    });
    assert_eq!(c.run(&data).unwrap(), c.run(&data).unwrap());
}

/// Ex. 6.9 and 6.11, the startup's hiring: `Y ~ Bern((X + Z − 2XZ)/5 + W/6)`.
/// The NDE is exactly zero although the direct path is active, while the
/// z-specific DE recovers `±1/5`. Built as an exact weighted population with
/// a symmetric `P(x, z)`, as the book's `X, Z ← Bern(expit(U))` gives.
fn startup(p_xz: [[f64; 2]; 2]) -> Dataset {
    let mut records = Vec::new();
    let mut weights = Vec::new();
    for (x, row) in p_xz.iter().enumerate() {
        for (z, pxz) in row.iter().enumerate() {
            for (w, pw) in [(0, 0.7), (1, 0.3)] {
                let py = (x + z - 2 * x * z) as f64 / 5.0 + w as f64 / 6.0;
                for (y, p) in [(1, py), (0, 1.0 - py)] {
                    records.push([x, z, w, y].map(|v| v.to_string()));
                    weights.push(pxz * pw * p);
                }
            }
        }
    }
    Dataset::from_records(["x", "z", "w", "y"], records)
        .unwrap()
        .with_weights(weights)
        .unwrap()
}

#[test]
fn startup_hiring_nde_vanishes_but_z_de_does_not() {
    let data = startup([[0.35, 0.15], [0.15, 0.35]]);
    let s = Sfm::new("x", ["z"], ["w"], "y");
    let g = general_measures(&data, &s, "0", "1").unwrap();
    assert!(g.nde.abs() < EPS, "{g:?}");
    assert!((g.tv - (g.nde - g.nie + g.exp_se_x1 - g.exp_se_x0)).abs() < EPS);

    let zs = z_specific_measures(&data, &s, "0", "1").unwrap();
    for zm in &zs {
        let e = zm.effects.unwrap();
        let expected = if zm.stratum == "z=0" { 0.2 } else { -0.2 };
        assert!((e.de - expected).abs() < EPS, "{zm:?}");
        assert!((e.de_sym - expected).abs() < EPS, "{zm:?}");
        assert!((e.te - (e.de_sym + e.ie_sym)).abs() < EPS);
        // W does not depend on X: no indirect effect anywhere.
        assert!(e.ie.abs() < EPS && e.ie_sym.abs() < EPS);
    }
    // x-DE(y | x0) weights the strata by P(z | x0) = (0.7, 0.3).
    let m = estimate(&data, &s, "0", "1").unwrap();
    assert!((m.de - 0.08).abs() < EPS && (m.de_sym - 0.08).abs() < EPS, "{m:?}");
}

/// With `X ⊥ Z` the population-level DE is zero too, so only Algorithm 15's
/// z-specific follow-up can find the direct discrimination.
#[test]
fn cookbook_finds_subgroup_discrimination() {
    let mut rng = SplitMix64::new(77);
    let records: Vec<[String; 4]> = (0..6000)
        .map(|_| {
            let x = rng.below(2);
            let z = rng.below(2);
            let w = usize::from(rng.unit() < 0.3);
            let p = (x + z - 2 * x * z) as f64 / 5.0 + w as f64 / 6.0;
            let y = usize::from(rng.unit() < p);
            [x, z, w, y].map(|v| v.to_string())
        })
        .collect();
    let data = Dataset::from_records(["x", "z", "w", "y"], records).unwrap();
    let r = Cookbook::new(Sfm::new("x", ["z"], ["w"], "y"), "0", "1")
        .with_bootstrap(BootstrapConfig {
            replicates: 300,
            seed: 2,
        })
        .run(&data)
        .unwrap();
    assert!(!r.de.rejects(), "{r}");
    assert_eq!(r.disparate_treatment, Verdict::SubgroupDiscrimination, "{r}");
    assert_eq!(r.subgroup_direct.len(), 2);
    assert!(r.subgroup_direct.iter().all(|t| t.test.rejects()), "{r}");
    assert!(r.to_string().contains("z-DE^sym [z=0]"), "{r}");
}

#[test]
fn general_and_z_specific_identities_on_samples() {
    let mut rng = SplitMix64::new(19);
    for _ in 0..10 {
        let scm = RandomSfm::new(&mut rng);
        let data = scm.sample(&mut rng, 4000);
        let g = general_measures(&data, &sfm(), "0", "1").unwrap();
        assert!((g.tv - (g.nde - g.nie + g.exp_se_x1 - g.exp_se_x0)).abs() < 1e-12);
        let zs = z_specific_measures(&data, &sfm(), "0", "1").unwrap();
        // The NDE is the P(z)-average of the z-DEs.
        let nde: f64 = zs.iter().map(|z| z.weight * z.effects.unwrap().de).sum();
        assert!((nde - g.nde).abs() < 1e-12);
        for z in &zs {
            let e = z.effects.unwrap();
            assert!((e.te - (e.de_sym + e.ie_sym)).abs() < 1e-12);
        }
    }
}

fn with_fit(data: &Dataset, c: Constraints) -> (FairPredictor, Measures) {
    let fair = FairPredictor::fit(data, &sfm(), "0", "1", c).unwrap();
    let out = fair.with_predictions(data, "f").unwrap();
    let m = estimate(&out, &sfm().with_outcome("f"), "0", "1").unwrap();
    (fair, m)
}

/// Thm 6.1.32: the constrained least-squares predictor zeroes exactly the
/// measures it is asked to, and matches the true outcome on BN pathways.
#[test]
fn in_processing_meets_its_constraints() {
    let mut rng = SplitMix64::new(23);
    for _ in 0..10 {
        let scm = RandomSfm::new(&mut rng);
        let data = scm.sample(&mut rng, 4000);
        let y = estimate(&data, &sfm(), "0", "1").unwrap();

        let (fair, m) = with_fit(&data, Constraints::Causal(BusinessNecessity::NONE));
        for v in [m.de, m.de_reverse, m.ie, m.ie_reverse, m.se, m.tv] {
            assert!(v.abs() < 1e-9, "{m:?}");
        }
        assert!(fair.loss >= fair.unconstrained_loss - 1e-12);

        let (_, m) = with_fit(&data, Constraints::Causal(BusinessNecessity::W));
        assert!(m.de.abs() < 1e-9 && m.de_reverse.abs() < 1e-9 && m.se.abs() < 1e-9);
        assert!((m.ie_sym - y.ie_sym).abs() < 1e-9, "{m:?} vs {y:?}");

        let (_, m) = with_fit(&data, Constraints::Causal(BusinessNecessity::Z));
        assert!(m.de_sym.abs() < 1e-9 && m.ie_sym.abs() < 1e-9);
        assert!((m.se - y.se).abs() < 1e-9);

        let (_, m) = with_fit(&data, Constraints::None);
        assert!((m.de - y.de).abs() < 1e-9 && (m.se - y.se).abs() < 1e-9);
    }
}

/// The Fair Prediction Theorem in action: demographic parity holds, yet the
/// causal measures do not vanish.
#[test]
fn tv_only_predictor_keeps_causal_disparities() {
    let mut rng = SplitMix64::new(29);
    let scm = RandomSfm::new(&mut rng);
    let data = scm.sample(&mut rng, 4000);
    let (_, m) = with_fit(&data, Constraints::TotalVariation);
    assert!(m.tv.abs() < 1e-9);
    assert!(m.de_sym.abs() + m.ie_sym.abs() + m.se.abs() > 1e-3, "{m:?}");
}

#[test]
fn predictor_rejects_unseen_cells() {
    let data = Dataset::from_csv("x,w,y\na,0,0\na,1,1\nb,0,1\nb,1,1\n").unwrap();
    let s = Sfm::new("x", Vec::<String>::new(), ["w"], "y");
    let fair = FairPredictor::fit(&data, &s, "a", "b", Constraints::None).unwrap();
    let other = Dataset::from_csv("x,w\na,2\n").unwrap();
    assert!(matches!(fair.predict(&other), Err(FairnessError::UnseenCell(_))));
}

/// The min-cost-flow solver finds the same optimum as the monotone coupling
/// on 1-D problems, and respects both marginals.
#[test]
fn transport_solver_is_optimal_in_one_dimension() {
    let mut rng = SplitMix64::new(31);
    for _ in 0..30 {
        let (n, m) = (1 + rng.below(6), 1 + rng.below(6));
        let av: Vec<f64> = (0..n).map(|_| rng.unit() * 10.0).collect();
        let bv: Vec<f64> = (0..m).map(|_| rng.unit() * 10.0).collect();
        let a = probs(&mut rng, n);
        let b = probs(&mut rng, m);
        let flow = transport::transport_plan(&a, &b, &|i, j| (av[i] - bv[j]).abs()).unwrap();
        let cost = |plan: &[Vec<(usize, f64)>]| -> f64 {
            plan.iter()
                .enumerate()
                .map(|(i, row)| {
                    row.iter().map(|&(j, p)| a[i] * p * (av[i] - bv[j]).abs()).sum::<f64>()
                })
                .sum()
        };
        let mono = transport_monotone(&av, &a, &bv, &b);
        assert!((cost(&flow) - cost(&mono)).abs() < 1e-9);
        for (j, bj) in b.iter().enumerate() {
            let into: f64 = flow
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    row.iter().filter(|e| e.0 == j).map(|e| a[i] * e.1).sum::<f64>()
                })
                .sum();
            assert!((into - bj).abs() < 1e-9);
        }
        for row in &flow {
            assert!((row.iter().map(|e| e.1).sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }
}

fn transport_monotone(av: &[f64], a: &[f64], bv: &[f64], b: &[f64]) -> Vec<Vec<(usize, f64)>> {
    // Brute force through the solver's public behaviour: a 1-D numeric
    // variable goes through the monotone path, so build the coupling the
    // same way `couple` would, via a tiny one-column dataset.
    let header = ["v"];
    let rows: Vec<[String; 1]> = av.iter().chain(bv).map(|v| [format!("{v}")]).collect();
    let data = Dataset::from_records(header, rows).unwrap();
    let col = data.column("v").unwrap();
    let key = |i: usize| vec![col.codes[i]];
    let src: std::collections::HashMap<Vec<u32>, f64> =
        (0..av.len()).map(|i| (key(i), a[i])).collect();
    let tgt: std::collections::HashMap<Vec<u32>, f64> =
        (0..bv.len()).map(|j| (key(av.len() + j), b[j])).collect();
    let metric_plan = transport::couple_for_tests(&data, "v", &src, &tgt);
    (0..av.len())
        .map(|i| {
            metric_plan[&key(i)]
                .iter()
                .map(|(k, p)| ((0..bv.len()).find(|&j| key(av.len() + j) == *k).unwrap(), *p))
                .collect()
        })
        .collect()
}

/// Thm 6.1.33: after Causal IF, the measures outside the BN-set vanish
/// exactly, and those inside it are untouched.
#[test]
fn causal_if_zeroes_the_unpermitted_pathways() {
    let mut rng = SplitMix64::new(37);
    for _ in 0..6 {
        let scm = RandomSfm::new(&mut rng);
        let data = scm.sample(&mut rng, 3000);
        let y = estimate(&data, &sfm(), "0", "1").unwrap();

        let fair = CausalIf::new(sfm(), "0", "1").transform(&data).unwrap();
        let m = estimate(&fair, &sfm(), "0", "1").unwrap();
        for v in [m.de, m.de_reverse, m.ie, m.ie_reverse, m.se, m.tv] {
            assert!(v.abs() < 1e-9, "{m:?}");
        }
        // The x1 group is the reference and does not move.
        assert!((m.factual_x1 - y.factual_x1).abs() < 1e-12);

        let fair = CausalIf::new(sfm(), "0", "1")
            .with_business_necessity(BusinessNecessity::Z)
            .transform(&data)
            .unwrap();
        let m = estimate(&fair, &sfm(), "0", "1").unwrap();
        assert!(m.de_sym.abs() < 1e-9 && m.ie_sym.abs() < 1e-9, "{m:?}");
        assert!((m.se - y.se).abs() < 1e-9, "{m:?} vs {y:?}");

        // W ∈ BN keeps x0's own mediator mechanism: W | x0, τ(z). The random
        // SCMs reach every W value under every parent, so positivity holds.
        let fair = CausalIf::new(sfm(), "0", "1")
            .with_business_necessity(BusinessNecessity::W)
            .transform(&data)
            .unwrap();
        let m = estimate(&fair, &sfm(), "0", "1").unwrap();
        assert!(m.de_sym.abs() < 1e-9 && m.se.abs() < 1e-9, "{m:?}");
    }
}

#[test]
fn causal_if_sampling_keeps_rows_and_reference_group() {
    let mut rng = SplitMix64::new(41);
    let scm = RandomSfm::new(&mut rng);
    let data = scm.sample(&mut rng, 20_000);
    let out = CausalIf::new(sfm(), "0", "1").transform_sampled(&data, 5).unwrap();
    assert_eq!(out.len(), data.len());
    for i in 0..data.len() {
        if data.value("x", i) == Some("1") {
            for c in ["z1", "z2", "w1", "w2", "y"] {
                assert_eq!(out.value(c, i), data.value(c, i));
            }
        }
    }
    // One draw per row: the measures are zero up to sampling noise.
    let m = estimate(&out, &sfm(), "0", "1").unwrap();
    assert!(m.de_sym.abs() < 0.03 && m.ie_sym.abs() < 0.03 && m.se.abs() < 0.03, "{m:?}");
}

fn policy_data(n: usize, seed: u64) -> (Dataset, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut records = Vec::new();
    let mut benefit = Vec::new();
    for _ in 0..n {
        let x = usize::from(rng.unit() < 0.4);
        // A handful of discrete benefit levels, some harmful, shifted by group.
        let delta = (rng.below(7) as f64 - 1.0) / 10.0 + 0.05 * x as f64;
        records.push([x.to_string()]);
        benefit.push(delta);
    }
    (Dataset::from_records(["x"], records).unwrap(), benefit)
}

#[test]
fn algorithm_17_spends_the_budget_on_the_largest_benefits() {
    let (data, benefit) = policy_data(2000, 43);
    let pol = benefit_fairness(&data, "x", "0", "1", &benefit, 0.3).unwrap();
    assert!((pol.rate - 0.3).abs() < 1e-9, "{pol:?}");
    let d = pol.thresholds[0];
    for (b, p) in benefit.iter().zip(&pol.treat_probability) {
        if *b > d {
            assert_eq!(*p, 1.0);
        } else if *b < d {
            assert_eq!(*p, 0.0);
        }
    }
    // A huge budget still never treats harmful or useless units.
    let pol = benefit_fairness(&data, "x", "0", "1", &benefit, 1.0).unwrap();
    for (b, p) in benefit.iter().zip(&pol.treat_probability) {
        assert_eq!(*p, if *b > 0.0 { 1.0 } else { 0.0 });
    }
}

#[test]
fn algorithm_18_is_optimal_and_respects_the_gap() {
    let (data, benefit) = policy_data(1500, 47);
    let a17 = benefit_fairness(&data, "x", "0", "1", &benefit, 0.35).unwrap();
    let loose = utilitarian_benefit_fairness(&data, "x", "0", "1", &benefit, 0.35, 1.0).unwrap();
    assert!((loose.expected_benefit - a17.expected_benefit).abs() < 1e-12);

    for m in [0.0, 0.05, 0.1, 0.2] {
        let pol = utilitarian_benefit_fairness(&data, "x", "0", "1", &benefit, 0.35, m).unwrap();
        assert!(pol.disparity().abs() <= m + 1e-9, "{m}: {pol:?}");
        assert!(pol.rate <= 0.35 + 1e-9);
        // No grid point does better.
        let n1 = (0..data.len()).filter(|&i| data.value("x", i) == Some("1")).count() as f64;
        let p1 = n1 / data.len() as f64;
        let gain = |g: &str, t: f64| -> f64 {
            let mut b: Vec<f64> = (0..data.len())
                .filter(|&i| data.value("x", i) == Some(g))
                .map(|i| benefit[i])
                .filter(|b| *b > 0.0)
                .collect();
            b.sort_by(|x, y| y.total_cmp(x));
            let count = if g == "1" { n1 } else { data.len() as f64 - n1 };
            let k = t * count;
            let full = k.floor() as usize;
            let mut s: f64 = b.iter().take(full).sum();
            if full < b.len() {
                s += b[full] * (k - full as f64);
            }
            s / count
        };
        let mut best: f64 = 0.0;
        for i in 0..=200 {
            for j in 0..=200 {
                let (t0, t1) = (i as f64 / 200.0, j as f64 / 200.0);
                if (t0 - t1).abs() <= m && (1.0 - p1) * t0 + p1 * t1 <= 0.35 {
                    let cap = |g: &str| {
                        (0..data.len())
                            .filter(|&r| data.value("x", r) == Some(g) && benefit[r] > 0.0)
                            .count() as f64
                    };
                    let (u0, u1) = (cap("0") / (data.len() as f64 - n1), cap("1") / n1);
                    if t0 <= u0 && t1 <= u1 {
                        best = best.max((1.0 - p1) * gain("0", t0) + p1 * gain("1", t1));
                    }
                }
            }
        }
        assert!(pol.expected_benefit >= best - 1e-9, "{m}: {} < {best}", pol.expected_benefit);
    }
}

#[test]
fn benefit_is_estimated_per_covariate_cell() {
    let csv = "x,w,d,y\n0,0,0,0\n0,0,1,1\n0,0,1,1\n0,1,0,1\n0,1,1,1\n1,0,0,0\n1,0,1,0\n";
    let data = Dataset::from_csv(csv).unwrap();
    let s = Sfm::new("x", Vec::<String>::new(), ["w"], "y");
    let delta = estimate_benefit(&data, &s, "d", "0", "1").unwrap();
    assert_eq!(delta, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let bad = Dataset::from_csv("x,w,d,y\n0,0,1,1\n").unwrap();
    assert!(matches!(
        estimate_benefit(&bad, &s, "d", "0", "1"),
        Err(FairnessError::UnknownLevel { .. })
    ));
    let bad = Dataset::from_csv("x,w,d,y\n0,0,1,1\n0,1,0,1\n").unwrap();
    assert!(matches!(
        estimate_benefit(&bad, &s, "d", "0", "1"),
        Err(FairnessError::Positivity { .. })
    ));
}
