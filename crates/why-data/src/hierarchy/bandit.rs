//! Counterfactual decision-making (Sec. 20.2.3): the strategies of the MAB
//! template (Fig. 20.20) and Algorithm 70 (MAB-OPT).

use std::collections::HashMap;

use super::HierarchyError;
use crate::fairness::SplitMix64;

/// One exogenous type of the MAB template (Fig. 20.20): context `Z`,
/// natural decision `X`, and the potential responses `D_x` and `Y_x`.
/// `Y` does not listen to `D`, so `Y_x` is well defined. `D` and `Y` share
/// the unit, so they can be confounded with each other and with `X`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MabUnit {
    /// The probability of the type.
    pub p: f64,
    /// The context.
    pub z: u8,
    /// The natural decision.
    pub x: u8,
    /// `D_{x=0}` and `D_{x=1}`.
    pub d: [u8; 2],
    /// `Y_{x=0}` and `Y_{x=1}`.
    pub y: [u8; 2],
}

/// An SCM on the MAB template, as a distribution over exogenous types.
/// Every variable is binary.
#[derive(Clone, Debug)]
pub struct MabScm {
    units: Vec<MabUnit>,
}

/// The expected reward of the four strategies of Sec. 20.2.3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strategies {
    /// `π_obs`: follow the natural decision, `E[Y]`.
    pub obs: f64,
    /// `π_int`: `do(x)` maximizing `E[Y_x | z]`.
    pub int: f64,
    /// `π_ett`: maximize `E[Y_x | z, x']` over the natural `x'`.
    pub ett: f64,
    /// `π_opt` (Eqs. 20.120-20.121): pick `x''` from `(z, x')`, read
    /// `D_{x''}`, then maximize `E[Y_x | z, x', D_{x''}]`.
    pub opt: f64,
}

impl MabScm {
    /// The SCM with these types.
    ///
    /// # Errors
    ///
    /// [`HierarchyError::InvalidInput`] when the probabilities are negative
    /// or do not sum to one, or a value is not 0 or 1.
    pub fn new(units: Vec<MabUnit>) -> Result<Self, HierarchyError> {
        let total: f64 = units.iter().map(|u| u.p).sum();
        if units.iter().any(|u| u.p < 0.0) || (total - 1.0).abs() > 1e-9 {
            return Err(HierarchyError::InvalidInput(
                "type probabilities must be non-negative and sum to one".into(),
            ));
        }
        if units
            .iter()
            .any(|u| u.z > 1 || u.x > 1 || u.d.iter().chain(&u.y).any(|&v| v > 1))
        {
            return Err(HierarchyError::InvalidInput("values must be 0 or 1".into()));
        }
        Ok(Self { units })
    }

    /// `(P(event), E[Y_x | event])`.
    fn cond(&self, x: usize, event: impl Fn(&MabUnit) -> bool) -> (f64, f64) {
        let (mut p, mut s) = (0.0, 0.0);
        for u in self.units.iter().filter(|u| event(u)) {
            p += u.p;
            s += u.p * f64::from(u.y[x]);
        }
        (p, if p > 0.0 { s / p } else { 0.0 })
    }

    fn best(&self, event: impl Fn(&MabUnit) -> bool) -> f64 {
        let (p, a) = self.cond(0, &event);
        let (_, b) = self.cond(1, &event);
        p * a.max(b)
    }

    /// The value of `π_opt` within context `(z, x')` when it reads
    /// `D_{x''}`.
    fn opt_given(&self, z: u8, xn: u8, xs: usize) -> f64 {
        (0..2u8)
            .map(|d| self.best(|u| u.z == z && u.x == xn && u.d[xs] == d))
            .sum()
    }

    /// `x''` for context `(z, x')` under `π_opt`.
    #[must_use]
    pub fn opt_probe(&self, z: u8, xn: u8) -> u8 {
        u8::from(self.opt_given(z, xn, 1) > self.opt_given(z, xn, 0) + 1e-12)
    }

    /// The exact expected reward of each strategy. Thm 20.2.8 and
    /// Cor. 20.2.9: `opt ≥ ett ≥ int` and `ett ≥ obs`.
    #[must_use]
    pub fn strategies(&self) -> Strategies {
        let obs = self
            .units
            .iter()
            .map(|u| u.p * f64::from(u.y[usize::from(u.x)]))
            .sum();
        let mut int = 0.0;
        let mut ett = 0.0;
        let mut opt = 0.0;
        for z in 0..2u8 {
            int += self.best(|u| u.z == z);
            for xn in 0..2u8 {
                ett += self.best(|u| u.z == z && u.x == xn);
                opt += self.opt_given(z, xn, 0).max(self.opt_given(z, xn, 1));
            }
        }
        Strategies { obs, int, ett, opt }
    }

    fn draw(&self, rng: &mut SplitMix64) -> MabUnit {
        let mut r = rng.unit();
        for u in &self.units {
            if r < u.p {
                return *u;
            }
            r -= u.p;
        }
        *self.units.last().expect("at least one type")
    }
}

/// The rewards an online run collected, one per round.
#[derive(Clone, Debug, PartialEq)]
pub struct MabRun {
    /// `Y` in each round.
    pub rewards: Vec<u8>,
}

impl MabRun {
    /// The mean reward over rounds `from..to`.
    #[must_use]
    pub fn mean(&self, from: usize, to: usize) -> f64 {
        let s = &self.rewards[from..to.min(self.rewards.len())];
        s.iter().map(|&y| f64::from(y)).sum::<f64>() / s.len().max(1) as f64
    }
}

/// A standard normal draw (Box-Muller).
fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = 1.0 - rng.unit();
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A Gamma(shape, 1) draw for `shape ≥ 1` (Marsaglia-Tsang).
fn gamma(shape: f64, rng: &mut SplitMix64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = 1.0 - rng.unit();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Thompson sampling over two Bernoulli arms with Beta(1, 1) priors.
#[derive(Clone, Copy, Debug, Default)]
struct Arms([[u32; 2]; 2]);

impl Arms {
    fn pick(&self, rng: &mut SplitMix64) -> usize {
        let sample = |[s, f]: [u32; 2], rng: &mut SplitMix64| {
            let a = gamma(f64::from(s) + 1.0, rng);
            let b = gamma(f64::from(f) + 1.0, rng);
            a / (a + b)
        };
        let a = sample(self.0[0], rng);
        let b = sample(self.0[1], rng);
        usize::from(b > a)
    }

    fn update(&mut self, arm: usize, y: u8) {
        self.0[arm][usize::from(y == 0)] += 1;
    }
}

/// Algorithm 70 (MAB-OPT) with Thompson sampling as the solver. Each
/// round draws a new unit and observes `(z, x')`. It picks `x''` from the
/// D-arms of `(z, x')` and reads `D_{x''}` through `CTF-WRITE(x'' → D)`.
/// Then it picks `x` from the Y-arms of `(z, x', x'', d)`, reads `Y_x`
/// through `CTF-WRITE(x → Y)`, and updates both bandits with `y`.
///
/// The hot start from `P(v)` (line 7) is left out: every arm starts from a
/// uniform prior.
///
/// ```
/// use why_data::hierarchy::{MabScm, MabUnit, mab_opt};
///
/// // A deterministic unit type: always rewarded by x = 1.
/// let unit = MabUnit { p: 1.0, z: 0, x: 0, d: [0, 0], y: [0, 1] };
/// let run = mab_opt(&MabScm::new(vec![unit]).unwrap(), 500, 7);
/// assert!(run.mean(250, 500) > 0.95);
/// ```
#[must_use]
pub fn mab_opt(scm: &MabScm, rounds: usize, seed: u64) -> MabRun {
    let mut rng = SplitMix64::new(seed);
    let mut d_arms: HashMap<(u8, u8), Arms> = HashMap::new();
    let mut y_arms: HashMap<(u8, u8, usize, u8), Arms> = HashMap::new();
    let mut rewards = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let u = scm.draw(&mut rng);
        let da = d_arms.entry((u.z, u.x)).or_default();
        let xs = da.pick(&mut rng);
        let d = u.d[xs];
        let ya = y_arms.entry((u.z, u.x, xs, d)).or_default();
        let x = ya.pick(&mut rng);
        let y = u.y[x];
        ya.update(x, y);
        d_arms.get_mut(&(u.z, u.x)).expect("inserted").update(xs, y);
        rewards.push(y);
    }
    MabRun { rewards }
}

/// The standard L2 baseline: Thompson sampling on `E[Y_x | z]`, choosing
/// `do(x)` from the context alone.
#[must_use]
pub fn mab_int(scm: &MabScm, rounds: usize, seed: u64) -> MabRun {
    let mut rng = SplitMix64::new(seed);
    let mut arms: HashMap<u8, Arms> = HashMap::new();
    let mut rewards = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let u = scm.draw(&mut rng);
        let a = arms.entry(u.z).or_default();
        let x = a.pick(&mut rng);
        let y = u.y[x];
        a.update(x, y);
        rewards.push(y);
    }
    MabRun { rewards }
}
