//! Online learning (Sec. 8.2.2): randomized controlled trials (Algorithm 20)
//! and the upper confidence bound (Algorithm 21), with their regret bounds.

use std::f64::consts::PI;

use super::cdm::uniform_for;
use super::{CrlError, Dataset, Deterministic, Environment, Plan, PolicySpace, Reward, State, optimize};
use crate::fairness::SplitMix64;

/// The trace of an online learner.
#[derive(Debug, Clone)]
pub struct OnlineRun {
    /// The actions taken in each episode, in stage order.
    pub actions: Vec<Vec<String>>,
    /// The realised reward `R(Y^{(t)})` of each episode.
    pub rewards: Vec<f64>,
    /// Every sample, in episode order. Exp-NUC holds in each one
    /// (Lemma 8.2.3), so this is valid input for [`super::ipw`] and
    /// [`super::dp`].
    pub data: Dataset,
    /// The policy committed to after exploring, for [`Rct`].
    pub committed: Option<Plan>,
}

impl OnlineRun {
    /// The cumulative pseudo-regret `Σ_{s ≤ t} Δ_{x^{(s)}}` of a bandit run
    /// after each episode (Eqs. 8.90 and 8.93), given the true `E_x[Y]` of
    /// every arm.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] if an arm played is missing from `arm_values`.
    pub fn pseudo_regret(&self, arm_values: &[(&str, f64)]) -> Result<Vec<f64>, CrlError> {
        let best = arm_values
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        self.actions
            .iter()
            .map(|a| {
                let arm = a.first().map_or("", String::as_str);
                let v = arm_values
                    .iter()
                    .find(|(x, _)| *x == arm)
                    .ok_or_else(|| CrlError::Unknown(arm.to_string()))?
                    .1;
                total += best - v;
                Ok(total)
            })
            .collect()
    }
}

/// Column positions of the actions and reward signals in a sample.
pub(super) fn positions(
    env: &dyn Environment,
    space: &PolicySpace,
    reward: &Reward,
) -> Result<(Vec<usize>, Vec<usize>), CrlError> {
    let vars = env.variables();
    let find = |name: &str| {
        vars.iter()
            .position(|v| v == name)
            .ok_or_else(|| CrlError::Unknown(name.to_string()))
    };
    let actions = space.actions().into_iter().map(find).collect::<Result<_, _>>()?;
    let signals = reward
        .signals()
        .iter()
        .map(|s| find(s))
        .collect::<Result<_, _>>()?;
    Ok((actions, signals))
}

/// Algorithm 20, randomized controlled trials: explore with `π_UNIF` for
/// `trials` episodes, then commit to `argmax_π Ê^{(N)}_π[R(Y)]`.
///
/// The exploration data satisfy NUC (Lemma 8.2.3), so the commitment is
/// [`optimize`] on them, which is DP (Thm 8.2.2) over `Π`. Histories that
/// exploration never reached fall back to the uniform rule.
#[derive(Debug, Clone, Copy)]
pub struct Rct {
    /// The number of exploration episodes `N`.
    pub trials: usize,
}

impl Rct {
    /// Runs for `episodes` episodes (`T`) in `env`.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] if `trials` is zero or exceeds `episodes`.
    /// [`CrlError::Unknown`] if the environment lacks a variable. Otherwise
    /// as [`optimize`].
    pub fn run(
        &self,
        env: &mut dyn Environment,
        space: &PolicySpace,
        reward: &Reward,
        episodes: usize,
        seed: u64,
    ) -> Result<OnlineRun, CrlError> {
        if self.trials == 0 || self.trials > episodes {
            return Err(CrlError::InvalidInput(format!(
                "need 0 < N ≤ T, got N = {} and T = {episodes}",
                self.trials
            )));
        }
        let (action_pos, signal_pos) = positions(env, space, reward)?;
        let explore = uniform_for(env, space)?;
        let mut rng = SplitMix64::new(seed);
        let mut uniform = || rng.unit();

        let mut rows = Vec::with_capacity(episodes);
        let mut committed: Option<Plan> = None;
        for t in 1..=episodes {
            let row = match &committed {
                None => env.intervene(space, &explore, &mut uniform)?,
                Some(plan) => env.intervene(space, &plan.policy, &mut uniform)?,
            };
            rows.push(row);
            if t == self.trials {
                let data = Dataset::from_records(env.variables(), rows.clone())?;
                committed = Some(optimize(&data, space, reward)?);
            }
        }
        finish(env, rows, &action_pos, &signal_pos, reward, committed)
    }
}

pub(super) fn finish(
    env: &dyn Environment,
    rows: Vec<Vec<String>>,
    action_pos: &[usize],
    signal_pos: &[usize],
    reward: &Reward,
    committed: Option<Plan>,
) -> Result<OnlineRun, CrlError> {
    let actions = rows
        .iter()
        .map(|r| action_pos.iter().map(|&p| r[p].clone()).collect())
        .collect();
    let rewards = rows
        .iter()
        .map(|r| reward.apply_str(signal_pos.iter().map(|&p| r[p].as_str())))
        .collect::<Result<_, _>>()?;
    let data = Dataset::from_records(env.variables(), rows)?;
    Ok(OnlineRun {
        actions,
        rewards,
        data,
        committed,
    })
}

/// Algorithm 21, the upper confidence bound for a multi-armed bandit
/// `Π = {⟨X, ∅⟩}`.
///
/// In episode `t` it plays the first arm maximising
///
/// ```text
/// UCB_{t−1}(x) = Ê_x[Y] + sqrt(log(1/δ) / (2 N_{t−1}(x))),   δ = t^{−4}
/// ```
///
/// with `UCB = ∞` for an arm not yet played (footnote 13). The width is
/// therefore `sqrt(2 ln t / N_{t−1}(x))`. The regret bound of Thm 8.2.5
/// assumes rewards in `[0, 1]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ucb;

impl Ucb {
    /// Runs for `episodes` episodes in `env`, pulling arms of `action`.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] if the environment lacks `action` or a reward
    /// signal.
    pub fn run(
        &self,
        env: &mut dyn Environment,
        action: &str,
        reward: &Reward,
        episodes: usize,
        seed: u64,
    ) -> Result<OnlineRun, CrlError> {
        index_policy(env, action, reward, episodes, seed, &|t, _, n, sum| {
            ucb_index(t, n, sum)
        })
    }
}

/// `Ê[Y] + sqrt(log(1/δ) / (2N))` with `δ = t^{−4}`, or `∞` when `N = 0`.
pub(super) fn ucb_index(t: usize, n: f64, sum: f64) -> f64 {
    if n <= 0.0 {
        f64::INFINITY
    } else {
        let log_inv_delta = 4.0 * (t as f64).ln();
        sum / n + (log_inv_delta / (2.0 * n)).sqrt()
    }
}

/// Plays, in each episode `t`, the first arm maximising
/// `index(t, arm, pulls, reward sum)`.
pub(super) fn index_policy(
    env: &mut dyn Environment,
    action: &str,
    reward: &Reward,
    episodes: usize,
    seed: u64,
    index: &dyn Fn(usize, usize, f64, f64) -> f64,
) -> Result<OnlineRun, CrlError> {
    let space = PolicySpace::bandit(action);
    let arms = env
        .domain(action)
        .ok_or_else(|| CrlError::Unknown(action.to_string()))?;
    let (action_pos, signal_pos) = positions(env, &space, reward)?;
    let mut rng = SplitMix64::new(seed);
    let mut uniform = || rng.unit();

    let mut pulls = vec![0usize; arms.len()];
    let mut sums = vec![0.0; arms.len()];
    let mut rows = Vec::with_capacity(episodes);
    for t in 1..=episodes {
        let mut best = 0;
        let mut best_index = f64::NEG_INFINITY;
        for x in 0..arms.len() {
            let v = index(t, x, pulls[x] as f64, sums[x]);
            if v > best_index {
                best = x;
                best_index = v;
            }
        }
        let arm = arms[best].clone();
        let pi = Deterministic(move |_: &str, _: &State<'_>| arm.clone());
        let row = env.intervene(&space, &pi, &mut uniform)?;
        let r = reward.apply_str(signal_pos.iter().map(|&p| row[p].as_str()))?;
        pulls[best] += 1;
        sums[best] += r;
        rows.push(row);
    }
    finish(env, rows, &action_pos, &signal_pos, reward, None)
}

/// Thm 8.2.4: the regret of [`Rct`] with `N` trials after `T` episodes, for
/// rewards in `[0, 1]`:
///
/// ```text
/// R(T) ≤ (N/K) Σ_{Δ_x>0} Δ_x + (T − N) Σ_{Δ_x>0} Δ_x exp(−N Δ_x² / (4K))
/// ```
///
/// `gaps` holds `Δ_x = E_{x*}[Y] − E_x[Y]` for all `K` arms, the optimal
/// one included as `0`.
#[must_use]
pub fn rct_regret_bound(gaps: &[f64], trials: usize, episodes: usize) -> f64 {
    let k = gaps.len() as f64;
    let n = trials as f64;
    let rest = episodes.saturating_sub(trials) as f64;
    gaps.iter()
        .filter(|d| **d > 0.0)
        .map(|d| n / k * d + rest * d * (-n * d * d / (4.0 * k)).exp())
        .sum()
}

/// Eq. 8.101: the number of trials minimising Eq. 8.100 (two arms, gap `Δ`),
/// `N = ⌈8/Δ² · ln(TΔ²/4)⌉`, or `0` when `TΔ² ≤ 4`. This uses the natural
/// log; the book's "N ≈ 878" for `T = 5000, Δ = 0.1` uses `log₁₀`.
#[must_use]
pub fn rct_optimal_trials(gap: f64, episodes: usize) -> usize {
    let x = episodes as f64 * gap * gap / 4.0;
    if x <= 1.0 || gap <= 0.0 {
        return 0;
    }
    (8.0 / (gap * gap) * x.ln()).ceil() as usize
}

/// Thm 8.2.5: the regret of [`Ucb`] after `T` episodes, for rewards in
/// `[0, 1]`:
///
/// ```text
/// R(T) ≤ 8 Σ_{Δ_x>0} log(T) / Δ_x + (1 + π²/3) Σ_{Δ_x>0} Δ_x
/// ```
#[must_use]
pub fn ucb_regret_bound(gaps: &[f64], episodes: usize) -> f64 {
    let log_t = (episodes as f64).ln();
    gaps.iter()
        .filter(|d| **d > 0.0)
        .map(|d| 8.0 * log_t / d + (1.0 + PI * PI / 3.0) * d)
        .sum()
}
