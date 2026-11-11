//! Causal offline-to-online learning (Sec. 9.2, CRL Task 1): causal bounds
//! from confounded observational data, and online learners that use them.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use super::lp::{Cmp, Lp, LpOutcome};
use super::online::{finish, index_policy, positions, ucb_index};
use super::{
    Cdm, CrlError, Dataset, Environment, OnlineRun, Policy, PolicySpace, Reward, State,
    TabularPolicy,
};
use crate::fairness::SplitMix64;

/// The most deterministic policies [`UcbCdm`] enumerates per episode when
/// the policy space lacks perfect recall.
const ONLINE_POLICY_CAP: usize = 4096;

/// Tolerance for rewards in `[0, 1]` and rules summing to one.
const TOL: f64 = 1e-9;

/// A closed interval `[lower, upper]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// The lower end.
    pub lower: f64,
    /// The upper end.
    pub upper: f64,
}

impl Interval {
    const UNIT: Self = Self {
        lower: 0.0,
        upper: 1.0,
    };

    fn clamped(lower: f64, upper: f64) -> Self {
        Self {
            lower: lower.clamp(0.0, 1.0),
            upper: upper.clamp(0.0, 1.0),
        }
    }

    /// Whether `v` lies in the interval, up to `tol`.
    #[must_use]
    pub fn contains(&self, v: f64, tol: f64) -> bool {
        v >= self.lower - tol && v <= self.upper + tol
    }
}

/// The natural bounds `E_x[Y] ∈ [l_x, r_x]` of one arm (Thm 9.2.2).
#[derive(Debug, Clone, PartialEq)]
pub struct ArmBound {
    /// The arm `x`.
    pub arm: String,
    /// `l_x = E[Y | x] P(x)`.
    pub lower: f64,
    /// `r_x = E[Y | x] P(x) + 1 − P(x)`.
    pub upper: f64,
}

/// `(values of vars, R(Y), weight)` for every row with positive weight,
/// checking that `R(Y) ∈ [0, 1]`.
fn unit_rows(
    data: &Dataset,
    vars: &[String],
    reward: &Reward,
) -> Result<Vec<(Vec<String>, f64, f64)>, CrlError> {
    for v in vars.iter().chain(reward.signals()) {
        if data.levels(v).is_none() {
            return Err(CrlError::Unknown(v.clone()));
        }
    }
    let mut out = Vec::with_capacity(data.len());
    for row in 0..data.len() {
        let w = data.weights().map_or(1.0, |ws| ws[row]);
        if w <= 0.0 {
            continue;
        }
        let r = reward.apply_str(
            reward
                .signals()
                .iter()
                .map(|s| data.value(s, row).expect("column checked")),
        )?;
        check_unit(r)?;
        let key = vars
            .iter()
            .map(|v| data.value(v, row).expect("column checked").to_string())
            .collect();
        out.push((key, r, w));
    }
    if out.is_empty() {
        return Err(CrlError::Data("no rows with positive weight".into()));
    }
    Ok(out)
}

fn check_unit(r: f64) -> Result<(), CrlError> {
    if (-TOL..=1.0 + TOL).contains(&r) {
        Ok(())
    } else {
        Err(CrlError::InvalidInput(format!(
            "causal bounds need rewards in [0, 1], got {r}"
        )))
    }
}

/// Thm 9.2.2, the natural bounds of every arm the data shows:
///
/// ```text
/// l_x = E[Y | x] P(x),   r_x = E[Y | x] P(x) + 1 − P(x)
/// ```
///
/// They hold in any SCM whatever the confounding between `X` and `Y`, and
/// cannot be tightened without further assumptions (Thm 5.6.2). An arm the
/// data never shows has the vacuous bound `[0, 1]`.
///
/// ```
/// use why_data::crl::{natural_bounds, Dataset, Reward};
///
/// // Table 9.2 with Δ = 0.1: P(X = 1) = 0.8, E[Y | X = 1] = 0.375, E[Y | X = 0] = 0.
/// let data = Dataset::from_csv("X,Y\n0,0\n1,1\n1,0\n").unwrap()
///     .with_weights(vec![0.2, 0.3, 0.5]).unwrap();
/// let b = natural_bounds(&data, "X", &Reward::total(["Y"])).unwrap();
/// assert!((b[0].upper - 0.8).abs() < 1e-12);
/// assert!((b[1].lower - 0.3).abs() < 1e-12 && (b[1].upper - 0.5).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// [`CrlError::Unknown`] for a missing column, and
/// [`CrlError::InvalidInput`] if a reward falls outside `[0, 1]`.
pub fn natural_bounds(
    data: &Dataset,
    action: &str,
    reward: &Reward,
) -> Result<Vec<ArmBound>, CrlError> {
    let rows = unit_rows(data, &[action.to_string()], reward)?;
    let total: f64 = rows.iter().map(|(_, _, w)| w).sum();
    let mut levels: Vec<String> = data.levels(action).expect("checked").to_vec();
    levels.sort();
    Ok(levels
        .into_iter()
        .map(|arm| {
            let (mut w, mut s) = (0.0, 0.0);
            for (k, r, wr) in &rows {
                if k[0] == arm {
                    w += wr;
                    s += wr * r;
                }
            }
            let lower = s / total;
            ArmBound {
                arm,
                lower,
                upper: lower + 1.0 - w / total,
            }
        })
        .collect())
}

/// Algorithm 23, UCB+: the UCB of Algorithm 21 with every index clipped to
/// the arm's causal bound (Eq. 9.22),
///
/// ```text
/// UCB⁺_t(x) = min{max{UCB_t(x), l_x}, r_x}
/// ```
///
/// An arm never played has `UCB = ∞`, so it starts at `r_x`. An arm whose
/// `r_x` is below the optimal reward is therefore played only while some
/// other index is lower still. Arms missing from `bounds` use `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct UcbPlus {
    /// The causal bounds `[l_x, r_x]`, e.g. from [`natural_bounds`].
    pub bounds: Vec<ArmBound>,
}

impl UcbPlus {
    /// Runs for `episodes` episodes in `env`, pulling arms of `action`.
    ///
    /// # Errors
    ///
    /// As [`super::Ucb::run`].
    pub fn run(
        &self,
        env: &mut dyn Environment,
        action: &str,
        reward: &Reward,
        episodes: usize,
        seed: u64,
    ) -> Result<OnlineRun, CrlError> {
        let arms = env
            .domain(action)
            .ok_or_else(|| CrlError::Unknown(action.to_string()))?;
        let clip: Vec<(f64, f64)> = arms
            .iter()
            .map(|a| {
                self.bounds
                    .iter()
                    .find(|b| b.arm == *a)
                    .map_or((0.0, 1.0), |b| (b.lower, b.upper))
            })
            .collect();
        index_policy(env, action, reward, episodes, seed, &|t, x, n, sum| {
            ucb_index(t, n, sum).max(clip[x].0).min(clip[x].1)
        })
    }
}

/// Algorithm 22, UCB−: UCB warm-started with observational data as if they
/// were interventional (Eqs. 9.4–9.5).
///
/// It is consistent only under NUC (Eq. 9.6). Without it the pooled means
/// stay biased until the online samples swamp the offline ones, which is
/// the negative transfer of Experiment 9.1.
#[derive(Debug, Clone)]
pub struct UcbTransfer {
    /// The observational samples `D_obs`. Weights count as sample sizes.
    pub observational: Dataset,
}

impl UcbTransfer {
    /// Runs for `episodes` episodes in `env`, pulling arms of `action`.
    ///
    /// # Errors
    ///
    /// As [`super::Ucb::run`], plus [`CrlError::Unknown`] if the data lack
    /// `action` or a reward signal.
    pub fn run(
        &self,
        env: &mut dyn Environment,
        action: &str,
        reward: &Reward,
        episodes: usize,
        seed: u64,
    ) -> Result<OnlineRun, CrlError> {
        let arms = env
            .domain(action)
            .ok_or_else(|| CrlError::Unknown(action.to_string()))?;
        let data = &self.observational;
        if data.levels(action).is_none() {
            return Err(CrlError::Unknown(action.to_string()));
        }
        let mut prior = vec![(0.0, 0.0); arms.len()];
        for row in 0..data.len() {
            let w = data.weights().map_or(1.0, |ws| ws[row]);
            let x = data.value(action, row).expect("checked");
            if let Some(i) = arms.iter().position(|a| a == x) {
                let mut signals = Vec::with_capacity(reward.signals().len());
                for s in reward.signals() {
                    signals.push(data.value(s, row).ok_or_else(|| CrlError::Unknown(s.clone()))?);
                }
                let r = reward.apply_str(signals)?;
                prior[i].0 += w;
                prior[i].1 += w * r;
            }
        }
        index_policy(env, action, reward, episodes, seed, &|t, x, n, sum| {
            ucb_index(t, n + prior[x].0, sum + prior[x].1)
        })
    }
}

/// Thm 9.2.3: the regret of [`UcbPlus`] after `T` episodes, for rewards in
/// `[0, 1]`:
///
/// ```text
/// R(T) ≤ 8 Σ_{Δ_x>0, r_x≥µ*} log(T)/Δ_x + (1 + π²/3) Σ_{Δ_x>0} Δ_x
/// ```
///
/// `means` holds `E_x[Y]` and `upper` holds `r_x`, both per arm. Only
/// suboptimal arms whose causal upper bound reaches `µ*` pay the
/// logarithmic term.
#[must_use]
pub fn ucb_plus_regret_bound(means: &[f64], upper: &[f64], episodes: usize) -> f64 {
    let best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_t = (episodes as f64).ln();
    means
        .iter()
        .zip(upper)
        .filter(|(m, _)| best - **m > 0.0)
        .map(|(m, r)| {
            let gap = best - m;
            let log_term = if *r >= best { 8.0 * log_t / gap } else { 0.0 };
            log_term + (1.0 + PI * PI / 3.0) * gap
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Sequential bounds (Thms 9.2.5 and 9.2.6)
// ---------------------------------------------------------------------------

/// The variables of a policy space laid out in temporal order, with a domain
/// for each.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    order: Vec<String>,
    /// `hist[i]` = length of the history `X̄_{i−1} ∪ S̄_i`; `X_i` follows.
    hist: Vec<usize>,
    domains: Vec<Vec<String>>,
    /// Positions in `order` of each `S_i`.
    state_pos: Vec<Vec<usize>>,
}

impl Layout {
    fn new(space: &PolicySpace, domains: Vec<Vec<String>>) -> Self {
        let order = space.order().to_vec();
        let state_pos = space
            .stages()
            .iter()
            .map(|s| {
                s.states
                    .iter()
                    .map(|v| order.iter().position(|o| o == v).expect("states are in order"))
                    .collect()
            })
            .collect();
        Self {
            hist: (0..space.horizon()).map(|i| space.history_len(i)).collect(),
            order,
            domains,
            state_pos,
        }
    }

    fn horizon(&self) -> usize {
        self.hist.len()
    }

    /// Every assignment to `order[from..to]`, lexicographically.
    fn assignments(&self, from: usize, to: usize) -> Vec<Vec<String>> {
        let mut out = vec![Vec::new()];
        for d in &self.domains[from..to] {
            out = out
                .into_iter()
                .flat_map(|prefix| {
                    d.iter().map(move |v| {
                        let mut p = prefix.clone();
                        p.push(v.clone());
                        p
                    })
                })
                .collect();
        }
        out
    }

    /// The `S_i` values inside a history.
    fn state<'a>(&'a self, i: usize, history: &'a [String]) -> Vec<(&'a str, &'a str)> {
        self.state_pos[i]
            .iter()
            .map(|&p| (self.order[p].as_str(), history[p].as_str()))
            .collect()
    }

    /// `π_i(· | s_i)` over the domain of `X_i`.
    fn rule(&self, policy: &dyn Policy, i: usize, history: &[String]) -> Result<Vec<f64>, CrlError> {
        let pairs = self.state(i, history);
        let state = State::new(&pairs);
        let action = &self.order[self.hist[i]];
        let probs: Vec<f64> = self.domains[self.hist[i]]
            .iter()
            .map(|x| policy.probability(action, &state, x))
            .collect();
        let total: f64 = probs.iter().sum();
        if probs.iter().any(|p| *p < 0.0) || (total - 1.0).abs() > TOL {
            return Err(CrlError::PolicyNotNormalized {
                action: action.clone(),
                state: state.to_string(),
                total,
            });
        }
        Ok(probs)
    }

    /// The new states revealed before `X_{i}` (`i ≥ 1`): `order[hist[i−1]+1 .. hist[i]]`.
    fn children(&self, i: usize) -> Vec<Vec<String>> {
        self.assignments(self.hist[i - 1] + 1, self.hist[i])
    }
}

fn concat(a: &[String], b: &[String]) -> Vec<String> {
    let mut out = a.to_vec();
    out.extend_from_slice(b);
    out
}

/// The causal bounds of Thms 9.2.5 and 9.2.6 for every history of a policy
/// space, estimated from observational data.
///
/// Write `h_i` for the history `X̄_{i−1} ∪ S̄_i` and `Γ` for Eq. 9.49:
///
/// ```text
/// Γ(h_1)          = P(h_1)
/// Γ(h_{i+1})      = P(h_{i+1}) − P(h_i, x_i) + Γ(h_i)
/// P_{x̄_i}(s_{i+1} | s̄_i)     ∈ [P(h_{i+1}) / Γ(h_i),  Γ(h_{i+1}) / Γ(h_i)]
/// E_{x̄_H}[R(Y) | s̄_H]        ∈ [E[R | h_H, x_H] P(h_H, x_H) / Γ(h_H),
///                                1 − (1 − E[R | h_H, x_H]) P(h_H, x_H) / Γ(h_H)]
/// ```
///
/// With one stage these are the natural bounds of Thm 9.2.2. The initial
/// distribution `P(s̄_1)` is not bounded but known: `S̄_1` precedes every
/// action. Histories are keyed by their values in temporal order, as in
/// [`PolicySpace::history`] followed by the action.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalBounds {
    space: PolicySpace,
    layout: Layout,
    initial: BTreeMap<Vec<String>, f64>,
    /// Index `i − 1` holds stage `i`'s transition bounds, keyed by `h_{i+1}`.
    transitions: Vec<BTreeMap<Vec<String>, Interval>>,
    /// Keyed by `(h_H, x_H)`.
    rewards: BTreeMap<Vec<String>, Interval>,
}

/// Estimates the causal bounds of Thms 9.2.5 and 9.2.6 from observational
/// data. Domains are the levels the data shows, sorted; a column showing
/// only `0`/`1` is taken as binary. A history the data never show gets the
/// vacuous bounds `[0, 1]`.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a missing column, and
/// [`CrlError::InvalidInput`] if a reward falls outside `[0, 1]`.
pub fn causal_bounds(
    data: &Dataset,
    space: &PolicySpace,
    reward: &Reward,
) -> Result<CausalBounds, CrlError> {
    let order = space.order().to_vec();
    let rows = unit_rows(data, &order, reward)?;
    let total: f64 = rows.iter().map(|(_, _, w)| w).sum();
    let domains: Vec<Vec<String>> = order
        .iter()
        .map(|v| {
            let mut d = data.levels(v).expect("checked").to_vec();
            d.sort();
            // A 0/1 column is binary even if the data show one value, as
            // `Cdm` widens booleans; the missing value gets vacuous bounds.
            if d.iter().all(|x| x == "0" || x == "1") {
                d = vec!["0".to_string(), "1".to_string()];
            }
            d
        })
        .collect();
    let layout = Layout::new(space, domains);
    let h = layout.horizon();

    let mut prob: HashMap<Vec<String>, f64> = HashMap::new();
    let mut reward_sum: HashMap<Vec<String>, f64> = HashMap::new();
    let full = layout.hist[h - 1] + 1;
    for (key, r, w) in &rows {
        for len in 0..=full {
            *prob.entry(key[..len].to_vec()).or_insert(0.0) += w / total;
        }
        *reward_sum.entry(key[..full].to_vec()).or_insert(0.0) += w / total * r;
    }
    let p = |k: &[String]| prob.get(k).copied().unwrap_or(0.0);

    // Γ over every history of every stage.
    let mut gamma: Vec<HashMap<Vec<String>, f64>> = Vec::with_capacity(h);
    for i in 0..h {
        let mut g = HashMap::new();
        for hist in layout.assignments(0, layout.hist[i]) {
            let v = if i == 0 {
                p(&hist)
            } else {
                let prev = layout.hist[i - 1];
                p(&hist) - p(&hist[..=prev]) + gamma[i - 1][&hist[..prev]]
            };
            g.insert(hist, v);
        }
        gamma.push(g);
    }

    let initial = layout
        .assignments(0, layout.hist[0])
        .into_iter()
        .map(|k| {
            let v = p(&k);
            (k, v)
        })
        .collect();
    let mut transitions = Vec::with_capacity(h.saturating_sub(1));
    for i in 1..h {
        let prev = layout.hist[i - 1];
        let mut map = BTreeMap::new();
        for child in layout.assignments(0, layout.hist[i]) {
            let g = gamma[i - 1][&child[..prev]];
            let iv = if g <= 0.0 {
                Interval::UNIT
            } else {
                Interval::clamped(p(&child) / g, gamma[i][&child] / g)
            };
            map.insert(child, iv);
        }
        transitions.push(map);
    }
    let mut rewards = BTreeMap::new();
    for cell in layout.assignments(0, full) {
        let g = gamma[h - 1][&cell[..full - 1]];
        let pc = p(&cell);
        let iv = if g <= 0.0 || pc <= 0.0 {
            Interval::UNIT
        } else {
            let e = reward_sum[&cell] / pc;
            Interval::clamped(e * pc / g, 1.0 - (1.0 - e) * pc / g)
        };
        rewards.insert(cell, iv);
    }
    Ok(CausalBounds {
        space: space.clone(),
        layout,
        initial,
        transitions,
        rewards,
    })
}

/// Maximises (or minimises) `Σ v_k p_k` over `{p : lo ≤ p ≤ hi, Σ p = 1}`
/// greedily: start at `lo`, then pour the remaining mass into the best
/// entries first.
fn box_simplex(values: &[f64], lo: &[f64], hi: &[f64], maximize: bool) -> f64 {
    let mut p = lo.to_vec();
    let mut mass = (1.0 - lo.iter().sum::<f64>()).max(0.0);
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        let o = values[a].total_cmp(&values[b]);
        if maximize { o.reverse() } else { o }
    });
    for k in idx {
        if mass <= 0.0 {
            break;
        }
        let add = (hi[k] - lo[k]).max(0.0).min(mass);
        p[k] += add;
        mass -= add;
    }
    p.iter().zip(values).map(|(a, b)| a * b).sum()
}

impl CausalBounds {
    /// The policy space the bounds cover.
    #[must_use]
    pub fn space(&self) -> &PolicySpace {
        &self.space
    }

    /// The domains used for each variable of `space.order()`.
    #[must_use]
    pub fn domain(&self, variable: &str) -> Option<&[String]> {
        self.layout
            .order
            .iter()
            .position(|v| v == variable)
            .map(|p| self.layout.domains[p].as_slice())
    }

    /// `P(s̄_1)`, given the values of the first history.
    #[must_use]
    pub fn initial(&self, history: &[&str]) -> Option<f64> {
        self.initial.get(&owned(history)).copied()
    }

    /// The bound on `P_{x̄_i}(s_{i+1} | s̄_i)` for stage `stage ≥ 1`
    /// (0-based; the transition into its states), given the history
    /// `h_{stage+1}` in temporal order.
    #[must_use]
    pub fn transition(&self, stage: usize, history: &[&str]) -> Option<Interval> {
        if stage == 0 {
            return None;
        }
        self.transitions
            .get(stage - 1)
            .and_then(|m| m.get(&owned(history)))
            .copied()
    }

    /// The bound on `E_{x̄_H}[R(Y) | s̄_H]`, given `(h_H, x_H)` in temporal
    /// order.
    #[must_use]
    pub fn reward(&self, history: &[&str]) -> Option<Interval> {
        self.rewards.get(&owned(history)).copied()
    }

    fn transition_box(&self, i: usize, hx: &[String]) -> (Vec<f64>, Vec<f64>) {
        let children = self.layout.children(i);
        let mut lo = Vec::with_capacity(children.len());
        let mut hi = Vec::with_capacity(children.len());
        for c in &children {
            let iv = self.transitions[i - 1]
                .get(&concat(hx, c))
                .copied()
                .unwrap_or(Interval::UNIT);
            lo.push(iv.lower);
            hi.push(iv.upper);
        }
        (lo, hi)
    }

    /// Eq. 9.67: `[l_π, r_π]`, the range of `E_π[R(Y); M]` over every SCM
    /// whose transitions and rewards lie within the bounds.
    ///
    /// The constraints are rectangular per history, so backward induction
    /// over the extreme transition at each history is exact.
    ///
    /// # Errors
    ///
    /// [`CrlError::PolicyNotNormalized`] if a rule is not a distribution over
    /// its action's domain.
    pub fn policy_bounds(&self, policy: &dyn Policy) -> Result<Interval, CrlError> {
        let lower = self.extreme(policy, false)?;
        let upper = self.extreme(policy, true)?;
        Ok(Interval { lower, upper })
    }

    fn extreme(&self, policy: &dyn Policy, maximize: bool) -> Result<f64, CrlError> {
        let l = &self.layout;
        let h = l.horizon();
        let mut next: HashMap<Vec<String>, f64> = HashMap::new();
        for i in (0..h).rev() {
            let mut cur = HashMap::new();
            for hist in l.assignments(0, l.hist[i]) {
                let probs = l.rule(policy, i, &hist)?;
                let mut v = 0.0;
                for (x, pi) in l.domains[l.hist[i]].iter().zip(&probs) {
                    if *pi == 0.0 {
                        continue;
                    }
                    let mut hx = hist.clone();
                    hx.push(x.clone());
                    let w = if i + 1 == h {
                        let iv = self.rewards.get(&hx).copied().unwrap_or(Interval::UNIT);
                        if maximize { iv.upper } else { iv.lower }
                    } else {
                        let (lo, hi) = self.transition_box(i + 1, &hx);
                        let vals: Vec<f64> = l
                            .children(i + 1)
                            .iter()
                            .map(|c| next[&concat(&hx, c)])
                            .collect();
                        box_simplex(&vals, &lo, &hi, maximize)
                    };
                    v += pi * w;
                }
                cur.insert(hist, v);
            }
            next = cur;
        }
        Ok(self
            .initial
            .iter()
            .map(|(k, p)| p * next[k])
            .sum())
    }
}

fn owned(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

// ---------------------------------------------------------------------------
// Algorithms 24 and 25
// ---------------------------------------------------------------------------

/// Algorithm 24 (UCB for CDMs), and Algorithm 25 (UCB+) when given causal
/// bounds.
///
/// In episode `t` it builds the plausible set `M_{t−1}(δ)`, `δ = t^{−4}`:
/// every transition distribution within an L1 ball of radius `ε_i(δ)`
/// (Eq. 9.38) of its empirical estimate, every conditional reward within
/// `ε_H(δ)` (Eq. 9.39). With bounds it intersects that set with `M_c`
/// (Eqs. 9.64–9.65). It then plays the optimal policy of the most
/// optimistic SCM (Eq. 9.34 or 9.66).
///
/// * With perfect recall the optimistic plan is the extended Bellman
///   equation (Eq. 9.45). Each inner maximum is a small linear program.
/// * Otherwise it enumerates deterministic policies (at most 4096) and
///   evaluates each optimistically. For a fixed policy the constraints are
///   rectangular per history, so that is again backward induction.
///
/// Departures: a history never visited has no confidence constraint (its
/// empirical estimate is undefined). If the confidence set and the causal
/// bounds are disjoint, which happens only on the low-probability event that
/// the confidence set misses `M*`, the causal bounds win, as in Eq. 9.22.
/// The initial distribution `P(s̄_1)` is pinned to the bounds' observational
/// estimate when bounds are given.
#[derive(Debug, Clone, Default)]
pub struct UcbCdm {
    /// The causal bounds of Algorithm 25, or `None` for Algorithm 24.
    pub bounds: Option<CausalBounds>,
}

/// The trace of [`UcbCdm`].
#[derive(Debug, Clone)]
pub struct CdmRun {
    /// Actions, rewards and samples, as for the bandit learners.
    pub run: OnlineRun,
    /// The optimistic policy `π^{(t)}` played in each episode.
    pub policies: Vec<TabularPolicy>,
}

impl CdmRun {
    /// The cumulative regret `t µ* − Σ_{s ≤ t} E_{π^{(s)}}[R(Y)]` (Eq. 9.27)
    /// after each episode, evaluated exactly in `cdm`.
    ///
    /// # Errors
    ///
    /// As [`Cdm::evaluate`] and [`Cdm::optimal`].
    pub fn regret(&self, cdm: &Cdm) -> Result<Vec<f64>, CrlError> {
        let best = cdm.optimal()?.value;
        let mut total = 0.0;
        let mut out = Vec::with_capacity(self.policies.len());
        let mut cache: Vec<(TabularPolicy, f64)> = Vec::new();
        for pi in &self.policies {
            let v = if let Some((_, v)) = cache.iter().find(|(p, _)| p == pi) {
                *v
            } else {
                let v = cdm.evaluate(pi)?;
                cache.push((pi.clone(), v));
                v
            };
            total += best - v;
            out.push(total);
        }
        Ok(out)
    }
}

/// Counts gathered from the episodes so far.
struct Counts {
    n: HashMap<Vec<String>, f64>,
    reward: HashMap<Vec<String>, f64>,
    episodes: f64,
}

impl UcbCdm {
    /// Runs for `episodes` episodes in `env` over `space`.
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] if the environment lacks a variable,
    /// [`CrlError::InvalidInput`] if the bounds cover another space or a
    /// reward falls outside `[0, 1]`, [`CrlError::TooManyPolicies`] past the
    /// enumeration cap.
    pub fn run(
        &self,
        env: &mut dyn Environment,
        space: &PolicySpace,
        reward: &Reward,
        episodes: usize,
        seed: u64,
    ) -> Result<CdmRun, CrlError> {
        if let Some(b) = &self.bounds
            && b.space != *space
        {
            return Err(CrlError::InvalidInput(
                "the causal bounds cover a different policy space".into(),
            ));
        }
        let domains: Vec<Vec<String>> = space
            .order()
            .iter()
            .map(|v| env.domain(v).ok_or_else(|| CrlError::Unknown(v.clone())))
            .collect::<Result<_, _>>()?;
        let layout = Layout::new(space, domains);
        let vars = env.variables();
        let order_pos: Vec<usize> = layout
            .order
            .iter()
            .map(|v| {
                vars.iter()
                    .position(|w| w == v)
                    .ok_or_else(|| CrlError::Unknown(v.clone()))
            })
            .collect::<Result<_, _>>()?;
        let (action_pos, signal_pos) = positions(env, space, reward)?;
        let action_domains: Vec<Vec<String>> =
            layout.hist.iter().map(|&a| layout.domains[a].clone()).collect();
        let mut rng = SplitMix64::new(seed);
        let mut uniform = || rng.unit();

        let mut counts = Counts {
            n: HashMap::new(),
            reward: HashMap::new(),
            episodes: 0.0,
        };
        let full = layout.hist[layout.horizon() - 1] + 1;
        let mut rows = Vec::with_capacity(episodes);
        let mut policies = Vec::with_capacity(episodes);
        for t in 1..=episodes {
            let delta = (t as f64).powi(-4);
            let policy = self.plan(&layout, space, &action_domains, &counts, delta)?;
            let row = env.intervene(space, &policy, &mut uniform)?;
            let r = reward.apply_str(signal_pos.iter().map(|&p| row[p].as_str()))?;
            check_unit(r)?;
            let key: Vec<String> = order_pos.iter().map(|&p| row[p].clone()).collect();
            for len in 0..=full {
                *counts.n.entry(key[..len].to_vec()).or_insert(0.0) += 1.0;
            }
            *counts.reward.entry(key).or_insert(0.0) += r;
            counts.episodes += 1.0;
            rows.push(row);
            policies.push(policy);
        }
        let run = finish(env, rows, &action_pos, &signal_pos, reward, None)?;
        Ok(CdmRun { run, policies })
    }

    /// The optimistic value of acting `x` after `hist` at stage `i`, given
    /// the values of the next stage's histories.
    fn optimistic(
        &self,
        layout: &Layout,
        counts: &Counts,
        delta: f64,
        i: usize,
        hx: &[String],
        next: &HashMap<Vec<String>, f64>,
    ) -> f64 {
        let h = layout.horizon() as f64;
        let cells: f64 = layout.domains[..hx.len()]
            .iter()
            .map(|d| d.len() as f64)
            .product();
        let n = counts.n.get(hx).copied().unwrap_or(0.0);
        if i + 1 == layout.horizon() {
            let causal = self
                .bounds
                .as_ref()
                .and_then(|b| b.rewards.get(hx).copied())
                .unwrap_or(Interval::UNIT);
            if n == 0.0 {
                return causal.upper;
            }
            let mean = counts.reward[hx] / n;
            let eps = ((8.0 * h * cells / delta).ln() / (2.0 * n)).sqrt();
            let (lo, hi) = (
                (mean - eps).max(causal.lower),
                (mean + eps).min(1.0).min(causal.upper),
            );
            return if lo <= hi { hi } else { causal.upper };
        }
        let children = layout.children(i + 1);
        let values: Vec<f64> = children.iter().map(|c| next[&concat(hx, c)]).collect();
        let (lo, hi) = match &self.bounds {
            Some(b) => b.transition_box(i + 1, hx),
            None => (vec![0.0; children.len()], vec![1.0; children.len()]),
        };
        let phat = (n > 0.0).then(|| {
            children
                .iter()
                .map(|c| counts.n.get(&concat(hx, c)).copied().unwrap_or(0.0) / n)
                .collect::<Vec<f64>>()
        });
        let k = children.len() as f64;
        let eps = phat
            .as_ref()
            .map_or(2.0, |_| (2.0 * k * (4.0 * h * cells / delta).ln() / n).sqrt());
        l1_box_max(&values, phat.as_deref(), eps, &lo, &hi)
    }

    fn initial_value(
        &self,
        layout: &Layout,
        counts: &Counts,
        delta: f64,
        v1: &HashMap<Vec<String>, f64>,
    ) -> f64 {
        let firsts = layout.assignments(0, layout.hist[0]);
        let values: Vec<f64> = firsts.iter().map(|k| v1[k]).collect();
        if let Some(b) = &self.bounds {
            let p: Vec<f64> = firsts
                .iter()
                .map(|k| b.initial.get(k).copied().unwrap_or(0.0))
                .collect();
            return p.iter().zip(&values).map(|(a, b)| a * b).sum();
        }
        let n = counts.episodes;
        let k = firsts.len() as f64;
        let ones = vec![1.0; firsts.len()];
        let zeros = vec![0.0; firsts.len()];
        if n == 0.0 {
            return box_simplex(&values, &zeros, &ones, true);
        }
        let phat: Vec<f64> = firsts
            .iter()
            .map(|c| counts.n.get(c).copied().unwrap_or(0.0) / n)
            .collect();
        let h = layout.horizon() as f64;
        let eps = (2.0 * k * (4.0 * h / delta).ln() / n).sqrt();
        l1_box_max(&values, Some(&phat), eps, &zeros, &ones)
    }

    fn plan(
        &self,
        layout: &Layout,
        space: &PolicySpace,
        action_domains: &[Vec<String>],
        counts: &Counts,
        delta: f64,
    ) -> Result<TabularPolicy, CrlError> {
        let h = layout.horizon();
        if space.has_perfect_recall() {
            let mut policy = TabularPolicy::new(space, action_domains.to_vec());
            let mut next: HashMap<Vec<String>, f64> = HashMap::new();
            for i in (0..h).rev() {
                let mut cur = HashMap::new();
                for hist in layout.assignments(0, layout.hist[i]) {
                    let mut best: Option<(usize, f64)> = None;
                    for (xi, x) in layout.domains[layout.hist[i]].iter().enumerate() {
                        let mut hx = hist.clone();
                        hx.push(x.clone());
                        let v = self.optimistic(layout, counts, delta, i, &hx, &next);
                        if best.is_none_or(|(_, b)| v > b) {
                            best = Some((xi, v));
                        }
                    }
                    let (xi, v) = best.expect("non-empty domain");
                    let state: Vec<&str> = layout.state(i, &hist).iter().map(|(_, v)| *v).collect();
                    policy.set(&space.stages()[i].action, &state, &layout.domains[layout.hist[i]][xi]);
                    cur.insert(hist, v);
                }
                next = cur;
            }
            return Ok(policy);
        }

        // Enumerate deterministic policies over the domain of each S_i.
        let states: Vec<Vec<Vec<String>>> = (0..h)
            .map(|i| {
                let mut out = vec![Vec::new()];
                for &p in &layout.state_pos[i] {
                    out = out
                        .into_iter()
                        .flat_map(|s: Vec<String>| {
                            layout.domains[p].iter().map(move |v| {
                                let mut s = s.clone();
                                s.push(v.clone());
                                s
                            })
                        })
                        .collect();
                }
                out
            })
            .collect();
        let digits: Vec<(usize, usize)> = states
            .iter()
            .enumerate()
            .flat_map(|(i, ss)| (0..ss.len()).map(move |j| (i, j)))
            .collect();
        let count: f64 = digits
            .iter()
            .map(|(i, _)| action_domains[*i].len() as f64)
            .product();
        if count > ONLINE_POLICY_CAP as f64 {
            return Err(CrlError::TooManyPolicies {
                count,
                cap: ONLINE_POLICY_CAP,
            });
        }
        let mut counter = vec![0usize; digits.len()];
        let mut best: Option<(TabularPolicy, f64)> = None;
        loop {
            let mut policy = TabularPolicy::new(space, action_domains.to_vec());
            for (d, (i, j)) in digits.iter().enumerate() {
                let s: Vec<&str> = states[*i][*j].iter().map(String::as_str).collect();
                policy.set(&space.stages()[*i].action, &s, &action_domains[*i][counter[d]]);
            }
            let mut next: HashMap<Vec<String>, f64> = HashMap::new();
            for i in (0..h).rev() {
                let mut cur = HashMap::new();
                for hist in layout.assignments(0, layout.hist[i]) {
                    let probs = layout.rule(&policy, i, &hist)?;
                    let mut v = 0.0;
                    for (x, pi) in layout.domains[layout.hist[i]].iter().zip(&probs) {
                        if *pi > 0.0 {
                            let mut hx = hist.clone();
                            hx.push(x.clone());
                            v += pi * self.optimistic(layout, counts, delta, i, &hx, &next);
                        }
                    }
                    cur.insert(hist, v);
                }
                next = cur;
            }
            let value = self.initial_value(layout, counts, delta, &next);
            if best.as_ref().is_none_or(|(_, b)| value > *b) {
                best = Some((policy, value));
            }
            let mut d = digits.len();
            loop {
                if d == 0 {
                    return Ok(best.expect("at least one policy").0);
                }
                d -= 1;
                counter[d] += 1;
                if counter[d] < action_domains[digits[d].0].len() {
                    break;
                }
                counter[d] = 0;
            }
        }
    }
}

/// `max Σ v_k p_k` over `{‖p − p̂‖₁ ≤ ε} ∩ {lo ≤ p ≤ hi} ∩ simplex`, falling
/// back to the box alone when there is no `p̂`, the ball covers the simplex,
/// or the intersection is empty.
fn l1_box_max(values: &[f64], phat: Option<&[f64]>, eps: f64, lo: &[f64], hi: &[f64]) -> f64 {
    let Some(phat) = phat.filter(|_| eps < 2.0) else {
        return box_simplex(values, lo, hi, true);
    };
    let k = values.len();
    // Variables: p_0..p_{k−1}, d_0..d_{k−1}.
    let mut lp = Lp::new(2 * k);
    let mut c = values.to_vec();
    c.extend(std::iter::repeat_n(0.0, k));
    lp.maximize(c);
    let unit = |j: usize, a: f64, b: Option<(usize, f64)>| {
        let mut row = vec![0.0; 2 * k];
        row[j] = a;
        if let Some((j2, a2)) = b {
            row[j2] = a2;
        }
        row
    };
    let mut sum = vec![0.0; 2 * k];
    let mut dsum = vec![0.0; 2 * k];
    for j in 0..k {
        sum[j] = 1.0;
        dsum[k + j] = 1.0;
        lp.constraint(unit(j, 1.0, None), Cmp::Le, hi[j]);
        lp.constraint(unit(j, 1.0, None), Cmp::Ge, lo[j]);
        lp.constraint(unit(j, 1.0, Some((k + j, -1.0))), Cmp::Le, phat[j]);
        lp.constraint(unit(j, -1.0, Some((k + j, -1.0))), Cmp::Le, -phat[j]);
    }
    lp.constraint(sum, Cmp::Eq, 1.0);
    lp.constraint(dsum, Cmp::Le, eps);
    match lp.solve() {
        LpOutcome::Optimal { value, .. } => value,
        _ => box_simplex(values, lo, hi, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimistic_transition_moves_mass_to_the_best_child() {
        // p̂ = (0.5, 0.5), ε = 0.4: move 0.2 to the better child.
        let v = l1_box_max(&[1.0, 0.0], Some(&[0.5, 0.5]), 0.4, &[0.0, 0.0], &[1.0, 1.0]);
        assert!((v - 0.7).abs() < 1e-9);
        // A causal cap of 0.6 binds first.
        let v = l1_box_max(&[1.0, 0.0], Some(&[0.5, 0.5]), 0.4, &[0.0, 0.0], &[0.6, 1.0]);
        assert!((v - 0.6).abs() < 1e-9);
        // Disjoint sets: the causal box wins.
        let v = l1_box_max(&[1.0, 0.0], Some(&[0.9, 0.1]), 0.1, &[0.0, 0.0], &[0.5, 1.0]);
        assert!((v - 0.5).abs() < 1e-9);
        assert!((box_simplex(&[1.0, 2.0, 3.0], &[0.2, 0.2, 0.2], &[1.0, 0.3, 1.0], false) - 1.6).abs() < 1e-9);
    }
}
