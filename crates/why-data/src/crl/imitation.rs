//! Causal imitation learning (Sec. 9.5, CRL Task 4): learning from expert
//! demonstrations when the reward is unknown and the expert may have seen
//! variables the imitator cannot.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use super::lp::{Cmp, Lp, LpOutcome};
use super::{CrlError, Dataset, Policy, PolicySpace, Reward, Stage, State, TabularPolicy, dp};
use crate::graph::dseparation::Admg;

/// The most subspaces [`imitation_subspace`] and
/// [`is_minimal_imitation_admissible`] enumerate.
const SUBSPACE_CAP: usize = 1 << 20;

/// The most deterministic policies [`causal_mwal`] enumerates.
const POLICY_CAP: usize = 20_000;

/// Why one stage satisfies, or fails, the imitation backdoor (Def. 9.5.3).
///
/// When both conditions hold the stage reports [`ImitationCondition::Blocked`],
/// which is the one [`imitation_distribution`] needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImitationCondition {
    /// Condition 2: `S_i` d-separates `X_i` from `Y` in
    /// `G_{X̲_i, π_{i+1}, …, π_H}`.
    Blocked,
    /// Condition 1 only: `X_i ∉ An(Y)` in `G_{π_{i+1}, …, π_H}`, while a
    /// back-door path stays open.
    NotAncestor,
    /// Neither holds; one open back-door path, rendered.
    Open(String),
}

/// One stage of an [`ImitationReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImitationStage {
    /// The action `X_i`.
    pub action: String,
    /// The outcome for this stage.
    pub condition: ImitationCondition,
}

/// The result of [`imitation_backdoor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImitationReport {
    /// One entry per stage, in temporal order.
    pub stages: Vec<ImitationStage>,
}

impl ImitationReport {
    /// Whether the space is imitation admissible.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.stages
            .iter()
            .all(|s| !matches!(s.condition, ImitationCondition::Open(_)))
    }
}

fn check_signals(g: &Admg, space: &PolicySpace, signals: &[&str]) -> Result<(), CrlError> {
    let actions = space.actions();
    for y in signals {
        if g.node(y).is_none() {
            return Err(CrlError::Unknown((*y).to_string()));
        }
        if actions.contains(y) {
            return Err(CrlError::InvalidInput(format!(
                "reward signal `{y}` is also an action"
            )));
        }
    }
    Ok(())
}

fn stage_condition(
    g: &Admg,
    space: &PolicySpace,
    signals: &[&str],
    i: usize,
    explain: bool,
) -> Result<ImitationCondition, CrlError> {
    let stage = &space.stages()[i];
    let x = stage.action.as_str();
    let manipulated = space.manipulated(g, i + 1)?;
    let z: Vec<&str> = stage.states.iter().map(String::as_str).collect();
    let ys: Vec<&str> = signals.iter().copied().filter(|y| !z.contains(y)).collect();
    let cut = manipulated.cut_edges_out_of(&[x])?;
    if ys.is_empty() || cut.is_d_separated(&[x], &ys, &z)? {
        return Ok(ImitationCondition::Blocked);
    }
    if !manipulated.ancestors_of(signals)?.contains(&x) {
        return Ok(ImitationCondition::NotAncestor);
    }
    let mut path = String::from("an open back-door path");
    if explain {
        for y in &ys {
            if let Some(p) = cut.paths(x, y, &z)?.into_iter().find(|p| p.is_open()) {
                path = p.to_string();
                break;
            }
        }
    }
    Ok(ImitationCondition::Open(path))
}

/// Def. 9.5.3, the imitation backdoor: for every stage, either `X_i` is no
/// longer an ancestor of `Y` once the later actions follow their policies,
/// or `S_i` blocks every back-door path from `X_i` to `Y` in that graph.
///
/// `G_{π_{i+1}, …, π_H}` replaces the parents of each later action `X_j`
/// with `S_j` ([`PolicySpace::manipulated`]). When the space is admissible,
/// cloning `P(X_i | S_i)` at every stage matches the expert's performance in
/// every SCM compatible with `G`, whatever the reward (Thm 9.5.3).
///
/// Neither this nor the sequential back-door (Def. 8.2.3) implies the
/// other (Exs. 9.21 and 9.22).
///
/// # Errors
///
/// [`CrlError::Unknown`] for a variable `g` lacks, and
/// [`CrlError::InvalidInput`] if a signal is an action.
pub fn imitation_backdoor(
    g: &Admg,
    space: &PolicySpace,
    signals: &[&str],
) -> Result<ImitationReport, CrlError> {
    check_signals(g, space, signals)?;
    let stages = (0..space.horizon())
        .map(|i| {
            Ok(ImitationStage {
                action: space.stages()[i].action.clone(),
                condition: stage_condition(g, space, signals, i, true)?,
            })
        })
        .collect::<Result<_, CrlError>>()?;
    Ok(ImitationReport { stages })
}

fn admissible(g: &Admg, space: &PolicySpace, signals: &[&str]) -> Result<bool, CrlError> {
    for i in (0..space.horizon()).rev() {
        if matches!(
            stage_condition(g, space, signals, i, false)?,
            ImitationCondition::Open(_)
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Every subspace `{⟨X_i, Z_i⟩}` with `Z_i ⊆ S_i` (Def. 9.5.4), by total
/// size, then lexicographically.
fn subspaces(space: &PolicySpace, proper_only: bool) -> Result<Vec<PolicySpace>, CrlError> {
    let sizes: Vec<usize> = space.stages().iter().map(|s| s.states.len()).collect();
    let bits: usize = sizes.iter().sum();
    if bits > 20 || (1usize << bits) > SUBSPACE_CAP {
        return Err(CrlError::TooManyPolicies {
            count: 2f64.powi(bits as i32),
            cap: SUBSPACE_CAP,
        });
    }
    let full = (1u64 << bits) - 1;
    let mut masks: Vec<u64> = (0..=full).filter(|m| !proper_only || *m != full).collect();
    masks.sort_by_key(|m| (m.count_ones(), (0..bits).map(|b| (m >> b) & 1 == 0).collect::<Vec<_>>()));
    masks
        .into_iter()
        .map(|m| {
            let mut offset = 0;
            let stages = space
                .stages()
                .iter()
                .map(|s| {
                    let states = s
                        .states
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| (m >> (offset + j)) & 1 == 1)
                        .map(|(_, v)| v.clone())
                        .collect();
                    offset += s.states.len();
                    Stage {
                        action: s.action.clone(),
                        states,
                    }
                })
                .collect();
            PolicySpace::from_stages(stages)
        })
        .collect()
}

/// Finds an imitation admissible subspace `Π′ ⊆ Π` with the fewest input
/// states, or `None` if there is none.
///
/// Every proper subspace of the result has fewer input states and was
/// checked first, so the result is minimal in the sense of Def. 9.5.5. By
/// Thm 9.5.4 every policy in it has an identifiable `P_π(Y)`
/// ([`imitation_distribution`]).
///
/// # Errors
///
/// As [`imitation_backdoor`], plus [`CrlError::TooManyPolicies`] if `Π`
/// has more than 20 input states in total.
pub fn imitation_subspace(
    g: &Admg,
    space: &PolicySpace,
    signals: &[&str],
) -> Result<Option<PolicySpace>, CrlError> {
    check_signals(g, space, signals)?;
    for sub in subspaces(space, false)? {
        if admissible(g, &sub, signals)? {
            return Ok(Some(sub));
        }
    }
    Ok(None)
}

/// Def. 9.5.5: whether `space` is imitation admissible and no proper
/// subspace is.
///
/// # Errors
///
/// As [`imitation_subspace`].
pub fn is_minimal_imitation_admissible(
    g: &Admg,
    space: &PolicySpace,
    signals: &[&str],
) -> Result<bool, CrlError> {
    check_signals(g, space, signals)?;
    if !admissible(g, space, signals)? {
        return Ok(false);
    }
    for sub in subspaces(space, true)? {
        if admissible(g, &sub, signals)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A stochastic policy given by one conditional table per stage, as
/// [`behavioral_cloning`] returns. A state the table lacks gets the uniform
/// distribution over the action's domain.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalPolicy {
    rules: Vec<ConditionalRule>,
}

/// One stage of a [`ConditionalPolicy`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalRule {
    /// The action `X_i`.
    pub action: String,
    /// The inputs `Z_i`.
    pub states: Vec<String>,
    /// The domain of `X_i`.
    pub domain: Vec<String>,
    /// `Z_i` values to `π_i(· | z_i)` over [`ConditionalRule::domain`].
    pub table: BTreeMap<Vec<String>, Vec<f64>>,
}

impl ConditionalPolicy {
    /// The rules, in stage order.
    #[must_use]
    pub fn rules(&self) -> &[ConditionalRule] {
        &self.rules
    }

    /// The rule of action `x`.
    #[must_use]
    pub fn rule(&self, x: &str) -> Option<&ConditionalRule> {
        self.rules.iter().find(|r| r.action == x)
    }
}

impl Policy for ConditionalPolicy {
    fn probability(&self, action: &str, state: &State<'_>, value: &str) -> f64 {
        let Some(rule) = self.rule(action) else {
            return 0.0;
        };
        let Some(i) = rule.domain.iter().position(|v| v == value) else {
            return 0.0;
        };
        let key: Vec<String> = rule
            .states
            .iter()
            .map(|s| state.get(s).unwrap_or("").to_string())
            .collect();
        match rule.table.get(&key) {
            Some(p) => p[i],
            None => 1.0 / rule.domain.len() as f64,
        }
    }
}

impl fmt::Display for ConditionalPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, r) in self.rules.iter().enumerate() {
            if k > 0 {
                writeln!(f)?;
            }
            write!(f, "π({} | {}):", r.action, r.states.join(", "))?;
            for (z, p) in &r.table {
                let dist: Vec<String> = r
                    .domain
                    .iter()
                    .zip(p)
                    .map(|(x, q)| format!("{x}: {q:.4}"))
                    .collect();
                write!(f, " [{}] → {{{}}}", z.join(", "), dist.join(", "))?;
            }
        }
        Ok(())
    }
}

/// Def. 9.5.2: the behavioural cloning policy `π_i(x_i | z_i) = P(x_i | z_i)`
/// for every stage `⟨X_i, Z_i⟩` of `space`, estimated from (weighted)
/// demonstrations.
///
/// It matches the expert under NUC with every action's observed parents in
/// its inputs (Thm 9.5.2), or when `space` is imitation admissible
/// (Thm 9.5.3). Otherwise it can fall far short (Ex. 9.20).
///
/// # Errors
///
/// [`CrlError::Unknown`] for a missing column.
pub fn behavioral_cloning(data: &Dataset, space: &PolicySpace) -> Result<ConditionalPolicy, CrlError> {
    let mut rules = Vec::with_capacity(space.horizon());
    for stage in space.stages() {
        let mut domain = data
            .levels(&stage.action)
            .ok_or_else(|| CrlError::Unknown(stage.action.clone()))?
            .to_vec();
        domain.sort();
        for s in &stage.states {
            if data.levels(s).is_none() {
                return Err(CrlError::Unknown(s.clone()));
            }
        }
        let mut table: BTreeMap<Vec<String>, Vec<f64>> = BTreeMap::new();
        for row in 0..data.len() {
            let w = data.weights().map_or(1.0, |ws| ws[row]);
            if w <= 0.0 {
                continue;
            }
            let key: Vec<String> = stage
                .states
                .iter()
                .map(|s| data.value(s, row).expect("checked").to_string())
                .collect();
            let x = data.value(&stage.action, row).expect("checked");
            let i = domain.iter().position(|d| d == x).expect("level");
            table.entry(key).or_insert_with(|| vec![0.0; domain.len()])[i] += w;
        }
        for p in table.values_mut() {
            let total: f64 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= total;
            }
        }
        rules.push(ConditionalRule {
            action: stage.action.clone(),
            states: stage.states.clone(),
            domain,
            table,
        });
    }
    Ok(ConditionalPolicy { rules })
}

/// The distinct values of `signals` in the data, sorted.
fn signal_values(data: &Dataset, signals: &[&str]) -> Result<Vec<Vec<String>>, CrlError> {
    for s in signals {
        if data.levels(s).is_none() {
            return Err(CrlError::Unknown((*s).to_string()));
        }
    }
    let mut out: Vec<Vec<String>> = Vec::new();
    for row in 0..data.len() {
        let y: Vec<String> = signals
            .iter()
            .map(|s| data.value(s, row).expect("checked").to_string())
            .collect();
        if !out.contains(&y) {
            out.push(y);
        }
    }
    out.sort();
    Ok(out)
}

/// The expert's distribution `P(y)` over the values of `signals`.
///
/// # Errors
///
/// [`CrlError::Unknown`] for a missing column.
pub fn expert_distribution(
    data: &Dataset,
    signals: &[&str],
) -> Result<Vec<(Vec<String>, f64)>, CrlError> {
    let values = signal_values(data, signals)?;
    let mut acc: HashMap<Vec<String>, f64> = HashMap::new();
    let mut total = 0.0;
    for row in 0..data.len() {
        let w = data.weights().map_or(1.0, |ws| ws[row]);
        let y: Vec<String> = signals
            .iter()
            .map(|s| data.value(s, row).expect("checked").to_string())
            .collect();
        *acc.entry(y).or_insert(0.0) += w;
        total += w;
    }
    Ok(values
        .into_iter()
        .map(|y| {
            let p = acc.get(&y).copied().unwrap_or(0.0) / total;
            (y, p)
        })
        .collect())
}

/// Thm 9.5.4: the imitator's distribution
///
/// ```text
/// P_π(y) = Σ_{x̄_H, z̄_H} P(y | x̄_H, z̄_H) Π_i P(z_i | x̄_{i−1}, z̄_{i−1}) π_i(x_i | z_i)
/// ```
///
/// over the values of `signals` the data shows. The formula is the
/// g-formula over the space's histories, so it is computed with [`dp`] on
/// the indicator of each `y`.
///
/// Thm 9.5.4 claims it equals the true `P_π(Y)` for every policy in a
/// minimal imitation admissible space. That needs every stage to satisfy
/// condition 2 ([`ImitationCondition::Blocked`]). A stage admitted by
/// condition 1 alone breaks it: in the bandit `X ↔ Y` with no edge
/// `X → Y`, `{⟨X, ∅⟩}` is minimal admissible and `P_π(y) = P(y)`, but the
/// formula gives `Σ_x P(y | x) π(x)`.
///
/// # Errors
///
/// As [`dp`].
pub fn imitation_distribution(
    data: &Dataset,
    space: &PolicySpace,
    signals: &[&str],
    policy: &dyn Policy,
) -> Result<Vec<(Vec<String>, f64)>, CrlError> {
    let names: Vec<String> = signals.iter().map(|s| (*s).to_string()).collect();
    // `dp` parses rewards as numbers, so compare through a numeric code.
    let coded = code_signals(data, &names)?;
    let mut out = Vec::with_capacity(coded.values.len());
    for y in &coded.values {
        let code_of = coded.code(y);
        let indicator = Reward::custom(coded.names.clone(), move |v| {
            f64::from(u8::from(v.first().copied() == Some(code_of)))
        });
        let p = dp(&coded.data, space, &indicator, policy)?.value;
        out.push((y.clone(), p));
    }
    Ok(out)
}

/// A copy of the data with the joint value of the signals as one numeric
/// column, so that non-numeric signals can feed [`dp`].
struct Coded {
    data: Dataset,
    names: Vec<String>,
    values: Vec<Vec<String>>,
}

impl Coded {
    fn code(&self, y: &[String]) -> f64 {
        self.values.iter().position(|v| v == y).map_or(-1.0, |i| i as f64)
    }
}

const CODE_COLUMN: &str = "__imitation_y__";

fn code_signals(data: &Dataset, signals: &[String]) -> Result<Coded, CrlError> {
    let refs: Vec<&str> = signals.iter().map(String::as_str).collect();
    let values = signal_values(data, &refs)?;
    let codes: Vec<String> = (0..data.len())
        .map(|row| {
            let y: Vec<String> = refs
                .iter()
                .map(|s| data.value(s, row).expect("checked").to_string())
                .collect();
            values.iter().position(|v| *v == y).expect("seen").to_string()
        })
        .collect();
    let data = data.clone().with_column(CODE_COLUMN, codes)?;
    Ok(Coded {
        data,
        names: vec![CODE_COLUMN.to_string()],
        values,
    })
}

/// A feature `φ_i` of the reward signals, for [`causal_mwal`].
pub type Feature<'a> = &'a dyn Fn(&[f64]) -> f64;

/// The result of [`causal_mwal`].
#[derive(Debug, Clone)]
pub struct MwalSolution {
    /// `ν* = min_π max_w wᵀ G π` (Eq. 9.273). At most zero means the
    /// imitator matches or beats the expert for every reward in the class.
    pub value: f64,
    /// The worst-case reward weights `w ∈ P_k`.
    pub weights: Vec<f64>,
    /// The imitating policy: a mixture of deterministic policies, drawn once
    /// per episode, with positive probabilities.
    pub mixture: Vec<(f64, TabularPolicy)>,
    /// `G`, one row per feature and one column per deterministic policy.
    pub game: Vec<Vec<f64>>,
}

/// Causal MWAL (Prop. 9.5.5): the canonical IRL program (Eq. 9.271) for the
/// reward class `R = {w · φ : w ∈ P_k}` over `space`, which should be a
/// minimal imitation admissible subspace ([`imitation_subspace`]) so that
/// every `P_π(Y)` is identified.
///
/// With deterministic policies `π^{(1)}, …, π^{(n)}` it solves the matrix
/// game
///
/// ```text
/// ν* = min_{λ ∈ Δ_n} max_{w ∈ P_k} wᵀ G λ,   G(i, j) = E[φ_i(Y)] − E_{π^{(j)}}[φ_i(Y)]
/// ```
///
/// exactly by linear programming; the multiplicative-weights algorithm of
/// MWAL approximates the same saddle point. `features` receive the signals'
/// values in the order given, parsed as numbers. Deterministic policies
/// range over the input values the data shows, and those violating
/// positivity are skipped.
///
/// # Errors
///
/// [`CrlError::InvalidInput`] without features,
/// [`CrlError::TooManyPolicies`] past 20 000 policies, and otherwise as
/// [`dp`].
pub fn causal_mwal(
    data: &Dataset,
    space: &PolicySpace,
    signals: &[&str],
    features: &[Feature<'_>],
) -> Result<MwalSolution, CrlError> {
    if features.is_empty() {
        return Err(CrlError::InvalidInput("Causal MWAL needs at least one feature".into()));
    }
    let names: Vec<String> = signals.iter().map(|s| (*s).to_string()).collect();
    let policies = deterministic_policies(data, space)?;
    let k = features.len();

    // Expert feature expectations and per-policy ones.
    let mut expert = vec![0.0; k];
    let mut total = 0.0;
    for row in 0..data.len() {
        let w = data.weights().map_or(1.0, |ws| ws[row]);
        let y = parse_signals(data, &names, row)?;
        for (i, f) in features.iter().enumerate() {
            expert[i] += w * f(&y);
        }
        total += w;
    }
    for e in &mut expert {
        *e /= total;
    }

    let mut columns: Vec<(TabularPolicy, Vec<f64>)> = Vec::new();
    let mut first_error = None;
    for pi in policies {
        let mut g = Vec::with_capacity(k);
        let mut ok = true;
        for (i, f) in features.iter().enumerate() {
            match dp_feature(data, space, &names, *f, &pi) {
                Ok(v) => g.push(expert[i] - v),
                Err(e @ CrlError::Positivity { .. }) => {
                    first_error.get_or_insert(e);
                    ok = false;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        if ok {
            columns.push((pi, g));
        }
    }
    if columns.is_empty() {
        return Err(first_error.unwrap_or_else(|| CrlError::Data("no policy to evaluate".into())));
    }
    let n = columns.len();
    let game: Vec<Vec<f64>> = (0..k).map(|i| columns.iter().map(|(_, g)| g[i]).collect()).collect();

    // min v s.t. (Gλ)_i ≤ v, Σλ = 1; v = v⁺ − v⁻.
    let mut lp = Lp::new(n + 2);
    let mut c = vec![0.0; n + 2];
    c[n] = -1.0;
    c[n + 1] = 1.0;
    lp.maximize(c);
    for row in &game {
        let mut a = row.clone();
        a.push(-1.0);
        a.push(1.0);
        lp.constraint(a, Cmp::Le, 0.0);
    }
    let mut sum = vec![1.0; n];
    sum.extend([0.0, 0.0]);
    lp.constraint(sum, Cmp::Eq, 1.0);
    let LpOutcome::Optimal { x, .. } = lp.solve() else {
        return Err(CrlError::Data("the matrix game has no solution".into()));
    };
    let value = x[n] - x[n + 1];

    // The adversary: max u s.t. (wᵀG)_j ≥ u, Σw = 1.
    let mut lp = Lp::new(k + 2);
    let mut c = vec![0.0; k + 2];
    c[k] = 1.0;
    c[k + 1] = -1.0;
    lp.maximize(c);
    for j in 0..n {
        let mut a: Vec<f64> = game.iter().map(|row| row[j]).collect();
        a.push(-1.0);
        a.push(1.0);
        lp.constraint(a, Cmp::Ge, 0.0);
    }
    let mut sum = vec![1.0; k];
    sum.extend([0.0, 0.0]);
    lp.constraint(sum, Cmp::Eq, 1.0);
    let weights = match lp.solve() {
        LpOutcome::Optimal { x, .. } => x[..k].to_vec(),
        _ => vec![1.0 / k as f64; k],
    };

    let mixture = columns
        .into_iter()
        .zip(&x[..n])
        .filter(|(_, l)| **l > 1e-9)
        .map(|((pi, _), l)| (*l, pi))
        .collect();
    Ok(MwalSolution {
        value,
        weights,
        mixture,
        game,
    })
}

fn parse_signals(data: &Dataset, names: &[String], row: usize) -> Result<Vec<f64>, CrlError> {
    names
        .iter()
        .map(|s| {
            let v = data.value(s, row).ok_or_else(|| CrlError::Unknown(s.clone()))?;
            v.trim().parse::<f64>().map_err(|_| CrlError::NonNumericReward {
                column: s.clone(),
                value: v.to_string(),
            })
        })
        .collect()
}

/// `E_π[φ(Y)]` by [`dp`], with `φ` borrowed for the duration of the call.
fn dp_feature(
    data: &Dataset,
    space: &PolicySpace,
    names: &[String],
    f: &dyn Fn(&[f64]) -> f64,
    pi: &TabularPolicy,
) -> Result<f64, CrlError> {
    // `Reward::custom` needs a 'static closure, so tabulate φ over the
    // signal values the data shows and look it up by value.
    let mut table: Vec<(Vec<u64>, f64)> = Vec::new();
    for row in 0..data.len() {
        let y = parse_signals(data, names, row)?;
        let key: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
        if !table.iter().any(|(k, _)| *k == key) {
            table.push((key, f(&y)));
        }
    }
    let reward = Reward::custom(names.to_vec(), move |y| {
        let key: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
        table
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(0.0, |(_, v)| *v)
    });
    Ok(dp(data, space, &reward, pi)?.value)
}

/// Every deterministic policy over the input values each stage shows in the
/// data.
fn deterministic_policies(data: &Dataset, space: &PolicySpace) -> Result<Vec<TabularPolicy>, CrlError> {
    let mut domains = Vec::with_capacity(space.horizon());
    let mut states: Vec<Vec<Vec<String>>> = Vec::with_capacity(space.horizon());
    for stage in space.stages() {
        let mut d = data
            .levels(&stage.action)
            .ok_or_else(|| CrlError::Unknown(stage.action.clone()))?
            .to_vec();
        d.sort();
        domains.push(d);
        let mut seen: Vec<Vec<String>> = Vec::new();
        for row in 0..data.len() {
            if data.weights().is_some_and(|w| w[row] <= 0.0) {
                continue;
            }
            let s: Vec<String> = stage
                .states
                .iter()
                .map(|v| {
                    data.value(v, row)
                        .map(str::to_string)
                        .ok_or_else(|| CrlError::Unknown(v.clone()))
                })
                .collect::<Result<_, _>>()?;
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen.sort();
        states.push(seen);
    }
    let digits: Vec<(usize, usize)> = states
        .iter()
        .enumerate()
        .flat_map(|(i, ss)| (0..ss.len()).map(move |j| (i, j)))
        .collect();
    let count: f64 = digits.iter().map(|(i, _)| domains[*i].len() as f64).product();
    if count > POLICY_CAP as f64 {
        return Err(CrlError::TooManyPolicies {
            count,
            cap: POLICY_CAP,
        });
    }
    let mut out = Vec::with_capacity(count as usize);
    let mut counter = vec![0usize; digits.len()];
    loop {
        let mut pi = TabularPolicy::new(space, domains.clone());
        for (d, (i, j)) in digits.iter().enumerate() {
            let s: Vec<&str> = states[*i][*j].iter().map(String::as_str).collect();
            pi.set(&space.stages()[*i].action, &s, &domains[*i][counter[d]]);
        }
        out.push(pi);
        let mut d = digits.len();
        loop {
            if d == 0 {
                return Ok(out);
            }
            d -= 1;
            counter[d] += 1;
            if counter[d] < domains[digits[d].0].len() {
                break;
            }
            counter[d] = 0;
        }
    }
}
