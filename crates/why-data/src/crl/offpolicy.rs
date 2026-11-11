//! Off-policy evaluation and learning from data: IPW (Thm 8.2.1), DP
//! (Thm 8.2.2) and policy search.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use super::{CrlError, Dataset, Policy, PolicySpace, Reward, State, TabularPolicy};

/// Tolerance for a decision rule summing to one.
const NORMALIZATION_TOL: f64 = 1e-9;

/// The most policies [`optimize`] will enumerate without perfect recall.
const POLICY_CAP: usize = 200_000;

/// A data set reduced to what the estimators need.
struct Prepared<'a> {
    space: &'a PolicySpace,
    /// Rows with positive weight, and their weights.
    rows: Vec<usize>,
    weights: Vec<f64>,
    /// Per variable of `space.order()`: one code per row, and the levels.
    codes: Vec<&'a [u32]>,
    levels: Vec<&'a [String]>,
    /// `R(Y)` per row (indexed like `rows`).
    reward: Vec<f64>,
    /// Positions in `space.order()` of each stage's `S_i`.
    state_pos: Vec<Vec<usize>>,
}

/// Why a cell has no value: the first positivity failure beneath it.
#[derive(Clone)]
struct Witness {
    stage: usize,
    key: Vec<u32>,
    action: String,
}

type Cell = Result<f64, Witness>;

impl<'a> Prepared<'a> {
    fn new(data: &'a Dataset, space: &'a PolicySpace, reward: &Reward) -> Result<Self, CrlError> {
        let order = space.order();
        let mut codes = Vec::with_capacity(order.len());
        let mut levels = Vec::with_capacity(order.len());
        for v in order {
            let col = data.column(v).map_err(|_| CrlError::Unknown(v.clone()))?;
            codes.push(col.codes.as_slice());
            levels.push(col.levels.as_slice());
        }
        let mut signal_cols = Vec::with_capacity(reward.signals().len());
        for y in reward.signals() {
            signal_cols.push(data.column(y).map_err(|_| CrlError::Unknown(y.clone()))?);
        }
        if reward.signals().iter().any(|y| order.contains(y)) {
            return Err(CrlError::InvalidInput(
                "a reward signal is also an action or input state".into(),
            ));
        }

        let mut rows = Vec::new();
        let mut weights = Vec::new();
        let mut rewards = Vec::new();
        for row in 0..data.len() {
            let w = data.weights().map_or(1.0, |ws| ws[row]);
            if w <= 0.0 {
                continue;
            }
            let r = reward.apply_str(
                signal_cols
                    .iter()
                    .map(|c| c.levels[c.codes[row] as usize].as_str()),
            )?;
            rows.push(row);
            weights.push(w);
            rewards.push(r);
        }
        if rows.is_empty() {
            return Err(CrlError::Data("no rows with positive weight".into()));
        }

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
        Ok(Self {
            space,
            rows,
            weights,
            codes,
            levels,
            reward: rewards,
            state_pos,
        })
    }

    /// The codes of the first `len` variables of `order` in row `k`.
    fn key(&self, k: usize, len: usize) -> Vec<u32> {
        let row = self.rows[k];
        self.codes[..len].iter().map(|c| c[row]).collect()
    }

    fn value(&self, pos: usize, code: u32) -> &str {
        &self.levels[pos][code as usize]
    }

    /// `(variable, value)` pairs for a prefix key.
    fn pairs(&self, key: &[u32]) -> Vec<(String, String)> {
        key.iter()
            .enumerate()
            .map(|(p, c)| (self.space.order()[p].clone(), self.value(p, *c).to_string()))
            .collect()
    }

    /// `π_i(x | s_i)` for every level `x` of `X_i`, given a history key.
    fn rule(&self, policy: &dyn Policy, i: usize, key: &[u32]) -> Result<Vec<f64>, CrlError> {
        let order = self.space.order();
        let pairs: Vec<(&str, &str)> = self.state_pos[i]
            .iter()
            .map(|&p| (order[p].as_str(), self.value(p, key[p])))
            .collect();
        let state = State::new(&pairs);
        let a = self.space.history_len(i);
        let action = &self.space.stages()[i].action;
        let probs: Vec<f64> = self.levels[a]
            .iter()
            .map(|x| policy.probability(action, &state, x))
            .collect();
        let total: f64 = probs.iter().sum();
        if probs.iter().any(|p| *p < 0.0) || (total - 1.0).abs() > NORMALIZATION_TOL {
            return Err(CrlError::PolicyNotNormalized {
                action: action.clone(),
                state: state.to_string(),
                total,
            });
        }
        Ok(probs)
    }

    fn positivity(&self, w: &Witness) -> CrlError {
        CrlError::Positivity {
            stage: w.stage + 1,
            history: self.pairs(&w.key),
            action: w.action.clone(),
        }
    }

    fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Weighted means of `per_row` within each cell of prefix length `len`.
    fn cell_means(&self, len: usize, per_row: &[Cell]) -> BTreeMap<Vec<u32>, Cell> {
        let mut acc: BTreeMap<Vec<u32>, (f64, Cell)> = BTreeMap::new();
        for (k, cell) in per_row.iter().enumerate() {
            let w = self.weights[k];
            let entry = acc.entry(self.key(k, len)).or_insert((0.0, Ok(0.0)));
            entry.0 += w;
            entry.1 = match (&entry.1, cell) {
                (Ok(s), Ok(v)) => Ok(s + w * v),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            };
        }
        acc.into_iter()
            .map(|(key, (w, s))| (key, s.map(|s| s / w)))
            .collect()
    }

    /// Backward recursion shared by evaluation and backward induction.
    ///
    /// `choose` maps the Q-values of one history (one per level of `X_i`,
    /// `None` where the data never shows that action) to the value-to-go.
    fn backward<F>(&self, mut choose: F) -> Result<(Cell, Vec<QTable>), CrlError>
    where
        F: FnMut(usize, &[u32], &[Option<Cell>]) -> Result<Cell, CrlError>,
    {
        let h = self.space.horizon();
        let mut to_go: Vec<Cell> = self.reward.iter().map(|r| Ok(*r)).collect();
        let mut tables = Vec::with_capacity(h);
        for i in (0..h).rev() {
            let len = self.space.history_len(i);
            let q = self.cell_means(len + 1, &to_go);
            tables.push(self.q_table(i, &q));

            let mut by_history: BTreeMap<Vec<u32>, Vec<Option<Cell>>> = BTreeMap::new();
            let n_actions = self.levels[len].len();
            for (key, cell) in &q {
                let slot = by_history
                    .entry(key[..len].to_vec())
                    .or_insert_with(|| vec![None; n_actions]);
                slot[key[len] as usize] = Some(cell.clone());
            }
            let mut v: HashMap<Vec<u32>, Cell> = HashMap::with_capacity(by_history.len());
            for (hist, cells) in &by_history {
                v.insert(hist.clone(), choose(i, hist, cells)?);
            }
            to_go = (0..self.rows.len())
                .map(|k| v[&self.key(k, len)].clone())
                .collect();
        }
        tables.reverse();

        let total = self.total_weight();
        let mut value: Cell = Ok(0.0);
        for (k, cell) in to_go.iter().enumerate() {
            value = match (&value, cell) {
                (Ok(s), Ok(v)) => Ok(s + self.weights[k] * v),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            };
        }
        Ok((value.map(|s| s / total), tables))
    }

    fn q_table(&self, i: usize, q: &BTreeMap<Vec<u32>, Cell>) -> QTable {
        let len = self.space.history_len(i) + 1;
        let mut rows: Vec<(Vec<String>, Option<f64>)> = q
            .iter()
            .map(|(key, cell)| {
                (
                    key.iter()
                        .enumerate()
                        .map(|(p, c)| self.value(p, *c).to_string())
                        .collect(),
                    cell.as_ref().ok().copied(),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        QTable {
            action: self.space.stages()[i].action.clone(),
            columns: self.space.order()[..len].to_vec(),
            rows,
        }
    }
}

/// A value function `Q^{(i)}(x̄_i, s̄_i)` tabulated over the cells the data
/// shows.
#[derive(Debug, Clone, PartialEq)]
pub struct QTable {
    /// The action `X_i` of this stage.
    pub action: String,
    /// The history `X̄_{i−1} ∪ S̄_i` followed by `X_i`.
    pub columns: Vec<String>,
    /// One row per cell, sorted by value. `None` marks a cell whose value
    /// depends on an action the data never shows after some later history.
    pub rows: Vec<(Vec<String>, Option<f64>)>,
}

impl QTable {
    /// The value at a cell, given values in [`QTable::columns`] order.
    #[must_use]
    pub fn get(&self, values: &[&str]) -> Option<f64> {
        self.rows
            .iter()
            .find(|(k, _)| k.iter().map(String::as_str).eq(values.iter().copied()))
            .and_then(|(_, v)| *v)
    }
}

impl fmt::Display for QTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths: Vec<usize> = self.columns.iter().map(|c| c.len().max(1)).collect();
        for (c, w) in self.columns.iter().zip(&widths) {
            write!(f, "{c:>w$}  ")?;
        }
        writeln!(f, "Q({})", self.action)?;
        for (i, (key, v)) in self.rows.iter().enumerate() {
            for (x, w) in key.iter().zip(&widths) {
                write!(f, "{x:>w$}  ")?;
            }
            match v {
                Some(v) => write!(f, "{v:.4}")?,
                None => write!(f, "  —")?,
            }
            if i + 1 < self.rows.len() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// The result of [`dp`].
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// The estimate of `E_π[R(Y)]`.
    pub value: f64,
    /// `Q^{(1)}_π, …, Q^{(H)}_π`, in stage order.
    pub q: Vec<QTable>,
}

/// The result of [`optimize`].
#[derive(Debug, Clone)]
pub struct Plan {
    /// The best deterministic policy found.
    pub policy: TabularPolicy,
    /// Its estimated value.
    pub value: f64,
    /// The value functions: `Q*` from backward induction, or `Q_π` of the
    /// winner when the search was exhaustive.
    pub q: Vec<QTable>,
}

/// Inverse propensity weighting (Thm 8.2.1):
///
/// ```text
/// E_π[R(Y)] = Σ E[R(Y) | x̄_H, s̄_H] P(x̄_H, s̄_H) Π_i π_i(x_i | s_i) / P(x_i | x̄_{i−1}, s̄_i)
/// ```
///
/// The propensities are the empirical (weighted) conditional frequencies.
/// On the exact table of `P(V)` this is the population estimand. On a
/// sample it is the plug-in IPW estimator.
///
/// # Errors
///
/// * [`CrlError::Unknown`] if a column is missing.
/// * [`CrlError::NonNumericReward`] if a reward signal does not parse.
/// * [`CrlError::PolicyNotNormalized`] if a rule does not sum to one over
///   the levels the data shows for its action.
/// * [`CrlError::Positivity`] if the policy may take an action the data
///   never shows after a history it reaches.
pub fn ipw(
    data: &Dataset,
    space: &PolicySpace,
    reward: &Reward,
    policy: &dyn Policy,
) -> Result<f64, CrlError> {
    let p = Prepared::new(data, space, reward)?;
    let h = space.horizon();

    // W(h_i) and W(h_i, x_i) for every stage.
    let mut hist_w: Vec<HashMap<Vec<u32>, f64>> = vec![HashMap::new(); h];
    let mut cell_w: Vec<HashMap<Vec<u32>, f64>> = vec![HashMap::new(); h];
    for k in 0..p.rows.len() {
        for i in 0..h {
            let len = space.history_len(i);
            let cell = p.key(k, len + 1);
            *hist_w[i].entry(cell[..len].to_vec()).or_insert(0.0) += p.weights[k];
            *cell_w[i].entry(cell).or_insert(0.0) += p.weights[k];
        }
    }

    let mut rules: Vec<HashMap<Vec<u32>, Vec<f64>>> = vec![HashMap::new(); h];
    let mut reached: Vec<HashSet<Vec<u32>>> = vec![HashSet::new(); h];
    let mut total = 0.0;
    for k in 0..p.rows.len() {
        let mut ratio = 1.0;
        for i in 0..h {
            let len = space.history_len(i);
            let cell = p.key(k, len + 1);
            let hist = &cell[..len];
            if !rules[i].contains_key(hist) {
                let probs = p.rule(policy, i, hist)?;
                rules[i].insert(hist.to_vec(), probs);
            }
            reached[i].insert(hist.to_vec());
            let pi = rules[i][hist][cell[len] as usize];
            if pi == 0.0 {
                ratio = 0.0;
                break;
            }
            let propensity = cell_w[i][&cell] / hist_w[i][hist];
            ratio *= pi / propensity;
        }
        total += p.weights[k] * p.reward[k] * ratio;
    }

    // Every action the policy may take after a reached history must have
    // been observed there, or its term is silently missing from the sum.
    for i in 0..h {
        let len = space.history_len(i);
        let mut hs: Vec<&Vec<u32>> = reached[i].iter().collect();
        hs.sort();
        for hist in hs {
            for (x, pi) in rules[i][hist].iter().enumerate() {
                let mut cell = hist.clone();
                cell.push(x as u32);
                if *pi > 0.0 && !cell_w[i].contains_key(&cell) {
                    return Err(p.positivity(&Witness {
                        stage: i,
                        key: hist.clone(),
                        action: p.levels[len][x].clone(),
                    }));
                }
            }
        }
    }
    Ok(total / p.total_weight())
}

/// Dynamic programming (Thm 8.2.2): for `i = H, …, 1`,
///
/// ```text
/// Q^{(H)}(x̄_H, s̄_H) = E[R(Y) | x̄_H, s̄_H]
/// Q^{(i)}(x̄_i, s̄_i) = E[Σ_{x_{i+1}} Q^{(i+1)}(x̄_{i+1}, s̄_{i+1}) π_{i+1}(x_{i+1} | S_{i+1}) | x̄_i, s̄_i]
/// E_π[R(Y)]          = E[Σ_{x_1} Q^{(1)}(x_1, S_1) π_1(x_1 | S_1)]
/// ```
///
/// Expectations are weighted means over the data. This is the same plug-in
/// functional as [`ipw`], so the two agree to rounding error on any data
/// set.
///
/// # Errors
///
/// As [`ipw`].
pub fn dp(
    data: &Dataset,
    space: &PolicySpace,
    reward: &Reward,
    policy: &dyn Policy,
) -> Result<Evaluation, CrlError> {
    let p = Prepared::new(data, space, reward)?;
    evaluate(&p, policy)
}

fn evaluate(p: &Prepared<'_>, policy: &dyn Policy) -> Result<Evaluation, CrlError> {
    let space = p.space;
    let (value, q) = p.backward(|i, hist, cells| {
        let probs = p.rule(policy, i, hist)?;
        let mut v = 0.0;
        for (x, pi) in probs.iter().enumerate() {
            if *pi == 0.0 {
                continue;
            }
            match &cells[x] {
                Some(Ok(qv)) => v += pi * qv,
                Some(Err(w)) => return Ok(Err(w.clone())),
                None => {
                    let a = space.history_len(i);
                    return Ok(Err(Witness {
                        stage: i,
                        key: hist.to_vec(),
                        action: p.levels[a][x].clone(),
                    }));
                }
            }
        }
        Ok(Ok(v))
    })?;
    let value = value.map_err(|w| p.positivity(&w))?;
    Ok(Evaluation { value, q })
}

/// Finds the deterministic policy in `space` with the highest estimated
/// value on `data`.
///
/// Deterministic policies are enough: in a fixed CDM a stochastic policy is
/// always matched by a deterministic one (footnote 10 of Sec. 8.2.2).
///
/// * **Perfect recall** ([`PolicySpace::has_perfect_recall`]): backward
///   induction (Eqs. 8.36–8.39). `π_i*(h) = argmax_x Q*^{(i)}(h, x)` over
///   the actions the data shows after `h`. It is exact, and its cost is
///   linear in the data.
/// * **Otherwise**, every assignment of an action to each observed value of
///   each `S_i` is evaluated with [`dp`]. Policies that violate positivity
///   are skipped. This is exponential, and capped at 200 000 policies.
///
/// Ties go to the earlier action, in the order the data first shows them.
///
/// # Errors
///
/// [`CrlError::TooManyPolicies`] past the cap. [`CrlError::Positivity`] if no
/// candidate is evaluable. Otherwise as [`ipw`].
pub fn optimize(data: &Dataset, space: &PolicySpace, reward: &Reward) -> Result<Plan, CrlError> {
    let p = Prepared::new(data, space, reward)?;
    let domains: Vec<Vec<String>> = (0..space.horizon())
        .map(|i| p.levels[space.history_len(i)].to_vec())
        .collect();
    if space.has_perfect_recall() {
        backward_induction(&p, domains)
    } else {
        exhaustive(&p, domains, POLICY_CAP)
    }
}

/// [`optimize`] forced to enumerate, to cross-check backward induction.
#[cfg(test)]
pub(super) fn optimize_exhaustive(
    data: &Dataset,
    space: &PolicySpace,
    reward: &Reward,
    cap: usize,
) -> Result<Plan, CrlError> {
    let p = Prepared::new(data, space, reward)?;
    let domains: Vec<Vec<String>> = (0..space.horizon())
        .map(|i| p.levels[space.history_len(i)].to_vec())
        .collect();
    exhaustive(&p, domains, cap)
}

fn backward_induction(p: &Prepared<'_>, domains: Vec<Vec<String>>) -> Result<Plan, CrlError> {
    let space = p.space;
    let mut policy = TabularPolicy::new(space, domains);
    let (value, q) = p.backward(|i, hist, cells| {
        let mut best: Option<(usize, f64)> = None;
        for (x, cell) in cells.iter().enumerate() {
            if let Some(Ok(v)) = cell
                && best.is_none_or(|(_, b)| *v > b)
            {
                best = Some((x, *v));
            }
        }
        let (x, v) = best.expect("every observed history has an observed action");
        let a = space.history_len(i);
        let state: Vec<&str> = p.state_pos[i].iter().map(|&s| p.value(s, hist[s])).collect();
        policy.set(&space.stages()[i].action, &state, &p.levels[a][x]);
        Ok(Ok(v))
    })?;
    let value = value.map_err(|w| p.positivity(&w))?;
    Ok(Plan { policy, value, q })
}

fn exhaustive(p: &Prepared<'_>, domains: Vec<Vec<String>>, cap: usize) -> Result<Plan, CrlError> {
    let space = p.space;
    // The observed values of each S_i.
    let states: Vec<Vec<Vec<String>>> = (0..space.horizon())
        .map(|i| {
            let mut seen: Vec<Vec<String>> = Vec::new();
            for k in 0..p.rows.len() {
                let row = p.rows[k];
                let s: Vec<String> = p.state_pos[i]
                    .iter()
                    .map(|&pos| p.value(pos, p.codes[pos][row]).to_string())
                    .collect();
                if !seen.contains(&s) {
                    seen.push(s);
                }
            }
            seen.sort();
            seen
        })
        .collect();

    // One digit per (stage, state), in base |D(X_i)|.
    let digits: Vec<(usize, usize)> = states
        .iter()
        .enumerate()
        .flat_map(|(i, ss)| (0..ss.len()).map(move |j| (i, j)))
        .collect();
    let count: f64 = digits
        .iter()
        .map(|(i, _)| domains[*i].len() as f64)
        .product();
    if count > cap as f64 {
        return Err(CrlError::TooManyPolicies { count, cap });
    }

    let mut counter = vec![0usize; digits.len()];
    let mut best: Option<Plan> = None;
    let mut first_error = None;
    loop {
        let mut policy = TabularPolicy::new(space, domains.clone());
        for (d, (i, j)) in digits.iter().enumerate() {
            let s: Vec<&str> = states[*i][*j].iter().map(String::as_str).collect();
            policy.set(&space.stages()[*i].action, &s, &domains[*i][counter[d]]);
        }
        match evaluate(p, &policy) {
            Ok(eval) => {
                if best.as_ref().is_none_or(|b| eval.value > b.value) {
                    best = Some(Plan {
                        policy,
                        value: eval.value,
                        q: eval.q,
                    });
                }
            }
            Err(e @ CrlError::Positivity { .. }) => {
                first_error.get_or_insert(e);
            }
            Err(e) => return Err(e),
        }

        // Lexicographic odometer, last digit fastest, so that ties keep the
        // earliest actions.
        let mut d = digits.len();
        loop {
            if d == 0 {
                return best.ok_or_else(|| first_error.expect("at least one policy was tried"));
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
