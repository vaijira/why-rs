//! Domain adaptation with labelled target samples (Sec. 11.4): Algorithms
//! 34–38.
//!
//! Every learner here is the non-parametric empirical risk minimiser of the
//! negative log-likelihood over all conditionals `µ(y | x)` on a finite
//! support, i.e. cell frequencies ([`CategoricalModel`]), optionally with
//! additive smoothing `alpha` so that held-out losses stay finite.

use std::collections::HashMap;

use super::{Dataset, TransportError, subsets};
use crate::fairness::SplitMix64;

/// The most candidate predictors the agnostic searches will fit.
const CANDIDATE_CAP: usize = 200_000;
/// The most completions of the intermediate tokens a sequential prediction
/// sums over.
const COMPLETION_CAP: usize = 1_000_000;

/// Rows of one table, read as `(features, outcome)` through a column mapping.
struct Sample<'a> {
    data: &'a Dataset,
    x: Vec<String>,
    y: String,
    rows: Option<Vec<usize>>,
}

impl<'a> Sample<'a> {
    fn all(data: &'a Dataset, x: &[&str], y: &str) -> Self {
        Self {
            data,
            x: x.iter().map(|s| (*s).to_string()).collect(),
            y: y.to_string(),
            rows: None,
        }
    }

    fn rows(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match &self.rows {
            Some(r) => Box::new(r.iter().copied()),
            None => Box::new(0..self.data.len()),
        }
    }

    fn check(&self) -> Result<(), TransportError> {
        for c in self.x.iter().chain(std::iter::once(&self.y)) {
            if self.data.levels(c).is_none() {
                return Err(TransportError::MissingData(format!("no column `{c}`")));
            }
        }
        Ok(())
    }
}

/// A conditional distribution `P̂(y | x)` on a finite support, estimated by
/// (smoothed) cell frequencies: the negative log-likelihood minimiser of
/// Eq. 11.245.
#[derive(Clone, Debug)]
pub struct CategoricalModel {
    alpha: f64,
    y_levels: Vec<String>,
    counts: HashMap<Vec<String>, Vec<f64>>,
}

impl CategoricalModel {
    fn fit(samples: &[Sample<'_>], y_levels: &[String], alpha: f64) -> Result<Self, TransportError> {
        let mut counts: HashMap<Vec<String>, Vec<f64>> = HashMap::new();
        for s in samples {
            s.check()?;
            for row in s.rows() {
                let key: Vec<String> = s
                    .x
                    .iter()
                    .map(|c| s.data.value(c, row).unwrap_or_default().to_string())
                    .collect();
                let yv = s.data.value(&s.y, row).unwrap_or_default();
                let Some(k) = y_levels.iter().position(|l| l == yv) else {
                    continue;
                };
                counts.entry(key).or_insert_with(|| vec![0.0; y_levels.len()])[k] +=
                    s.data.weight(row);
            }
        }
        Ok(Self {
            alpha,
            y_levels: y_levels.to_vec(),
            counts,
        })
    }

    /// The outcome levels, in the order [`CategoricalModel::distribution`]
    /// reports them.
    #[must_use]
    pub fn levels(&self) -> &[String] {
        &self.y_levels
    }

    /// `P̂(· | x)` over [`CategoricalModel::levels`]. A cell never seen gets
    /// the uniform distribution.
    #[must_use]
    pub fn distribution(&self, x: &[&str]) -> Vec<f64> {
        let k = self.y_levels.len();
        let key: Vec<String> = x.iter().map(|s| (*s).to_string()).collect();
        match self.counts.get(&key) {
            Some(c) => {
                let total: f64 = c.iter().sum::<f64>() + self.alpha * k as f64;
                if total <= 0.0 {
                    vec![1.0 / k as f64; k]
                } else {
                    c.iter().map(|n| (n + self.alpha) / total).collect()
                }
            }
            None => vec![1.0 / k as f64; k],
        }
    }

    /// `P̂(y | x)`; zero for a level never seen in training.
    #[must_use]
    pub fn probability(&self, x: &[&str], y: &str) -> f64 {
        self.y_levels
            .iter()
            .position(|l| l == y)
            .map_or(0.0, |k| self.distribution(x)[k])
    }

    /// The most likely level given `x`; ties go to the earlier level.
    #[must_use]
    pub fn predict(&self, x: &[&str]) -> &str {
        let d = self.distribution(x);
        let mut best = 0;
        for (i, p) in d.iter().enumerate() {
            if *p > d[best] {
                best = i;
            }
        }
        &self.y_levels[best]
    }

    /// The mean negative log-likelihood on `data`, reading the features from
    /// `x` (in the model's order) and the outcome from `y`.
    ///
    /// # Errors
    ///
    /// [`TransportError::MissingData`] for a missing column or no rows.
    pub fn log_loss(&self, data: &Dataset, x: &[&str], y: &str) -> Result<f64, TransportError> {
        self.loss(&Sample::all(data, x, y))
    }

    fn loss(&self, s: &Sample<'_>) -> Result<f64, TransportError> {
        s.check()?;
        let mut total = 0.0;
        let mut weight = 0.0;
        for row in s.rows() {
            let key: Vec<&str> = s
                .x
                .iter()
                .map(|c| s.data.value(c, row).unwrap_or_default())
                .collect();
            let w = s.data.weight(row);
            let p = self.probability(&key, s.data.value(&s.y, row).unwrap_or_default());
            total -= w * p.ln();
            weight += w;
        }
        if weight <= 0.0 {
            return Err(TransportError::MissingData("no rows to evaluate on".into()));
        }
        Ok(total / weight)
    }
}

fn outcome_levels<'a>(data: impl IntoIterator<Item = (&'a Dataset, &'a str)>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (d, y) in data {
        for l in d.levels(y).unwrap_or_default() {
            if !out.contains(l) {
                out.push(l.clone());
            }
        }
    }
    out
}

/// A random half of the target rows for training, the rest for validation.
fn split(n: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    let test = idx.split_off(n / 2);
    (idx, test)
}

/// `J = { j : Y ∉ Δ^{j,*} }`.
fn shared(delta: &[&[&str]], y: &str, k: usize) -> Result<Vec<usize>, TransportError> {
    if delta.len() != k {
        return Err(TransportError::InvalidInput(format!(
            "{} discrepancy sets for {k} sources",
            delta.len()
        )));
    }
    Ok((0..k).filter(|j| !delta[*j].contains(&y)).collect())
}

/// Algorithm 34, structure-informed domain adaptation (`X → Y`): pool the
/// target data with every source `j` whose mechanism for `Y` matches the
/// target (`Y ∉ Δ^{j,*}`, so `P^j(y | x) = P*(y | x)` in a Markovian model),
/// and fit `P̂(y | x)` on the pool. Returns the model and `J`.
///
/// With `J ≠ ∅` the excess risk decays with the source size `N`
/// (Lemma 11.4.1); otherwise only the target's `n` samples count.
///
/// # Errors
///
/// [`TransportError::InvalidInput`] if `delta` has the wrong length, and
/// [`TransportError::MissingData`] for a missing column.
pub fn structure_informed_da(
    sources: &[&Dataset],
    target: &Dataset,
    delta: &[&[&str]],
    features: &[&str],
    y: &str,
    alpha: f64,
) -> Result<(CategoricalModel, Vec<usize>), TransportError> {
    let j = shared(delta, y, sources.len())?;
    let mut samples: Vec<Sample<'_>> = j
        .iter()
        .map(|&j| Sample::all(sources[j], features, y))
        .collect();
    samples.push(Sample::all(target, features, y));
    let levels = outcome_levels(sources.iter().map(|d| (*d, y)).chain([(target, y)]));
    Ok((CategoricalModel::fit(&samples, &levels, alpha)?, j))
}

/// The predictor a structure-agnostic search selected.
#[derive(Clone, Debug)]
pub struct AgnosticChoice {
    /// The sources pooled with the target's training half.
    pub sources: Vec<usize>,
    /// The selected predictor, fitted on those sources and the training half.
    pub model: CategoricalModel,
    /// Its loss on the validation half.
    pub loss: f64,
    /// The validation loss of every candidate, in the order tried.
    pub losses: Vec<(Vec<usize>, f64)>,
}

/// Algorithm 35, structure-agnostic domain adaptation (`X → Y`): without
/// `Δ`, fit one predictor per subset `S` of sources on `S` plus a random half
/// of the target, and keep the one with the lowest negative log-likelihood
/// on the other half. Its target risk is within `O(√(K/n))` of
/// [`structure_informed_da`]'s (Thm 11.4.2).
///
/// # Errors
///
/// [`TransportError::TooLarge`] beyond 16 sources, and
/// [`TransportError::MissingData`] for a missing column or a target too small
/// to split.
pub fn structure_agnostic_da(
    sources: &[&Dataset],
    target: &Dataset,
    features: &[&str],
    y: &str,
    alpha: f64,
    seed: u64,
) -> Result<AgnosticChoice, TransportError> {
    if sources.len() > 16 {
        return Err(TransportError::TooLarge(format!(
            "{} sources give too many subsets",
            sources.len()
        )));
    }
    let (train, test) = split(target.len(), seed);
    let levels = outcome_levels(sources.iter().map(|d| (*d, y)).chain([(target, y)]));
    let mut held = Sample::all(target, features, y);
    held.rows = Some(test);
    let ids: Vec<usize> = (0..sources.len()).collect();
    let mut best: Option<AgnosticChoice> = None;
    let mut losses = Vec::new();
    for s in subsets(&ids)? {
        let mut samples: Vec<Sample<'_>> = s
            .iter()
            .map(|&j| Sample::all(sources[j], features, y))
            .collect();
        let mut tr = Sample::all(target, features, y);
        tr.rows = Some(train.clone());
        samples.push(tr);
        let model = CategoricalModel::fit(&samples, &levels, alpha)?;
        let loss = model.loss(&held)?;
        losses.push((s.clone(), loss));
        if best.as_ref().is_none_or(|b| loss < b.loss) {
            best = Some(AgnosticChoice {
                sources: s,
                model,
                loss,
                losses: Vec::new(),
            });
        }
    }
    let mut best = best.expect("the empty subset is always tried");
    best.losses = losses;
    Ok(best)
}

/// Algorithm 36, structure-informed domain adaptation with several causes:
/// `Y`'s parents may differ across domains while its mechanism is shared.
///
/// Every source `j` with `Y ∉ Δ^{j,*}` contributes its rows with its ordered
/// parents `source_parents[j]` renamed to canonical positions `R_1..R_c`,
/// `c = |Pa*_Y|`, and so does the target with `target_parents`. The returned
/// model is `µ(y | r_1..r_c)`; predict a target row by passing the values of
/// `target_parents` in order. Returns the model and `J`.
///
/// # Errors
///
/// [`TransportError::InvalidInput`] if a pooled source's parent list has a
/// different length from the target's (a shared mechanism has one arity),
/// and [`TransportError::MissingData`] for a missing column.
pub fn multi_cause_da(
    sources: &[&Dataset],
    target: &Dataset,
    delta: &[&[&str]],
    source_parents: &[&[&str]],
    target_parents: &[&str],
    y: &str,
    alpha: f64,
) -> Result<(CategoricalModel, Vec<usize>), TransportError> {
    let j = shared(delta, y, sources.len())?;
    if source_parents.len() != sources.len() {
        return Err(TransportError::InvalidInput(format!(
            "{} parent lists for {} sources",
            source_parents.len(),
            sources.len()
        )));
    }
    let c = target_parents.len();
    let mut samples = Vec::new();
    for &jj in &j {
        if source_parents[jj].len() != c {
            return Err(TransportError::InvalidInput(format!(
                "source {jj} shares Y's mechanism but has {} parents, the target {c}",
                source_parents[jj].len()
            )));
        }
        samples.push(Sample::all(sources[jj], source_parents[jj], y));
    }
    samples.push(Sample::all(target, target_parents, y));
    let levels = outcome_levels(sources.iter().map(|d| (*d, y)).chain([(target, y)]));
    Ok((CategoricalModel::fit(&samples, &levels, alpha)?, j))
}

/// The predictor [`multi_cause_agnostic_da`] selected.
#[derive(Clone, Debug)]
pub struct MultiCauseChoice {
    /// The ordered covariates read in the target.
    pub target_parents: Vec<String>,
    /// The ordered covariates read in the pooled sources, if any were pooled.
    pub source_parents: Option<Vec<String>>,
    /// The pooled sources.
    pub sources: Vec<usize>,
    /// The selected predictor over canonical positions.
    pub model: CategoricalModel,
    /// Its validation loss.
    pub loss: f64,
    /// How many candidates were fitted.
    pub candidates: usize,
}

/// Every ordered selection of `c` items.
fn arrangements(items: &[String], c: usize) -> Vec<Vec<String>> {
    if c == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for (i, it) in items.iter().enumerate() {
        let rest: Vec<String> = items
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.clone())
            .collect();
        for mut tail in arrangements(&rest, c - 1) {
            tail.insert(0, it.clone());
            out.push(tail);
        }
    }
    out
}

/// Algorithm 37, structure-agnostic domain adaptation with several causes:
/// without `Δ` or the parent sets, try every ordered covariate list for the
/// target, alone and pooled with every non-empty source subset read through
/// every ordered list of the same length, and keep the predictor with the
/// lowest validation loss. Each pooled source subset is read through one
/// shared ordering.
///
/// For three covariates and one source that is the 98 candidates of
/// Eq. 11.253. The excess risk over Algorithm 36 is
/// `O(√(K·M·log M / n))` (Prop. 11.4.4).
///
/// # Errors
///
/// [`TransportError::TooLarge`] beyond 200 000 candidates, and
/// [`TransportError::MissingData`] for a missing column.
pub fn multi_cause_agnostic_da(
    sources: &[&Dataset],
    target: &Dataset,
    covariates: &[&str],
    y: &str,
    alpha: f64,
    seed: u64,
) -> Result<MultiCauseChoice, TransportError> {
    let cov: Vec<String> = covariates.iter().map(|s| (*s).to_string()).collect();
    let ids: Vec<usize> = (0..sources.len()).collect();
    if sources.len() > 16 {
        return Err(TransportError::TooLarge(format!(
            "{} sources give too many subsets",
            sources.len()
        )));
    }
    let nonempty: Vec<Vec<usize>> = subsets(&ids)?.into_iter().skip(1).collect();
    let mut total = 0usize;
    for c in 0..=cov.len() {
        let p = arrangements(&cov, c).len();
        total = total.saturating_add(p.saturating_mul(1 + p.saturating_mul(nonempty.len())));
    }
    if total > CANDIDATE_CAP {
        return Err(TransportError::TooLarge(format!(
            "{total} candidates exceed the cap of {CANDIDATE_CAP}"
        )));
    }
    let (train, test) = split(target.len(), seed);
    let levels = outcome_levels(sources.iter().map(|d| (*d, y)).chain([(target, y)]));
    let mut best: Option<MultiCauseChoice> = None;
    let mut fitted = 0;
    let mut consider = |tp: &[String],
                        sp: Option<&[String]>,
                        s: &[usize],
                        best: &mut Option<MultiCauseChoice>|
     -> Result<(), TransportError> {
        let tp_refs: Vec<&str> = tp.iter().map(String::as_str).collect();
        let mut samples = Vec::new();
        if let Some(sp) = sp {
            let sp_refs: Vec<&str> = sp.iter().map(String::as_str).collect();
            for &j in s {
                samples.push(Sample::all(sources[j], &sp_refs, y));
            }
        }
        let mut tr = Sample::all(target, &tp_refs, y);
        tr.rows = Some(train.clone());
        samples.push(tr);
        let model = CategoricalModel::fit(&samples, &levels, alpha)?;
        let mut held = Sample::all(target, &tp_refs, y);
        held.rows = Some(test.clone());
        let loss = model.loss(&held)?;
        fitted += 1;
        if best.as_ref().is_none_or(|b| loss < b.loss) {
            *best = Some(MultiCauseChoice {
                target_parents: tp.to_vec(),
                source_parents: sp.map(<[String]>::to_vec),
                sources: s.to_vec(),
                model,
                loss,
                candidates: 0,
            });
        }
        Ok(())
    };
    for c in 0..=cov.len() {
        let orders = arrangements(&cov, c);
        for tp in &orders {
            consider(tp, None, &[], &mut best)?;
            for sp in &orders {
                for s in &nonempty {
                    consider(tp, Some(sp), s, &mut best)?;
                }
            }
        }
    }
    let mut best = best.expect("the empty covariate list is always tried");
    best.candidates = fitted;
    Ok(best)
}

/// One domain of a sequential prediction task (Sec. 11.4.3): tokens
/// `V_1 … V_T` in temporal order, each generated by a mechanism.
#[derive(Clone, Debug)]
pub struct SequenceDomain<'a> {
    /// The data, one column per position.
    pub data: &'a Dataset,
    /// The column of each position, in order.
    pub positions: Vec<String>,
    /// The parents of each position, as indices of earlier positions in the
    /// order the mechanism reads them.
    pub parents: Vec<Vec<usize>>,
    /// A label per position naming its mechanism. This is the discrepancy
    /// oracle of Def. 11.4.1: `Δ(i, j; i′, j′) = 0` exactly when the labels of
    /// position `i` in domain `j` and `i′` in `j′` are equal.
    pub mechanisms: Vec<String>,
}

impl SequenceDomain<'_> {
    fn validate(&self, what: &str) -> Result<(), TransportError> {
        let t = self.positions.len();
        if self.parents.len() != t || self.mechanisms.len() != t {
            return Err(TransportError::InvalidInput(format!(
                "{what}: {t} positions, {} parent lists, {} mechanisms",
                self.parents.len(),
                self.mechanisms.len()
            )));
        }
        for (i, pa) in self.parents.iter().enumerate() {
            if let Some(p) = pa.iter().find(|p| **p >= i) {
                return Err(TransportError::InvalidInput(format!(
                    "{what}: position {i} has parent {p}, which does not precede it"
                )));
            }
        }
        Ok(())
    }
}

/// Where a pooled module's data came from: `None` is the target.
pub type PoolSource = (Option<usize>, usize);

/// The predictor [`sequential_da`] builds:
/// `µ(v_T | v_{1:M}) = sum_{v_{M+1:T−1}} ∏_i P̂(v_i | pa*_i)`.
#[derive(Clone, Debug)]
pub struct SequentialPredictor {
    m: usize,
    alphabet: Vec<String>,
    parents: Vec<Vec<usize>>,
    modules: Vec<Option<CategoricalModel>>,
    pooled: Vec<Vec<PoolSource>>,
}

impl SequentialPredictor {
    /// The module `P̂(v_i | pa*_i)` for target position `i` (0-based), if
    /// `i ≥ M`.
    #[must_use]
    pub fn module(&self, i: usize) -> Option<&CategoricalModel> {
        self.modules.get(i).and_then(Option::as_ref)
    }

    /// The `(source, position)` pairs pooled for target position `i`; a
    /// source of `None` is the target itself. This is `J_i ∪ {(i, *)}`.
    #[must_use]
    pub fn pooled(&self, i: usize) -> &[PoolSource] {
        self.pooled.get(i).map_or(&[], Vec::as_slice)
    }

    /// The token alphabet.
    #[must_use]
    pub fn alphabet(&self) -> &[String] {
        &self.alphabet
    }

    /// `µ(v_T = a | v_{1:M} = prefix)` for every token `a` of the alphabet,
    /// summing out the intermediate tokens.
    ///
    /// # Errors
    ///
    /// [`TransportError::InvalidInput`] if `prefix` does not have `M` tokens,
    /// and [`TransportError::TooLarge`] beyond a million completions.
    pub fn distribution(&self, prefix: &[&str]) -> Result<Vec<f64>, TransportError> {
        if prefix.len() != self.m {
            return Err(TransportError::InvalidInput(format!(
                "a prefix of {} tokens for M = {}",
                prefix.len(),
                self.m
            )));
        }
        let t = self.parents.len();
        let v = self.alphabet.len();
        let hidden = t - 1 - self.m;
        let completions = v
            .checked_pow(u32::try_from(hidden).unwrap_or(u32::MAX))
            .filter(|n| *n <= COMPLETION_CAP)
            .ok_or_else(|| TransportError::TooLarge(format!("{v}^{hidden} completions")))?;
        let mut tokens: Vec<&str> = prefix.to_vec();
        tokens.resize(t, "");
        let mut out = vec![0.0; v];
        let mut idx = vec![0usize; hidden];
        for _ in 0..completions {
            for (k, &a) in idx.iter().enumerate() {
                tokens[self.m + k] = &self.alphabet[a];
            }
            let mut weight = 1.0;
            for i in self.m..t - 1 {
                weight *= self.factor(i, &tokens);
                if weight == 0.0 {
                    break;
                }
            }
            if weight > 0.0 {
                for (a, slot) in out.iter_mut().enumerate() {
                    tokens[t - 1] = &self.alphabet[a];
                    *slot += weight * self.factor(t - 1, &tokens);
                }
            }
            for slot in &mut idx {
                *slot += 1;
                if *slot < v {
                    break;
                }
                *slot = 0;
            }
        }
        Ok(out)
    }

    fn factor(&self, i: usize, tokens: &[&str]) -> f64 {
        let module = self.modules[i].as_ref().expect("positions ≥ M have modules");
        let pa: Vec<&str> = self.parents[i].iter().map(|&p| tokens[p]).collect();
        module.probability(&pa, tokens[i])
    }
}

/// Algorithm 38, structure-informed domain adaptation for sequences: for each
/// target position `i > M`, pool the data of every `(i′, j′)` the discrepancy
/// oracle says shares its mechanism, together with the target's own `(i, *)`,
/// fit `P̂(v_i | pa*_i)` on parents aligned by order, and compose the modules
/// into `µ(v_T | v_{1:M})`.
///
/// When every position has a match in some source the risk decays with the
/// source size `N` (zero-shot); otherwise with the target's `n`
/// (Thm 11.4.5). `m` is `M`, the length of the observed prefix.
///
/// # Errors
///
/// [`TransportError::InvalidInput`] for inconsistent domains, a prefix as
/// long as the target sequence, or two positions sharing a mechanism with
/// different numbers of parents, and [`TransportError::MissingData`] for a
/// missing column.
pub fn sequential_da(
    sources: &[SequenceDomain<'_>],
    target: &SequenceDomain<'_>,
    m: usize,
    alpha: f64,
) -> Result<SequentialPredictor, TransportError> {
    target.validate("target")?;
    for (j, s) in sources.iter().enumerate() {
        s.validate(&format!("source {j}"))?;
    }
    let t = target.positions.len();
    if m >= t {
        return Err(TransportError::InvalidInput(format!(
            "a prefix of {m} tokens leaves nothing to predict in a sequence of {t}"
        )));
    }
    let mut alphabet: Vec<String> = Vec::new();
    for dom in sources.iter().chain(std::iter::once(target)) {
        for col in &dom.positions {
            for l in dom.data.levels(col).unwrap_or_default() {
                if !alphabet.contains(l) {
                    alphabet.push(l.clone());
                }
            }
        }
    }
    let mut modules = vec![None; t];
    let mut pooled = vec![Vec::new(); t];
    for i in m..t {
        let label = &target.mechanisms[i];
        let arity = target.parents[i].len();
        let column = |d: &SequenceDomain<'_>, p: usize| d.positions[p].clone();
        let mut samples = Vec::new();
        let mut from = Vec::new();
        for (j, s) in sources.iter().enumerate() {
            for (ip, l) in s.mechanisms.iter().enumerate() {
                if l != label {
                    continue;
                }
                if s.parents[ip].len() != arity {
                    return Err(TransportError::InvalidInput(format!(
                        "mechanism `{label}` reads {} parents at source {j} position {ip} \
                         but {arity} at target position {i}",
                        s.parents[ip].len()
                    )));
                }
                samples.push(Sample {
                    data: s.data,
                    x: s.parents[ip].iter().map(|&p| column(s, p)).collect(),
                    y: column(s, ip),
                    rows: None,
                });
                from.push((Some(j), ip));
            }
        }
        samples.push(Sample {
            data: target.data,
            x: target.parents[i].iter().map(|&p| column(target, p)).collect(),
            y: column(target, i),
            rows: None,
        });
        from.push((None, i));
        modules[i] = Some(CategoricalModel::fit(&samples, &alphabet, alpha)?);
        pooled[i] = from;
    }
    Ok(SequentialPredictor {
        m,
        alphabet,
        parents: target.parents.clone(),
        modules,
        pooled,
    })
}
