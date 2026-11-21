//! Identification and estimation with NCMs: Algorithms 44–47 and 49.

use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Tensor, activation};
use why_data::fairness::Dataset;
use why_data::graph::cinfer::Input;
use why_data::graph::cinfer::counterfactual::{CtfError, CtfQuery, ctf_infer};
use why_data::graph::dseparation::Admg;
use why_data::ncm::{Expanded, expand};

use crate::data::Table;
use crate::model::{Ncm, values};
use crate::{Experiment, Kind, NcmConfig, NcmError, Rng};

/// The quantity to identify: `Σ_i c_i P(y_{*,i} | x_{*,i})`, with each value
/// name (`X`, `X'`, …) mapped to a level.
///
/// A single query has one term with `c = 1`. The effects the book studies
/// are differences, such as the average treatment effect
/// `P(y_x) − P(y_{x'})`. Each term is pushed up or down according to the
/// sign of its coefficient, as in Table 13.6.
#[derive(Clone, Debug)]
pub struct Objective {
    terms: Vec<(f64, CtfQuery)>,
    values: Vec<(String, String)>,
}

impl Objective {
    /// `P(query)`.
    #[must_use]
    pub fn new(query: CtfQuery, values: &[(&str, &str)]) -> Self {
        Self {
            terms: vec![(1.0, query)],
            values: values
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        }
    }

    /// `P(a) − P(b)`, the shape of the ATE, ETT, NDE and Ctf-DE (Table 13.7).
    #[must_use]
    pub fn difference(a: CtfQuery, b: CtfQuery, values: &[(&str, &str)]) -> Self {
        Self::new(a, values).term(-1.0, b)
    }

    /// The same objective plus `coefficient · P(query)`.
    #[must_use]
    pub fn term(mut self, coefficient: f64, query: CtfQuery) -> Self {
        self.terms.push((coefficient, query));
        self
    }

    /// The terms, with their coefficients.
    #[must_use]
    pub fn terms(&self) -> &[(f64, CtfQuery)] {
        &self.terms
    }

    fn expanded(&self) -> Result<Vec<(f64, Expanded)>, NcmError> {
        let values: Vec<(&str, &str)> = self
            .values
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.terms
            .iter()
            .map(|(c, q)| Ok((*c, expand(q, &values)?)))
            .collect()
    }

    /// Its value under `model`, estimated from `m` draws of `Û`.
    ///
    /// # Errors
    ///
    /// Those of [`Ncm::estimate`].
    pub fn evaluate(&self, model: &Ncm, m: usize, seed: u64) -> Result<f64, NcmError> {
        let mut total = 0.0;
        for (c, e) in self.expanded()? {
            total += c * model.estimate_expanded(&e, m, seed)?;
        }
        Ok(total)
    }
}

impl Ncm {
    /// `P̂(y_* | x_*)`, with each value name mapped to a level as in
    /// [`why_data::ncm::expand`]. The explicit model computes it from
    /// Eq. 13.147 with `m` draws of `Û` and the worlds enumerated; the
    /// implicit one counts `m` samples (Eq. 13.151). Either way this is the
    /// mutilation procedure of Corols. 13.1.9 and 13.2.6.
    ///
    /// # Errors
    ///
    /// [`NcmError::ImpossibleCondition`] when no draw satisfies `x_*`;
    /// [`NcmError::TooLarge`] when the explicit enumeration needs too many
    /// cells; errors for unknown variables, levels or value names.
    pub fn estimate(
        &self,
        query: &CtfQuery,
        values: &[(&str, &str)],
        m: usize,
        seed: u64,
    ) -> Result<f64, NcmError> {
        self.estimate_expanded(&expand(query, values)?, m, seed)
    }

    pub(crate) fn estimate_expanded(
        &self,
        e: &Expanded,
        m: usize,
        seed: u64,
    ) -> Result<f64, NcmError> {
        match self.kind {
            Kind::Explicit => {
                let mut rng = Rng::new(seed);
                // Chunks of draws keep the enumerated batch small.
                let chunk = 2048.min(m.max(1));
                let (mut total, mut n) = (0.0, 0usize);
                while n < m.max(1) {
                    let noise = self.draw(chunk, &mut rng);
                    total +=
                        f64::from(values(self.query_tensor(e, &noise, 0.0)?)[0]) * chunk as f64;
                    n += chunk;
                }
                Ok(total / n as f64)
            }
            Kind::Implicit => self.monte_carlo(e, m.max(1), seed),
        }
    }

    /// The observational distribution `P̂(V) = E_Û[Π_V P(v | pa_V, û_V)]`
    /// over every joint level, from `m` draws of `Û`, as a data set whose
    /// weights are the probabilities. Gumbel-max sampling draws from the same
    /// softmax, so this holds for both kinds.
    ///
    /// # Errors
    ///
    /// [`NcmError::TooLarge`] when the joint has more than
    /// [`NcmConfig::max_cells`] levels.
    pub fn distribution(&self, m: usize, seed: u64) -> Result<Dataset, NcmError> {
        let spec = &self.spec;
        let n = spec.names.len();
        let cells = spec
            .levels
            .iter()
            .try_fold(1usize, |a, l| a.checked_mul(l.len()))
            .filter(|c| *c <= self.max_cells)
            .ok_or(NcmError::TooLarge(self.max_cells))?;
        let m = m.max(1);
        let noise = self.draw(m, &mut Rng::new(seed));
        // probs[v][(cfg * m + s) * d + level]
        let mut probs = Vec::with_capacity(n);
        for v in 0..n {
            let mut configs: Vec<Vec<usize>> = vec![Vec::new()];
            for &p in &spec.parents[v] {
                configs = configs
                    .into_iter()
                    .flat_map(|c| {
                        (0..spec.levels[p].len()).map(move |l| {
                            let mut c = c.clone();
                            c.push(l);
                            c
                        })
                    })
                    .collect();
            }
            let (logits, _) = self.split(v, self.logits(v, &configs, &noise));
            probs.push(values(activation::softmax(logits, 1)));
        }
        let mut rows = Vec::with_capacity(cells);
        let mut weights = Vec::with_capacity(cells);
        let mut a = vec![0usize; n];
        for _ in 0..cells {
            let mut per_draw = vec![1.0f64; m];
            for v in 0..n {
                let d = spec.levels[v].len();
                let cfg = spec.parents[v]
                    .iter()
                    .fold(0, |at, &p| at * spec.levels[p].len() + a[p]);
                for (s, w) in per_draw.iter_mut().enumerate() {
                    *w *= f64::from(probs[v][(cfg * m + s) * d + a[v]]);
                }
            }
            rows.push(
                (0..n)
                    .map(|v| spec.levels[v][a[v]].clone())
                    .collect::<Vec<_>>(),
            );
            weights.push(per_draw.iter().sum::<f64>() / m as f64);
            for v in (0..n).rev() {
                a[v] += 1;
                if a[v] < spec.levels[v].len() {
                    break;
                }
                a[v] = 0;
            }
        }
        Dataset::from_records(spec.names.clone(), rows)
            .and_then(|d| d.with_weights(weights))
            .map_err(|_| NcmError::NoData)
    }

    /// The mean log-likelihood `(1/n) Σ_i log P̂(v_i | do(z_i))` of an
    /// experiment's rows under the explicit reading (Eq. 13.144), from `m`
    /// draws of `Û`.
    ///
    /// # Errors
    ///
    /// [`NcmError::MissingColumn`] and [`NcmError::UnknownLevel`] for data
    /// the model cannot read.
    pub fn log_likelihood(
        &self,
        experiment: &Experiment,
        m: usize,
        seed: u64,
    ) -> Result<f64, NcmError> {
        let t = Table::new(&self.spec, experiment)?;
        let noise = self.draw(m.max(1), &mut Rng::new(seed));
        Ok(f64::from(values(self.table_log_likelihood(&t, &noise))[0]))
    }

    /// How far the model is from the data: the KL divergence `Σ_k KL(P_k ‖
    /// P̂_k)` over the experiments for the explicit model, and the summed
    /// squared MMD for the implicit one.
    fn misfit(&self, tables: &[Table], m: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        tables
            .iter()
            .map(|t| {
                let noise = self.draw(m, &mut rng);
                match self.kind {
                    Kind::Explicit => {
                        -f64::from(values(self.table_log_likelihood(t, &noise))[0]) - t.entropy()
                    }
                    Kind::Implicit => f64::from(values(self.mmd(t, &noise, &mut rng))[0]),
                }
            })
            .sum()
    }
}

/// The outcome of [`neural_id`].
#[derive(Clone, Debug)]
pub struct NeuralIdResult {
    /// The objective under the model trained to minimize it.
    pub min: f64,
    /// The objective under the model trained to maximize it.
    pub max: f64,
    /// `max − min`, the left side of Eq. 13.149.
    pub gap: f64,
    /// Whether `gap < τ`.
    pub identifiable: bool,
    /// The midpoint of `min` and `max` when identifiable, else `None`
    /// (FAIL).
    pub estimate: Option<f64>,
    /// How far each model is from the data after training,
    /// `[min model, max model]`: the summed KL divergence `Σ_k KL(P_k ‖ P̂_k)`
    /// for the explicit model, the summed squared MMD for the implicit one.
    /// A large value means the optimization, not the identification,
    /// failed.
    pub misfit: [f64; 2],
    /// The two trained models, `[θ*_min, θ*_max]`, which can be queried and
    /// sampled further.
    pub models: [Ncm; 2],
}

/// The straight-through leak of an empty cross-world overlap while training.
const LEAK: f64 = 0.05;

/// Trains `model` to fit `tables`, optionally pushing an objective up or
/// down with weight `λ` (Algorithms 47 and 49).
fn train(
    model: &mut Ncm,
    tables: &[Table],
    objective: Option<(&[(f64, Expanded)], bool)>,
    config: &NcmConfig,
    seed: u64,
) -> Result<(), NcmError> {
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng::new(seed);
    let epochs = config.epochs.max(1);
    let (l0, l1) = config.lambda;
    for t in 0..epochs {
        let frac = if epochs == 1 {
            1.0
        } else {
            t as f64 / (epochs - 1) as f64
        };
        let lambda = l0 * (l1 / l0).powf(frac);
        let mut loss: Tensor<1> = Tensor::zeros([1], &model.device);
        for table in tables {
            let noise = model.draw(config.samples, &mut rng);
            loss = loss
                + match model.kind {
                    Kind::Explicit => model.table_log_likelihood(table, &noise).neg(),
                    // MMD itself, not its square: the square is flat at the
                    // optimum, so the query term would win however small λ.
                    Kind::Implicit => model
                        .mmd(table, &noise, &mut rng)
                        .clamp_min(0.0)
                        .add_scalar(1e-6)
                        .sqrt(),
                };
        }
        if let Some((terms, up)) = objective {
            for (c, e) in terms {
                // Push this term the way that moves the objective.
                let push_up = (*c > 0.0) == up;
                let noise = model.draw(config.samples, &mut rng);
                let term = match model.kind {
                    Kind::Explicit => {
                        // No clamp: it would cut the gradient exactly where
                        // the query is stuck at 0 or 1.
                        let q = model.query_tensor(e, &noise, LEAK)?;
                        if push_up {
                            q.add_scalar(1e-6).log().neg()
                        } else {
                            q.neg().add_scalar(1.0 + 1e-6).log().neg()
                        }
                    }
                    Kind::Implicit => model.query_log_loss(e, &noise, push_up)?,
                };
                loss = loss + term.mul_scalar(lambda * c.abs());
            }
        }
        let grads = GradientsParams::from_grads(loss.backward(), &model.nets);
        // The step shrinks to a tenth by the end, so the last epochs settle
        // instead of bouncing on the Monte-Carlo noise.
        let rate = config.learning_rate * 0.1f64.powf(frac);
        model.nets = opt.step(rate, model.nets.clone(), grads);
    }
    Ok(())
}

fn tables(model: &Ncm, experiments: &[Experiment]) -> Result<Vec<Table>, NcmError> {
    if experiments.is_empty() {
        return Err(NcmError::NoData);
    }
    experiments
        .iter()
        .map(|e| Table::new(&model.spec, e))
        .collect()
}

/// Fits a G-NCM to the data alone (line 3 of Algorithm 45): maximum
/// likelihood for [`Kind::Explicit`], MMD for [`Kind::Implicit`].
///
/// # Errors
///
/// [`NcmError::NoData`] without data, and errors for data the graph cannot
/// read.
pub fn fit(g: &Admg, experiments: &[Experiment], config: &NcmConfig) -> Result<Ncm, NcmError> {
    let mut model = Ncm::from_data(g, experiments, config)?;
    let tables = tables(&model, experiments)?;
    train(&mut model, &tables, None, config, config.seed ^ 0xF17)?;
    Ok(model)
}

/// Algorithms 44 and 46: decides whether `objective` is identifiable from
/// `G` and the data, and estimates it when it is.
///
/// Two G-NCMs are trained on the experiments (`Z = {P(V_{z_k})}`; one
/// observational data set gives Algorithm 44). One also maximizes the
/// objective and the other minimizes it, with weight `λ` decaying from
/// `config.lambda.0` to `config.lambda.1`. Once both fit the data, the
/// objective is identifiable exactly when they agree (Thms 13.1.8 and
/// 13.2.5); here, when their values differ by less than `config.tolerance`
/// (Eq. 13.149).
///
/// ```no_run
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
/// use why_data::fairness::Dataset;
/// use why_ncm::{Experiment, NcmConfig, Objective, neural_id};
///
/// // The bow graph: X -> Y with X <-> Y. P(y | do(x)) is not identifiable.
/// let mut g = Admg::new();
/// g.directed("X", "Y");
/// g.bidirected("X", "Y");
/// let rows = [["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]];
/// let data = Dataset::from_records(["X", "Y"], rows)
///     .unwrap()
///     .with_weights(vec![0.3, 0.2, 0.1, 0.4])
///     .unwrap();
/// let q = CtfQuery::parse(&g, "P(y_x)").unwrap();
/// let objective = Objective::new(q, &[("X", "1"), ("Y", "1")]);
/// let r = neural_id(&g, &objective, &[Experiment::observational(data)], &NcmConfig::default()).unwrap();
/// assert!(!r.identifiable); // the bounds are [0.4, 0.9]
/// ```
///
/// # Errors
///
/// [`NcmError::NoData`] without data; [`NcmError::TooLarge`] when an
/// explicit query needs too many cells; errors for unknown variables,
/// levels or value names.
pub fn neural_id(
    g: &Admg,
    objective: &Objective,
    experiments: &[Experiment],
    config: &NcmConfig,
) -> Result<NeuralIdResult, NcmError> {
    let terms = objective.expanded()?;
    let base = Ncm::from_data(g, experiments, config)?;
    let tables = tables(&base, experiments)?;
    let mut models = [
        base.clone(),
        Ncm::from_data(g, experiments, &reseed(config, 1))?,
    ];
    for (i, model) in models.iter_mut().enumerate() {
        train(
            model,
            &tables,
            Some((&terms, i == 1)),
            config,
            config.seed ^ (0xA11 + i as u64),
        )?;
    }
    let seed = config.seed ^ 0xE5;
    let min = objective.evaluate(&models[0], config.eval_samples, seed)?;
    let max = objective.evaluate(&models[1], config.eval_samples, seed)?;
    let gap = max - min;
    let identifiable = gap.abs() < config.tolerance;
    let misfit = [
        models[0].misfit(&tables, config.eval_samples.min(4096), seed),
        models[1].misfit(&tables, config.eval_samples.min(4096), seed),
    ];
    Ok(NeuralIdResult {
        min,
        max,
        gap,
        identifiable,
        estimate: identifiable.then_some(0.5 * (min + max)),
        misfit,
        models,
    })
}

fn reseed(config: &NcmConfig, k: u64) -> NcmConfig {
    NcmConfig {
        seed: config.seed.wrapping_add(k.wrapping_mul(0x9E37_79B9)),
        ..config.clone()
    }
}

/// Algorithm 45: decides identifiability symbolically, with C-INFER on `G`
/// and the experiments' regimes, and only then estimates the objective with
/// a G-NCM fitted to the data. Returns `None` (FAIL) when some term is not
/// identifiable, and otherwise the estimate and the fitted model.
///
/// # Errors
///
/// [`NcmError::Structure`] for a malformed query or graph; the errors of
/// [`fit`] and [`Ncm::estimate`].
pub fn symbolic_then_neural(
    g: &Admg,
    objective: &Objective,
    experiments: &[Experiment],
    config: &NcmConfig,
) -> Result<Option<(f64, Ncm)>, NcmError> {
    let inputs: Vec<Input> = experiments
        .iter()
        .map(|e| Input::experimental(&e.intervened.iter().map(String::as_str).collect::<Vec<_>>()))
        .collect();
    for (_, q) in objective.terms() {
        match ctf_infer(g, q, &inputs) {
            Ok(_) => {}
            Err(CtfError::NotIdentifiable { .. } | CtfError::Inconsistent { .. }) => {
                return Ok(None);
            }
            Err(e) => return Err(why_data::ncm::NcmError::from(e).into()),
        }
    }
    let model = fit(g, experiments, config)?;
    let value = objective.evaluate(&model, config.eval_samples, config.seed ^ 0xE5)?;
    Ok(Some((value, model)))
}
