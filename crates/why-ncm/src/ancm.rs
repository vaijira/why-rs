//! Algorithm 50 (ANCM): a G-NCM that generates labelled images, trained as
//! a variational autoencoder (Sec. 14.3.1).

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Tensor, activation};
use why_data::fairness::Dataset;
use why_data::graph::dseparation::Admg;

use crate::data::Table;
use crate::model::{Mlp, Ncm, Nets, forward_rows, seeded, values};
use crate::{Experiment, Kind, NcmConfig, NcmError, Rng};

/// Labelled images: the generative factors `V` of each sample, and its
/// pixels `I`, all images the same length.
#[derive(Clone, Debug)]
pub struct ImageData {
    labels: Dataset,
    images: Vec<Vec<f32>>,
}

impl ImageData {
    /// Pairs row `k` of `labels` with `images[k]`.
    ///
    /// # Errors
    ///
    /// [`NcmError::NoData`] without rows, and [`NcmError::BadImages`] when
    /// the counts differ or an image has a different length or a value that
    /// is not finite.
    pub fn new(labels: Dataset, images: Vec<Vec<f32>>) -> Result<Self, NcmError> {
        if images.is_empty() || labels.is_empty() {
            return Err(NcmError::NoData);
        }
        if labels.len() != images.len() {
            return Err(NcmError::BadImages(format!(
                "{} label rows for {} images",
                labels.len(),
                images.len()
            )));
        }
        let m = images[0].len();
        if m == 0 || images.iter().any(|i| i.len() != m) {
            return Err(NcmError::BadImages(
                "images must be non-empty and of one length".into(),
            ));
        }
        if images.iter().flatten().any(|x| !x.is_finite()) {
            return Err(NcmError::BadImages("a pixel is not finite".into()));
        }
        Ok(Self { labels, images })
    }

    /// The labels.
    #[must_use]
    pub fn labels(&self) -> &Dataset {
        &self.labels
    }

    /// The images.
    #[must_use]
    pub fn images(&self) -> &[Vec<f32>] {
        &self.images
    }
}

/// Architecture and training settings of an [`Ancm`].
#[derive(Clone, Debug)]
pub struct AncmConfig {
    /// Width of each hidden layer, in every network.
    pub hidden: usize,
    /// Number of hidden layers per network.
    pub layers: usize,
    /// Number of uniforms in each `Û_C`.
    pub noise_dim: usize,
    /// Dimension of the image noise `Û_I`, the unlabelled factors.
    pub image_noise: usize,
    /// Passes over the data `T`.
    pub epochs: usize,
    /// Rows per gradient step.
    pub batch: usize,
    /// Adam's learning rate `η` at the start. It decays geometrically to a
    /// tenth of this by the end.
    pub learning_rate: f64,
    /// `λ`, the weight of the label loss `L2` (Eq. 14.29).
    pub lambda: f64,
    /// Draws of `Û_C` per step for the labels' likelihood `log P̂(v)`.
    pub samples: usize,
    /// Draws of `Û_C` from which an edit resamples its posterior.
    pub posterior_samples: usize,
    /// The standard deviation of the Gaussian pixel likelihood.
    pub pixel_scale: f64,
    /// Seeds initialization, shuffling and every draw.
    pub seed: u64,
}

impl Default for AncmConfig {
    fn default() -> Self {
        Self {
            hidden: 64,
            layers: 2,
            noise_dim: 2,
            image_noise: 2,
            epochs: 40,
            batch: 128,
            learning_rate: 0.003,
            lambda: 1.0,
            samples: 256,
            posterior_samples: 4096,
            pixel_scale: 0.1,
            seed: 0,
        }
    }
}

/// The trainable parts together, for one optimizer.
#[derive(Module, Debug)]
struct Parts {
    nets: Nets,
    encoder: Mlp,
    decoder: Mlp,
    labeller: Mlp,
}

/// An augmented NCM: a G-constrained NCM over the labelled factors
/// (Def. 13.1.11, explicit reading) whose labels a decoder turns into an
/// image together with an image noise `Û_I`. An encoder `Q_ω(Û_I | I)`
/// inverts the decoder, and a labeller reads `V` off an image.
///
/// By Thm 14.3.2 a trained ANCM that fits `P(V, I)` is a ctf-consistent
/// estimator for every care set. By Lemma 14.3.1 its F-ctf queries are
/// counterfactuals of the generative level, [`Ancm::ncm`], which
/// [`Ncm::estimate`] evaluates.
#[derive(Clone, Debug)]
pub struct Ancm {
    ncm: Ncm,
    encoder: Mlp,
    decoder: Mlp,
    labeller: Mlp,
    pixels: usize,
    image_noise: usize,
    posterior_samples: usize,
}

/// A factual image and its edit, with their labels in the order of
/// [`Ancm::variables`].
#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    /// The factual labels `v`.
    pub labels: Vec<String>,
    /// The factual image `i`: the decoded one for a sample, the given one
    /// for an edit.
    pub image: Vec<f32>,
    /// The labels `v′` in the edited world.
    pub edited_labels: Vec<String>,
    /// The edited image `i′_{x′}`.
    pub edited_image: Vec<f32>,
}

/// Algorithm 50: trains an ANCM on labelled images, with the generative
/// level constrained by `g` over the label columns.
///
/// The loss is Eq. 14.29, `L1 + λ L2`. `L1` is the negative evidence lower
/// bound of `log P̂(v, i)`, which equals `log P̂(i)` since `v = h(i)`. Its
/// variational posterior is `p̂(û_C | v) · Q_ω(û_I | i)`: the posterior of the
/// generative level's noise is the exact one, so that part of the bound is
/// tight and `L1` splits into
///
/// * `−log P̂(v)`, the G-NCM's exact likelihood of the labels
///   (Eq. 13.142, the MLE-NCM of chapter 13), and
/// * the negative ELBO of `log P̂(i | v)`: a Gaussian around the decoder's
///   output at `(v, û_I)`, with `û_I ~ Q_ω(· | i)` and a standard normal
///   prior.
///
/// `L2` is the cross-entropy of a labeller's prediction from the image: the
/// fit of `P(V | I)` that `λ` weighs (Eq. 14.28).
///
/// Taking `p̂(û_C | v)` exact rather than learned is the one departure from
/// the text, where `Q_ω` encodes all of `Û`. With a learned posterior the
/// bound leaves `Û_C` unused when the confounding it carries is weak. On
/// Ex. 14.2, whose `F ↔ Y` mutual information is 0.02 nats, `P̂(F, Y)` then
/// stays at the product of its marginals, so Thm 14.3.2's premise fails.
///
/// # Errors
///
/// [`NcmError::MissingColumn`] when the labels lack a variable of `g`, and
/// the errors of [`Ncm::from_data`].
pub fn ancm(g: &Admg, data: &ImageData, config: &AncmConfig) -> Result<Ancm, NcmError> {
    let ncm_config = NcmConfig {
        kind: Kind::Explicit,
        hidden: config.hidden,
        layers: config.layers,
        noise_dim: config.noise_dim,
        seed: config.seed,
        ..NcmConfig::default()
    };
    let observational = Experiment::observational(data.labels.clone());
    let mut ncm = Ncm::from_data(g, std::slice::from_ref(&observational), &ncm_config)?;
    let spec = ncm.spec.clone();
    let table = Table::new(&spec, &observational)?;
    let rows: Vec<Vec<usize>> = (0..data.labels.len())
        .map(|r| {
            spec.names
                .iter()
                .enumerate()
                .map(|(v, name)| {
                    let x = data
                        .labels
                        .value(name, r)
                        .ok_or_else(|| NcmError::MissingColumn(name.clone()))?;
                    spec.level(v, x)
                })
                .collect()
        })
        .collect::<Result<_, _>>()?;
    let pixels = data.images[0].len();
    let dz = config.image_noise;
    let device = ncm.device.clone();
    let (encoder, decoder, labeller) = seeded(&device, config.seed ^ 0x14, || {
        (
            Mlp::new(pixels, config.hidden, config.layers, 2 * dz.max(1), &device),
            Mlp::new(
                spec.width + dz,
                config.hidden,
                config.layers,
                pixels,
                &device,
            ),
            Mlp::new(pixels, config.hidden, config.layers, spec.width, &device),
        )
    });
    let mut parts = Parts {
        nets: ncm.nets.clone(),
        encoder,
        decoder,
        labeller,
    };
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng::new(config.seed ^ 0xa9c3);
    let batch = config.batch.max(1);
    let total = (config.epochs.max(1) * rows.len().div_ceil(batch)).max(2);
    let variance2 = 2.0 * config.pixel_scale * config.pixel_scale;
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut step = 0usize;
    for _ in 0..config.epochs.max(1) {
        shuffle(&mut order, &mut rng);
        for chunk in order.chunks(batch) {
            let b = chunk.len();
            let x = ncm.matrix(
                chunk
                    .iter()
                    .flat_map(|&k| data.images[k].iter().copied())
                    .collect(),
                b,
                pixels,
            );
            let labels: Vec<Vec<usize>> = chunk.iter().map(|&k| rows[k].clone()).collect();
            let onehots = one_hots(&ncm, &labels);

            // −log P̂(v), over the whole table each step.
            ncm.nets = parts.nets.clone();
            let noise = ncm.draw(config.samples.max(1), &mut rng);
            let generative = ncm.table_log_likelihood(&table, &noise).neg();

            // The negative ELBO of log P̂(i | v).
            let image = if dz == 0 {
                let recon = decode(&parts.decoder, &onehots, None);
                (x.clone() - recon)
                    .powi_scalar(2)
                    .sum_dim(1)
                    .div_scalar(variance2)
                    .mean()
            } else {
                let enc = parts.encoder.forward(x.clone());
                let mu = enc.clone().narrow(1, 0, dz);
                let log_var = enc.narrow(1, dz, dz).clamp(-8.0, 4.0);
                let eps = ncm.matrix((0..b * dz).map(|_| normal(&mut rng)).collect(), b, dz);
                let z = mu.clone() + log_var.clone().mul_scalar(0.5).exp() * eps;
                let recon = decode(&parts.decoder, &onehots, Some(z));
                let nll = (x.clone() - recon)
                    .powi_scalar(2)
                    .sum_dim(1)
                    .div_scalar(variance2);
                let kl = (mu.powi_scalar(2) + log_var.clone().exp() - log_var)
                    .sub_scalar(1.0)
                    .sum_dim(1)
                    .mul_scalar(0.5);
                (nll + kl).mean()
            };

            // L2: the labeller's cross-entropy.
            let logits = parts.labeller.forward(x);
            let mut label_loss: Tensor<1> = Tensor::zeros([1], &device);
            for v in 0..spec.names.len() {
                let d = spec.levels[v].len();
                let logp = activation::log_softmax(logits.clone().narrow(1, spec.offsets[v], d), 1);
                label_loss = label_loss + (logp * onehots[v].clone()).sum_dim(1).mean().neg();
            }

            let loss = generative + image + label_loss.mul_scalar(config.lambda);
            let grads = GradientsParams::from_grads(loss.backward(), &parts);
            let frac = step as f64 / (total - 1) as f64;
            let rate = config.learning_rate * 0.1f64.powf(frac);
            parts = opt.step(rate, parts, grads);
            step += 1;
        }
    }
    ncm.nets = parts.nets;
    Ok(Ancm {
        ncm,
        encoder: parts.encoder,
        decoder: parts.decoder,
        labeller: parts.labeller,
        pixels,
        image_noise: dz,
        posterior_samples: config.posterior_samples.max(1),
    })
}

impl Ancm {
    /// The generative level: the G-NCM over the labelled factors. Its
    /// counterfactuals are the ANCM's F-ctf queries (Lemma 14.3.1).
    #[must_use]
    pub fn ncm(&self) -> &Ncm {
        &self.ncm
    }

    /// The labelled factors, in the order labels are reported.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        self.ncm.variables()
    }

    /// The number of pixels.
    #[must_use]
    pub fn pixels(&self) -> usize {
        self.pixels
    }

    /// The labeller trained by `L2`: the most likely level of each
    /// variable given the image.
    ///
    /// # Errors
    ///
    /// [`NcmError::BadImages`] for an image of the wrong length.
    pub fn label(&self, image: &[f32]) -> Result<Vec<String>, NcmError> {
        self.check(image)?;
        Ok(self.names_of(&self.label_levels(image)))
    }

    /// `n` samples of `P̂(I, I_{x′})` (Sec. 14.3.1): `Û` from its prior, the
    /// factual image from `M̂`, the edited one from the submodel `M̂_{x′}`
    /// with the same `Û`. `intervention` maps variables to levels.
    ///
    /// # Errors
    ///
    /// [`NcmError::UnknownVariable`] and [`NcmError::UnknownLevel`] for a
    /// bad intervention.
    pub fn sample(
        &self,
        n: usize,
        intervention: &[(&str, &str)],
        seed: u64,
    ) -> Result<Vec<Edit>, NcmError> {
        let x = self.intervention(intervention)?;
        let k = self.ncm.spec.names.len();
        let mut rng = Rng::new(seed);
        let u = self.prior_noise(n, &mut rng);
        let z = self.image_prior(n, &mut rng);
        let eps: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        let delta: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        let factual = self.simulate(&u, &[], &eps, &delta, n);
        let edited = self.simulate(&u, &x, &eps, &delta, n);
        Ok(self.finish(factual, edited, z, None))
    }

    /// Edits `image`, whose labels are `labels` (or the labeller's, when
    /// `None`), to the world `do(intervention)`.
    ///
    /// Abduction follows the training posterior. `Û_C` is resampled from
    /// [`AncmConfig::posterior_samples`] prior draws in proportion to
    /// `P̂(v | û_C)`. Each variable's sampling noise is drawn from its
    /// posterior given its label, and `Û_I` from the encoder `Q_ω(· | i)`.
    ///
    /// # Errors
    ///
    /// [`NcmError::BadImages`] for an image of the wrong length or a label
    /// list of the wrong size, [`NcmError::ImpossibleCondition`] when no
    /// draw of `Û_C` can produce the labels, and errors for unknown
    /// variables or levels.
    pub fn edit(
        &self,
        image: &[f32],
        labels: Option<&[&str]>,
        intervention: &[(&str, &str)],
        seed: u64,
    ) -> Result<Edit, NcmError> {
        self.check(image)?;
        let x = self.intervention(intervention)?;
        let spec = &self.ncm.spec;
        let n = spec.names.len();
        let levels: Vec<usize> = match labels {
            Some(l) if l.len() != n => {
                return Err(NcmError::BadImages(format!(
                    "{} labels for {n} variables",
                    l.len()
                )));
            }
            Some(l) => l
                .iter()
                .enumerate()
                .map(|(v, x)| spec.level(v, x))
                .collect::<Result<_, _>>()?,
            None => self.label_levels(image),
        };
        let mut rng = Rng::new(seed);

        // Û_C ∝ prior × P̂(v | û_C), by importance resampling.
        let m = self.posterior_samples;
        let draws = self.prior_noise(m, &mut rng);
        let rows = vec![levels.clone(); m];
        let mut weight = vec![1.0f64; m];
        let mut cache = Vec::with_capacity(n);
        for v in 0..n {
            let parents = self.parent_one_hots(v, &rows);
            let (logits, gate) = self
                .ncm
                .split(v, forward_rows(spec, &self.ncm.nets, v, parents, &draws));
            let probs = values(activation::softmax(logits, 1));
            let d = spec.levels[v].len();
            for (s, w) in weight.iter_mut().enumerate() {
                *w *= f64::from(probs[s * d + levels[v]]);
            }
            cache.push((probs, values(gate)));
        }
        let total: f64 = weight.iter().sum();
        if total <= 0.0 {
            return Err(NcmError::ImpossibleCondition);
        }
        let mut target = f64::from(rng.unit()) * total;
        let pick = weight
            .iter()
            .position(|w| {
                target -= w;
                target <= 0.0
            })
            .unwrap_or(m - 1);
        let u = draws.narrow(0, pick, 1);

        // (δ_V, ε_V) given v: δ from its prior, then ε uniform on v's
        // interval of the inverse CDF in that order.
        let mut eps = vec![0.0f32; n];
        let mut delta = vec![0.0f32; n];
        for v in 0..n {
            let d = spec.levels[v].len();
            let (probs, gates) = &cache[v];
            let row = &probs[pick * d..(pick + 1) * d];
            delta[v] = rng.unit();
            let reverse = delta[v] < gates[pick];
            let mut lo = 0.0f32;
            let mut order: Vec<usize> = (0..d).collect();
            if reverse {
                order.reverse();
            }
            for &i in &order {
                if i == levels[v] {
                    break;
                }
                lo += row[i];
            }
            eps[v] = (lo + rng.unit() * row[levels[v]]).min(0.999_999);
        }
        let edited = self.simulate(&u, &x, &eps, &delta, 1);

        // Û_I from the encoder.
        let z = (self.image_noise > 0).then(|| {
            let dz = self.image_noise;
            let enc = self
                .encoder
                .forward(self.ncm.matrix(image.to_vec(), 1, self.pixels));
            let mu = enc.clone().narrow(1, 0, dz);
            let log_var = enc.narrow(1, dz, dz).clamp(-8.0, 4.0);
            let e = self
                .ncm
                .matrix((0..dz).map(|_| normal(&mut rng)).collect(), 1, dz);
            mu + log_var.mul_scalar(0.5).exp() * e
        });
        Ok(self.finish(vec![levels], edited, z, Some(image)).remove(0))
    }

    fn check(&self, image: &[f32]) -> Result<(), NcmError> {
        if image.len() == self.pixels {
            Ok(())
        } else {
            Err(NcmError::BadImages(format!(
                "an image of {} pixels for a model of {}",
                image.len(),
                self.pixels
            )))
        }
    }

    fn label_levels(&self, image: &[f32]) -> Vec<usize> {
        let spec = &self.ncm.spec;
        let logits = values(
            self.labeller
                .forward(self.ncm.matrix(image.to_vec(), 1, self.pixels)),
        );
        (0..spec.names.len())
            .map(|v| {
                let at = spec.offsets[v];
                argmax(&logits[at..at + spec.levels[v].len()])
            })
            .collect()
    }

    fn intervention(&self, x: &[(&str, &str)]) -> Result<Vec<(usize, usize)>, NcmError> {
        let spec = &self.ncm.spec;
        x.iter()
            .map(|(k, l)| {
                let v = spec.var(k)?;
                Ok((v, spec.level(v, l)?))
            })
            .collect()
    }

    fn names_of(&self, levels: &[usize]) -> Vec<String> {
        let spec = &self.ncm.spec;
        levels
            .iter()
            .enumerate()
            .map(|(v, l)| spec.levels[v][*l].clone())
            .collect()
    }

    /// `n` prior draws of every `Û_C`, side by side.
    fn prior_noise(&self, n: usize, rng: &mut Rng) -> Tensor<2> {
        let spec = &self.ncm.spec;
        let width = spec.n_noises() * spec.noise_dim;
        self.ncm
            .matrix((0..n * width).map(|_| rng.unit()).collect(), n, width)
    }

    fn image_prior(&self, n: usize, rng: &mut Rng) -> Option<Tensor<2>> {
        let dz = self.image_noise;
        (dz > 0).then(|| {
            self.ncm
                .matrix((0..n * dz).map(|_| normal(rng)).collect(), n, dz)
        })
    }

    /// One-hot encodings of `var`'s parents for each row of levels.
    fn parent_one_hots(&self, var: usize, rows: &[Vec<usize>]) -> Vec<Tensor<2>> {
        let spec = &self.ncm.spec;
        spec.parents[var]
            .iter()
            .map(|&p| one_hot(&self.ncm, rows.iter().map(|r| r[p]), spec.levels[p].len()))
            .collect()
    }

    /// Every variable's level in the world that fixes `x`, by the explicit
    /// model's inverse CDF at `eps`, reversed where `delta < g_V` (the
    /// coupling of [`Ncm::estimate`]).
    fn simulate(
        &self,
        u: &Tensor<2>,
        x: &[(usize, usize)],
        eps: &[f32],
        delta: &[f32],
        n: usize,
    ) -> Vec<Vec<usize>> {
        let spec = &self.ncm.spec;
        let k = spec.names.len();
        let mut rows = vec![vec![0usize; k]; n];
        for v in 0..k {
            if let Some((_, l)) = x.iter().find(|(w, _)| *w == v) {
                for r in &mut rows {
                    r[v] = *l;
                }
                continue;
            }
            let parents = self.parent_one_hots(v, &rows);
            let (logits, gate) = self
                .ncm
                .split(v, forward_rows(spec, &self.ncm.nets, v, parents, u));
            let probs = values(activation::softmax(logits, 1));
            let gates = values(gate);
            let d = spec.levels[v].len();
            for (s, r) in rows.iter_mut().enumerate() {
                let row = &probs[s * d..(s + 1) * d];
                let reverse = delta[s * k + v] < gates[s];
                let mut acc = 0.0;
                let mut pick = |i: &usize| {
                    acc += row[*i];
                    eps[s * k + v] < acc
                };
                r[v] = if reverse {
                    (0..d).rev().find(&mut pick).unwrap_or(0)
                } else {
                    (0..d).find(&mut pick).unwrap_or(d - 1)
                };
            }
        }
        rows
    }

    /// Decodes both worlds with the same `Û_I` and packages the edits.
    fn finish(
        &self,
        factual: Vec<Vec<usize>>,
        edited: Vec<Vec<usize>>,
        z: Option<Tensor<2>>,
        given: Option<&[f32]>,
    ) -> Vec<Edit> {
        let decode_all = |rows: &[Vec<usize>]| -> Vec<f32> {
            values(decode(&self.decoder, &one_hots(&self.ncm, rows), z.clone()))
        };
        let a = match given {
            Some(i) => i.to_vec(),
            None => decode_all(&factual),
        };
        let b = decode_all(&edited);
        let p = self.pixels;
        factual
            .iter()
            .zip(&edited)
            .enumerate()
            .map(|(s, (f, e))| Edit {
                labels: self.names_of(f),
                image: if given.is_some() {
                    a.clone()
                } else {
                    a[s * p..(s + 1) * p].to_vec()
                },
                edited_labels: self.names_of(e),
                edited_image: b[s * p..(s + 1) * p].to_vec(),
            })
            .collect()
    }
}

/// The decoder's pixels for labels `onehots` and image noise `z`.
fn decode(decoder: &Mlp, onehots: &[Tensor<2>], z: Option<Tensor<2>>) -> Tensor<2> {
    let mut parts: Vec<Tensor<2>> = onehots.to_vec();
    parts.extend(z);
    decoder.forward(Tensor::cat(parts, 1))
}

fn one_hots(ncm: &Ncm, rows: &[Vec<usize>]) -> Vec<Tensor<2>> {
    let spec = &ncm.spec;
    (0..spec.names.len())
        .map(|v| one_hot(ncm, rows.iter().map(|r| r[v]), spec.levels[v].len()))
        .collect()
}

fn one_hot(ncm: &Ncm, levels: impl ExactSizeIterator<Item = usize>, d: usize) -> Tensor<2> {
    let n = levels.len();
    let mut x = vec![0.0f32; n * d];
    for (s, l) in levels.enumerate() {
        x[s * d + l] = 1.0;
    }
    ncm.matrix(x, n, d)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn normal(rng: &mut Rng) -> f32 {
    let (a, b) = (f64::from(rng.unit()).max(1e-12), f64::from(rng.unit()));
    ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
}

fn shuffle(order: &mut [usize], rng: &mut Rng) {
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}
