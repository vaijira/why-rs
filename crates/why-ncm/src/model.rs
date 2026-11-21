//! The G-constrained NCM: one MLP per variable (Def. 13.1.11, Eq. 13.141).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData, activation};
use why_data::graph::dseparation::Admg;
use why_data::ncm::NcmStructure;

use crate::{Experiment, Kind, NcmConfig, NcmError, Rng};

/// Everything about an NCM except its parameters.
#[derive(Debug)]
pub(crate) struct Spec {
    pub structure: NcmStructure,
    /// The variables in topological order; a variable's index is its rank.
    pub names: Vec<String>,
    pub index: BTreeMap<String, usize>,
    pub levels: Vec<Vec<String>>,
    pub parents: Vec<Vec<usize>>,
    pub noises: Vec<Vec<usize>>,
    pub noise_dim: usize,
    /// Where each variable's levels start in a row of all levels.
    pub offsets: Vec<usize>,
    /// The number of levels of all variables together.
    pub width: usize,
}

impl Spec {
    pub fn var(&self, name: &str) -> Result<usize, NcmError> {
        self.index
            .get(name)
            .copied()
            .ok_or_else(|| NcmError::UnknownVariable(name.to_string()))
    }

    pub fn level(&self, var: usize, level: &str) -> Result<usize, NcmError> {
        self.levels[var]
            .iter()
            .position(|l| l == level)
            .ok_or_else(|| NcmError::UnknownLevel {
                var: self.names[var].clone(),
                level: level.to_string(),
            })
    }

    pub fn input_dim(&self, var: usize) -> usize {
        self.parents[var]
            .iter()
            .map(|p| self.levels[*p].len())
            .sum::<usize>()
            + self.noises[var].len() * self.noise_dim
    }

    pub fn n_noises(&self) -> usize {
        self.structure.confounders.len()
    }
}

/// Serializes parameter initialization. burn's Flex backend keeps one
/// global RNG, so two models initialized at once (say, by tests running in
/// parallel) would interleave their draws and stop being reproducible.
static INIT: Mutex<()> = Mutex::new(());

/// Seeds the backend and runs `init` while no other model initializes.
pub(crate) fn seeded<T>(device: &Device, seed: u64, init: impl FnOnce() -> T) -> T {
    let _guard = INIT.lock().unwrap_or_else(PoisonError::into_inner);
    device.seed(seed);
    init()
}

/// A feedforward network with ReLU hidden layers and linear output.
#[derive(Module, Debug)]
pub(crate) struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub(crate) fn new(
        input: usize,
        hidden: usize,
        depth: usize,
        output: usize,
        device: &Device,
    ) -> Self {
        let mut sizes = vec![input.max(1)];
        sizes.extend(std::iter::repeat_n(hidden, depth));
        sizes.push(output);
        let layers = sizes
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(x);
            if i < last {
                x = activation::relu(x);
            }
        }
        x
    }
}

/// `φ_V` of `nets` on per-row inputs: `parents[i]` is `rows × |D_{P_i}|`
/// and `noise` is `rows × (n_noises · noise_dim)`, every `Û_C` side by side.
/// Unlike [`Ncm::logits_soft`] the noise is a tensor, so gradients reach
/// whatever produced it (the ANCM's encoder).
pub(crate) fn forward_rows(
    spec: &Spec,
    nets: &Nets,
    var: usize,
    parents: Vec<Tensor<2>>,
    noise: &Tensor<2>,
) -> Tensor<2> {
    let rows = noise.dims()[0];
    let d = spec.noise_dim;
    let mut parts = parents;
    for &k in &spec.noises[var] {
        parts.push(noise.clone().narrow(1, k * d, d));
    }
    let input = if parts.is_empty() {
        Tensor::<2>::zeros([rows, 1], &noise.device())
    } else {
        Tensor::cat(parts, 1)
    };
    nets.mlps[var].forward(input)
}

/// The trainable part: `φ_V` for each variable, in topological order.
#[derive(Module, Debug)]
pub(crate) struct Nets {
    pub mlps: Vec<Mlp>,
}

/// One batch of `m` draws of every `Û_C`, each `noise_dim` uniforms.
#[derive(Clone, Debug)]
pub(crate) struct Noise {
    pub m: usize,
    /// `m × n_noises × noise_dim`, row-major.
    pub values: Vec<f32>,
    /// One `ε_V` per draw and variable, `m × n_vars`: the explicit model's
    /// sampling noise, shared by every world.
    pub eps: Vec<f32>,
    /// One `δ_V` per draw and variable, `m × n_vars`: whether the explicit
    /// model reads `V`'s levels in reverse in a world (when `δ_V < g_V`).
    pub delta: Vec<f32>,
    /// One standard Gumbel `G_{V=v}` per draw and level, `m × width`: the
    /// implicit model's sampling noise (Eq. 13.141), shared by every world.
    pub gumbel: Vec<f32>,
}

/// A G-constrained neural causal model over categorical variables.
///
/// Each variable `V` has a network `φ_V` whose inputs are its parents in `G`
/// (one-hot) and the noises of the C²-components that contain it, so the
/// model is G-constrained by construction (Def. 13.1.11) and satisfies `G`'s
/// constraints on all three layers (Thms 13.1.6, 13.2.3).
#[derive(Clone, Debug)]
pub struct Ncm {
    pub(crate) spec: Arc<Spec>,
    pub(crate) nets: Nets,
    pub(crate) device: Device,
    pub(crate) kind: Kind,
    pub(crate) temperature: f64,
    pub(crate) max_cells: usize,
}

impl Ncm {
    /// An untrained G-NCM for `g`, with each variable's levels from
    /// `levels`. Initialization is seeded by `config.seed`.
    ///
    /// # Errors
    ///
    /// [`NcmError::Structure`] when `g` is cyclic, and
    /// [`NcmError::MissingColumn`] when a variable has no levels.
    pub fn new(
        g: &Admg,
        levels: &BTreeMap<String, Vec<String>>,
        config: &NcmConfig,
    ) -> Result<Self, NcmError> {
        let structure = NcmStructure::from_graph(g)?;
        let names = structure.order.clone();
        let index: BTreeMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let levels: Vec<Vec<String>> = names
            .iter()
            .map(|v| {
                levels
                    .get(v)
                    .filter(|l| !l.is_empty())
                    .cloned()
                    .ok_or_else(|| NcmError::MissingColumn(v.clone()))
            })
            .collect::<Result<_, _>>()?;
        let parents = names
            .iter()
            .map(|v| structure.parents[v].iter().map(|p| index[p]).collect())
            .collect();
        let noises = names.iter().map(|v| structure.noises_of(v)).collect();
        let offsets: Vec<usize> = levels
            .iter()
            .scan(0, |at, l| {
                let here = *at;
                *at += l.len();
                Some(here)
            })
            .collect();
        let width = levels.iter().map(Vec::len).sum();
        let spec = Spec {
            structure,
            names,
            index,
            levels,
            parents,
            noises,
            noise_dim: config.noise_dim.max(1),
            offsets,
            width,
        };
        let device = Device::flex().autodiff();
        let mlps = seeded(&device, config.seed, || {
            (0..spec.names.len())
                .map(|v| {
                    Mlp::new(
                        spec.input_dim(v),
                        config.hidden,
                        config.layers,
                        spec.levels[v].len() + 1,
                        &device,
                    )
                })
                .collect()
        });
        Ok(Self {
            spec: Arc::new(spec),
            nets: Nets { mlps },
            device,
            kind: config.kind,
            temperature: config.temperature,
            max_cells: config.max_cells,
        })
    }

    /// An untrained G-NCM for `g`, with each variable's levels read off the
    /// data sets (their union, sorted).
    ///
    /// # Errors
    ///
    /// [`NcmError::NoData`] without data sets, and those of [`Ncm::new`].
    pub fn from_data(
        g: &Admg,
        experiments: &[Experiment],
        config: &NcmConfig,
    ) -> Result<Self, NcmError> {
        if experiments.is_empty() {
            return Err(NcmError::NoData);
        }
        let mut levels: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for v in g.variables() {
            let set = levels.entry(v.to_string()).or_default();
            for e in experiments {
                let l = e
                    .data
                    .levels(v)
                    .ok_or_else(|| NcmError::MissingColumn(v.to_string()))?;
                set.extend(l.iter().cloned());
            }
        }
        let levels = levels
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect();
        Self::new(g, &levels, config)
    }

    /// The variables, in the topological order the model simulates them.
    #[must_use]
    pub fn variables(&self) -> &[String] {
        &self.spec.names
    }

    /// The levels of `var`, if it is a variable of the model.
    #[must_use]
    pub fn levels(&self, var: &str) -> Option<&[String]> {
        self.spec
            .index
            .get(var)
            .map(|i| self.spec.levels[*i].as_slice())
    }

    /// Whether the networks are read as probabilities or as samples.
    #[must_use]
    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub(crate) fn draw(&self, m: usize, rng: &mut Rng) -> Noise {
        let width = self.spec.n_noises() * self.spec.noise_dim;
        let values = (0..m * width).map(|_| rng.unit()).collect();
        let eps = (0..m * self.spec.names.len()).map(|_| rng.unit()).collect();
        let gumbel = (0..m * self.spec.width)
            .map(|_| -(-(rng.unit().max(1e-7)).ln()).ln())
            .collect();
        let delta = (0..m * self.spec.names.len()).map(|_| rng.unit()).collect();
        Noise {
            m,
            values,
            eps,
            delta,
            gumbel,
        }
    }

    /// The noise part of `var`'s input for draw `s`.
    fn push_noise(&self, var: usize, noise: &Noise, s: usize, out: &mut Vec<f32>) {
        let d = self.spec.noise_dim;
        let width = self.spec.n_noises() * d;
        for &k in &self.spec.noises[var] {
            let at = s * width + k * d;
            out.extend_from_slice(&noise.values[at..at + d]);
        }
    }

    /// The network of `var` (logits and gate logit, see [`Ncm::split`]) on
    /// every pair of a parent configuration and a draw: row
    /// `c * m + s` reads `parents[c]` (level indices, in the order of the
    /// variable's parents) and draw `s`.
    pub(crate) fn logits(&self, var: usize, parents: &[Vec<usize>], noise: &Noise) -> Tensor<2> {
        let dim = self.spec.input_dim(var).max(1);
        let rows = parents.len() * noise.m;
        let mut x = Vec::with_capacity(rows * dim);
        for pa in parents {
            for s in 0..noise.m {
                let start = x.len();
                for (p, &level) in self.spec.parents[var].iter().zip(pa) {
                    let n = self.spec.levels[*p].len();
                    x.extend((0..n).map(|i| f32::from(u8::from(i == level))));
                }
                self.push_noise(var, noise, s, &mut x);
                x.resize(start + dim, 0.0);
            }
        }
        let input = Tensor::<2>::from_data(TensorData::new(x, [rows, dim]), &self.device);
        self.nets.mlps[var].forward(input)
    }

    /// The network of `var` on per-draw parent encodings: `parents[i]` is `m × |D_{P_i}|`
    /// (one-hot, or a straight-through relaxation of one).
    pub(crate) fn logits_soft(
        &self,
        var: usize,
        parents: Vec<Tensor<2>>,
        noise: &Noise,
    ) -> Tensor<2> {
        let d = self.spec.noises[var].len() * self.spec.noise_dim;
        let mut parts = parents;
        if d > 0 {
            let mut x = Vec::with_capacity(noise.m * d);
            for s in 0..noise.m {
                self.push_noise(var, noise, s, &mut x);
            }
            parts.push(Tensor::<2>::from_data(
                TensorData::new(x, [noise.m, d]),
                &self.device,
            ));
        }
        let input = if parts.is_empty() {
            Tensor::<2>::zeros([noise.m, 1], &self.device)
        } else {
            Tensor::cat(parts, 1)
        };
        self.nets.mlps[var].forward(input)
    }

    /// A network's output split into the logits over `var`'s levels and,
    /// for the explicit model, the order gate `g_V = sigmoid(·)`.
    pub(crate) fn split(&self, var: usize, out: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let d = self.spec.levels[var].len();
        let gate = activation::sigmoid(out.clone().narrow(1, d, 1));
        (out.narrow(1, 0, d), gate)
    }

    /// A constant `rows × cols` matrix.
    pub(crate) fn matrix(&self, values: Vec<f32>, rows: usize, cols: usize) -> Tensor<2> {
        Tensor::<2>::from_data(TensorData::new(values, [rows, cols]), &self.device)
    }
}

/// A tensor's values.
pub(crate) fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("f32 data")
}
