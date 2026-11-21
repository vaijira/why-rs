//! NCMs built from binary-step units, as in the book's worked examples, with
//! exact query evaluation and Algorithm 48.

use std::collections::{BTreeMap, BTreeSet};

use super::{Expanded, Fix, NcmError, c2_components, expand};
use crate::fairness::SplitMix64;
use crate::graph::cinfer::counterfactual::CtfQuery;
use crate::graph::dseparation::Admg;

/// What feeds a [`Unit`].
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// An exogenous variable, uniform on `[0, 1]`.
    Noise(String),
    /// A parent's value, as its level's index (`0`, `1`, …).
    Parent(String),
    /// Another unit's output.
    Unit(Box<Unit>),
}

impl From<Unit> for Input {
    fn from(u: Unit) -> Self {
        Self::Unit(Box::new(u))
    }
}

/// A unit's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// The binary step: 1 when the pre-activation is `>= 0`, else 0.
    Step,
    /// No activation, for output layers such as Eqs. 13.40–13.41.
    Identity,
}

/// One neuron: `activation(Σ w_i x_i + bias)`. A mechanism is a tree of
/// them, which is a feedforward network.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    /// The weighted inputs.
    pub inputs: Vec<(Input, f64)>,
    /// The bias.
    pub bias: f64,
    /// The activation.
    pub activation: Activation,
}

/// The exogenous input `name`.
#[must_use]
pub fn noise(name: &str) -> Input {
    Input::Noise(name.to_string())
}

/// The parent `name`'s value, as a unit with weight 1 and no activation,
/// so it composes with the other units directly.
#[must_use]
pub fn parent(name: &str) -> Unit {
    linear([(Input::Parent(name.to_string()), 1.0)], 0.0)
}

/// `f̂_≥z(x) = σ(x − z)`: 1 when `x >= z`.
#[must_use]
pub fn at_least(x: impl Into<Input>, z: f64) -> Unit {
    Unit {
        inputs: vec![(x.into(), 1.0)],
        bias: -z,
        activation: Activation::Step,
    }
}

/// `f̂_<z(x) = f̂_NOT(f̂_≥z(x))`: 1 when `x < z`.
#[must_use]
pub fn below(x: impl Into<Input>, z: f64) -> Unit {
    not(at_least(x, z))
}

/// `f̂_AND(x_1, …, x_n) = σ(Σ x_i − n)`.
#[must_use]
pub fn and<I: Into<Input>>(xs: impl IntoIterator<Item = I>) -> Unit {
    let inputs: Vec<(Input, f64)> = xs.into_iter().map(|x| (x.into(), 1.0)).collect();
    let n = inputs.len() as f64;
    Unit {
        inputs,
        bias: -n,
        activation: Activation::Step,
    }
}

/// `f̂_OR(x_1, …, x_n) = σ(Σ x_i − 1)`.
#[must_use]
pub fn or<I: Into<Input>>(xs: impl IntoIterator<Item = I>) -> Unit {
    Unit {
        inputs: xs.into_iter().map(|x| (x.into(), 1.0)).collect(),
        bias: -1.0,
        activation: Activation::Step,
    }
}

/// `f̂_NOT(x) = σ(−x)`.
#[must_use]
pub fn not(x: impl Into<Input>) -> Unit {
    Unit {
        inputs: vec![(x.into(), -1.0)],
        bias: 0.0,
        activation: Activation::Step,
    }
}

/// `Σ w_i x_i + bias` with no activation.
#[must_use]
pub fn linear<I: Into<Input>>(terms: impl IntoIterator<Item = (I, f64)>, bias: f64) -> Unit {
    Unit {
        inputs: terms.into_iter().map(|(x, w)| (x.into(), w)).collect(),
        bias,
        activation: Activation::Identity,
    }
}

impl Unit {
    fn eval(&self, parents: &BTreeMap<&str, usize>, noise: &BTreeMap<&str, f64>) -> f64 {
        let mut s = self.bias;
        for (x, w) in &self.inputs {
            let v = match x {
                Input::Noise(n) => noise[n.as_str()],
                Input::Parent(p) => parents[p.as_str()] as f64,
                Input::Unit(u) => u.eval(parents, noise),
            };
            s += w * v;
        }
        match self.activation {
            Activation::Step => f64::from(u8::from(s >= 0.0)),
            Activation::Identity => s,
        }
    }

    fn walk(&self, f: &mut impl FnMut(&Self)) {
        f(self);
        for (x, _) in &self.inputs {
            if let Input::Unit(u) = x {
                u.walk(f);
            }
        }
    }

    fn inputs_named(&self, parents: &mut BTreeSet<String>, noises: &mut BTreeSet<String>) {
        self.walk(&mut |u| {
            for (x, _) in &u.inputs {
                match x {
                    Input::Noise(n) => {
                        noises.insert(n.clone());
                    }
                    Input::Parent(p) => {
                        parents.insert(p.clone());
                    }
                    Input::Unit(_) => {}
                }
            }
        });
    }
}

/// An NCM whose mechanisms are networks of [`Unit`]s, with named noises
/// `Û ~ Unif(0, 1)`, each shared by the variables in its scope
/// (Def. 13.1.7).
///
/// When every unit that reads a noise reads only that noise through a step,
/// each mechanism is constant on the boxes between those units' thresholds.
/// [`StepNcm::probability`] integrates over the boxes, so it is exact. All the
/// book's hand-built NCMs have this form.
///
/// ```
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
/// use why_data::ncm::{StepNcm, and, at_least, noise, not, or, parent};
///
/// // M̂_1 of Ex. 13.6 (Eqs. 13.47–13.48): no confounding.
/// let m = StepNcm::new()
///     .noise("Ud", &["D"])
///     .noise("Ub", &["B"])
///     .mechanism("D", at_least(noise("Ud"), 112.0 / 256.0))
///     .mechanism(
///         "B",
///         or([
///             and([not(parent("D")), at_least(noise("Ub"), 34.0 / 112.0)]),
///             and([parent("D"), at_least(noise("Ub"), 90.0 / 144.0)]),
///         ]),
///     );
/// let g = m.graph().unwrap();
/// let q = CtfQuery::parse(&g, "P(b_d)").unwrap();
/// let p = m.probability(&q, &[("B", "1"), ("D", "1")]).unwrap();
/// assert!((p - 54.0 / 144.0).abs() < 1e-12); // Eq. 13.58
/// ```
#[derive(Clone, Debug, Default)]
pub struct StepNcm {
    noises: Vec<(String, Vec<String>)>,
    mechanisms: BTreeMap<String, Unit>,
    levels: BTreeMap<String, Vec<String>>,
}

impl StepNcm {
    /// No variables yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the noise `name`, which the mechanisms of `scope` may read.
    #[must_use]
    pub fn noise(mut self, name: &str, scope: &[&str]) -> Self {
        self.noises.push((
            name.to_string(),
            scope.iter().map(|s| (*s).to_string()).collect(),
        ));
        self
    }

    /// Sets the mechanism `f̂_var`. Its output is the index of `var`'s level.
    #[must_use]
    pub fn mechanism(mut self, var: &str, f: Unit) -> Self {
        self.mechanisms.insert(var.to_string(), f);
        self
    }

    /// Names `var`'s levels; the default is `["0", "1"]`.
    #[must_use]
    pub fn levels(mut self, var: &str, levels: &[&str]) -> Self {
        self.levels.insert(
            var.to_string(),
            levels.iter().map(|s| (*s).to_string()).collect(),
        );
        self
    }

    fn levels_of(&self, var: &str) -> Vec<String> {
        self.levels
            .get(var)
            .cloned()
            .unwrap_or_else(|| vec!["0".into(), "1".into()])
    }

    /// The parents `var`'s mechanism reads.
    #[must_use]
    pub fn parents(&self, var: &str) -> Vec<String> {
        let (mut p, mut n) = (BTreeSet::new(), BTreeSet::new());
        if let Some(f) = self.mechanisms.get(var) {
            f.inputs_named(&mut p, &mut n);
        }
        p.into_iter().collect()
    }

    /// The causal diagram the NCM induces: an edge `P -> V` for each parent
    /// a mechanism reads, and `A <-> B` for each pair of variables whose
    /// mechanisms read a common noise.
    ///
    /// # Errors
    ///
    /// The errors of [`StepNcm::probability`] for an ill-formed model.
    pub fn graph(&self) -> Result<Admg, NcmError> {
        let order = self.order()?;
        let mut g = Admg::new();
        for v in &order {
            g.add_node(v);
        }
        for v in &order {
            for p in self.parents(v) {
                g.directed(&p, v);
            }
        }
        for (name, _) in &self.noises {
            let readers: Vec<&String> = order.iter().filter(|v| self.reads(v, name)).collect();
            for (i, a) in readers.iter().enumerate() {
                for b in &readers[i + 1..] {
                    g.bidirected(a, b);
                }
            }
        }
        Ok(g)
    }

    fn reads(&self, var: &str, noise: &str) -> bool {
        let (mut p, mut n) = (BTreeSet::new(), BTreeSet::new());
        if let Some(f) = self.mechanisms.get(var) {
            f.inputs_named(&mut p, &mut n);
        }
        n.contains(noise)
    }

    /// Whether this NCM is G-constrained (Def. 13.1.11), up to unused
    /// inputs: every mechanism reads only parents it has in `g`, and every
    /// noise is read only by variables of one C²-component of `g`.
    ///
    /// # Errors
    ///
    /// The errors of [`StepNcm::probability`] for an ill-formed model.
    pub fn is_g_constrained(&self, g: &Admg) -> Result<bool, NcmError> {
        let theirs = super::NcmStructure::from_graph(g)?;
        for v in self.order()? {
            let Some(pa) = theirs.parents.get(&v) else {
                return Ok(false);
            };
            if self.parents(&v).iter().any(|p| !pa.contains(p)) {
                return Ok(false);
            }
        }
        let c2 = c2_components(g);
        for (name, _) in &self.noises {
            let readers: Vec<String> = self
                .mechanisms
                .keys()
                .filter(|v| self.reads(v, name))
                .cloned()
                .collect();
            if !c2.iter().any(|c| readers.iter().all(|r| c.contains(r))) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The variables in an order where every mechanism's parents come first.
    fn order(&self) -> Result<Vec<String>, NcmError> {
        for (v, f) in &self.mechanisms {
            let (mut p, mut n) = (BTreeSet::new(), BTreeSet::new());
            f.inputs_named(&mut p, &mut n);
            for q in &p {
                if !self.mechanisms.contains_key(q) {
                    return Err(NcmError::UnknownVariable(q.clone()));
                }
            }
            for u in &n {
                match self.noises.iter().find(|(name, _)| name == u) {
                    None => return Err(NcmError::UnknownVariable(u.clone())),
                    Some((_, scope)) if !scope.contains(v) => {
                        return Err(NcmError::InvalidInput(format!(
                            "the mechanism of {v} reads {u}, whose scope does not contain it"
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        let mut order: Vec<String> = Vec::new();
        let mut done: BTreeSet<String> = BTreeSet::new();
        while order.len() < self.mechanisms.len() {
            let next = self
                .mechanisms
                .keys()
                .find(|v| !done.contains(*v) && self.parents(v).iter().all(|p| done.contains(p)));
            match next {
                Some(v) => {
                    done.insert(v.clone());
                    order.push(v.clone());
                }
                None => {
                    let stuck = self
                        .mechanisms
                        .keys()
                        .find(|v| !done.contains(*v))
                        .expect("some left");
                    return Err(NcmError::Cyclic(stuck.clone()));
                }
            }
        }
        Ok(order)
    }

    /// The thresholds at which some unit's output can change, per noise.
    fn breakpoints(&self) -> Result<Vec<Vec<f64>>, NcmError> {
        let mut out: Vec<Vec<f64>> = vec![vec![0.0, 1.0]; self.noises.len()];
        for (v, f) in &self.mechanisms {
            let mut err = None;
            f.walk(&mut |u| {
                let reads_noise = u.inputs.iter().any(|(x, _)| matches!(x, Input::Noise(_)));
                if !reads_noise {
                    return;
                }
                match (u.inputs.as_slice(), u.activation) {
                    ([(Input::Noise(n), w)], Activation::Step) if *w != 0.0 => {
                        let i = self
                            .noises
                            .iter()
                            .position(|(name, _)| name == n)
                            .expect("checked");
                        let t = -u.bias / w;
                        if t > 0.0 && t < 1.0 {
                            out[i].push(t);
                        }
                    }
                    _ => err = Some(NcmError::NotAxisAligned(v.clone())),
                }
            });
            if let Some(e) = err {
                return Err(e);
            }
        }
        for b in &mut out {
            b.sort_by(f64::total_cmp);
            b.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        }
        Ok(out)
    }

    /// The level indices every world's variables take under one draw `u`.
    fn simulate(
        &self,
        order: &[String],
        expanded: &Expanded,
        u: &BTreeMap<&str, f64>,
    ) -> Result<Vec<BTreeMap<String, usize>>, NcmError> {
        let mut out: Vec<BTreeMap<String, usize>> = Vec::with_capacity(expanded.worlds.len());
        for w in &expanded.worlds {
            let mut vals: BTreeMap<String, usize> = BTreeMap::new();
            for v in order {
                let x = match w.fixed.get(v) {
                    Some(Fix::Level(l)) => self.index(v, l)?,
                    Some(Fix::Copy(k)) => out[*k][v],
                    None => {
                        let pa: BTreeMap<&str, usize> =
                            vals.iter().map(|(k, x)| (k.as_str(), *x)).collect();
                        let y = self.mechanisms[v].eval(&pa, u);
                        let r = y.round();
                        let n = self.levels_of(v).len();
                        if (y - r).abs() > 1e-9 || r < 0.0 || r as usize >= n {
                            return Err(NcmError::OutOfDomain {
                                var: v.clone(),
                                value: y.to_string(),
                            });
                        }
                        r as usize
                    }
                };
                vals.insert(v.clone(), x);
            }
            out.push(vals);
        }
        Ok(out)
    }

    fn index(&self, var: &str, level: &str) -> Result<usize, NcmError> {
        if !self.mechanisms.contains_key(var) {
            return Err(NcmError::UnknownVariable(var.to_string()));
        }
        self.levels_of(var)
            .iter()
            .position(|l| l == level)
            .ok_or_else(|| NcmError::OutOfDomain {
                var: var.to_string(),
                value: level.to_string(),
            })
    }

    fn targets(&self, a: &[super::Assignment]) -> Result<Vec<(usize, String, usize)>, NcmError> {
        a.iter()
            .map(|a| Ok((a.world, a.var.clone(), self.index(&a.var, &a.level)?)))
            .collect()
    }

    /// The exact `P(y_* | x_*)` (Eq. 13.152), with each value name mapped to
    /// a level by `values` as in [`expand`].
    ///
    /// # Errors
    ///
    /// [`NcmError::ImpossibleCondition`] when `P(x_*) = 0`;
    /// [`NcmError::NotAxisAligned`] when a unit mixes a noise with other
    /// inputs; [`NcmError::TooLarge`] past ten million boxes; the errors of
    /// [`expand`]; and others for an ill-formed model.
    pub fn probability(&self, query: &CtfQuery, values: &[(&str, &str)]) -> Result<f64, NcmError> {
        const LIMIT: usize = 10_000_000;
        let expanded = expand(query, values)?;
        let order = self.order()?;
        let events = self.targets(&expanded.events)?;
        let given = self.targets(&expanded.given)?;
        let cuts = self.breakpoints()?;
        let total = cuts
            .iter()
            .try_fold(1usize, |acc, b| acc.checked_mul(b.len() - 1));
        if total.is_none_or(|t| t > LIMIT) {
            return Err(NcmError::TooLarge(LIMIT));
        }
        let names: Vec<&str> = self.noises.iter().map(|(n, _)| n.as_str()).collect();
        let mut idx = vec![0usize; cuts.len()];
        let (mut num, mut den) = (0.0, 0.0);
        loop {
            let mut weight = 1.0;
            let mut u = BTreeMap::new();
            for (k, b) in cuts.iter().enumerate() {
                let (lo, hi) = (b[idx[k]], b[idx[k] + 1]);
                weight *= hi - lo;
                u.insert(names[k], 0.5 * (lo + hi));
            }
            let vals = self.simulate(&order, &expanded, &u)?;
            let holds =
                |list: &[(usize, String, usize)]| list.iter().all(|(w, v, x)| vals[*w][v] == *x);
            if holds(&given) {
                den += weight;
                if holds(&events) {
                    num += weight;
                }
            }
            // Advance the mixed-radix counter over boxes.
            let mut k = 0;
            loop {
                if k == idx.len() {
                    if den <= 0.0 {
                        return Err(NcmError::ImpossibleCondition);
                    }
                    return Ok(num / den);
                }
                idx[k] += 1;
                if idx[k] + 1 < cuts[k].len() {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
        }
    }

    /// Algorithm 48 (NCM counterfactual sampling): draws `Û` until `m` draws
    /// satisfy `x_*`, and returns the levels the events' counterfactual
    /// variables take in each, in the order of `query.events`.
    ///
    /// # Errors
    ///
    /// [`NcmError::ImpossibleCondition`] when a million draws in a row
    /// reject; the errors of [`expand`]; others for an ill-formed model.
    pub fn sample(
        &self,
        query: &CtfQuery,
        values: &[(&str, &str)],
        m: usize,
        seed: u64,
    ) -> Result<Vec<Vec<String>>, NcmError> {
        let expanded = expand(query, values)?;
        let order = self.order()?;
        let given = self.targets(&expanded.given)?;
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(m);
        let mut misses = 0usize;
        while out.len() < m {
            let u: BTreeMap<&str, f64> = self
                .noises
                .iter()
                .map(|(n, _)| (n.as_str(), rng.unit()))
                .collect();
            let vals = self.simulate(&order, &expanded, &u)?;
            if given.iter().all(|(w, v, x)| vals[*w][v] == *x) {
                misses = 0;
                out.push(
                    expanded
                        .events
                        .iter()
                        .map(|a| self.levels_of(&a.var)[vals[a.world][&a.var]].clone())
                        .collect(),
                );
            } else {
                misses += 1;
                if misses >= 1_000_000 {
                    return Err(NcmError::ImpossibleCondition);
                }
            }
        }
        Ok(out)
    }

    /// Eq. 13.151: the Monte-Carlo estimate of `P(y_* | x_*)` from `m`
    /// accepted draws of [`StepNcm::sample`].
    ///
    /// # Errors
    ///
    /// Those of [`StepNcm::sample`].
    pub fn estimate(
        &self,
        query: &CtfQuery,
        values: &[(&str, &str)],
        m: usize,
        seed: u64,
    ) -> Result<f64, NcmError> {
        let expanded = expand(query, values)?;
        let wanted: Vec<&str> = expanded.events.iter().map(|a| a.level.as_str()).collect();
        let draws = self.sample(query, values, m, seed)?;
        let hits = draws
            .iter()
            .filter(|d| d.iter().map(String::as_str).eq(wanted.iter().copied()))
            .count();
        Ok(hits as f64 / m.max(1) as f64)
    }
}
