//! Augmented SCMs (Def. 14.1.1) and image counterfactual distributions
//! (Eq. 14.4, Def. 14.3.1).

use std::collections::HashMap;

use super::AscmError;
use crate::fairness::Dataset;
use crate::scm::{Model, Value, WorldId};

/// An augmented SCM (Def. 14.1.1) over discrete variables: a structural
/// model whose endogenous variables include the labelled generative factors
/// `V` and the pixels of the image `I`.
///
/// The pixels' mechanisms play the role of `f_I`. The constructor checks
/// that `f_I` is invertible in `V` on the model's support, so the labeller
/// `h` (Eq. 14.1) is a function: [`Ascm::label`].
#[derive(Clone, Debug)]
pub struct Ascm {
    model: Model,
    factors: Vec<String>,
    image: Vec<String>,
    labels: HashMap<Vec<Value>, Vec<Value>>,
}

impl Ascm {
    /// The ASCM of `model` with generative factors `factors` and pixels
    /// `image`, both endogenous variables of `model`.
    ///
    /// # Errors
    ///
    /// [`AscmError::UnknownVariable`] for a name that is not endogenous,
    /// [`AscmError::NotInvertible`] when two images with positive probability
    /// coincide while their labels differ, and the model's own errors.
    pub fn new(model: Model, factors: &[&str], image: &[&str]) -> Result<Self, AscmError> {
        let obs = model.world(model.observational());
        for v in factors.iter().chain(image) {
            if !obs.world().variables().iter().any(|n| n == v) {
                return Err(AscmError::UnknownVariable((*v).to_string()));
            }
        }
        let mut columns: Vec<&str> = image.to_vec();
        columns.extend_from_slice(factors);
        let table = obs.probability_table(Some(&columns))?;
        let mut labels: HashMap<Vec<Value>, Vec<Value>> = HashMap::new();
        for (row, p) in &table.rows {
            if *p <= 0.0 {
                continue;
            }
            let (i, v) = row.split_at(image.len());
            if labels.entry(i.to_vec()).or_insert_with(|| v.to_vec()) != v {
                return Err(AscmError::NotInvertible);
            }
        }
        Ok(Self {
            model,
            factors: factors.iter().map(|s| (*s).to_string()).collect(),
            image: image.iter().map(|s| (*s).to_string()).collect(),
            labels,
        })
    }

    /// The underlying structural model.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The generative factors `V`.
    #[must_use]
    pub fn factors(&self) -> &[String] {
        &self.factors
    }

    /// The pixels of `I`.
    #[must_use]
    pub fn image(&self) -> &[String] {
        &self.image
    }

    /// The labeller `h`: the factors an image shows, in the order of
    /// [`Ascm::factors`], or `None` for an image outside the support.
    #[must_use]
    pub fn label(&self, image: &[Value]) -> Option<&[Value]> {
        self.labels.get(image).map(Vec::as_slice)
    }

    /// The labeller `h_W` restricted to the factors `w`.
    ///
    /// # Errors
    ///
    /// [`AscmError::UnknownVariable`] when a name in `w` is not a factor.
    pub fn label_of(&self, image: &[Value], w: &[&str]) -> Result<Option<Vec<Value>>, AscmError> {
        let at = w
            .iter()
            .map(|n| {
                self.factors
                    .iter()
                    .position(|f| f == n)
                    .ok_or_else(|| AscmError::UnknownVariable((*n).to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .label(image)
            .map(|v| at.iter().map(|&i| v[i]).collect()))
    }

    /// The observational distribution `P(V)` of the factors, as a data set
    /// whose weights are the probabilities. Levels are written as integers
    /// (`0`/`1` for booleans).
    ///
    /// # Errors
    ///
    /// The model's errors.
    pub fn observational(&self) -> Result<Dataset, AscmError> {
        let names: Vec<&str> = self.factors.iter().map(String::as_str).collect();
        let table = self
            .model
            .world(self.model.observational())
            .probability_table(Some(&names))?;
        let rows: Vec<(Vec<String>, f64)> = table
            .rows
            .iter()
            .filter(|(_, p)| *p > 0.0)
            .map(|(r, p)| (r.iter().map(|v| v.as_int().to_string()).collect(), *p))
            .collect();
        let weights = rows.iter().map(|(_, p)| *p).collect();
        Ok(
            Dataset::from_records(names, rows.into_iter().map(|(r, _)| r))?
                .with_weights(weights)?,
        )
    }

    /// The names the endogenous variables `names` have in the world `do(x)`.
    fn names_in(
        &mut self,
        x: &[(&str, Value)],
        names: &[String],
    ) -> Result<Vec<String>, AscmError> {
        if x.is_empty() {
            return Ok(names.to_vec());
        }
        let w: WorldId = self.model.intervene(x)?;
        let base = self
            .model
            .world(self.model.observational())
            .world()
            .variables()
            .to_vec();
        let renamed = self.model.world(w).world().variables();
        Ok(names
            .iter()
            .map(|n| {
                let i = base.iter().position(|b| b == n).expect("endogenous");
                renamed[i].clone()
            })
            .collect())
    }

    /// The image counterfactual distribution `P(I, I_x)` (Eq. 14.4): the
    /// joint of the factual image and the image in the world `do(x)`, under
    /// one shared draw of the exogenous variables.
    ///
    /// # Errors
    ///
    /// The model's errors, for instance for an `x` that names no endogenous
    /// variable.
    pub fn image_counterfactual(
        &mut self,
        x: &[(&str, Value)],
    ) -> Result<ImageCounterfactual, AscmError> {
        let image = self.image.clone();
        let edited = self.names_in(x, &image)?;
        let mut columns: Vec<&str> = image.iter().map(String::as_str).collect();
        columns.extend(edited.iter().map(String::as_str));
        let table = self
            .model
            .world(self.model.observational())
            .probability_table(Some(&columns))?;
        let n = image.len();
        let rows = table
            .rows
            .into_iter()
            .filter(|(_, p)| *p > 0.0)
            .map(|(r, p)| (r[..n].to_vec(), r[n..].to_vec(), p))
            .collect();
        Ok(ImageCounterfactual { rows })
    }

    /// The generative-level counterfactual `P(w′_x | w)`: `events` hold in
    /// the world `do(x)` and `given` in the factual world. With an empty
    /// `given` this is the joint `P(w′_x)`; put the factual events in
    /// `events` under an empty `x` to get the joint `P(w, w′_x)` of
    /// Lemma 14.3.1 through [`Ascm::joint_counterfactual`].
    ///
    /// # Errors
    ///
    /// [`AscmError::ImpossibleCondition`] when `given` has probability zero,
    /// and the model's errors.
    pub fn counterfactual(
        &mut self,
        x: &[(&str, Value)],
        events: &[(&str, Value)],
        given: &[(&str, Value)],
    ) -> Result<f64, AscmError> {
        let num = self.joint_counterfactual(x, events, given)?;
        let den = self.joint_counterfactual(&[], given, &[])?;
        if den <= 0.0 {
            return Err(AscmError::ImpossibleCondition);
        }
        Ok(num / den)
    }

    /// The joint `P(w′_x, w)` of `events` in the world `do(x)` and `factual`
    /// in the factual world.
    ///
    /// # Errors
    ///
    /// The model's errors.
    pub fn joint_counterfactual(
        &mut self,
        x: &[(&str, Value)],
        events: &[(&str, Value)],
        factual: &[(&str, Value)],
    ) -> Result<f64, AscmError> {
        let names: Vec<String> = events.iter().map(|(n, _)| (*n).to_string()).collect();
        let renamed = self.names_in(x, &names)?;
        let mut columns: Vec<&str> = renamed.iter().map(String::as_str).collect();
        columns.extend(factual.iter().map(|(n, _)| *n));
        let wanted: Vec<i64> = events
            .iter()
            .chain(factual)
            .map(|(_, v)| v.as_int())
            .collect();
        let table = self
            .model
            .world(self.model.observational())
            .probability_table(Some(&columns))?;
        Ok(table
            .rows
            .iter()
            .filter(|(r, _)| r.iter().map(|v| v.as_int()).eq(wanted.iter().copied()))
            .map(|(_, p)| p)
            .sum::<f64>()
            + 0.0)
    }
}

/// An image counterfactual distribution `P(I, I_{x′})`: each pair of a
/// factual and an edited image with positive probability (Fig. 14.5).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageCounterfactual {
    /// `(i, i′, P(I = i, I_{x′} = i′))`, ordered by the images.
    pub rows: Vec<(Vec<Value>, Vec<Value>, f64)>,
}

impl ImageCounterfactual {
    /// The image counterfactual query `P(I = i, I_{x′} = i′)`.
    #[must_use]
    pub fn probability(&self, i: &[Value], edited: &[Value]) -> f64 {
        self.feature(
            |im| im.iter().map(|v| v.as_int()).collect::<Vec<_>>(),
            &ints(i),
            &ints(edited),
        )
    }

    /// The F-ctf query `λ_φ(P(I, I_{x′}), w, w′)` (Def. 14.3.1): the mass of
    /// every image pair whose cared features are `w` before the edit and
    /// `w′` after.
    pub fn feature<W: PartialEq>(&self, phi: impl Fn(&[Value]) -> W, w: &W, edited: &W) -> f64 {
        self.rows
            .iter()
            .filter(|(a, b, _)| phi(a) == *w && phi(b) == *edited)
            .map(|(_, _, p)| p)
            .sum()
    }

    /// The F-ctf query relative to the image query `P(i, i′_{x′})`,
    /// `λ_φ(P(i, i′_{x′}))`, which takes `w = φ(i)` and `w′ = φ(i′)`.
    pub fn feature_of<W: PartialEq>(
        &self,
        phi: impl Fn(&[Value]) -> W,
        i: &[Value],
        edited: &[Value],
    ) -> f64 {
        let (w, w2) = (phi(i), phi(edited));
        self.feature(phi, &w, &w2)
    }
}

fn ints(v: &[Value]) -> Vec<i64> {
    v.iter().map(|x| x.as_int()).collect()
}
