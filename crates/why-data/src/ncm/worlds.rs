//! Counterfactual queries as the worlds an NCM simulates (Eq. 13.150,
//! Def. 13.3.1).

use std::collections::{BTreeMap, BTreeSet};

use super::{NcmError, NcmStructure};
use crate::graph::cinfer::counterfactual::{CtfError, CtfQuery, Event, Setting};

/// What an intervention sets a variable to in one world.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fix {
    /// A constant: the atomic intervention `do(V = level)`.
    Level(String),
    /// The value the same variable takes in an earlier world, as `W_{x'}`
    /// does in `Y_{x, W_{x'}}`: a natural intervention (Def. 13.3.1).
    Copy(usize),
}

/// One submodel `M_x̃`: the variables an intervention fixes, and how.
/// The natural world fixes nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct World {
    /// The fixed variables.
    pub fixed: BTreeMap<String, Fix>,
}

/// `V = level` in world `world`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Assignment {
    /// Index into [`Expanded::worlds`].
    pub world: usize,
    /// The variable.
    pub var: String,
    /// Its level.
    pub level: String,
}

/// A counterfactual query `P(y_* | x_*)` laid out over worlds that share
/// one draw of the exogenous variables.
///
/// A world's [`Fix::Copy`] entries always point to earlier worlds, so the
/// worlds can be simulated in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expanded {
    /// The distinct submodels the query mentions.
    pub worlds: Vec<World>,
    /// The events `y_*` whose probability is asked for.
    pub events: Vec<Assignment>,
    /// The conditioning events `x_*`; empty for a joint.
    pub given: Vec<Assignment>,
}

/// Lays `query` out over worlds, with each value name (`X`, `X'`, …: the
/// variable followed by the value's primes) mapped to a level by `values`.
///
/// ```
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
/// use why_data::ncm::{Fix, expand};
///
/// let mut g = Admg::new();
/// g.directed("X", "W");
/// g.directed("W", "Y");
/// g.directed("X", "Y");
/// // The first term of the natural direct effect.
/// let q = CtfQuery::parse(&g, "P(y_{x, W_{x'}})").unwrap();
/// let e = expand(&q, &[("X", "1"), ("X'", "0"), ("Y", "1")]).unwrap();
/// assert_eq!(e.worlds.len(), 2);
/// // World 0 is do(X = 0); world 1 sets X = 1 and W to its value in world 0.
/// assert_eq!(e.worlds[0].fixed["X"], Fix::Level("0".into()));
/// assert_eq!(e.worlds[1].fixed["W"], Fix::Copy(0));
/// assert_eq!((e.events[0].world, e.events[0].level.as_str()), (1, "1"));
/// ```
///
/// # Errors
///
/// [`NcmError::MissingValue`] for a value name `values` lacks, and
/// [`NcmError::Query`] for a nested subscript entry that nests a different
/// variable than the one it sets.
pub fn expand(query: &CtfQuery, values: &[(&str, &str)]) -> Result<Expanded, NcmError> {
    let values: BTreeMap<&str, &str> = values.iter().copied().collect();
    let mut worlds: Vec<World> = Vec::new();
    let mut convert = |e: &Event| -> Result<Assignment, NcmError> {
        let world = world_of(&e.ctf.subscript, &values, &mut worlds)?;
        Ok(Assignment {
            world,
            var: e.ctf.var.clone(),
            level: level(&values, &e.ctf.var, e.value)?,
        })
    };
    let events = query
        .events
        .iter()
        .map(&mut convert)
        .collect::<Result<Vec<_>, _>>()?;
    let given = query
        .given
        .iter()
        .map(&mut convert)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Expanded {
        worlds,
        events,
        given,
    })
}

fn level(values: &BTreeMap<&str, &str>, var: &str, primes: usize) -> Result<String, NcmError> {
    let name = format!("{var}{}", "'".repeat(primes));
    values
        .get(name.as_str())
        .map(|v| (*v).to_string())
        .ok_or(NcmError::MissingValue(name))
}

fn world_of(
    subscript: &BTreeMap<String, Setting>,
    values: &BTreeMap<&str, &str>,
    worlds: &mut Vec<World>,
) -> Result<usize, NcmError> {
    let mut w = World::default();
    for (var, s) in subscript {
        let fix = match s {
            Setting::Value(p) => Fix::Level(level(values, var, *p)?),
            Setting::Nested(inner) => {
                if inner.var != *var {
                    return Err(CtfError::MisplacedNesting(var.clone()).into());
                }
                Fix::Copy(world_of(&inner.subscript, values, worlds)?)
            }
        };
        w.fixed.insert(var.clone(), fix);
    }
    if let Some(i) = worlds.iter().position(|x| *x == w) {
        return Ok(i);
    }
    worlds.push(w);
    Ok(worlds.len() - 1)
}

impl Expanded {
    /// For each world, the variables whose values the query depends on, in
    /// the structure's order: the ancestors of the variables the events
    /// mention there, or that a later world copies from there, with the
    /// edges into the world's fixed variables cut. Everything else sums out.
    #[must_use]
    pub fn relevant(&self, s: &NcmStructure) -> Vec<Vec<String>> {
        let mut mentioned: Vec<BTreeSet<String>> = vec![BTreeSet::new(); self.worlds.len()];
        for a in self.events.iter().chain(&self.given) {
            mentioned[a.world].insert(a.var.clone());
        }
        let mut out = vec![Vec::new(); self.worlds.len()];
        for j in (0..self.worlds.len()).rev() {
            let cut: BTreeSet<String> = self.worlds[j].fixed.keys().cloned().collect();
            let rel = s.ancestors(&mentioned[j], &cut);
            for v in &rel {
                if let Some(Fix::Copy(k)) = self.worlds[j].fixed.get(v) {
                    mentioned[*k].insert(v.clone());
                }
            }
            out[j] = rel;
        }
        out
    }

    /// Every variable the query names, in any world or subscript.
    #[must_use]
    pub fn variables(&self) -> BTreeSet<String> {
        let mut out: BTreeSet<String> = self
            .events
            .iter()
            .chain(&self.given)
            .map(|a| a.var.clone())
            .collect();
        for w in &self.worlds {
            out.extend(w.fixed.keys().cloned());
        }
        out
    }

    /// The levels the query writes for each variable, fixed or observed.
    #[must_use]
    pub fn levels(&self) -> BTreeMap<String, BTreeSet<String>> {
        let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for a in self.events.iter().chain(&self.given) {
            out.entry(a.var.clone())
                .or_default()
                .insert(a.level.clone());
        }
        for w in &self.worlds {
            for (v, f) in &w.fixed {
                if let Fix::Level(l) = f {
                    out.entry(v.clone()).or_default().insert(l.clone());
                }
            }
        }
        out
    }
}
