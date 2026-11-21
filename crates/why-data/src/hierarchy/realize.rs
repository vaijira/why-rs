//! Counterfactual realizability (Sec. 20.2.2): Algorithms 68 (CTF-REALIZE)
//! and 69 (COMPATIBLE).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::layers::Dag;
use super::{HierarchyError, Term};
use crate::graph::dseparation::Admg;

/// A physical action an agent can apply to a unit (Defs. 20.1.8, 20.2.1).
/// Reading a variable is always available.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// `RAND(X)`: Fisherian randomization. It erases the unit's natural
    /// `X`, and every child receives the randomized value.
    Rand(String),
    /// `CTF-RAND(X → C)`: the children in `C` receive a randomized value of
    /// `X`, while `X` and its other children keep the natural one.
    CtfRand(String, BTreeSet<String>),
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rand(x) => write!(f, "RAND({x})"),
            Self::CtfRand(x, c) => {
                let c: Vec<&str> = c.iter().map(String::as_str).collect();
                write!(f, "CTF-RAND({x} → {{{}}})", c.join(", "))
            }
        }
    }
}

/// What an action must deliver in a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tag {
    /// The children must receive the natural value: the action is not
    /// performed.
    Natural,
    /// The children must receive this value.
    Value(u8),
}

/// The maximal feasible action set `A†(G)`: a separate counterfactual
/// randomization of every variable for each of its children (Cor. 20.2.6).
///
/// # Errors
///
/// [`HierarchyError::Graph`] for a cyclic graph.
pub fn maximal_actions(g: &Admg) -> Result<Vec<Action>, HierarchyError> {
    let d = Dag::new(g)?;
    let mut out = Vec::new();
    for v in 0..d.names.len() {
        for &c in &d.ch[v] {
            out.push(Action::CtfRand(
                d.names[v].clone(),
                BTreeSet::from([d.names[c].clone()]),
            ));
        }
    }
    Ok(out)
}

/// Counterfactual randomization that passes one value to all children of
/// a variable, `CTF-RAND(V → Ch(V))` for every `V`: the regime of L2.25.
///
/// # Errors
///
/// [`HierarchyError::Graph`] for a cyclic graph.
pub fn layer225_actions(g: &Admg) -> Result<Vec<Action>, HierarchyError> {
    let d = Dag::new(g)?;
    Ok((0..d.names.len())
        .filter(|&v| !d.ch[v].is_empty())
        .map(|v| {
            Action::CtfRand(
                d.names[v].clone(),
                d.ch[v].iter().map(|&c| d.names[c].clone()).collect(),
            )
        })
        .collect())
}

/// Fisherian randomization of every variable, `RAND(V)`: the regime of L2.
///
/// # Errors
///
/// [`HierarchyError::Graph`] for a cyclic graph.
pub fn fisher_actions(g: &Admg) -> Result<Vec<Action>, HierarchyError> {
    let d = Dag::new(g)?;
    Ok(d.names.iter().map(|v| Action::Rand(v.clone())).collect())
}

/// The output of Phase 1 of CTF-REALIZE: which actions to perform with
/// which tags, and which reads fill which terms.
#[derive(Clone, Debug)]
pub struct Plan {
    dag: Dag,
    terms: Vec<Term>,
    actions: Vec<Action>,
    /// Per variable: (index into `actions`, tag).
    int: BTreeMap<usize, BTreeMap<usize, Tag>>,
    /// Per variable: the query terms its read fills.
    outputs: BTreeMap<usize, Vec<usize>>,
}

impl Plan {
    /// The actions with a value tag, in topological order.
    #[must_use]
    pub fn interventions(&self) -> Vec<(Action, u8)> {
        let mut out = Vec::new();
        for tags in self.int.values() {
            for (&a, tag) in tags {
                if let Tag::Value(v) = tag {
                    out.push((self.actions[a].clone(), *v));
                }
            }
        }
        out
    }

    /// The value of `p` that its child `c` receives when it differs from
    /// the natural one.
    fn perceived(&self, p: usize, c: usize) -> Option<u8> {
        let tags = self.int.get(&p)?;
        let c = &self.dag.names[c];
        let mut best: Option<(usize, u8)> = None;
        let mut rand = None;
        for (&a, tag) in tags {
            let Tag::Value(v) = *tag else { continue };
            match &self.actions[a] {
                Action::CtfRand(_, group) if group.contains(c) => {
                    if best.is_none_or(|(size, _)| group.len() < size) {
                        best = Some((group.len(), v));
                    }
                }
                Action::Rand(_) => rand = Some(v),
                Action::CtfRand(..) => {}
            }
        }
        best.map(|(_, v)| v).or(rand)
    }

    /// Phase 2 of CTF-REALIZE on one unit, whose mechanisms are
    /// `mechanism(v, parents)`: performs the tagged actions, evaluates
    /// every mechanism once in topological order, and reads the outputs.
    /// The result lists the value of each query term.
    pub fn sample(&self, mechanism: impl Fn(&str, &BTreeMap<String, u8>) -> u8) -> Vec<u8> {
        let d = &self.dag;
        let mut actual = vec![0u8; d.names.len()];
        for &v in &d.order {
            let randomized = self.int.get(&v).and_then(|tags| {
                tags.iter()
                    .find_map(|(&a, tag)| match (&self.actions[a], tag) {
                        (Action::Rand(_), Tag::Value(x)) => Some(*x),
                        _ => None,
                    })
            });
            actual[v] = match randomized {
                Some(x) => x,
                None => {
                    let parents: BTreeMap<String, u8> = d.pa[v]
                        .iter()
                        .map(|&p| {
                            (
                                d.names[p].clone(),
                                self.perceived(p, v).unwrap_or(actual[p]),
                            )
                        })
                        .collect();
                    mechanism(&d.names[v], &parents)
                }
            };
        }
        let mut out = vec![0u8; self.terms.len()];
        for (&v, idx) in &self.outputs {
            for &i in idx {
                out[i] = actual[v];
            }
        }
        out
    }
}

/// Algorithm 68 (CTF-REALIZE), Phase 1, with Algorithm 69 (COMPATIBLE):
/// a plan that draws i.i.d. samples of `P(W*)` from the actions, or
/// [`HierarchyError::NotRealizable`].
///
/// Each term `W_t` needs three things (Thm 20.2.5):
/// 1. every child of `T` on the way to `W` receives the value in `t`;
/// 2. every other ancestor of `W` in `G_T̄` reaches its children on the
///    way naturally;
/// 3. `f_W` is not overwritten by a `RAND(W)`.
///
/// Two terms whose requirements clash on one action make the query not
/// realizable.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::hierarchy::{Term, ctf_realize, fisher_actions, maximal_actions};
///
/// // Ex. 20.19: the ETT's P(Y_x, X) in X -> Y.
/// let mut g = Admg::new();
/// g.directed("X", "Y");
/// let q = [Term::new("Y", &[("X", 1)]), Term::natural("X")];
/// assert!(ctf_realize(&g, &q, &maximal_actions(&g).unwrap()).is_ok());
/// assert!(ctf_realize(&g, &q, &fisher_actions(&g).unwrap()).is_err());
/// // Ex. 20.20: the probability of sufficiency's P(Y_x, X, Y) is not
/// // realizable even with every action.
/// let ps = [Term::new("Y", &[("X", 1)]), Term::natural("X"), Term::natural("Y")];
/// assert!(ctf_realize(&g, &ps, &maximal_actions(&g).unwrap()).is_err());
/// ```
///
/// # Errors
///
/// [`HierarchyError::NotRealizable`] when two requirements clash or an
/// action is missing; [`HierarchyError::UnknownVariable`] and
/// [`HierarchyError::InvalidInput`] for malformed input.
pub fn ctf_realize(g: &Admg, query: &[Term], actions: &[Action]) -> Result<Plan, HierarchyError> {
    let d = Dag::new(g)?;
    for a in actions {
        match a {
            Action::Rand(x) => {
                d.idx(x)?;
            }
            Action::CtfRand(x, group) => {
                let v = d.idx(x)?;
                for c in group {
                    let c = d.idx(c)?;
                    if !d.ch[v].contains(&c) {
                        return Err(HierarchyError::InvalidInput(format!(
                            "{a}: {} is not a child of {x}",
                            d.names[c]
                        )));
                    }
                }
                if group.is_empty() {
                    return Err(HierarchyError::InvalidInput(format!("{a} has no target")));
                }
            }
        }
    }
    let terms: Vec<Term> = query
        .iter()
        .map(|t| {
            d.check(t)?;
            Ok(d.restrict(&t.var, &t.sub))
        })
        .collect::<Result<_, HierarchyError>>()?;
    let mut int: BTreeMap<usize, BTreeMap<usize, Tag>> = BTreeMap::new();
    let mut outputs: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut tag = |v: usize, a: usize, t: Tag| -> Result<(), HierarchyError> {
        let slot = int.entry(v).or_default();
        match slot.get(&a) {
            Some(old) if *old != t => Err(HierarchyError::NotRealizable(format!(
                "{} must deliver both {old:?} and {t:?}",
                actions[a]
            ))),
            _ => {
                slot.insert(a, t);
                Ok(())
            }
        }
    };
    let rand_of = |v: &str| {
        actions
            .iter()
            .position(|a| matches!(a, Action::Rand(x) if x == v))
    };
    for &v in &d.order {
        let name = &d.names[v];
        for (i, t) in terms.iter().enumerate() {
            let w = d.idx(&t.var)?;
            if v == w {
                outputs.entry(v).or_default().push(i);
                continue;
            }
            let cut: BTreeSet<usize> = t.sub.keys().map(|k| d.idx(k)).collect::<Result<_, _>>()?;
            let an = d.an(w, &cut);
            if !an.contains(&v) {
                continue;
            }
            // COMPATIBLE(V, W_t).
            for &c in &d.ch[v] {
                if cut.contains(&c) || !an.contains(&c) {
                    continue;
                }
                let child = &d.names[c];
                let groups: Vec<usize> = actions
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| matches!(a, Action::CtfRand(x, gr) if x == name && gr.contains(child)))
                    .map(|(j, _)| j)
                    .collect();
                if let Some(&value) = t.sub.get(name) {
                    let smallest = groups.iter().copied().min_by_key(|&j| match &actions[j] {
                        Action::CtfRand(_, gr) => gr.len(),
                        Action::Rand(_) => usize::MAX,
                    });
                    match (smallest, rand_of(name)) {
                        (Some(j), _) => tag(v, j, Tag::Value(value))?,
                        (None, Some(j)) => tag(v, j, Tag::Value(value))?,
                        (None, None) => {
                            return Err(HierarchyError::NotRealizable(format!(
                                "no action sets {name} for {child} (needed by {t})"
                            )));
                        }
                    }
                } else {
                    for &j in &groups {
                        tag(v, j, Tag::Natural)?;
                    }
                    // The natural value cannot survive a Fisherian
                    // randomization (missing from Algorithm 69).
                    if let Some(j) = rand_of(name) {
                        tag(v, j, Tag::Natural)?;
                    }
                }
            }
        }
    }
    for &v in outputs.keys() {
        if let Some(j) = rand_of(&d.names[v])
            && matches!(int.get(&v).and_then(|s| s.get(&j)), Some(Tag::Value(_)))
        {
            return Err(HierarchyError::NotRealizable(format!(
                "{} must be read but RAND({}) overwrites it",
                d.names[v], d.names[v]
            )));
        }
    }
    Ok(Plan {
        dag: d,
        terms,
        actions: actions.to_vec(),
        int,
        outputs,
    })
}
