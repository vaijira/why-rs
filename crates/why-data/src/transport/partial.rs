//! Statistical transportability under partial observability (Sec. 11.3):
//! source data `P^k(V)` plus target measurements `P*(W)`.

use std::collections::{BTreeSet, HashMap, VecDeque};

use super::{Domain, Estimand, SelectionDiagram, TransportError, as_refs, subsets};
use crate::graph::dseparation::Admg;

/// A transport formula from one of the graphical scenarios of Sec. 11.3.1.
#[derive(Clone, Debug)]
pub struct ScenarioFormula {
    /// The separating set `Z ⊆ W` the theorem needs.
    pub z: Vec<String>,
    /// `P*(y | x)` in terms of `P^source` and `P*(W)`.
    pub estimand: Estimand,
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

/// Searches subsets of `pool`, smallest first, for one passing `test`.
fn first_subset(
    pool: Vec<String>,
    mut test: impl FnMut(&[&str]) -> Result<bool, TransportError>,
) -> Result<Option<Vec<String>>, TransportError> {
    for z in subsets(&pool)? {
        if test(&as_refs(&z))? {
            return Ok(Some(z));
        }
    }
    Ok(None)
}

fn validate(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &str,
    w: &[&str],
) -> Result<(), TransportError> {
    d.delta(source)?;
    d.check(x)?;
    d.check(&[y])?;
    d.check(w)?;
    if x.contains(&y) {
        return Err(TransportError::InvalidInput(format!(
            "`{y}` is both a feature and the outcome"
        )));
    }
    Ok(())
}

/// Thm 11.3.1, the features are measured in the target (`X ⊆ W`): if some
/// `Z ⊆ W \ X` has `S ⊥ Y | X, Z`, then
/// `P*(y | x) = sum_z P(y | x, z) P*(z | x)`.
///
/// Returns the smallest such `Z`, or `None` when there is none or `X ⊄ W`.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source or variable.
pub fn features_measured(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &str,
    w: &[&str],
) -> Result<Option<ScenarioFormula>, TransportError> {
    validate(d, source, x, y, w)?;
    if !x.iter().all(|v| w.contains(v)) {
        return Ok(None);
    }
    let pool: Vec<String> = w
        .iter()
        .filter(|v| !x.contains(v) && **v != y)
        .map(|s| (*s).to_string())
        .collect();
    let found = first_subset(pool, |z| {
        let mut xz = x.to_vec();
        xz.extend_from_slice(z);
        d.is_s_admissible(source, &[y], &xz)
    })?;
    Ok(found.map(|z| {
        let src = Domain::source(source);
        let mut xz = strings(x);
        xz.extend(z.iter().cloned());
        let body = Estimand::product(vec![
            Estimand::P {
                domain: src,
                vars: vec![y.to_string()],
                given: xz,
            },
            Estimand::P {
                domain: Domain::Target,
                vars: z.clone(),
                given: strings(x),
            },
        ]);
        ScenarioFormula {
            estimand: Estimand::sum(z.clone(), body),
            z,
        }
    }))
}

/// Thm 11.3.2, the outcome is measured in the target (`Y ∈ W`): if some
/// `Z ⊆ W \ {Y}` has `(S, Y ⊥ X | Z)`, then
/// `P*(y | x) = sum_z P*(y | z) P(x | z) P*(z) / sum_z P(x | z) P*(z)`.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source or variable.
pub fn outcome_measured(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &str,
    w: &[&str],
) -> Result<Option<ScenarioFormula>, TransportError> {
    validate(d, source, x, y, w)?;
    if !w.contains(&y) {
        return Ok(None);
    }
    let pool: Vec<String> = w
        .iter()
        .filter(|v| !x.contains(v) && **v != y)
        .map(|s| (*s).to_string())
        .collect();
    let s = d.s_node(source);
    let g = d.augmented();
    let found = first_subset(pool, |z| Ok(g.is_d_separated(&[&s, y], x, z)?))?;
    Ok(found.map(|z| {
        let src = Domain::source(source);
        let px = Estimand::P {
            domain: src,
            vars: strings(x),
            given: z.clone(),
        };
        let pz = Estimand::P {
            domain: Domain::Target,
            vars: z.clone(),
            given: vec![],
        };
        let py = Estimand::P {
            domain: Domain::Target,
            vars: vec![y.to_string()],
            given: z.clone(),
        };
        let num = Estimand::sum(z.clone(), Estimand::product(vec![py, px.clone(), pz.clone()]));
        let den = Estimand::sum(z.clone(), Estimand::product(vec![px, pz]));
        ScenarioFormula {
            estimand: Estimand::ratio(num, den),
            z,
        }
    }))
}

/// Thm 11.3.3, neither the features nor the outcome need be measured: if
/// some `Z ⊆ W` has `(Y, X ⊥ S | Z)`, then
/// `P*(y | x) = sum_z P(y | x, z) P(x | z) P*(z) / sum_z P(x | z) P*(z)`.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an unknown source or variable.
pub fn auxiliary_measured(
    d: &SelectionDiagram,
    source: &str,
    x: &[&str],
    y: &str,
    w: &[&str],
) -> Result<Option<ScenarioFormula>, TransportError> {
    validate(d, source, x, y, w)?;
    let pool: Vec<String> = w
        .iter()
        .filter(|v| !x.contains(v) && **v != y)
        .map(|s| (*s).to_string())
        .collect();
    let found = first_subset(pool, |z| {
        let mut xy = x.to_vec();
        xy.push(y);
        d.is_s_admissible(source, &xy, z)
    })?;
    Ok(found.map(|z| {
        let src = Domain::source(source);
        let mut xz = strings(x);
        xz.extend(z.iter().cloned());
        let py = Estimand::P {
            domain: src.clone(),
            vars: vec![y.to_string()],
            given: xz,
        };
        let px = Estimand::P {
            domain: src,
            vars: strings(x),
            given: z.clone(),
        };
        let pz = Estimand::P {
            domain: Domain::Target,
            vars: z.clone(),
            given: vec![],
        };
        let num = Estimand::sum(z.clone(), Estimand::product(vec![py, px.clone(), pz.clone()]));
        let den = Estimand::sum(z.clone(), Estimand::product(vec![px, pz]));
        ScenarioFormula {
            estimand: Estimand::ratio(num, den),
            z,
        }
    }))
}

/// The variables connected to `from` by any path, ignoring edge marks.
fn connected(g: &Admg, from: &str) -> BTreeSet<String> {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let gr = g.graph();
    for e in gr.edge_indices() {
        let (a, b) = gr.edge_endpoints(e).expect("valid edge");
        let (a, b) = (gr[a].as_str(), gr[b].as_str());
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let mut seen: BTreeSet<String> = BTreeSet::from([from.to_string()]);
    let mut queue = VecDeque::from([from]);
    while let Some(n) = queue.pop_front() {
        for m in adj.get(n).into_iter().flatten() {
            if seen.insert((*m).to_string()) {
                queue.push_back(m);
            }
        }
    }
    seen
}

/// Thm 11.3.4, the statistical transportability product: transports
/// `P*(y | x)` from every source's `P^k(V)` and the target measurements
/// `P*(W)`, or reports that it is not transportable. The theorem states the
/// procedure as necessary and sufficient.
///
/// 1. `D` is the set of variables connected to `Y` in `D^Δ[An(X, Y)]` with the
///    edges out of `X` removed: only their distribution matters.
/// 2. Split `D` into c-components `D_j`.
/// 3. `D*` is the union of the components every source differs on.
/// 4. If `D* ∪ Pa(D*) ⊆ W`, every component is read from a source it does not
///    differ on, or else from the target, and
///    `P*(y | x) = sum_{D \ (X ∪ Y)} ∏ / sum_{D \ X} ∏`.
/// 5. Otherwise project the diagram onto
///    `(An(X, Y) \ (D_w \ W)) ∪ X ∪ Y` and start again, until the projection
///    no longer shrinks it.
///
/// Each component's factor is Tian's `∏ P(v_i | Pa(T_i) \ {v_i})`, where
/// `T_i` is the c-component of `V_i` among the variables up to it in
/// topological order (see the module docs for why this departs from the
/// printed `pa⁺`).
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::{SelectionDiagram, statistical_transport};
///
/// // Ex. 11.31: X → Z → R → Y, with R differing; P*(Z, R) is measured.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "R");
/// g.directed("R", "Y");
/// let d = SelectionDiagram::new(g).with_source("1", &["R"]).unwrap();
/// let e = statistical_transport(&d, &["X"], "Y", &["Z", "R"]).unwrap();
/// assert_eq!(e.to_string(), "sum_{r, z} P^1(z | x) P^*(r | z) P^1(y | r)");
/// ```
///
/// # Errors
///
/// [`TransportError::NotTransportable`] when the procedure stops without a
/// formula, [`TransportError::Unknown`] for an unknown variable, and
/// [`TransportError::InvalidInput`] if `y` is in `x`.
pub fn statistical_transport(
    d: &SelectionDiagram,
    x: &[&str],
    y: &str,
    w: &[&str],
) -> Result<Estimand, TransportError> {
    d.check(x)?;
    d.check(&[y])?;
    d.check(w)?;
    if x.contains(&y) {
        return Err(TransportError::InvalidInput(format!(
            "`{y}` is both a feature and the outcome"
        )));
    }
    let w: BTreeSet<String> = w.iter().map(|s| (*s).to_string()).collect();
    let xs: BTreeSet<String> = x.iter().map(|s| (*s).to_string()).collect();
    let mut cur = d.clone();
    loop {
        let g = cur.graph();
        let mut seeds = x.to_vec();
        seeds.push(y);
        let a: Vec<String> = g
            .ancestors_of(&seeds)?
            .into_iter()
            .map(str::to_string)
            .collect();
        let ga = g.induced_subgraph(&as_refs(&a))?;
        let dset = connected(&ga.cut_edges_out_of(x)?, y);
        let gd = ga.induced_subgraph(&as_refs(&dset.iter().cloned().collect::<Vec<_>>()))?;
        let comps = gd.c_components();

        let differs_everywhere =
            |c: &[String]| cur.sources.iter().all(|(_, delta)| c.iter().any(|v| delta.contains(v)));
        let mut dstar: BTreeSet<String> = BTreeSet::new();
        for c in &comps {
            if differs_everywhere(c) {
                dstar.extend(c.iter().cloned());
            }
        }
        let mut dw = dstar.clone();
        for v in &dstar {
            dw.extend(cur.parents(v));
        }
        let missing: BTreeSet<String> = dw.difference(&w).cloned().collect();
        if missing.is_empty() {
            return product_formula(&cur, &ga, &dset, &comps, &xs, y);
        }
        let mut next: BTreeSet<String> = a.iter().filter(|v| !missing.contains(*v)).cloned().collect();
        next.extend(xs.iter().cloned());
        next.insert(y.to_string());
        let current: BTreeSet<String> = g.variables().into_iter().map(str::to_string).collect();
        if next == current {
            return Err(TransportError::NotTransportable(format!(
                "the target must measure {{{}}}",
                missing.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }
        cur = cur.project(&as_refs(&next.into_iter().collect::<Vec<_>>()))?;
    }
}

/// A topological order that breaks ties by insertion order (Kahn's
/// algorithm), so the order variables were added to the graph decides how a
/// formula prints.
fn stable_order(g: &Admg) -> Result<Vec<String>, TransportError> {
    let vars: Vec<&str> = g.variables();
    let gr = g.graph();
    let mut parents: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in gr.edge_indices() {
        if gr[e] == crate::graph::dseparation::EdgeKind::Directed {
            let (a, b) = gr.edge_endpoints(e).expect("valid edge");
            parents.entry(gr[b].as_str()).or_default().push(gr[a].as_str());
        }
    }
    let mut placed: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::with_capacity(vars.len());
    while out.len() < vars.len() {
        let next = vars.iter().find(|v| {
            !placed.contains(*v)
                && parents
                    .get(*v)
                    .is_none_or(|ps| ps.iter().all(|p| placed.contains(p)))
        });
        let Some(v) = next else {
            let stuck = vars.iter().find(|v| !placed.contains(*v)).expect("unplaced");
            return Err(crate::graph::dseparation::DSepError::Cyclic((*stuck).to_string()).into());
        };
        placed.insert(v);
        out.push((*v).to_string());
    }
    Ok(out)
}

/// Eq. 11.234 on the current (possibly projected) diagram.
fn product_formula(
    cur: &SelectionDiagram,
    ga: &Admg,
    dset: &BTreeSet<String>,
    comps: &[Vec<String>],
    xs: &BTreeSet<String>,
    y: &str,
) -> Result<Estimand, TransportError> {
    let full = stable_order(ga)?;
    let rank: HashMap<String, usize> = full
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();
    let order: Vec<String> = full.into_iter().filter(|v| dset.contains(v)).collect();
    let tau = |c: &[String]| -> Domain {
        cur.sources
            .iter()
            .find(|(_, delta)| c.iter().all(|v| !delta.contains(v)))
            .map_or(Domain::Target, |(s, _)| Domain::Source(s.clone()))
    };
    let mut factors = Vec::new();
    for (i, v) in order.iter().enumerate() {
        let comp = comps
            .iter()
            .find(|c| c.contains(v))
            .expect("every variable of D is in a component");
        let prefix = ga.induced_subgraph(&as_refs(&order[..=i]))?;
        let t = prefix
            .c_components()
            .into_iter()
            .find(|c| c.contains(v))
            .expect("v is in the prefix");
        let mut given: BTreeSet<String> = t.iter().cloned().collect();
        for m in &t {
            given.extend(cur.parents(m));
        }
        given.remove(v);
        let given: Vec<String> = given.into_iter().collect();
        factors.push(Estimand::P {
            domain: tau(comp),
            vars: vec![v.clone()],
            given,
        });
    }
    let prod = Estimand::product(factors);
    let num_over: Vec<String> = order
        .iter()
        .filter(|v| !xs.contains(*v) && *v != y)
        .cloned()
        .collect();
    let den_over: Vec<String> = order.iter().filter(|v| !xs.contains(*v)).cloned().collect();
    let mut e = Estimand::ratio(
        Estimand::sum(num_over, prod.clone()),
        Estimand::sum(den_over, prod),
    );
    e.sort_given(&rank);
    Ok(e)
}
