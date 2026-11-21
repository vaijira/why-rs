//! Optimal bounds of a counterfactual query from `P(V)` and `G`, by the
//! canonical model (see the module docs).

use std::collections::{BTreeMap, HashMap};

use super::{AscmError, Interval};
use crate::crl::lp::{Cmp, Lp, LpOutcome};
use crate::fairness::Dataset;
use crate::graph::cinfer::counterfactual::CtfQuery;
use crate::graph::dseparation::Admg;
use crate::ncm::{Fix, NcmStructure, expand};

/// The most response types one c-component may have.
const MAX_TYPES: usize = 50_000;
/// The most joint assignments of the query's worlds.
const MAX_ASSIGNMENTS: usize = 1_000_000;
/// Cell probabilities closer than this count as equal.
const TIE: f64 = 1e-9;

/// The optimal bounds on `P(y_* | x_*)` over every SCM compatible with `g`
/// that induces the distribution `p` over `V`.
///
/// `query` is in textbook notation (see
/// [`CtfQuery::parse`](crate::graph::cinfer::counterfactual::CtfQuery::parse)),
/// with each value name (`X`, `X'`, …) mapped to a level by `values` as in
/// [`expand`]. Events may live in any world, nested subscripts included. The
/// conditioning events must be factual, so the query has the shape of a
/// conditional F-ctf query `P(w′_{x′} | w)` (Eq. 14.18).
///
/// `p` is a data set whose row weights are the probabilities of an exact
/// distribution over the variables of `g`. Each variable's levels are those
/// the data show.
///
/// ```
/// use why_data::ascm::optimal_bounds;
/// use why_data::fairness::Dataset;
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
///
/// // X -> Y with P(x) = 1/2, P(y | x) = 0.7 and P(y | x') = 0.2.
/// let mut g = Admg::new();
/// g.directed("X", "Y");
/// let rows = [["1", "1"], ["1", "0"], ["0", "1"], ["0", "0"]];
/// let p = Dataset::from_records(["X", "Y"], rows)
///     .unwrap()
///     .with_weights(vec![0.35, 0.15, 0.1, 0.4])
///     .unwrap();
/// // The probability of necessity, P(y'_{x'} | x, y): Fréchet bounds.
/// let q = CtfQuery::parse(&g, "P(y'_{x'} | x, y)").unwrap();
/// let b = optimal_bounds(&g, &p, &q, &[("X", "1"), ("X'", "0"), ("Y", "1"), ("Y'", "0")])
///     .unwrap();
/// assert!((b.lower - 0.5 / 0.7).abs() < 1e-9);
/// assert!((b.upper - 1.0).abs() < 1e-9);
/// ```
///
/// # Errors
///
/// [`AscmError::Nonlinear`] when the free response types of more than one
/// c-component meet in the query. [`AscmError::Incompatible`] when `p`
/// breaks a constraint of `g`, and [`AscmError::Positivity`] when Tian's
/// factorization divides by zero. [`AscmError::CounterfactualEvidence`] for
/// a counterfactual conditioning event, [`AscmError::ImpossibleCondition`]
/// when the conditioning event has probability zero, and
/// [`AscmError::TooLarge`] past the size limits. Also errors for unknown
/// variables, levels or value names.
pub fn optimal_bounds(
    g: &Admg,
    p: &Dataset,
    query: &CtfQuery,
    values: &[(&str, &str)],
) -> Result<Interval, AscmError> {
    let model = Canonical::new(g, p)?;
    model.bounds(query, values)
}

/// The variables, their levels and the joint distribution, with the
/// c-components of the graph.
struct Canonical {
    structure: NcmStructure,
    names: Vec<String>,
    index: HashMap<String, usize>,
    levels: Vec<Vec<String>>,
    parents: Vec<Vec<usize>>,
    /// Joint probabilities of full assignments (level indices), summing to 1.
    joint: Vec<(Vec<usize>, f64)>,
    components: Vec<Vec<usize>>,
    component_of: Vec<usize>,
}

/// One coordinate of a response type: variable `V` at parent configuration
/// `cfg`, and the level it responds with.
type Entry = (usize, usize, usize);

impl Canonical {
    fn new(g: &Admg, p: &Dataset) -> Result<Self, AscmError> {
        let structure = NcmStructure::from_graph(g)?;
        let names = structure.order.clone();
        let index: HashMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let levels: Vec<Vec<String>> = names
            .iter()
            .map(|v| {
                p.levels(v)
                    .map(<[String]>::to_vec)
                    .ok_or_else(|| AscmError::MissingColumn(v.clone()))
            })
            .collect::<Result<_, _>>()?;
        let parents = names
            .iter()
            .map(|v| structure.parents[v].iter().map(|p| index[p]).collect())
            .collect();
        let mut table: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for r in 0..p.len() {
            let row = names
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let x = p.value(v, r).expect("column checked above");
                    levels[i].iter().position(|l| l == x).expect("a level")
                })
                .collect();
            *table.entry(row).or_default() += p.weights().map_or(1.0, |w| w[r]);
        }
        let total: f64 = table.values().sum();
        if total <= 0.0 {
            return Err(AscmError::Data("the distribution has no mass".into()));
        }
        let joint = table
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .map(|(r, w)| (r, w / total))
            .collect();
        let components: Vec<Vec<usize>> = g
            .c_components()
            .iter()
            .map(|c| {
                let mut c: Vec<usize> = c.iter().map(|v| index[v]).collect();
                c.sort_unstable();
                c
            })
            .collect();
        let mut component_of = vec![0; names.len()];
        for (k, c) in components.iter().enumerate() {
            for &v in c {
                component_of[v] = k;
            }
        }
        Ok(Self {
            structure,
            names,
            index,
            levels,
            parents,
            joint,
            components,
            component_of,
        })
    }

    fn level(&self, var: usize, level: &str) -> Result<usize, AscmError> {
        self.levels[var]
            .iter()
            .position(|l| l == level)
            .ok_or_else(|| AscmError::UnknownLevel {
                var: self.names[var].clone(),
                level: level.to_string(),
            })
    }

    fn var(&self, name: &str) -> Result<usize, AscmError> {
        self.index
            .get(name)
            .copied()
            .ok_or_else(|| AscmError::UnknownVariable(name.to_string()))
    }

    /// `P(V_i = l_i for every (i, l_i) in fixed)`.
    fn prob(&self, fixed: &[(usize, usize)]) -> f64 {
        self.joint
            .iter()
            .filter(|(row, _)| fixed.iter().all(|(v, l)| row[*v] == *l))
            .map(|(_, w)| w)
            .sum()
    }

    fn n_configs(&self, var: usize) -> usize {
        self.parents[var]
            .iter()
            .map(|p| self.levels[*p].len())
            .product()
    }

    /// The configuration index of `var`'s parents, read by `value`.
    fn config(&self, var: usize, value: impl Fn(usize) -> usize) -> usize {
        self.parents[var]
            .iter()
            .fold(0, |at, &p| at * self.levels[p].len() + value(p))
    }

    /// The parents' levels of configuration `cfg` of `var`.
    fn decode(&self, var: usize, mut cfg: usize) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self.parents[var]
            .iter()
            .rev()
            .map(|&p| {
                let n = self.levels[p].len();
                let l = cfg % n;
                cfg /= n;
                (p, l)
            })
            .collect();
        out.reverse();
        out
    }

    fn describe(&self, fixed: &[(usize, usize)]) -> String {
        fixed
            .iter()
            .map(|(v, l)| format!("{} = {}", self.names[*v], self.levels[*v][*l]))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Tian's c-factor `Q[C](v) = Π_{V_i ∈ C} P(v_i | v^{(i−1)})`, for a full
    /// assignment `v` in topological order.
    fn c_factor(&self, c: &[usize], v: &[usize]) -> Result<f64, AscmError> {
        let mut q = 1.0;
        for &i in c {
            let prefix: Vec<(usize, usize)> = (0..i).map(|j| (j, v[j])).collect();
            let den = self.prob(&prefix);
            if den <= 0.0 {
                return Err(AscmError::Positivity(self.describe(&prefix)));
            }
            let mut with = prefix;
            with.push((i, v[i]));
            q *= self.prob(&with) / den;
        }
        Ok(q)
    }

    fn bounds(&self, query: &CtfQuery, values: &[(&str, &str)]) -> Result<Interval, AscmError> {
        let e = expand(query, values)?;
        let n_worlds = e.worlds.len();
        // What each world fixes, by variable index.
        let mut fixed: Vec<HashMap<usize, Fixed>> = vec![HashMap::new(); n_worlds];
        for (j, w) in e.worlds.iter().enumerate() {
            for (name, f) in &w.fixed {
                let v = self.var(name)?;
                let f = match f {
                    Fix::Level(l) => Fixed::Level(self.level(v, l)?),
                    Fix::Copy(k) => Fixed::Copy(*k),
                };
                fixed[j].insert(v, f);
            }
        }
        let natural = |j: usize| e.worlds[j].fixed.is_empty();
        // The events and the evidence, as required levels per (world, var).
        let mut required: HashMap<(usize, usize), usize> = HashMap::new();
        let mut contradictory = false;
        let mut given = Vec::new();
        for (a, is_given) in e
            .events
            .iter()
            .map(|a| (a, false))
            .chain(e.given.iter().map(|a| (a, true)))
        {
            let v = self.var(&a.var)?;
            let l = self.level(v, &a.level)?;
            if is_given {
                if !natural(a.world) {
                    return Err(AscmError::CounterfactualEvidence(a.var.clone()));
                }
                given.push((v, l));
            }
            if let Some(&old) = required.get(&(a.world, v)) {
                contradictory |= old != l;
            }
            required.insert((a.world, v), l);
        }
        let den = if given.is_empty() {
            1.0
        } else {
            self.prob(&given)
        };
        if den <= 0.0 {
            return Err(AscmError::ImpossibleCondition);
        }
        if contradictory {
            return Ok(Interval {
                lower: 0.0,
                upper: 0.0,
            });
        }

        let relevant: Vec<Vec<usize>> = e
            .relevant(&self.structure)
            .iter()
            .map(|r| r.iter().map(|v| self.index[v]).collect())
            .collect();
        let slots: Vec<(usize, usize)> = relevant
            .iter()
            .enumerate()
            .flat_map(|(j, r)| r.iter().map(move |&v| (j, v)))
            .collect();

        // Every joint assignment of the worlds, as one cell per component.
        let mut cells: Vec<HashMap<Vec<Entry>, usize>> =
            vec![HashMap::new(); self.components.len()];
        let mut assignments: Vec<Vec<Option<usize>>> = Vec::new();
        let mut vals = vec![vec![None; self.names.len()]; n_worlds];
        let mut walk = Walk {
            model: self,
            slots: &slots,
            fixed: &fixed,
            required: &required,
            cells: &mut cells,
            out: &mut assignments,
        };
        walk.go(0, &mut vals)?;

        // Each involved component's polytope, and the range of each cell.
        let mut lps: Vec<Option<Component>> = Vec::with_capacity(self.components.len());
        let mut ranges: Vec<Vec<(f64, f64)>> = Vec::with_capacity(self.components.len());
        for (k, table) in cells.iter().enumerate() {
            if table.is_empty() {
                lps.push(None);
                ranges.push(Vec::new());
                continue;
            }
            let empty = Vec::new();
            let mut by_id: Vec<&Vec<Entry>> = vec![&empty; table.len()];
            for (cell, &id) in table {
                by_id[id] = cell;
            }
            let comp = Component::new(self, k, &by_id)?;
            let r = by_id
                .iter()
                .map(|cell| comp.range(&comp.indicator(cell)))
                .collect::<Result<Vec<_>, _>>()?;
            ranges.push(r);
            lps.push(Some(comp));
        }
        let free: Vec<usize> = (0..self.components.len())
            .filter(|&k| ranges[k].iter().any(|(lo, hi)| hi - lo > TIE))
            .collect();
        if free.len() > 1 {
            return Err(AscmError::Nonlinear(
                free.iter()
                    .map(|&k| {
                        self.components[k]
                            .iter()
                            .map(|&v| self.names[v].clone())
                            .collect()
                    })
                    .collect(),
            ));
        }
        let point = |k: usize, id: usize| {
            let (lo, hi) = ranges[k][id];
            (lo + hi) / 2.0
        };
        let (lo, hi) = match free.first() {
            None => {
                let total: f64 = assignments
                    .iter()
                    .map(|a| {
                        a.iter()
                            .enumerate()
                            .filter_map(|(k, c)| c.map(|id| point(k, id)))
                            .product::<f64>()
                    })
                    .sum();
                (total, total)
            }
            Some(&u) => {
                let comp = lps[u].as_ref().expect("involved");
                let mut by_id: Vec<Vec<Entry>> = vec![Vec::new(); cells[u].len()];
                for (cell, &id) in &cells[u] {
                    by_id[id].clone_from(cell);
                }
                let mut objective = vec![0.0; comp.n_types];
                for a in &assignments {
                    let rest: f64 = a
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != u)
                        .filter_map(|(k, c)| c.map(|id| point(k, id)))
                        .product();
                    if rest == 0.0 {
                        continue;
                    }
                    let id = a[u].expect("the free component is involved");
                    for (o, x) in objective.iter_mut().zip(comp.indicator(&by_id[id])) {
                        *o += rest * x;
                    }
                }
                comp.range(&objective)?
            }
        };
        // `+ 0.0` turns a `-0.0` (an empty float sum) into `0.0`.
        Ok(Interval {
            lower: (lo / den).clamp(0.0, 1.0) + 0.0,
            upper: (hi / den).clamp(0.0, 1.0) + 0.0,
        })
    }
}

/// What a world sets a variable to, by level index.
#[derive(Clone, Copy, Debug)]
enum Fixed {
    Level(usize),
    Copy(usize),
}

/// The depth-first enumeration of the worlds' joint assignments.
struct Walk<'a> {
    model: &'a Canonical,
    slots: &'a [(usize, usize)],
    fixed: &'a [HashMap<usize, Fixed>],
    required: &'a HashMap<(usize, usize), usize>,
    cells: &'a mut Vec<HashMap<Vec<Entry>, usize>>,
    out: &'a mut Vec<Vec<Option<usize>>>,
}

impl Walk<'_> {
    fn go(&mut self, at: usize, vals: &mut Vec<Vec<Option<usize>>>) -> Result<(), AscmError> {
        if at == self.slots.len() {
            return self.leaf(vals);
        }
        let (j, v) = self.slots[at];
        let choices: Vec<usize> = match self.fixed[j].get(&v) {
            Some(Fixed::Level(l)) => vec![*l],
            Some(Fixed::Copy(k)) => vec![vals[*k][v].expect("copied from an earlier world")],
            None => (0..self.model.levels[v].len()).collect(),
        };
        for l in choices {
            if self.required.get(&(j, v)).is_some_and(|r| *r != l) {
                continue;
            }
            vals[j][v] = Some(l);
            self.go(at + 1, vals)?;
        }
        vals[j][v] = None;
        Ok(())
    }

    fn leaf(&mut self, vals: &[Vec<Option<usize>>]) -> Result<(), AscmError> {
        let m = self.model;
        let mut coords: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for &(j, v) in self.slots {
            if self.fixed[j].contains_key(&v) {
                continue;
            }
            let cfg = m.config(v, |p| vals[j][p].expect("parents are relevant"));
            let l = vals[j][v].expect("assigned");
            if *coords.entry((v, cfg)).or_insert(l) != l {
                // The same unit responds differently to the same inputs.
                return Ok(());
            }
        }
        let mut split: Vec<Vec<Entry>> = vec![Vec::new(); m.components.len()];
        for ((v, cfg), l) in coords {
            split[m.component_of[v]].push((v, cfg, l));
        }
        let assignment = split
            .into_iter()
            .enumerate()
            .map(|(k, cell)| {
                (!cell.is_empty()).then(|| {
                    let next = self.cells[k].len();
                    *self.cells[k].entry(cell).or_insert(next)
                })
            })
            .collect();
        if self.out.len() == MAX_ASSIGNMENTS {
            return Err(AscmError::TooLarge(MAX_ASSIGNMENTS));
        }
        self.out.push(assignment);
        Ok(())
    }
}

/// The polytope of one component's response-type distributions `q_C`
/// consistent with `Q[C]`.
struct Component {
    /// The coordinates `(V, cfg)` a response type assigns a level to.
    coords: HashMap<(usize, usize), usize>,
    /// The number of levels of each coordinate, and the stride of each in a
    /// type's index.
    radix: Vec<usize>,
    stride: Vec<usize>,
    n_types: usize,
    rows: Vec<(Vec<f64>, f64)>,
}

impl Component {
    fn new(m: &Canonical, k: usize, cells: &[&Vec<Entry>]) -> Result<Self, AscmError> {
        let vars = &m.components[k];
        // A lone variable's coordinates are constrained one at a time, so
        // only those the query reads matter. A larger component's c-factor
        // ties all of them together.
        let mut list: Vec<(usize, usize)> = if vars.len() == 1 {
            let mut l: Vec<(usize, usize)> = cells
                .iter()
                .flat_map(|c| c.iter().map(|(v, cfg, _)| (*v, *cfg)))
                .collect();
            l.sort_unstable();
            l.dedup();
            l
        } else {
            vars.iter()
                .flat_map(|&v| (0..m.n_configs(v)).map(move |c| (v, c)))
                .collect()
        };
        list.sort_unstable();
        let radix: Vec<usize> = list.iter().map(|(v, _)| m.levels[*v].len()).collect();
        let mut stride = vec![1usize; radix.len()];
        let mut n_types = 1usize;
        for (s, r) in stride.iter_mut().zip(&radix) {
            *s = n_types;
            n_types = n_types
                .checked_mul(*r)
                .filter(|n| *n <= MAX_TYPES)
                .ok_or(AscmError::TooLarge(MAX_TYPES))?;
        }
        let coords: HashMap<(usize, usize), usize> =
            list.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let mut comp = Self {
            coords,
            radix,
            stride,
            n_types,
            rows: Vec::new(),
        };
        if vars.len() == 1 {
            let v = vars[0];
            for &(_, cfg) in &list {
                let pa = m.decode(v, cfg);
                let den = m.prob(&pa);
                if den <= 0.0 {
                    return Err(AscmError::Positivity(m.describe(&pa)));
                }
                for l in 0..m.levels[v].len() {
                    let mut with = pa.clone();
                    with.push((v, l));
                    comp.add_row(&[(v, cfg, l)], m.prob(&with) / den);
                }
            }
        } else {
            // Q[C](c; o) for every level of C and of its outside parents O.
            let mut outside: Vec<usize> = vars
                .iter()
                .flat_map(|v| m.parents[*v].iter().copied())
                .filter(|p| !vars.contains(p))
                .collect();
            outside.sort_unstable();
            outside.dedup();
            let free: Vec<usize> = vars.iter().chain(&outside).copied().collect();
            let mut v = vec![0usize; m.names.len()];
            loop {
                let entries: Vec<Entry> = vars
                    .iter()
                    .map(|&x| (x, m.config(x, |p| v[p]), v[x]))
                    .collect();
                comp.add_row(&entries, m.c_factor(vars, &v)?);
                // Next assignment of `free`, odometer style.
                let mut i = 0;
                loop {
                    if i == free.len() {
                        return comp.check();
                    }
                    let x = free[i];
                    v[x] += 1;
                    if v[x] < m.levels[x].len() {
                        break;
                    }
                    v[x] = 0;
                    i += 1;
                }
            }
        }
        comp.check()
    }

    fn check(self) -> Result<Self, AscmError> {
        match self.lp(vec![0.0; self.n_types]).solve() {
            LpOutcome::Infeasible => Err(AscmError::Incompatible),
            _ => Ok(self),
        }
    }

    fn add_row(&mut self, entries: &[Entry], rhs: f64) {
        let a = self.indicator(entries);
        self.rows.push((a, rhs));
    }

    /// The response types that satisfy every entry, as a 0/1 vector.
    fn indicator(&self, entries: &[Entry]) -> Vec<f64> {
        let fixed: Vec<(usize, usize)> = entries
            .iter()
            .map(|(v, cfg, l)| (self.coords[&(*v, *cfg)], *l))
            .collect();
        (0..self.n_types)
            .map(|t| {
                let hit = fixed
                    .iter()
                    .all(|&(i, l)| t / self.stride[i] % self.radix[i] == l);
                f64::from(u8::from(hit))
            })
            .collect()
    }

    fn lp(&self, objective: Vec<f64>) -> Lp {
        let mut lp = Lp::new(self.n_types);
        lp.maximize(objective);
        lp.constraint(vec![1.0; self.n_types], Cmp::Eq, 1.0);
        for (a, b) in &self.rows {
            lp.constraint(a.clone(), Cmp::Eq, *b);
        }
        lp
    }

    /// The least and the greatest value of `c · q` over the polytope.
    fn range(&self, c: &[f64]) -> Result<(f64, f64), AscmError> {
        let solve = |sign: f64| -> Result<f64, AscmError> {
            match self.lp(c.iter().map(|x| sign * x).collect()).solve() {
                LpOutcome::Optimal { value, .. } => Ok(sign * value),
                LpOutcome::Infeasible => Err(AscmError::Incompatible),
                LpOutcome::Unbounded => unreachable!("a distribution is bounded"),
            }
        };
        Ok((solve(-1.0)?, solve(1.0)?))
    }
}
