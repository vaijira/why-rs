//! Policy transportability by C-INFER (Sec. 12.1.2, Algorithms 41 and 43).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use super::TrError;
use crate::graph::cinfer::{
    CFactor, CTree, Input, Operator, VarSet, c_components, evaluate, grow_input_tree,
};
use crate::graph::dseparation::{Admg, DSepError};
use crate::graph::identification::Formula;
use crate::transport::{Domain, Estimand, Intervention, Regime, SelectionDiagram, TransportError};

fn join(s: &VarSet) -> String {
    s.iter().cloned().collect::<Vec<_>>().join(", ")
}

fn var_set(names: &[&str]) -> VarSet {
    names.iter().map(ToString::to_string).collect()
}

fn refs(s: &VarSet) -> Vec<&str> {
    s.iter().map(String::as_str).collect()
}

/// The target `P*(y | w; σ_X)`: the distribution of `y` given `w` in the
/// target domain under the regime `σ_X` (signature 12.35).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyQuery {
    /// The outcomes.
    pub y: VarSet,
    /// What is conditioned on; empty for an unconditional query.
    pub w: VarSet,
    /// The target regime `σ_X`.
    pub regime: Regime,
}

impl PolicyQuery {
    /// `P*(y; σ_X)`.
    #[must_use]
    pub fn new(y: &[&str], regime: Regime) -> Self {
        Self {
            y: var_set(y),
            w: VarSet::new(),
            regime,
        }
    }

    /// The same query, additionally conditioned on `w`.
    #[must_use]
    pub fn given(mut self, w: &[&str]) -> Self {
        self.w = var_set(w);
        self
    }

    fn validate(&self, g: &Admg) -> Result<(), DSepError> {
        if self.y.is_empty() {
            return Err(DSepError::Empty("Y"));
        }
        for v in self.y.iter().chain(&self.w) {
            if g.node(v).is_none() {
                return Err(DSepError::Unknown(v.clone()));
            }
        }
        if let Some(v) = self.y.intersection(&self.w).next() {
            return Err(DSepError::Overlap(v.clone()));
        }
        let atomic = self.regime.atomic_variables();
        for v in self.regime.variables() {
            if self.y.contains(v) || (atomic.contains(v) && self.w.contains(v)) {
                return Err(DSepError::Overlap(v.to_string()));
            }
        }
        Ok(())
    }
}

impl fmt::Display for PolicyQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P*({}", join(&self.y))?;
        if !self.w.is_empty() {
            write!(f, " | {}", join(&self.w))?;
        }
        if !self.regime.is_empty() {
            write!(f, "; {}", self.regime)?;
        }
        f.write_str(")")
    }
}

/// An available distribution `P^k(V; σ_Z)`: the target's or a source's,
/// in its natural regime or under the interventions `σ_Z`.
///
/// Every input is a distribution over all of `V`. An atomic `do(z)` stands
/// for an experiment at every level of `Z`, which is equivalent to a trial
/// that randomizes `Z`; that is how its estimand terms are evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainInput {
    /// The domain the data come from: [`Domain::Target`] or a
    /// [`Domain::Source`].
    pub domain: Domain,
    /// The regime it was collected under.
    pub regime: Regime,
}

impl DomainInput {
    /// `P^domain(V; σ_Z)`.
    #[must_use]
    pub fn new(domain: Domain, regime: Regime) -> Self {
        Self { domain, regime }
    }

    /// `P^domain(V)`, the natural regime.
    #[must_use]
    pub fn observational(domain: Domain) -> Self {
        Self::new(domain, Regime::new())
    }

    /// The domain an estimand term read from this input names.
    #[must_use]
    pub fn estimand_domain(&self) -> Domain {
        self.domain.clone().under(self.regime.clone())
    }
}

impl fmt::Display for DomainInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.regime.is_empty() {
            write!(f, "P^{}(V)", self.domain)
        } else {
            write!(f, "P^{}(V; {})", self.domain, self.regime)
        }
    }
}

/// A c-factor `Q^k[C; σ_Z]` of domain `k` under regime `σ_Z`.
///
/// For a set `C` disjoint from the regime's variables it does not depend on
/// the regime (σ-operator, Lemma 12.1.11), and for one disjoint from
/// `Δ^{k,*}` it does not depend on the domain (π-operator, Lemma 12.1.10).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrFactor {
    /// The domain.
    pub domain: Domain,
    /// The variables the factor is a joint over.
    pub scope: VarSet,
    /// The regime.
    pub regime: Regime,
}

impl fmt::Display for TrFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q^{}[{}", self.domain, join(&self.scope))?;
        if !self.regime.is_empty() {
            write!(f, "; {}", self.regime)?;
        }
        f.write_str("]")
    }
}

/// What a transportability tree node stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrTerm {
    /// The query, at the root of a q-tree.
    Query(PolicyQuery),
    /// A c-factor.
    Factor(TrFactor),
}

impl fmt::Display for TrTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query(q) => q.fmt(f),
            Self::Factor(c) => c.fmt(f),
        }
    }
}

/// A c-operator on an edge of a transportability tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrOperator {
    /// `Σ`, marginalisation over `sum_out`.
    Sigma {
        /// The variables summed out.
        sum_out: VarSet,
    },
    /// `δ`, c-component factorisation.
    Delta,
    /// `Γ`, unconditioning: the query is the child normalised over `y`.
    Gamma {
        /// The conditioning variables.
        given: VarSet,
    },
    /// The π- and σ-operators: the same c-factor read in the target domain
    /// (π, Lemma 12.1.10) and in the query's regime (σ, Lemma 12.1.11).
    Transfer {
        /// Whether the domain changes (π).
        domain: bool,
        /// Whether the regime changes (σ).
        regime: bool,
    },
}

impl fmt::Display for TrOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sigma { sum_out } => write!(f, "Σ_{{{}}}", join(sum_out)),
            Self::Delta => f.write_str("δ"),
            Self::Gamma { given } => write!(f, "Γ | {}", join(given)),
            Self::Transfer { domain, regime } => f.write_str(match (domain, regime) {
                (true, true) => "π, σ",
                (true, false) => "π",
                _ => "σ",
            }),
        }
    }
}

/// A transportability tree: a q-tree for a query in `π*`, or a d-tree for one
/// input distribution.
pub type TrTree = CTree<TrTerm, TrOperator>;

impl TrTree {
    /// The c-factor at `idx`, if it holds one rather than the query.
    #[must_use]
    pub fn tr_factor(&self, idx: usize) -> Option<&TrFactor> {
        match &self.nodes[idx].term {
            TrTerm::Factor(c) => Some(c),
            TrTerm::Query(_) => None,
        }
    }

    fn find_factor(&self, factor: &TrFactor) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| matches!(&n.term, TrTerm::Factor(c) if c == factor))
    }
}

/// Where a q-tree leaf's value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrSource {
    /// `Q*[X_i; do(x)]` for an atomic treatment: the indicator `1[X_i = x_i]`.
    Indicator,
    /// `Q*[X_i; σ_X]` for a soft treatment: the policy `P̂(x_i | pa*_i)`.
    Policy,
    /// Node `node` of the d-tree of input `input`.
    Input {
        /// The index into the inputs.
        input: usize,
        /// The index into that input's d-tree.
        node: usize,
    },
}

/// A successful transport, with the trees that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrDerivation {
    /// The q-tree.
    pub query_tree: TrTree,
    /// One d-tree per input, in the order the inputs were given.
    pub input_trees: Vec<TrTree>,
    /// The source of every q-tree leaf.
    pub mapping: BTreeMap<usize, TrSource>,
    /// The transport formula.
    pub estimand: Estimand,
}

/// A d-tree together with what evaluating its nodes needs: the raw layer-2
/// tree it was converted from, and the regime's graph.
pub(crate) struct Grown {
    pub(crate) tree: TrTree,
    raw: CTree,
    /// The raw node behind each node of `tree`.
    raw_of: Vec<usize>,
    graph: Admg,
    order: Vec<String>,
}

impl Grown {
    /// The formula for node `node`, its terms tagged with `tag`.
    pub(crate) fn formula(&self, node: usize, tag: &str) -> Result<Formula, DSepError> {
        let mut f = evaluate(
            &self.graph,
            &self.raw,
            self.raw_of[node],
            &Input::observational(),
            &self.order,
        )?;
        retag(&mut f, tag);
        Ok(f)
    }
}

/// Checks that an input names the target or a declared source.
pub(crate) fn delta_of<'a>(
    d: &'a SelectionDiagram,
    input: &DomainInput,
) -> Result<Option<&'a BTreeSet<String>>, TransportError> {
    match &input.domain {
        Domain::Target => Ok(None),
        Domain::Source(s) => d
            .discrepancies(s)
            .map(Some)
            .ok_or_else(|| TransportError::Unknown(s.clone())),
        other => Err(TransportError::InvalidInput(format!(
            "an input's domain must be the target or a source, not {other}"
        ))),
    }
}

/// `GENINPUTTREE` (Algorithm 41) towards the c-factors over `targets`, which
/// are read in the target domain under `to`.
///
/// The search runs in `G_{σ_Z}`, with the input itself, `Q^k[V; σ_Z]`, at the
/// root: every variable, intervened ones included, each of those a
/// c-component of its own. Line 3 gives up on a target that meets `Z` or
/// `Δ^{k,*}`. Otherwise the layer-2 search applies `Σ` and `δ` until it
/// reaches the target's scope, and the last edge carries π when the domain
/// changes and σ when the regime does.
pub(crate) fn grow(
    d: &SelectionDiagram,
    input: &DomainInput,
    targets: &[VarSet],
    to: &Regime,
) -> Result<Grown, TransportError> {
    let delta = delta_of(d, input)?;
    let graph = input.regime.apply(d.graph())?;
    let order = graph.topological_order()?;
    let z: VarSet = input.regime.variables().map(str::to_string).collect();
    let usable: Vec<CFactor> = targets
        .iter()
        .filter(|c| c.is_disjoint(&z) && delta.is_none_or(|dl| c.is_disjoint(dl)))
        .map(|c| CFactor {
            scope: c.clone(),
            doing: VarSet::new(),
        })
        .collect();
    let raw = grow_input_tree(&graph, &Input::observational(), &usable)?;

    let factor = |scope: &VarSet| TrFactor {
        domain: input.domain.clone(),
        scope: scope.clone(),
        regime: input.regime.clone(),
    };
    let root_scope = &raw.factor(raw.root).expect("d-trees hold c-factors").scope;
    let mut tree = TrTree::new(TrTerm::Factor(factor(root_scope)));
    let mut raw_of = vec![raw.root];
    let mut map = vec![0usize; raw.nodes.len()];
    for i in 1..raw.nodes.len() {
        let node = &raw.nodes[i];
        let parent = map[node.parent.expect("only the root has no parent")];
        let scope = &raw.factor(i).expect("d-trees hold c-factors").scope;
        let (term, op) = match node.op_from_parent.as_ref().expect("not the root") {
            Operator::Sigma { sum_out } => (
                factor(scope),
                TrOperator::Sigma {
                    sum_out: sum_out.clone(),
                },
            ),
            Operator::Delta { .. } => (factor(scope), TrOperator::Delta),
            Operator::Regime { .. } => {
                let across_domains = input.domain != Domain::Target;
                let across_regimes = input.regime != *to;
                if !across_domains && !across_regimes {
                    map[i] = parent;
                    continue;
                }
                (
                    TrFactor {
                        domain: Domain::Target,
                        scope: scope.clone(),
                        regime: to.clone(),
                    },
                    TrOperator::Transfer {
                        domain: across_domains,
                        regime: across_regimes,
                    },
                )
            }
            Operator::Gamma { .. } => unreachable!("Γ only appears in q-trees"),
        };
        let idx = tree.add_child(parent, TrTerm::Factor(term), op);
        if idx == raw_of.len() {
            raw_of.push(i);
        }
        map[i] = idx;
    }
    Ok(Grown {
        tree,
        raw,
        raw_of,
        graph,
        order,
    })
}

/// The tag marking terms read from input `k` while the estimand is a
/// [`Formula`]. A [`Formula`] has no domain, so the domain rides in the
/// `doing` slot, which keeps the chain rule from merging terms of different
/// inputs; [`to_estimand`] reads it back. No variable can have this name.
pub(crate) fn input_tag(k: usize) -> String {
    format!("\u{0}{k}")
}

const POLICY_TAG: &str = "\u{0}policy";

fn retag(f: &mut Formula, tag: &str) {
    match f {
        Formula::P { doing, .. } => *doing = vec![tag.to_string()],
        Formula::Product(fs) => fs.iter_mut().for_each(|f| retag(f, tag)),
        Formula::Sum { body, .. } => retag(body, tag),
        Formula::Ratio { num, den } => {
            retag(num, tag);
            retag(den, tag);
        }
    }
}

/// Converts a tagged [`Formula`] into an [`Estimand`], each term read from
/// the domain its tag names, with conditioning sets in `g`'s order.
pub(crate) fn to_estimand(
    f: &Formula,
    inputs: &[DomainInput],
    g: &Admg,
) -> Result<Estimand, DSepError> {
    fn walk(f: &Formula, domains: &HashMap<String, Domain>) -> Estimand {
        match f {
            Formula::P { vars, given, doing } => Estimand::P {
                domain: domains[&doing[0]].clone(),
                vars: vars.clone(),
                given: given.clone(),
            },
            Formula::Product(fs) => Estimand::product(fs.iter().map(|f| walk(f, domains)).collect()),
            Formula::Sum { over, body } => Estimand::sum(over.clone(), walk(body, domains)),
            Formula::Ratio { num, den } => Estimand::ratio(walk(num, domains), walk(den, domains)),
        }
    }
    let mut domains: HashMap<String, Domain> = inputs
        .iter()
        .enumerate()
        .map(|(k, i)| (input_tag(k), i.estimand_domain()))
        .collect();
    domains.insert(POLICY_TAG.to_string(), Domain::Policy);
    let mut e = walk(f, &domains);
    let rank: HashMap<String, usize> = g
        .topological_order()?
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    e.sort_given(&rank);
    Ok(e)
}

/// The kind of a q-tree leaf.
enum Leaf<'a> {
    Indicator,
    Policy(&'a str, &'a BTreeSet<String>),
    Target,
}

fn leaf_kind<'a>(query: &'a PolicyQuery, scope: &VarSet) -> Leaf<'a> {
    if scope.len() == 1 {
        let v = scope.iter().next().expect("one element");
        if let Some((name, i)) = query.regime.iter().find(|(n, _)| *n == v.as_str()) {
            return match i {
                Intervention::Atomic => Leaf::Indicator,
                Intervention::Policy { parents } => Leaf::Policy(name, parents),
            };
        }
    }
    Leaf::Target
}

/// The q-tree leaves that must come from an input.
fn target_leaves(query: &PolicyQuery, t_q: &TrTree) -> Vec<(usize, VarSet)> {
    t_q.leaves()
        .into_iter()
        .filter_map(|i| t_q.tr_factor(i).map(|c| (i, c.scope.clone())))
        .filter(|(_, s)| matches!(leaf_kind(query, s), Leaf::Target))
        .collect()
}

/// `GENQUERYTREE` for `P*(y | w; σ_X)`.
///
/// With `D = An(Y ∪ W)` in `G_{σ_X}`:
///
/// ```text
/// P*(y; σ_X) <-Σ_{D \ Y}- Q*[D; σ_X] <-δ- { Q*[D_i; σ_X] }
/// ```
///
/// where the `D_i` are the c-components of `G_{σ_X}[D]`. Each intervened
/// `X_i` is a component of its own: an atomic one is the indicator
/// `1[X_i = x_i]`, a soft one the policy `P̂(x_i | pa*_i)` (Eqs. 12.62–12.63).
/// A conditional query replaces `Σ` with `Γ`, the ratio of the sums over
/// `D \ (Y ∪ W)` and `D \ W`.
///
/// # Errors
///
/// [`TransportError::Graph`] for unknown names, overlapping sets, or a policy
/// that would make the diagram cyclic.
pub fn gen_query_tree(d: &SelectionDiagram, query: &PolicyQuery) -> Result<TrTree, TransportError> {
    query.validate(d.graph())?;
    let gx = query.regime.apply(d.graph())?;
    let yw: VarSet = query.y.union(&query.w).cloned().collect();
    let scope: VarSet = gx
        .ancestors_of(&refs(&yw))?
        .into_iter()
        .map(str::to_string)
        .collect();
    let op = if query.w.is_empty() {
        TrOperator::Sigma {
            sum_out: scope.difference(&query.y).cloned().collect(),
        }
    } else {
        TrOperator::Gamma {
            given: query.w.clone(),
        }
    };
    let factor = |s: &VarSet| {
        TrTerm::Factor(TrFactor {
            domain: Domain::Target,
            scope: s.clone(),
            regime: query.regime.clone(),
        })
    };
    let mut tree = TrTree::new(TrTerm::Query(query.clone()));
    let top = tree.add_child(tree.root, factor(&scope), op);
    let components = c_components(&gx.induced_subgraph(&refs(&scope))?);
    if components.len() > 1 {
        for c in &components {
            tree.add_child(top, factor(c), TrOperator::Delta);
        }
    }
    Ok(tree)
}

/// `GENINPUTTREE` (Algorithm 41): what `input` can produce towards the
/// q-tree's leaves, as [`grow`] describes.
///
/// # Errors
///
/// [`TransportError::Unknown`] for an undeclared source, and
/// [`TransportError::Graph`] for an input regime naming unknown variables
/// or making the diagram cyclic.
pub fn gen_input_tree(
    d: &SelectionDiagram,
    input: &DomainInput,
    query: &PolicyQuery,
    t_q: &TrTree,
) -> Result<TrTree, TransportError> {
    let targets: Vec<VarSet> = target_leaves(query, t_q).into_iter().map(|(_, s)| s).collect();
    Ok(grow(d, input, &targets, &query.regime)?.tree)
}

/// `MAPFACTORS`: a source for each q-tree leaf that has one. Indicator and
/// policy leaves always map; any other leaf maps to the first input whose
/// d-tree reached it.
#[must_use]
pub fn map_factors(
    query: &PolicyQuery,
    t_q: &TrTree,
    t_ps: &[TrTree],
) -> BTreeMap<usize, TrSource> {
    let mut mapping = BTreeMap::new();
    for i in t_q.leaves() {
        let Some(leaf) = t_q.tr_factor(i) else { continue };
        match leaf_kind(query, &leaf.scope) {
            Leaf::Indicator => {
                mapping.insert(i, TrSource::Indicator);
            }
            Leaf::Policy(..) => {
                mapping.insert(i, TrSource::Policy);
            }
            Leaf::Target => {
                if let Some((input, node)) = t_ps
                    .iter()
                    .enumerate()
                    .find_map(|(k, t)| t.find_factor(leaf).map(|n| (k, n)))
                {
                    mapping.insert(i, TrSource::Input { input, node });
                }
            }
        }
    }
    mapping
}

/// `C-INFER` for σ-TR (Algorithm 43 with Algorithm 41): transports
/// `P*(y | w; σ_X)` from the inputs and the selection diagram.
///
/// The g-TR task of Thm 12.1.12 is the special case where every regime is
/// atomic. For unconditional queries the procedure is complete
/// (Thms 12.1.12–12.1.13): it fails exactly when the query is not
/// transportable. A conditional query is transported as the ratio of two
/// unconditional ones; that is sound but may miss a query that rule 2 of the
/// σ-calculus would rescue.
///
/// ```
/// use why_data::generalize::{DomainInput, PolicyQuery, sigma_tr};
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::{Domain, Regime, SelectionDiagram};
///
/// // Example 12.11: X → Z → Y, X <-> Y, Δ¹ = {X, Y}, Δ² = {X, Z}.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// g.bidirected("X", "Y");
/// let d = SelectionDiagram::new(g)
///     .with_source("1", &["X", "Y"]).unwrap()
///     .with_source("2", &["X", "Z"]).unwrap();
/// let do_x = Regime::new().atomic(&["X"]);
/// let inputs = [
///     DomainInput::new(Domain::source("1"), do_x.clone()),
///     DomainInput::new(Domain::source("2"), do_x.clone()),
/// ];
/// let t = sigma_tr(&d, &PolicyQuery::new(&["Y"], do_x), &inputs).unwrap();
/// assert_eq!(t.estimand.to_string(), "sum_{z} P^1(z; do(x)) P^2(y | z; do(x))");
/// ```
///
/// # Errors
///
/// [`TrError::NotTransportable`] when some target c-factor appears in no
/// d-tree, carrying the d-trees and the missing factors;
/// [`TrError::Invalid`] for malformed queries or inputs.
pub fn sigma_tr(
    d: &SelectionDiagram,
    query: &PolicyQuery,
    inputs: &[DomainInput],
) -> Result<TrDerivation, TrError> {
    let query_tree = gen_query_tree(d, query)?;
    let targets = target_leaves(query, &query_tree);
    let scopes: Vec<VarSet> = targets.iter().map(|(_, s)| s.clone()).collect();
    let grown = inputs
        .iter()
        .map(|p| grow(d, p, &scopes, &query.regime))
        .collect::<Result<Vec<_>, _>>()?;
    let input_trees: Vec<TrTree> = grown.iter().map(|g| g.tree.clone()).collect();
    let mapping = map_factors(query, &query_tree, &input_trees);
    let unmapped: Vec<TrFactor> = targets
        .iter()
        .filter(|(i, _)| !mapping.contains_key(i))
        .map(|(i, _)| query_tree.tr_factor(*i).expect("a factor").clone())
        .collect();
    if !unmapped.is_empty() {
        return Err(TrError::NotTransportable {
            input_trees,
            unmapped,
        });
    }
    let estimand = compose(d.graph(), query, &query_tree, inputs, &grown, &mapping)?;
    Ok(TrDerivation {
        query_tree,
        input_trees,
        mapping,
        estimand,
    })
}

/// `COMPOSEQUERY`: the product of the leaves' factors — each evaluated along
/// its d-tree path, or the policy's own conditional — summed over the
/// ancestral set minus `Y` and the atomic treatments, which stay fixed.
fn compose(
    g: &Admg,
    query: &PolicyQuery,
    t_q: &TrTree,
    inputs: &[DomainInput],
    grown: &[Grown],
    mapping: &BTreeMap<usize, TrSource>,
) -> Result<Estimand, DSepError> {
    let order = g.topological_order()?;
    // Parents before children, as the book prints them.
    let mut leaves = t_q.leaves();
    leaves.sort_by_key(|&i| {
        let scope = &t_q.tr_factor(i).expect("leaves are factors").scope;
        order.iter().position(|v| scope.contains(v))
    });
    let mut factors = Vec::new();
    for i in leaves {
        let scope = &t_q.tr_factor(i).expect("leaves are factors").scope;
        match mapping[&i] {
            TrSource::Indicator => {}
            TrSource::Policy => {
                let Leaf::Policy(x, parents) = leaf_kind(query, scope) else {
                    unreachable!("mapped as a policy")
                };
                factors.push(Formula::P {
                    vars: vec![x.to_string()],
                    given: order.iter().filter(|v| parents.contains(*v)).cloned().collect(),
                    doing: vec![POLICY_TAG.to_string()],
                });
            }
            TrSource::Input { input, node } => {
                factors.push(grown[input].formula(node, &input_tag(input))?);
            }
        }
    }
    factors.sort_by_key(|f| !matches!(f, Formula::P { .. }));
    let product = match Formula::product(factors) {
        p @ Formula::Product(_) => p,
        other => Formula::Product(vec![other]),
    };
    let top = t_q.nodes[t_q.root].children[0];
    let scope = &t_q.tr_factor(top).expect("a factor").scope;
    let fixed = query.regime.atomic_variables();
    let without = |drop: &[&VarSet]| -> Vec<String> {
        scope
            .iter()
            .filter(|v| !drop.iter().any(|s| s.contains(*v)))
            .cloned()
            .collect()
    };
    let formula = if query.w.is_empty() {
        Formula::sum(without(&[&query.y, &fixed]), product)
    } else {
        let num = Formula::sum(without(&[&query.y, &query.w, &fixed]), product.clone());
        let den = Formula::sum(without(&[&query.w, &fixed]), product);
        if matches!(&den, Formula::Product(f) if f.is_empty()) {
            num
        } else {
            Formula::Ratio {
                num: Box::new(num),
                den: Box::new(den),
            }
        }
    };
    let mut formula = formula.simplify();
    let mut free = formula.free_vars();
    free.extend(query.y.iter().chain(&query.w).chain(&fixed).cloned());
    formula.avoid_capture(&free);
    to_estimand(&formula, inputs, g)
}
