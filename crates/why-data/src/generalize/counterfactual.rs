//! Counterfactual transportability (Sec. 12.2.1, Algorithms 42 and 43).

use std::collections::BTreeMap;
use std::fmt;

use super::TrError;
use super::policy::{DomainInput, TrFactor, TrTree, grow, input_tag, to_estimand};
use crate::graph::cinfer::VarSet;
use crate::graph::cinfer::counterfactual::{Built, CtfError, CtfQuery, CtfTree, build, compose};
use crate::transport::{Domain, Estimand, Regime, SelectionDiagram, TransportError};

/// What a transported counterfactual query evaluates to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtfTrEstimand {
    /// The query contradicts itself, as `P(y_y')` does.
    Zero,
    /// Every queried event is certain given the rest, as in `P(y_y)`.
    One,
    /// An expression in the input distributions.
    Estimand(Estimand),
}

impl fmt::Display for CtfTrEstimand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => f.write_str("0"),
            Self::One => f.write_str("1"),
            Self::Estimand(e) => e.fmt(f),
        }
    }
}

/// A successful counterfactual transport, with the trees that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CtfTrDerivation {
    /// The counterfactual q-tree (Algorithm 42), down to the Ξ leaves.
    pub query_tree: CtfTree,
    /// One d-tree per input (Algorithm 41), in the order given.
    pub input_trees: Vec<TrTree>,
    /// For each Ξ leaf of the q-tree, the input and d-tree node it is read
    /// from.
    pub mapping: BTreeMap<usize, (usize, usize)>,
    /// The transport formula.
    pub estimand: CtfTrEstimand,
}

impl From<CtfError> for TrError {
    fn from(e: CtfError) -> Self {
        match e {
            CtfError::Inconsistent { factor, .. } => Self::Inconsistent { factor },
            CtfError::ImpossibleCondition => Self::ImpossibleCondition,
            CtfError::MisplacedNesting(v) => Self::Invalid(TransportError::InvalidInput(format!(
                "the subscript entry for {v} nests a different variable"
            ))),
            CtfError::Graph(e) => Self::Invalid(TransportError::Graph(e)),
            CtfError::NotIdentifiable { .. } => {
                unreachable!("building a q-tree never consults the inputs")
            }
        }
    }
}

/// `C-INFER` for ctf-TR (Algorithm 43 with Algorithms 42 and 41):
/// transports the counterfactual `P*(y_* | x_*)` of the target domain from
/// distributions of any domain and regime (Def. 12.2.1, signature 12.91).
///
/// The q-tree is chapter 5's: CUT unnests, Σ takes the counterfactual
/// ancestors, Γ conditions, δ splits them into ctf-factors and Ξ collapses
/// each consistent one to a c-factor `Q*[C]` of the target's natural regime.
/// An inconsistent ctf-factor is not transportable from any data. Each `Q*[C]`
/// is then sought in the inputs' d-trees, which may read it from a source
/// whose discrepancies miss `C` (π-operator, Lemma 12.2.2) and from a regime
/// that leaves `C` alone (σ-operator).
///
/// ```
/// use why_data::generalize::{DomainInput, ctf_tr};
/// use why_data::graph::cinfer::counterfactual::CtfQuery;
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::{Domain, Regime, SelectionDiagram};
///
/// // Example 12.15 (Fig. 12.15): the US differs from Australia in how
/// // well-being Y responds; Australia ran an experiment on schooling X.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// g.directed("X", "Y");
/// g.bidirected("X", "Z");
/// let d = SelectionDiagram::new(g).with_source("AU", &["Y"]).unwrap();
/// let inputs = [
///     DomainInput::observational(Domain::Target),
///     DomainInput::new(Domain::source("AU"), Regime::new().atomic(&["X"])),
/// ];
/// // The nested counterfactual of the natural direct effect.
/// let q = CtfQuery::parse(d.graph(), "P(y_{x', Z_x})").unwrap();
/// let t = ctf_tr(&d, &q, &inputs).unwrap();
/// assert_eq!(t.estimand.to_string(), "sum_{z} P^AU(z; do(x)) P^*(y | x', z)");
/// ```
///
/// # Errors
///
/// [`TrError::Inconsistent`] when a ctf-factor cannot be collapsed;
/// [`TrError::NotTransportable`] when some c-factor appears in no d-tree;
/// [`TrError::ImpossibleCondition`] and [`TrError::Invalid`] for malformed
/// queries or inputs.
pub fn ctf_tr(
    d: &SelectionDiagram,
    query: &CtfQuery,
    inputs: &[DomainInput],
) -> Result<CtfTrDerivation, TrError> {
    let g = d.graph();
    let plan = match build(g, query)? {
        Built::Constant(query_tree, value) => {
            use crate::graph::cinfer::counterfactual::CtfEstimand;
            return Ok(CtfTrDerivation {
                query_tree,
                input_trees: Vec::new(),
                mapping: BTreeMap::new(),
                estimand: match value {
                    CtfEstimand::Zero => CtfTrEstimand::Zero,
                    _ => CtfTrEstimand::One,
                },
            });
        }
        Built::Plan(p) => p,
    };
    let mut scopes: Vec<VarSet> = Vec::new();
    for (_, c, _) in &plan.leaves {
        if !scopes.contains(&c.scope) {
            scopes.push(c.scope.clone());
        }
    }
    let natural = Regime::new();
    let grown = inputs
        .iter()
        .map(|p| grow(d, p, &scopes, &natural))
        .collect::<Result<Vec<_>, _>>()?;
    let input_trees: Vec<TrTree> = grown.iter().map(|g| g.tree.clone()).collect();

    let mut mapping = BTreeMap::new();
    let mut unmapped = Vec::new();
    for (leaf, c, _) in &plan.leaves {
        let target = TrFactor {
            domain: Domain::Target,
            scope: c.scope.clone(),
            regime: natural.clone(),
        };
        let found = input_trees.iter().enumerate().find_map(|(k, t)| {
            t.nodes
                .iter()
                .position(|n| matches!(&n.term, super::TrTerm::Factor(f) if *f == target))
                .map(|n| (k, n))
        });
        match found {
            Some(src) => {
                mapping.insert(*leaf, src);
            }
            None if !unmapped.contains(&target) => unmapped.push(target),
            None => {}
        }
    }
    if !unmapped.is_empty() {
        return Err(TrError::NotTransportable {
            input_trees,
            unmapped,
        });
    }
    let formula = compose(g, query, &plan, |leaf| {
        let (input, node) = mapping[&leaf];
        grown[input].formula(node, &input_tag(input))
    })
    .map_err(TransportError::Graph)?;
    let estimand = to_estimand(&formula, inputs, g).map_err(TransportError::Graph)?;
    Ok(CtfTrDerivation {
        query_tree: plan.tree,
        input_trees,
        mapping,
        estimand: CtfTrEstimand::Estimand(estimand),
    })
}
