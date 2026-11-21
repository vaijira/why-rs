//! Statistical generalization and transportability (layer 1): chapter 11,
//! "Generalizability and Transportability of Statistical Quantities".
//!
//! A predictor `P(y | x)` learned in source domains `π¹ … π^K` is only useful
//! in a target `π*` if the target quantity `P*(y | x)` can be computed from
//! what the sources (and possibly partial target data) reveal. A
//! [`SelectionDiagram`] says where the domains may differ: `Δ^{k,*}` lists the
//! variables whose mechanism or exogenous distribution may change between
//! source `k` and the target, and the diagram draws one S-node pointing at
//! each of them.
//!
//! # Domain generalization (Sec. 11.2)
//!
//! * [`is_directly_transportable`] is Thm 11.2.1: `P*(y | x) = P^k(y | x)`
//!   exactly when `S_k ⊥ Y | X` (S-admissibility, Def. 11.2.2).
//!   [`list_dtr_dist`] is Algorithm 32 (ListDTRDist), which lists every such
//!   conditioning set with polynomial delay. [`structural_covariate_shift`]
//!   checks Def. 11.2.3.
//! * [`score_tr`] is Algorithm 33 (Score-TR): whether the score function
//!   `E_{P*}[Y | z]` of a feature set is transportable from several sources
//!   at once (Thm 11.2.4), and the estimand when it is.
//!   [`transportable_feature_sets`] enumerates all of them and the maximal
//!   ones (Def. 11.2.5).
//! * [`Representation`] classifies the features a representation `φ(X)`
//!   determines, constrains or leaves free (Def. 11.2.8), and
//!   [`SelectionDiagram::augment`] adds the `R†` node of Def. 11.2.9, after
//!   which [`score_tr`] decides Thm 11.2.11.
//! * [`source_invariance`] measures the source invariance property
//!   (Def. 11.2.11) on data, which under CMS ([`cms_holds`],
//!   Assumption 11.2.12) and r-faithfulness is equivalent to transportability
//!   (Thm 11.2.16). [`invariant_feature_sets`] searches feature subsets for it.
//!
//! # Partial observability (Sec. 11.3)
//!
//! [`features_measured`], [`outcome_measured`] and [`auxiliary_measured`] are
//! Thms 11.3.1–11.3.3. [`statistical_transport`] is the complete procedure of
//! Thm 11.3.4 (the statistical transportability product), which also takes
//! target measurements `P*(W)`.
//!
//! Every estimand is an [`Estimand`] whose factors name the domain they are
//! read from. [`Estimand::evaluate`] computes it against any
//! [`Distributions`], such as [`EmpiricalDistributions`] built from one
//! [`Dataset`] per domain: the plug-in estimator.
//!
//! # Domain adaptation (Sec. 11.4)
//!
//! With a few labelled target samples, [`structure_informed_da`]
//! (Algorithm 34) pools the sources whose mechanism for `Y` matches the
//! target, and [`structure_agnostic_da`] (Algorithm 35) selects the source
//! subset on held-out target data. [`multi_cause_da`] (Algorithm 36) and
//! [`multi_cause_agnostic_da`] (Algorithm 37) align differing parent sets
//! with a shared mechanism. [`sequential_da`] (Algorithm 38) composes
//! per-position modules pooled through a discrepancy oracle (Def. 11.4.1).
//!
//! # Departures from the text
//!
//! * **Thm 11.3.4's `pa⁺` is Tian's conditioning set.** The text conditions
//!   each `V_i` on its parents and the earlier members of its c-component.
//!   The c-factor needs the parents of those members too: `Q[C] = ∏ P(v_i |
//!   Pa(T_i) \ {v_i})`, with `T_i` the c-component of `V_i` among the
//!   variables up to it. In Ex. 11.34 that turns `P²(a | x1)` into
//!   `P²(a | x1, x2, r)`, since `A ↔ X1 ← X2` is open given `X1`; the test
//!   `example_11_34_needs_tians_conditioning_set` shows that the printed
//!   factor gives a wrong answer.
//! * **Score-TR computes each c-factor with ID.** Line 11 of Algorithm 33 is
//!   the Markovian c-factor, but the figures it is applied to have
//!   bidirected edges. [`score_tr`] obtains `Q^i[C_t]` from `P^i(V)` with
//!   [`identify`](crate::graph::identification::identify). The denominator on
//!   line 16 sums over `W \ (Z \ Z̃)`: as printed (`W \ {y}`) it would also
//!   sum out the features and be a constant.
//! * **Thm 11.3.4 repeats from step 1** after projecting, so `D` is recomputed
//!   on the smaller diagram, and S-nodes are projected with the variables, so
//!   a discrepancy on a projected-out variable passes to its children. S-nodes
//!   never connect variables in step 1: components only they join cancel
//!   between numerator and denominator.
//! * **Ex. 11.30 misses a separator.** Thm 11.3.3 does apply to
//!   Fig. 11.32(a) with `W = {Z, R}`: `Z = {Z, R}` gives `(Y, X ⊥ S | Z)`,
//!   because `X → Z → R ← S` is blocked at `Z`. The resulting formula is
//!   sound, so this is only a missed opportunity, not a wrong one.
//! * **Ex. 11.4's `3/8` is a joint.** `P^{2*}(Y = 1, X = 1) = 3/8`; the
//!   conditional is `3/4`. Either way it differs from the sources' `1/2`.
//! * **Ex. 11.40 is inconsistent**: Eq. 11.255 makes the target's `V3` a
//!   unary function of `V2` and Eq. 11.256 a binary one of `V1, V2`. With
//!   Eq. 11.256's parents the noiseless chain gives `V6 = 14·V1 mod 10`, not
//!   the caption's `13·V1` (which `V5 ← V4, V1` would give). The tests build
//!   their own sequential instance.
//!
//! Not implemented: the structure-agnostic sequential search (Algorithm 39,
//! which enumerates every partition of positions and every set of graphs),
//! the optimization-based approach of Sec. 11.4.4, and learning
//! representations (UIRM, Def. 11.2.13), which needs a model class.

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use crate::graph::dseparation::{Admg, DSepError, EdgeKind};

mod adaptation;
mod direct;
mod estimand;
pub(crate) mod invariance;
mod partial;
mod representation;
mod score;
#[cfg(test)]
mod tests;

pub use crate::fairness::Dataset;
pub use adaptation::{
    AgnosticChoice, CategoricalModel, MultiCauseChoice, PoolSource, SequenceDomain,
    SequentialPredictor, multi_cause_agnostic_da, multi_cause_da, sequential_da,
    structure_agnostic_da, structure_informed_da,
};
pub use direct::{is_directly_transportable, list_dtr_dist, structural_covariate_shift};
pub use estimand::{Distributions, EmpiricalDistributions, Estimand};

pub use invariance::{
    InvarianceReport, LevelScores, cms_holds, invariant_feature_sets, source_invariance,
    source_invariance_by,
};
pub use partial::{
    ScenarioFormula, auxiliary_measured, features_measured, outcome_measured,
    statistical_transport,
};
pub use representation::{Representation, VariableRole};
pub use score::{FeatureSets, ScoreTr, score_tr, transportable_feature_sets};

/// The most variables any subset enumeration in this module will range over.
const SUBSET_CAP: usize = 20;

/// What can go wrong in this module.
#[derive(Clone, Debug, PartialEq)]
pub enum TransportError {
    /// A graph query failed: an unknown name, overlapping sets, or a cycle.
    Graph(DSepError),
    /// A name is not a variable of the diagram, or not a source domain.
    Unknown(String),
    /// A source domain was declared twice, or under the reserved name `*`.
    DuplicateSource(String),
    /// The query is not transportable; names the obstruction.
    NotTransportable(String),
    /// A distribution the estimand needs is missing or does not cover a
    /// variable.
    MissingData(String),
    /// A conditional is undefined because its conditioning event has
    /// probability zero.
    Positivity(String),
    /// Malformed input.
    InvalidInput(String),
    /// An enumeration would exceed its cap.
    TooLarge(String),
}

impl From<DSepError> for TransportError {
    fn from(e: DSepError) -> Self {
        Self::Graph(e)
    }
}

impl From<crate::fairness::FairnessError> for TransportError {
    fn from(e: crate::fairness::FairnessError) -> Self {
        Self::InvalidInput(e.to_string())
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => write!(f, "{e}"),
            Self::Unknown(v) => write!(f, "unknown variable or domain `{v}`"),
            Self::DuplicateSource(s) => write!(f, "source domain `{s}` is declared twice or reserved"),
            Self::NotTransportable(why) => write!(f, "not transportable: {why}"),
            Self::MissingData(why) => write!(f, "missing data: {why}"),
            Self::Positivity(why) => write!(f, "positivity violated: {why}"),
            Self::InvalidInput(why) => write!(f, "invalid input: {why}"),
            Self::TooLarge(why) => write!(f, "too large: {why}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// A domain: the target `π*` or a named source, possibly under an
/// intervention regime.
///
/// Chapter 11 only reads the natural regime of each domain. Chapter 12
/// ([`crate::generalize`]) also reads distributions collected under
/// interventions, `P^k(V; σ_Z)`, and factors that the target policy itself
/// defines, `P̂(x | pa*)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Domain {
    /// The target domain `π*`.
    Target,
    /// A source domain, by name.
    Source(String),
    /// A domain (never itself `Under` or `Policy`) under a non-empty
    /// intervention regime: the distribution `P^k(V; σ_Z)`.
    Under(Box<Domain>, Regime),
    /// The target policy `σ*_X`: its factors `P̂(x | pa*)` are known by
    /// definition rather than estimated.
    Policy,
}

impl Domain {
    /// The source domain called `name`.
    pub fn source(name: impl Into<String>) -> Self {
        Self::Source(name.into())
    }

    /// This domain under `regime`: itself if `regime` is empty, and the
    /// regime replaces any the domain already had.
    #[must_use]
    pub fn under(self, regime: Regime) -> Self {
        let base = match self {
            Self::Under(base, _) => *base,
            other => other,
        };
        if regime.is_empty() || base == Self::Policy {
            base
        } else {
            Self::Under(Box::new(base), regime)
        }
    }

    /// The domain without its regime.
    #[must_use]
    pub fn base(&self) -> &Self {
        match self {
            Self::Under(base, _) => base,
            other => other,
        }
    }

    /// The regime the distribution was collected under, if any.
    #[must_use]
    pub fn regime(&self) -> Option<&Regime> {
        match self {
            Self::Under(_, r) => Some(r),
            _ => None,
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target => f.write_str("*"),
            Self::Source(s) => f.write_str(s),
            Self::Under(d, r) => write!(f, "{d}; {r}"),
            Self::Policy => f.write_str("σ*"),
        }
    }
}

/// How a regime sets one variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Intervention {
    /// `do(x)`: the variable is fixed. In an *input* distribution an atomic
    /// intervention stands for a trial at every level, which is the same as
    /// randomizing it, so its value appears as an ordinary conditioning
    /// variable when the estimand is evaluated.
    Atomic,
    /// A soft intervention `σ_X = P̂(x | pa*)`: the variable is drawn from a
    /// known policy of `parents`, which replace its natural parents, and it
    /// no longer listens to the unobserved confounders it had. A randomized
    /// trial is a policy with no parents.
    Policy {
        /// The policy's inputs `pa*`.
        parents: BTreeSet<String>,
    },
}

/// A regime `σ_Z`: the interventions in force, keyed by the variable each
/// one sets. The empty regime is the natural (observational) one.
///
/// ```
/// use why_data::transport::Regime;
///
/// let r = Regime::new().atomic(&["X1"]).policy("X2", &["X1", "Z"]);
/// assert_eq!(r.to_string(), "do(x1), σ_{X2}");
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Regime {
    interventions: std::collections::BTreeMap<String, Intervention>,
}

impl Regime {
    /// The natural regime: nothing intervened.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same regime with `vars` also fixed by `do(·)`.
    #[must_use]
    pub fn atomic(mut self, vars: &[&str]) -> Self {
        for v in vars {
            self.interventions
                .insert((*v).to_string(), Intervention::Atomic);
        }
        self
    }

    /// The same regime with `var` also drawn from a policy of `parents`.
    #[must_use]
    pub fn policy(mut self, var: &str, parents: &[&str]) -> Self {
        self.interventions.insert(
            var.to_string(),
            Intervention::Policy {
                parents: parents.iter().map(|p| (*p).to_string()).collect(),
            },
        );
        self
    }

    /// Whether nothing is intervened on.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interventions.is_empty()
    }

    /// Every intervened variable, sorted.
    pub fn variables(&self) -> impl Iterator<Item = &str> {
        self.interventions.keys().map(String::as_str)
    }

    /// How `var` is set, if it is intervened on.
    #[must_use]
    pub fn get(&self, var: &str) -> Option<&Intervention> {
        self.interventions.get(var)
    }

    /// Every intervention, keyed by its variable.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Intervention)> {
        self.interventions.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The variables fixed by `do(·)`.
    #[must_use]
    pub fn atomic_variables(&self) -> BTreeSet<String> {
        self.iter()
            .filter(|(_, i)| **i == Intervention::Atomic)
            .map(|(v, _)| v.to_string())
            .collect()
    }

    /// `G_σ`: the diagram with every edge into an intervened variable
    /// removed, bidirected ones included, and an edge from each policy
    /// parent added.
    ///
    /// # Errors
    ///
    /// [`DSepError::Unknown`] for a name not in `g`, and
    /// [`DSepError::Cyclic`] when a policy reads one of its variable's
    /// descendants.
    pub fn apply(&self, g: &Admg) -> Result<Admg, DSepError> {
        for (v, i) in self.iter() {
            if g.node(v).is_none() {
                return Err(DSepError::Unknown(v.to_string()));
            }
            if let Intervention::Policy { parents } = i
                && let Some(p) = parents.iter().find(|p| g.node(p).is_none())
            {
                return Err(DSepError::Unknown(p.clone()));
            }
        }
        let vars: Vec<&str> = self.variables().collect();
        let mut out = g.cut_edges_into(&vars)?;
        for (v, i) in self.iter() {
            if let Intervention::Policy { parents } = i {
                for p in parents {
                    out.directed(p, v);
                }
            }
        }
        out.topological_order()?;
        Ok(out)
    }
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("∅");
        }
        let atomic: Vec<String> = self
            .atomic_variables()
            .iter()
            .map(|v| v.to_lowercase())
            .collect();
        let soft: Vec<&str> = self
            .iter()
            .filter(|(_, i)| matches!(i, Intervention::Policy { .. }))
            .map(|(v, _)| v)
            .collect();
        let mut parts = Vec::new();
        if !atomic.is_empty() {
            parts.push(format!("do({})", atomic.join(", ")));
        }
        if !soft.is_empty() {
            parts.push(format!("σ_{{{}}}", soft.join(", ")));
        }
        f.write_str(&parts.join(", "))
    }
}

/// A selection diagram `D^Δ` (Def. 10.1.2): one causal diagram shared by every
/// domain, plus, for each source `k`, the discrepancy set `Δ^{k,*}` of
/// variables whose mechanism `f_V` or exogenous distribution `P(U_V)` may
/// differ between `π^k` and the target.
///
/// The S-nodes are implicit. [`SelectionDiagram::augmented`] materialises
/// them, one per source, for d-separation queries.
///
/// ```
/// use why_data::graph::dseparation::Admg;
/// use why_data::transport::SelectionDiagram;
///
/// // Fig. 11.6(a): S → X → Z → Y.
/// let mut g = Admg::new();
/// g.directed("X", "Z");
/// g.directed("Z", "Y");
/// let d = SelectionDiagram::new(g).with_source("1", &["X"]).unwrap();
/// assert!(d.is_s_admissible("1", &["Y"], &["X"]).unwrap());
/// assert!(!d.is_s_admissible("1", &["Y"], &[]).unwrap());
/// ```
#[derive(Clone, Debug)]
pub struct SelectionDiagram {
    graph: Admg,
    sources: Vec<(String, BTreeSet<String>)>,
}

impl SelectionDiagram {
    /// A diagram with no source domains yet.
    #[must_use]
    pub fn new(graph: Admg) -> Self {
        Self {
            graph,
            sources: Vec::new(),
        }
    }

    /// Adds source `name` with discrepancy set `Δ^{name,*} = delta`.
    ///
    /// # Errors
    ///
    /// [`TransportError::Unknown`] for a variable not in the graph, and
    /// [`TransportError::DuplicateSource`] for a repeated name or `*`.
    pub fn with_source(mut self, name: &str, delta: &[&str]) -> Result<Self, TransportError> {
        if name == "*" || self.sources.iter().any(|(s, _)| s == name) {
            return Err(TransportError::DuplicateSource(name.to_string()));
        }
        for v in delta {
            if self.graph.node(v).is_none() {
                return Err(TransportError::Unknown((*v).to_string()));
            }
        }
        self.sources
            .push((name.to_string(), delta.iter().map(|s| (*s).to_string()).collect()));
        Ok(self)
    }

    /// The causal diagram over the observed variables.
    #[must_use]
    pub fn graph(&self) -> &Admg {
        &self.graph
    }

    /// The source domains, in declaration order.
    pub fn sources(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().map(|(s, _)| s.as_str())
    }

    /// `Δ^{source,*}`.
    #[must_use]
    pub fn discrepancies(&self, source: &str) -> Option<&BTreeSet<String>> {
        self.sources
            .iter()
            .find(|(s, _)| s == source)
            .map(|(_, d)| d)
    }

    fn delta(&self, source: &str) -> Result<&BTreeSet<String>, TransportError> {
        self.discrepancies(source)
            .ok_or_else(|| TransportError::Unknown(source.to_string()))
    }

    /// The name of `source`'s S-node in [`SelectionDiagram::augmented`]:
    /// `S_<source>`, primed until it collides with no variable.
    #[must_use]
    pub fn s_node(&self, source: &str) -> String {
        let mut name = format!("S_{source}");
        while self.graph.node(&name).is_some() {
            name.push('\'');
        }
        name
    }

    /// The diagram with one S-node per source pointing at its discrepancies.
    ///
    /// For a d-separation query about one source use
    /// [`SelectionDiagram::augmented_for`]: here every other source's S-node
    /// acts as a common cause of its children.
    #[must_use]
    pub fn augmented(&self) -> Admg {
        let mut g = self.graph.clone();
        for (s, delta) in &self.sources {
            let node = self.s_node(s);
            g.add_node(&node);
            for v in delta {
                g.directed(&node, v);
            }
        }
        g
    }

    /// The diagram with only `source`'s S-node, and that node's name; `None`
    /// when `Δ^{source,*}` is empty and there is no S-node to draw.
    ///
    /// Unlike [`SelectionDiagram::augmented`], the other sources' S-nodes are
    /// left out: when `π^source` is compared with `π*` they are constants, and
    /// drawn in they would act as common causes of their children.
    ///
    /// # Errors
    ///
    /// [`TransportError::Unknown`] for an unknown source.
    pub fn augmented_for(&self, source: &str) -> Result<(Admg, Option<String>), TransportError> {
        let delta = self.delta(source)?;
        if delta.is_empty() {
            return Ok((self.graph.clone(), None));
        }
        let mut g = self.graph.clone();
        let node = self.s_node(source);
        g.add_node(&node);
        for v in delta {
            g.directed(&node, v);
        }
        Ok((g, Some(node)))
    }

    /// As [`SelectionDiagram::augmented_for`], but the S-node is always
    /// drawn, isolated when `Δ^{source,*}` is empty, so that it can appear in
    /// a separation query either way.
    pub(crate) fn with_s_node(&self, source: &str) -> Result<(Admg, String), TransportError> {
        let (mut g, s) = self.augmented_for(source)?;
        let s = s.unwrap_or_else(|| {
            let node = self.s_node(source);
            g.add_node(&node);
            node
        });
        Ok((g, s))
    }

    /// S-admissibility (Def. 11.2.2): `S_source ⊥ A | Z` in the diagram, which
    /// makes `P^source(a | z) = P*(a | z)`.
    ///
    /// # Errors
    ///
    /// [`TransportError::Unknown`] for an unknown source, and
    /// [`TransportError::Graph`] for unknown or overlapping variables.
    pub fn is_s_admissible(
        &self,
        source: &str,
        a: &[&str],
        z: &[&str],
    ) -> Result<bool, TransportError> {
        let delta = self.delta(source)?;
        if delta.is_empty() {
            // Validate the names anyway, so typos do not pass silently.
            for v in a.iter().chain(z) {
                if self.graph.node(v).is_none() {
                    return Err(TransportError::Graph(DSepError::Unknown((*v).to_string())));
                }
            }
            return Ok(true);
        }
        let (g, s) = self.with_s_node(source)?;
        Ok(g.is_d_separated(&[&s], a, z)?)
    }

    /// Adds the representation node `R†` of Def. 11.2.9 with arrows from the
    /// constrained variables. Its mechanism is `φ†`, the same in every domain,
    /// so no S-node points at it.
    ///
    /// # Errors
    ///
    /// [`TransportError::Unknown`] for an unknown parent, and
    /// [`TransportError::InvalidInput`] if `name` is already a variable.
    pub fn augment(&self, name: &str, constrained: &[&str]) -> Result<Self, TransportError> {
        if self.graph.node(name).is_some() {
            return Err(TransportError::InvalidInput(format!(
                "`{name}` is already a variable"
            )));
        }
        let mut out = self.clone();
        out.graph.add_node(name);
        for p in constrained {
            if self.graph.node(p).is_none() {
                return Err(TransportError::Unknown((*p).to_string()));
            }
            out.graph.directed(p, name);
        }
        Ok(out)
    }

    /// The latent projection onto `keep`, carrying the S-nodes along: a
    /// variable that loses a discrepant parent to the projection inherits its
    /// S-node.
    pub(crate) fn project(&self, keep: &[&str]) -> Result<Self, TransportError> {
        let s_nodes: Vec<(String, String)> = self
            .sources
            .iter()
            .map(|(s, _)| (s.clone(), self.s_node(s)))
            .collect();
        let mut all: Vec<&str> = keep.to_vec();
        all.extend(s_nodes.iter().map(|(_, n)| n.as_str()));
        let projected = crate::crl::latent_projection(&self.augmented(), &all)
            .map_err(|e| TransportError::InvalidInput(e.to_string()))?;
        let s_names: HashSet<&str> = s_nodes.iter().map(|(_, n)| n.as_str()).collect();
        let mut graph = Admg::new();
        for v in projected.variables() {
            if !s_names.contains(v) {
                graph.add_node(v);
            }
        }
        let mut sources: Vec<(String, BTreeSet<String>)> = s_nodes
            .iter()
            .map(|(s, _)| (s.clone(), BTreeSet::new()))
            .collect();
        let g = projected.graph();
        for e in g.edge_indices() {
            let (a, b) = g.edge_endpoints(e).expect("valid edge");
            let (a, b) = (g[a].as_str(), g[b].as_str());
            if let Some(k) = s_nodes.iter().position(|(_, n)| n == a) {
                sources[k].1.insert(b.to_string());
            } else if !s_names.contains(b) {
                graph.add_edge(a, b, g[e]);
            }
        }
        Ok(Self { graph, sources })
    }

    /// Parents of `v` in the diagram (directed edges only).
    pub(crate) fn parents(&self, v: &str) -> BTreeSet<String> {
        let g = self.graph.graph();
        let Some(idx) = self.graph.node(v) else {
            return BTreeSet::new();
        };
        g.edges_directed(idx, petgraph::Direction::Incoming)
            .filter(|e| *e.weight() == EdgeKind::Directed)
            .map(|e| g[petgraph::visit::EdgeRef::source(&e)].clone())
            .collect()
    }

    pub(crate) fn check(&self, names: &[&str]) -> Result<(), TransportError> {
        for n in names {
            if self.graph.node(n).is_none() {
                return Err(TransportError::Unknown((*n).to_string()));
            }
        }
        Ok(())
    }
}

/// Every subset of `items`, smallest first, as index masks.
pub(crate) fn subsets<T: Clone>(items: &[T]) -> Result<Vec<Vec<T>>, TransportError> {
    if items.len() > SUBSET_CAP {
        return Err(TransportError::TooLarge(format!(
            "{} candidates exceed the cap of {SUBSET_CAP}",
            items.len()
        )));
    }
    let mut masks: Vec<u32> = (0..(1u32 << items.len())).collect();
    masks.sort_by_key(|m| (m.count_ones(), *m));
    Ok(masks
        .into_iter()
        .map(|m| {
            items
                .iter()
                .enumerate()
                .filter(|(i, _)| (m >> i) & 1 == 1)
                .map(|(_, v)| v.clone())
                .collect()
        })
        .collect())
}

pub(crate) fn as_refs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}
