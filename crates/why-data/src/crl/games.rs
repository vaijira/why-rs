//! Causal game theory (Sec. 9.6): causal normal form games, the PCH layer
//! selection game, causal Nash equilibria (Algorithm 30) and learning them
//! from counterfactual randomization (Algorithm 31). Two players.

use std::collections::BTreeMap;
use std::fmt;

use super::cdm::{endogenous_domains, evaluate_world};
use super::lp::{Cmp, Lp, LpOutcome};
use super::CrlError;
use crate::scm::{Model, Value};

/// The most strategies per player [`Bimatrix::nash_equilibria`] accepts.
const STRATEGY_CAP: usize = 12;

/// A two-player game in normal form: `payoffs[i][j] = (u₁, u₂)` when the row
/// player plays `i` and the column player `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bimatrix {
    /// The row player's strategies.
    pub rows: Vec<String>,
    /// The column player's strategies.
    pub cols: Vec<String>,
    /// `(u₁, u₂)` per cell.
    pub payoffs: Vec<Vec<(f64, f64)>>,
}

/// A Nash equilibrium in mixed strategies.
#[derive(Debug, Clone, PartialEq)]
pub struct Equilibrium {
    /// The row player's mixed strategy.
    pub row: Vec<f64>,
    /// The column player's mixed strategy.
    pub col: Vec<f64>,
    /// The expected payoffs `(u₁, u₂)`.
    pub payoff: (f64, f64),
}

impl Equilibrium {
    /// The indices of the strategies played with positive probability by
    /// the row and the column player.
    #[must_use]
    pub fn support(&self) -> (Vec<usize>, Vec<usize>) {
        let s = |p: &[f64]| p.iter().enumerate().filter(|(_, q)| **q > 1e-9).map(|(i, _)| i).collect();
        (s(&self.row), s(&self.col))
    }
}

const TOL: f64 = 1e-9;

impl Bimatrix {
    /// Builds a game, checking that the payoff table matches the strategies.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] for a ragged or empty table.
    pub fn new(
        rows: Vec<String>,
        cols: Vec<String>,
        payoffs: Vec<Vec<(f64, f64)>>,
    ) -> Result<Self, CrlError> {
        if rows.is_empty() || cols.is_empty() || payoffs.len() != rows.len()
            || payoffs.iter().any(|r| r.len() != cols.len())
        {
            return Err(CrlError::InvalidInput("the payoff table does not match the strategies".into()));
        }
        Ok(Self { rows, cols, payoffs })
    }

    /// The subgame on the given row and column indices.
    #[must_use]
    pub fn restrict(&self, rows: &[usize], cols: &[usize]) -> Self {
        Self {
            rows: rows.iter().map(|&i| self.rows[i].clone()).collect(),
            cols: cols.iter().map(|&j| self.cols[j].clone()).collect(),
            payoffs: rows
                .iter()
                .map(|&i| cols.iter().map(|&j| self.payoffs[i][j]).collect())
                .collect(),
        }
    }

    fn expected(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let mut u = (0.0, 0.0);
        for (i, xi) in x.iter().enumerate() {
            for (j, yj) in y.iter().enumerate() {
                u.0 += xi * yj * self.payoffs[i][j].0;
                u.1 += xi * yj * self.payoffs[i][j].1;
            }
        }
        u
    }

    /// The pure equilibria, in row-major order.
    #[must_use]
    pub fn pure_equilibria(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for i in 0..self.rows.len() {
            for j in 0..self.cols.len() {
                let (u1, u2) = self.payoffs[i][j];
                let row_best = (0..self.rows.len()).all(|k| self.payoffs[k][j].0 <= u1 + TOL);
                let col_best = (0..self.cols.len()).all(|k| self.payoffs[i][k].1 <= u2 + TOL);
                if row_best && col_best {
                    out.push((i, j));
                }
            }
        }
        out
    }

    /// Every Nash equilibrium found by support enumeration: for each pair of
    /// supports, a linear feasibility problem finds strategies that make
    /// every supported strategy a best response. Degenerate games are
    /// handled because supports of different sizes are tried too; a
    /// continuum of equilibria contributes its vertices. Duplicates are
    /// removed. Pure equilibria come first.
    ///
    /// # Errors
    ///
    /// [`CrlError::TooManyPolicies`] beyond 12 strategies per player.
    pub fn nash_equilibria(&self) -> Result<Vec<Equilibrium>, CrlError> {
        let (m, n) = (self.rows.len(), self.cols.len());
        if m > STRATEGY_CAP || n > STRATEGY_CAP {
            return Err(CrlError::TooManyPolicies {
                count: (m * n) as f64,
                cap: STRATEGY_CAP * STRATEGY_CAP,
            });
        }
        let mut out: Vec<Equilibrium> = Vec::new();
        let push = |eq: Equilibrium, out: &mut Vec<Equilibrium>| {
            let same = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-7);
            if !out.iter().any(|e| same(&e.row, &eq.row) && same(&e.col, &eq.col)) {
                out.push(eq);
            }
        };
        for (i, j) in self.pure_equilibria() {
            let mut row = vec![0.0; m];
            let mut col = vec![0.0; n];
            row[i] = 1.0;
            col[j] = 1.0;
            let payoff = self.payoffs[i][j];
            push(Equilibrium { row, col, payoff }, &mut out);
        }
        let masks = |k: usize| -> Vec<u32> {
            let mut v: Vec<u32> = (1..(1u32 << k)).collect();
            v.sort_by_key(|m| m.count_ones());
            v
        };
        for rs in masks(m) {
            for cs in masks(n) {
                let row_support: Vec<usize> = (0..m).filter(|i| rs >> i & 1 == 1).collect();
                let col_support: Vec<usize> = (0..n).filter(|j| cs >> j & 1 == 1).collect();
                // y on the column support makes every supported row optimal.
                let Some(col) = self.indifference(&col_support, &row_support, true) else {
                    continue;
                };
                let Some(row) = self.indifference(&row_support, &col_support, false) else {
                    continue;
                };
                let payoff = self.expected(&row, &col);
                push(Equilibrium { row, col, payoff }, &mut out);
            }
        }
        Ok(out)
    }

    /// A mixed strategy of one player on `own` that makes every strategy in
    /// `other` a best response for the opponent, or `None`.
    fn indifference(&self, own: &[usize], other: &[usize], own_is_col: bool) -> Option<Vec<f64>> {
        let (n_own, n_other) = if own_is_col {
            (self.cols.len(), self.rows.len())
        } else {
            (self.rows.len(), self.cols.len())
        };
        // Variables: p over own support, then u = u⁺ − u⁻.
        let k = own.len();
        let mut lp = Lp::new(k + 2);
        let payoff = |o: usize, t: usize| {
            if own_is_col {
                self.payoffs[t][o].0
            } else {
                self.payoffs[o][t].1
            }
        };
        for t in 0..n_other {
            let mut a: Vec<f64> = own.iter().map(|&o| payoff(o, t)).collect();
            a.push(-1.0);
            a.push(1.0);
            let cmp = if other.contains(&t) { Cmp::Eq } else { Cmp::Le };
            lp.constraint(a, cmp, 0.0);
        }
        let mut sum = vec![1.0; k];
        sum.extend([0.0, 0.0]);
        lp.constraint(sum, Cmp::Eq, 1.0);
        match lp.solve() {
            LpOutcome::Optimal { x, .. } => {
                let mut p = vec![0.0; n_own];
                for (idx, &o) in own.iter().enumerate() {
                    p[o] = x[idx];
                }
                Some(p)
            }
            _ => None,
        }
    }

    /// The equilibrium [`learn_cne`] uses for a game's value: the first pure
    /// equilibrium in row-major order, or else the first mixed one found.
    ///
    /// The book leaves the choice among several equilibria open (Def. 9.6.8
    /// takes "a Nash equilibrium payoff"). Row-major order reproduces
    /// Tables 9.15 and 9.37(b).
    ///
    /// # Errors
    ///
    /// As [`Bimatrix::nash_equilibria`].
    pub fn equilibrium(&self) -> Result<Equilibrium, CrlError> {
        Ok(self
            .nash_equilibria()?
            .into_iter()
            .next()
            .expect("every finite game has an equilibrium"))
    }
}

impl fmt::Display for Bimatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w0 = self.rows.iter().map(String::len).max().unwrap_or(0);
        write!(f, "{:w0$}", "")?;
        for c in &self.cols {
            write!(f, "  {c:>18}")?;
        }
        for (r, row) in self.rows.iter().zip(&self.payoffs) {
            writeln!(f)?;
            write!(f, "{r:w0$}")?;
            for (a, b) in row {
                write!(f, "  {:>18}", format!("({a:.4}, {b:.4})"))?;
            }
        }
        Ok(())
    }
}

/// The PCH layer a strategy lives in (Defs. 9.6.2–9.6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    /// `L1`: act on the natural mechanism, `X ← f_X(·)` (`a₀`).
    Natural,
    /// `L2`: a hard intervention `do(x)`.
    Do,
    /// `L3`: any other map `X ← h(X′)` from the intuition to the action.
    Counterfactual,
}

/// The action spaces a player may restrict itself to (Def. 9.6.6):
/// `A¹`, `A²`, `A¹ ∪ A²` or `A³`. `A³` contains every map `h : D(X) → D(X)`,
/// including the identity (`A¹`) and the constants (`A²`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionSpace {
    /// `A¹ = {a₀}`.
    L1,
    /// `A² = {do(x)}`.
    L2,
    /// `A¹ ∪ A²`.
    L1L2,
    /// `A³`.
    L3,
}

impl ActionSpace {
    /// All four, in the book's order.
    pub const ALL: [Self; 4] = [Self::L1, Self::L2, Self::L1L2, Self::L3];

    /// Whether a strategy of `layer` belongs to this space.
    #[must_use]
    pub fn contains(self, layer: Layer) -> bool {
        match self {
            Self::L1 => layer == Layer::Natural,
            Self::L2 => layer == Layer::Do,
            Self::L1L2 => layer != Layer::Counterfactual,
            Self::L3 => true,
        }
    }

    /// The spaces contained in this one, in the book's order.
    #[must_use]
    pub fn subspaces(self) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|s| match self {
                Self::L1 => *s == Self::L1,
                Self::L2 => *s == Self::L2,
                Self::L1L2 => *s != Self::L3,
                Self::L3 => true,
            })
            .collect()
    }
}

impl fmt::Display for ActionSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::L1 => "A1",
            Self::L2 => "A2",
            Self::L1L2 => "A1∪A2",
            Self::L3 => "A3",
        })
    }
}

/// A bimatrix over each player's full strategy set, every strategy tagged
/// with its PCH layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayeredGame {
    /// The game over all strategies.
    pub game: Bimatrix,
    /// The layer of each row strategy.
    pub row_layers: Vec<Layer>,
    /// The layer of each column strategy.
    pub col_layers: Vec<Layer>,
}

impl LayeredGame {
    /// Builds a layered game.
    ///
    /// # Errors
    ///
    /// [`CrlError::InvalidInput`] if the layers do not match the strategies.
    pub fn new(game: Bimatrix, row_layers: Vec<Layer>, col_layers: Vec<Layer>) -> Result<Self, CrlError> {
        if row_layers.len() != game.rows.len() || col_layers.len() != game.cols.len() {
            return Err(CrlError::InvalidInput("one layer per strategy".into()));
        }
        Ok(Self {
            game,
            row_layers,
            col_layers,
        })
    }

    fn indices(layers: &[Layer], space: ActionSpace) -> Vec<usize> {
        layers
            .iter()
            .enumerate()
            .filter(|(_, l)| space.contains(**l))
            .map(|(i, _)| i)
            .collect()
    }

    /// The PCH projection `Γ(A₁, A₂)` (Def. 9.6.7), or `None` if a player
    /// would have no strategy left.
    #[must_use]
    pub fn projection(&self, row: ActionSpace, col: ActionSpace) -> Option<Bimatrix> {
        let r = Self::indices(&self.row_layers, row);
        let c = Self::indices(&self.col_layers, col);
        (!r.is_empty() && !c.is_empty()).then(|| self.game.restrict(&r, &c))
    }
}

/// The output of [`learn_cne`].
#[derive(Debug, Clone, PartialEq)]
pub struct CausalNash {
    /// The PCH layer selection game `L_Γ` (Def. 9.6.8).
    pub selection: Bimatrix,
    /// The equilibrium `s*` of `L_Γ`.
    pub selection_equilibrium: Equilibrium,
    /// `A*_i = ⋃_{A ∈ supp(s*_i)} A`, as the strategy indices of the full
    /// game.
    pub strategies: (Vec<usize>, Vec<usize>),
    /// The projection `Γ(A*)`.
    pub game: Bimatrix,
    /// Its equilibrium: the causal Nash equilibrium (Def. 9.6.9).
    pub equilibrium: Equilibrium,
}

/// Algorithm 30, Learn-CNE, for players whose available spaces are
/// `available`.
///
/// It builds the layer selection game, whose cells are the equilibrium
/// payoffs of the PCH projections, solves it, unions the action spaces in
/// the support of each player's equilibrium strategy, and returns the
/// equilibrium of that projection. Equilibria are chosen by
/// [`Bimatrix::equilibrium`].
///
/// # Errors
///
/// As [`Bimatrix::nash_equilibria`], plus [`CrlError::InvalidInput`] if a
/// player has no strategy in some space it may choose.
pub fn learn_cne(
    game: &LayeredGame,
    available: (ActionSpace, ActionSpace),
) -> Result<CausalNash, CrlError> {
    let rows: Vec<ActionSpace> = available
        .0
        .subspaces()
        .into_iter()
        .filter(|s| !LayeredGame::indices(&game.row_layers, *s).is_empty())
        .collect();
    let cols: Vec<ActionSpace> = available
        .1
        .subspaces()
        .into_iter()
        .filter(|s| !LayeredGame::indices(&game.col_layers, *s).is_empty())
        .collect();
    if rows.is_empty() || cols.is_empty() {
        return Err(CrlError::InvalidInput("a player has no strategy".into()));
    }
    let mut payoffs = Vec::with_capacity(rows.len());
    for r in &rows {
        let mut line = Vec::with_capacity(cols.len());
        for c in &cols {
            let sub = game.projection(*r, *c).expect("non-empty spaces");
            line.push(sub.equilibrium()?.payoff);
        }
        payoffs.push(line);
    }
    let selection = Bimatrix::new(
        rows.iter().map(ToString::to_string).collect(),
        cols.iter().map(ToString::to_string).collect(),
        payoffs,
    )?;
    let selection_equilibrium = selection.equilibrium()?;
    let (rs, cs) = selection_equilibrium.support();
    let union = |layers: &[Layer], spaces: &[ActionSpace], picked: &[usize]| -> Vec<usize> {
        (0..layers.len())
            .filter(|i| picked.iter().any(|&p| spaces[p].contains(layers[*i])))
            .collect()
    };
    let r_idx = union(&game.row_layers, &rows, &rs);
    let c_idx = union(&game.col_layers, &cols, &cs);
    let projected = game.game.restrict(&r_idx, &c_idx);
    let equilibrium = projected.equilibrium()?;
    Ok(CausalNash {
        selection,
        selection_equilibrium,
        strategies: (r_idx, c_idx),
        game: projected,
        equilibrium,
    })
}

/// A map `h : D(X) → D(X)` from the intuition to the action, as the image of
/// each domain value in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strategy {
    /// `h(d)` for each value `d` of the domain, in domain order.
    pub map: Vec<usize>,
    /// Its layer.
    pub layer: Layer,
}

/// Every map of a `k`-value domain: the identity first, then the constants,
/// then the rest lexicographically.
fn all_maps(k: usize) -> Vec<Strategy> {
    let mut maps: Vec<Vec<usize>> = vec![Vec::new()];
    for _ in 0..k {
        maps = maps
            .into_iter()
            .flat_map(|m| {
                (0..k).map(move |v| {
                    let mut m = m.clone();
                    m.push(v);
                    m
                })
            })
            .collect();
    }
    let identity: Vec<usize> = (0..k).collect();
    let mut out = vec![Strategy {
        map: identity.clone(),
        layer: Layer::Natural,
    }];
    for v in 0..k {
        let constant = vec![v; k];
        if constant != identity {
            out.push(Strategy {
                map: constant,
                layer: Layer::Do,
            });
        }
    }
    for m in maps {
        if m != identity && m.iter().any(|v| *v != m[0]) {
            out.push(Strategy {
                map: m,
                layer: Layer::Counterfactual,
            });
        }
    }
    out
}

fn strategy_name(action: &str, s: &Strategy, domain: &[String]) -> String {
    match s.layer {
        Layer::Natural => format!("{action} = {action}′"),
        Layer::Do => format!("do({action} = {})", domain[s.map[0]]),
        Layer::Counterfactual => {
            let parts: Vec<String> = s
                .map
                .iter()
                .enumerate()
                .map(|(d, v)| format!("{}→{}", domain[d], domain[*v]))
                .collect();
            format!("{action} = h({action}′): {}", parts.join(", "))
        }
    }
}

/// A two-player causal normal form game built from an SCM (Def. 9.6.6):
/// each player owns one action variable and one reward variable, and may
/// play any map from its intuition to its action.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalGame {
    /// `(action, reward)` per player.
    pub players: [(String, String); 2],
    /// The action domains, rendered.
    pub domains: [Vec<String>; 2],
    /// Each player's strategies, the identity first, then the constants.
    pub strategies: [Vec<Strategy>; 2],
    /// The game over all `L3` strategies.
    pub layered: LayeredGame,
}

impl CausalGame {
    /// Computes the payoff of every pair of `L3` strategies exactly:
    ///
    /// ```text
    /// u(h₁, h₂) = Σ_u P(u) Y_{X₁ = h₁(X₁(u)), X₂ = h₂(X₂(u))}(u)
    /// ```
    ///
    /// where the intuitions `X_i(u)` come from the natural world and the
    /// rewards from the world where both players act (Eq. 9.296).
    ///
    /// # Errors
    ///
    /// [`CrlError::Unknown`] for a variable the model lacks,
    /// [`CrlError::InvalidInput`] for a shared action variable, and
    /// [`CrlError::Scm`] if an equation cannot be evaluated.
    pub fn from_model(model: &Model, players: [(&str, &str); 2]) -> Result<Self, CrlError> {
        let domains = endogenous_domains(model)?;
        for (x, y) in players {
            for v in [x, y] {
                if !domains.contains_key(v) {
                    return Err(CrlError::Unknown(v.to_string()));
                }
            }
        }
        if players[0].0 == players[1].0 {
            return Err(CrlError::InvalidInput("the players share an action".into()));
        }
        let dom: [Vec<Value>; 2] = [domains[players[0].0].clone(), domains[players[1].0].clone()];
        let strategies = [all_maps(dom[0].len()), all_maps(dom[1].len())];
        let n = [strategies[0].len(), strategies[1].len()];
        let mut payoffs = vec![vec![(0.0, 0.0); n[1]]; n[0]];
        for (u, p) in model.exogenous_assignments() {
            if p <= 0.0 {
                continue;
            }
            let natural = evaluate_world(model, &u, &[])?;
            let intuition = [
                dom[0].iter().position(|v| *v == natural[players[0].0]).expect("in domain"),
                dom[1].iter().position(|v| *v == natural[players[1].0]).expect("in domain"),
            ];
            // Cache the outcome of each pair of realised actions.
            let mut outcome: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
            for (i, s1) in strategies[0].iter().enumerate() {
                for (j, s2) in strategies[1].iter().enumerate() {
                    let a = (s1.map[intuition[0]], s2.map[intuition[1]]);
                    let y = match outcome.get(&a) {
                        Some(y) => *y,
                        None => {
                            let world = evaluate_world(
                                model,
                                &u,
                                &[(players[0].0, dom[0][a.0]), (players[1].0, dom[1][a.1])],
                            )?;
                            let y = (
                                world[players[0].1].as_int() as f64,
                                world[players[1].1].as_int() as f64,
                            );
                            outcome.insert(a, y);
                            y
                        }
                    };
                    payoffs[i][j].0 += p * y.0;
                    payoffs[i][j].1 += p * y.1;
                }
            }
        }
        let names: [Vec<String>; 2] = [
            dom[0].iter().map(ToString::to_string).collect(),
            dom[1].iter().map(ToString::to_string).collect(),
        ];
        let game = Bimatrix::new(
            strategies[0]
                .iter()
                .map(|s| strategy_name(players[0].0, s, &names[0]))
                .collect(),
            strategies[1]
                .iter()
                .map(|s| strategy_name(players[1].0, s, &names[1]))
                .collect(),
            payoffs,
        )?;
        let layered = LayeredGame::new(
            game,
            strategies[0].iter().map(|s| s.layer).collect(),
            strategies[1].iter().map(|s| s.layer).collect(),
        )?;
        Ok(Self {
            players: [
                (players[0].0.to_string(), players[0].1.to_string()),
                (players[1].0.to_string(), players[1].1.to_string()),
            ],
            domains: names,
            strategies,
            layered,
        })
    }

    /// Algorithm 30 on this game with both players allowed `available`.
    ///
    /// # Errors
    ///
    /// As [`learn_cne`].
    pub fn causal_nash(&self, available: (ActionSpace, ActionSpace)) -> Result<CausalNash, CrlError> {
        learn_cne(&self.layered, available)
    }
}

/// The mixture components of one cell: `(weight, (mean y₁, mean y₂))`.
pub type Components = Vec<(f64, (f64, f64))>;

/// One sample of the counterfactual randomization of Algorithm 31: player
/// 1's intuition and both realised actions, with both rewards.
#[derive(Debug, Clone, PartialEq)]
pub struct CtfGameSample {
    /// Player 1's intended action `x₁′`.
    pub intended: String,
    /// Player 1's realised action `x₁`.
    pub action1: String,
    /// Player 2's realised action `x₂`.
    pub action2: String,
    /// `(y₁, y₂)`.
    pub reward: (f64, f64),
}

/// The output of [`ctf_nash_learning`].
#[derive(Debug, Clone, PartialEq)]
pub struct CtfNashLearned {
    /// The mixture components per cell `(x₁′, x₁, x₂)`: `(weight, mean)`,
    /// heaviest first.
    pub components: BTreeMap<(String, String, String), Components>,
    /// The learned game: rows are player 1's maps `f : X₁′ → X₁`, columns
    /// player 2's maps `g : [k] → X₂` (Eq. 9.301), with the payoffs of
    /// Eq. 9.302.
    pub game: LayeredGame,
    /// Learn-CNE on it. Player 1 chooses among all four action spaces.
    /// Player 2 has no natural strategy, because its intuitions are known
    /// only up to a permutation of the components, so its choice is
    /// between the constants (`A²`) and every map (`A³`).
    pub cne: CausalNash,
}

/// Algorithm 31, Ctf-Nash-Learning, from player 1's point of view, for
/// deterministic rewards (Ex. 9.34).
///
/// Data come from both players running Ctf-RCT. For a fixed
/// `(x₁′, x₁, x₂)` the reward is a mixture over player 2's hidden intuition
/// `x₂′`, with weights `P(x₂′ | x₁′)` and point masses at
/// `Y_{x₁,x₂} | x₁′, x₂′`. Each distinct reward value is one component
/// (step 3), components are ordered by weight, and a cell that shows other
/// than `k = |D(X₂)|` distinct values falls back to step 4: its mean for
/// every component, with weights borrowed from a cell sharing `x₁′`, or
/// `1/k`. Step 6 then fills the payoff of every pair `(f, g)`.
///
/// Ordering by weight labels components consistently across cells only
/// when the weights `P(x₂′ | x₁′)` are distinct (Assumption 9.6.6).
///
/// # Errors
///
/// [`CrlError::InvalidInput`] for empty data or an empty domain, and as
/// [`learn_cne`].
pub fn ctf_nash_learning(
    samples: &[CtfGameSample],
    domain1: &[String],
    domain2: &[String],
) -> Result<CtfNashLearned, CrlError> {
    if samples.is_empty() || domain1.is_empty() || domain2.is_empty() {
        return Err(CrlError::InvalidInput("Ctf-Nash-Learning needs data and domains".into()));
    }
    let k = domain2.len();
    let pos = |d: &[String], v: &str| {
        d.iter()
            .position(|w| w == v)
            .ok_or_else(|| CrlError::InvalidInput(format!("value `{v}` is outside the domain")))
    };
    let mut p_intent = vec![0.0; domain1.len()];
    let mut cells: BTreeMap<(usize, usize, usize), Vec<(f64, f64)>> = BTreeMap::new();
    for s in samples {
        let c = (pos(domain1, &s.intended)?, pos(domain1, &s.action1)?, pos(domain2, &s.action2)?);
        p_intent[c.0] += 1.0;
        cells.entry(c).or_default().push(s.reward);
    }
    let total: f64 = p_intent.iter().sum();
    for p in &mut p_intent {
        *p /= total;
    }

    // Step 3: distinct values as components, heaviest first.
    let mut comps: BTreeMap<(usize, usize, usize), Option<Components>> = BTreeMap::new();
    let mut means: BTreeMap<(usize, usize, usize), (f64, f64)> = BTreeMap::new();
    for (c, ys) in &cells {
        let n = ys.len() as f64;
        let mut distinct: Vec<((f64, f64), f64)> = Vec::new();
        for y in ys {
            match distinct.iter_mut().find(|(v, _)| (v.0 - y.0).abs() < 1e-9 && (v.1 - y.1).abs() < 1e-9) {
                Some((_, cnt)) => *cnt += 1.0,
                None => distinct.push((*y, 1.0)),
            }
        }
        distinct.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.0.total_cmp(&b.0.0)).then(a.0.1.total_cmp(&b.0.1)));
        means.insert(
            *c,
            (ys.iter().map(|y| y.0).sum::<f64>() / n, ys.iter().map(|y| y.1).sum::<f64>() / n),
        );
        comps.insert(
            *c,
            (distinct.len() == k).then(|| distinct.into_iter().map(|(v, cnt)| (cnt / n, v)).collect()),
        );
    }
    // Step 4: fall back for cells that do not show k components.
    let mut resolved: BTreeMap<(usize, usize, usize), Components> = BTreeMap::new();
    for (c, comp) in &comps {
        let v = match comp {
            Some(v) => v.clone(),
            None => {
                let borrowed = comps
                    .iter()
                    .find(|(c2, v2)| c2.0 == c.0 && v2.is_some())
                    .and_then(|(_, v2)| v2.as_ref())
                    .map(|v2| v2.iter().map(|(w, _)| *w).collect::<Vec<f64>>());
                let weights = borrowed.unwrap_or_else(|| vec![1.0 / k as f64; k]);
                weights.into_iter().map(|w| (w, means[c])).collect()
            }
        };
        resolved.insert(*c, v);
    }

    // Steps 5–6: the payoff of every (f, g).
    let fs = all_maps(domain1.len());
    let gs: Vec<Strategy> = {
        let mut maps: Vec<Vec<usize>> = vec![Vec::new()];
        for _ in 0..k {
            maps = maps
                .into_iter()
                .flat_map(|m| {
                    (0..domain2.len()).map(move |v| {
                        let mut m = m.clone();
                        m.push(v);
                        m
                    })
                })
                .collect();
        }
        let mut constants: Vec<Strategy> = Vec::new();
        let mut others: Vec<Strategy> = Vec::new();
        for m in maps {
            if m.iter().all(|v| *v == m[0]) {
                constants.push(Strategy { map: m, layer: Layer::Do });
            } else {
                others.push(Strategy {
                    map: m,
                    layer: Layer::Counterfactual,
                });
            }
        }
        constants.extend(others);
        constants
    };
    let mut payoffs = vec![vec![(0.0, 0.0); gs.len()]; fs.len()];
    for (i, f) in fs.iter().enumerate() {
        for (j, g) in gs.iter().enumerate() {
            let mut u = (0.0, 0.0);
            for (xp, pxp) in p_intent.iter().enumerate() {
                if *pxp == 0.0 {
                    continue;
                }
                for comp in 0..k {
                    let c = (xp, f.map[xp], g.map[comp]);
                    if let Some(v) = resolved.get(&c) {
                        let (w, m) = v[comp];
                        u.0 += pxp * w * m.0;
                        u.1 += pxp * w * m.1;
                    }
                }
            }
            payoffs[i][j] = u;
        }
    }
    let row_names = fs.iter().map(|s| strategy_name("X1", s, domain1)).collect();
    let col_names = gs
        .iter()
        .map(|s| {
            let parts: Vec<String> = s
                .map
                .iter()
                .enumerate()
                .map(|(c, v)| format!("{}→{}", c + 1, domain2[*v]))
                .collect();
            format!("g: {}", parts.join(", "))
        })
        .collect();
    let game = LayeredGame::new(
        Bimatrix::new(row_names, col_names, payoffs)?,
        fs.iter().map(|s| s.layer).collect(),
        gs.iter().map(|s| s.layer).collect(),
    )?;
    let cne = learn_cne(&game, (ActionSpace::L3, ActionSpace::L3))?;
    let components = resolved
        .into_iter()
        .map(|((a, b, c), v)| ((domain1[a].clone(), domain1[b].clone(), domain2[c].clone()), v))
        .collect();
    Ok(CtfNashLearned {
        components,
        game,
        cne,
    })
}
