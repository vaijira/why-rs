# Causal reinforcement learning

[crates/why-data/src/crl.rs](../../crates/why-data/src/crl.rs) implements chapter 8,
sections 8.1–8.2.3.1. Like `fairness`, it has no new dependency, so it is
WASM-safe. It reuses `fairness::Dataset` (re-exported as `crl::Dataset`) and
`fairness::SplitMix64`.

| File | Book |
| --- | --- |
| `space.rs` | `PolicySpace` (Def. 8.1.1, `validate` checks the non-descendant constraints), `Reward` (Def. 8.1.2) |
| `policy.rs` | `Policy` trait: `Deterministic`, `Stochastic`, `Uniform` (π_UNIF), `TabularPolicy` |
| `conditions.rs` | `nuc` (Def. 8.2.1, on an `scm::Model`), `sequential_backdoor` (Def. 8.2.3, on an `Admg`) |
| `offpolicy.rs` | `ipw` (Thm 8.2.1), `dp` (Thm 8.2.2), `optimize` (backward induction or enumeration) |
| `cdm.rs` | `Cdm` (Def. 8.1.3, exact evaluation and planning, also an `Environment`), `Task` (Def. 8.1.4) with `run` = Algorithm 19 |
| `online.rs` | `Rct` (Algorithm 20), `Ucb` (Algorithm 21), Thm 8.2.4/8.2.5 bounds, Eq. 8.101 |

Facts that are easy to get wrong:

- **Propensities and Q-functions condition on the history
  `X̄_{i−1} ∪ S̄_i`**, not on `S_i`. `PolicySpace` keeps one temporal ordering
  of all `S`/`X` variables, so every history is a prefix of it. The
  estimators key cells by prefix.
- **IPW and DP are the same plug-in functional.** They agree to 1e-10 on any
  data set, whether or not NUC or SBC holds. The random-SCM test asserts
  this, and that SBC ⇒ both equal `Cdm::evaluate` (Thm 8.2.8),
  NUC ⇒ SBC (Cor. 8.2.7), and that data under any policy evaluate any other
  (Lemma 8.2.3).
- **`Cdm::optimal` runs `optimize` on the exact table under π_UNIF.** That
  table satisfies Exp-NUC and full positivity, so the plan is exact.
  `optimize` uses backward induction only with perfect recall
  (`S_i = X̄_{i−1} ∪ S̄_i`). Otherwise it enumerates deterministic policies,
  capped at 200 000.
- **Positivity is an error, not a dropped term**: `CrlError::Positivity`
  names the stage, history and action. Rules must sum to 1 over the levels
  the data shows for the action.
- The DTR of Eq. 8.2 has continuous noise, so it cannot be an `scm::Model`.
  The tests and the example build its exact table in closed form. Y has an
  extra logistic noise `U5` (listed in Eq. 8.80, missing from Eq. 8.2),
  without which Table 8.5 cannot be reproduced. Its exact optimum is
  0.67574. The book prints 0.6758, which comes from rounded Q-values.

Book errata found while porting: Eq. 8.119 should read `Y ← S1 ⊕ X2 ⊕ U3`,
not `S1 ⊕ X1 ⊕ U3`. Tables 8.7–8.8, Eq. 8.124 and the text's back-door path
all need X2. Eq. 8.101's "N ≈ 878" uses `log₁₀`; the natural log, which the
derivation implies, gives 2021.

Not implemented: do-calculus policy identification (Sec. 8.2.3.2). When SBC
fails, `Task::run` returns `NotIdentifiable`. Infinite-horizon MDP planning
(Ex. 8.5) exists only for tabular MDPs, through chapter 9's
`CtfMdp::experimental_value_iteration`, which reproduces Table 8.1.
