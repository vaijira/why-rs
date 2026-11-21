# Causal fairness

[crates/why-data/src/fairness.rs](../../crates/why-data/src/fairness.rs) implements
chapter 6, section 6.1, on tabular data. There is no dependency, so it is
WASM-safe, with its own CSV reader, RNG and transport solver. `X`/`Z`/`W`
are categorical (multi-column `Z`/`W` use joint levels), and `Y` is numeric.

| File | Book |
| --- | --- |
| `estimate.rs` | x-specific (both transitions + symmetric, Def. 6.1.32), z-specific (Def. 6.1.19), general TE/NDE/NIE/Exp-SE |
| `cookbook.rs` | Algorithm 15. Decision rule is "bootstrap CI excludes 0"; z-specific follow-ups are Bonferroni-corrected across strata |
| `predict.rs` | Thm 6.1.32 in-processing, solved exactly (see below); `Constraints::TotalVariation` is the FPT baseline |
| `transport.rs` | Algorithm 16 (Causal IF): min-cost-flow OT, monotone coupling for 1-D numeric |
| `decision.rs` | Algorithms 17 and 18, `estimate_benefit`, Prop. 6.1.40 bounds |

Departures from the text, each deliberate:

- **Signs follow Plečko & Bareinboim**, so `TV = DE − IE − SE` and
  `TV = DE^sym + IE^sym − SE` hold exactly. `ie_sym` has the sign of the
  `x0 → x1` transition. Summaries that flip the brackets of x-IE/x-SE are
  inconsistent with the identity they quote. Eq. 6.305 has a typo (it should
  read `x-IE_{x0,x1}`).
- **In-processing with W ∈ BN matches `x-IE^sym`, not both IE transitions.**
  The two TV decompositions imply `IE₀₁ + IE₁₀ = DE₀₁ + DE₁₀`. Once both DEs
  are forced to 0, matching each IE to `Y` is infeasible in general. With
  discrete covariates every ID constraint is linear in the cell table, so the
  program is solved in closed form by KKT. The five causal constraints have
  rank 4, which `solve_consistent` tolerates.
- **Causal IF is evaluated on the transformed data** `(X, Z̃, W̃, Ỹ)`, not on
  `(X, Z, W, Ỹ)`. The theorem only holds on the former. `transform` is exact
  (a weighted table); `transform_sampled` draws one value per row.
- **Algorithms 17/18 never treat `Δ ≤ 0`.** This follows the optimality
  statement of Prop. 6.1.41, so Eq. 6.493's budget equality becomes `≤ b`.
  Algorithm 18 maximises `E[Δ·d]`, not Eq. 6.494's sum of conditional means.
  It enumerates the vertices of the feasible polygon cut by the gain
  breakpoints, so it is exact.
- `Measures`/`ZEffects`/`GeneralMeasures` snap differences within 1e-9 of the
  outcome scale to 0. Without that, a measure that vanishes by construction
  comes out as ±1e-17 with an equally tiny bootstrap SE, and gets flagged.

Tests to run after touching a file: `estimands_match_brute_force_counterfactuals`
(estimate.rs, checks all five counterfactual means against noise
enumeration), `in_processing_meets_its_constraints` (predict.rs),
`causal_if_zeroes_the_unpermitted_pathways` and
`transport_solver_is_optimal_in_one_dimension` (transport.rs), and
`algorithm_18_is_optimal_and_respects_the_gap` (decision.rs, against a grid
search). `startup_hiring_nde_vanishes_but_z_de_does_not` is Ex. 6.9/6.11 with
exact numbers.

There is no chapter 6 code in the book's companion Python (`causalai-book`),
so nothing was ported from it; the source is section 6.1.4 of the book
(pp. 450–487 of the Oct 2025 draft). Ex. 6.21 (college admissions over time) is not ported because the
book gives no initial values for κ, λ, α, β.

Not implemented here: BN-sets over individual columns rather than whole blocks, principal
fairness (Def. 6.1.36, which the book argues against), and continuous
covariates (bin them).
