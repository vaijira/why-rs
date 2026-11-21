# Image counterfactuals (chapter 14)

Chapter 14 is split like chapter 13. The tensor-free part is
[crates/why-data/src/ascm.rs](../../crates/why-data/src/ascm.rs). It adds no
dependency and builds for WASM.

| File | Book |
| --- | --- |
| `model.rs` | `Ascm` (Def. 14.1.1) on an `scm::Model` whose pixels are endogenous variables, the labeller `h` (Eq. 14.1), `ImageCounterfactual` = `P(I, I_{x′})` (Eq. 14.4, Fig. 14.5) with the F-ctf query `λ_φ` (Def. 14.3.1) |
| `bounds.rs` | `optimal_bounds`: the optimal bounds of Def. 14.3.2 (2) from `P(V)` and `G`, by the canonical model |
| `consistency.rs` | `ctf_consistency` (Def. 14.3.2) and `total_variation` |

Algorithm 50 (ANCM) is
[crates/why-ncm/src/ancm.rs](../../crates/why-ncm/src/ancm.rs): `ancm` trains
it on `ImageData`. `Ancm::sample` draws image pairs from `P̂(I, I_{x′})` and
`Ancm::edit` edits a given image. `Ncm::distribution` (in `identify.rs`) is
the fitted `P̂(V)` that Def. 14.3.2 (1) compares.

Facts that are easy to get wrong:

- **The bounds come from a canonical model.** Each c-component gets a
  joint distribution over its variables' response types, constrained by
  Tian's c-factor `Q[C]`. A lone variable keeps only the parent
  configurations the query reads, which is exact. A larger component keeps
  all of them, capped at 50 000 types. Each component's cells are
  bracketed by two LPs (`crl::lp`, now `pub(crate)`). Components whose
  cells are all pinned contribute constants. At most one free component
  may remain, or the program is multilinear and the result is
  `AscmError::Nonlinear`. Every book example leaves one.
- **Tight only for clique c-components.** When a district has a
  non-adjacent pair, `q_C` may couple what `G` keeps independent, and the
  interval is a valid outer bound. The random test
  `bounds_contain_the_truth_and_collapse_when_identified` asserts
  containment on all graphs, and collapse of `ctf_infer`-identified queries
  only on clique graphs. Run it after touching `bounds.rs`.
- **Evidence must be factual** (`CounterfactualEvidence` otherwise), so the
  denominator is a constant `P(w)` (Eq. 14.18). `P(V)` must be exact and
  satisfy `G`. A violation inside a component is `Incompatible`, and one
  across components goes unnoticed.
- **ANCM's posterior for `Û_C` is exact, not encoded.** Eq. 14.27's ELBO
  is kept with the variational family `p̂(û_C | v) · Q_ω(û_I | i)`, so it
  splits into the chapter 13 MLE `log P̂(v)` plus a conditional VAE for
  `P(I | V)`. L2 trains a labeller head. With `Q_ω` encoding all of `Û`, as
  printed, the VAE ignored `Û_{FY}` on Ex. 14.2 (its `F ↔ Y` mutual
  information is 0.02 nats). `P̂(F, Y)` stayed at the product of its
  marginals (TV 0.096), and L2 slowed it further. `edit` resamples
  `Û_C` from 4096 prior draws by `P̂(v | û_C)`.
- **Edits and `Ncm::estimate` share one coupling**: an inverse CDF at
  `ε_V`, reversed when `δ_V < g_V`. The example checks that 1240 edits lose
  the bar at the model's own rate (0.641 vs 0.643). Keep them in step if you
  touch either.
- MNIST models: `U_D` is uniform on `0..10`. `C`'s Bernoulli is
  `U_C < 5 + 10·U_D` (backdoor) or `U_C < 95 − 10·D` (frontdoor), with
  `U_C` uniform on `0..100`.

Book errata, also listed in the module docs:

- Ex. 14.2's `Y = 0: young` is reversed.
- Ex. 14.3 writes `do(Y = 1)` for `do(Y = 0)`.
- Ex. 14.4 swaps "same" and "different".
- Eqs. 14.33, 14.35 and 14.41 condition on the wrong factual image.
- Eqs. 14.31/14.37 write `U_D` where the diagrams need `D` (backdoor `B`,
  frontdoor `C`).
- `B`'s precedence must be `((·) ∨ (C ⊕ U2)) ∧ U3`.

With those readings, every bound in Figs. 14.8–14.12 and Ex. 14.9
reproduces to the printed digits.

There is no chapter 14 code in the book's companion Python.

Not implemented:

- The baselines (CVAE, CGN, DEAR). The example's Markovian-graph ANCM plays
  DEAR's role.
- Real image data (MNIST, CelebA-HQ). The tests and example render small
  synthetic images.
- DiffuseVAE refinement.
- Multilinear bounds with two or more free components.
