# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup docs
live in [README.md](README.md).

## What this is

A Cargo workspace of Rust causal-inference tools plus a WASM web UI. The root
[Cargo.toml](Cargo.toml) is a **virtual manifest** — `[workspace]` and profiles
only, no `[package]`. Every crate lives under [crates/](crates/).

| Crate | Role |
| --- | --- |
| [crates/why-data/](crates/why-data/) | Graphs, d-separation, back-door, do-calculus, σ-calculus, ID, C-INFER, symbolic SCMs, causal fairness and explanations (chapter 6: Algorithms 15–18, L3 Shapley values), causal RL (chapter 8: CDMs, NUC/SBC, IPW/DP, Algorithms 19–21; chapter 9: causal bounds, POMIS, counterfactual policies, causal imitation, causal games, Algorithms 22–31) |
| [crates/why-parser/](crates/why-parser/) | Parsers: dagitty (pest) and the book's section format (`<NODES>`, `<EDGES>`, `<TASK>`, `<QUERY>`, `<INTERVENTIONS>`) |
| [crates/why-ui/](crates/why-ui/) | `cdylib`+`rlib` WASM front end built with `dominator` |

The browser entry point is [crates/why-ui/src/lib.rs](crates/why-ui/src/lib.rs);
static assets (`index.html`, images) live in
[crates/why-ui/dist/](crates/why-ui/dist/) and generated output goes to
`crates/why-ui/dist/js/` (gitignored via the `**/dist/js/` pattern — note the
`**`, a single-level `*/dist/js/` would no longer match at this depth).

## The `yarn` vs `yarnpkg` gotcha

On Ubuntu/Debian the Yarn binary is installed as **`yarnpkg`** — `yarn` is taken
by the unrelated `cmdtest` package. The maintainer has a shell alias, but that
alias is **not** visible to non-interactive tool shells, so `yarn` fails with
`command not found`.

**Always invoke `yarnpkg` directly.** If it is missing, fall back to `yarn`
before concluding the environment is broken.

## Running the web UI

Run all three from the repository root:

```sh
yarnpkg install    # first time only, or after package.json changes
yarnpkg build      # release build -> crates/why-ui/dist/js/
yarnpkg start      # dev server + watch + live reload
```

- `build` = `rimraf crates/why-ui/dist/js && rollup --config`. Compiles
  `why-ui` to WASM in the release profile, runs `wasm-opt -Oz`, minifies with
  terser.
- `start` = same, plus `--watch`. Serves `crates/why-ui/dist` on
  **<http://localhost:10001>**, uses the `dev` profile, skips terser, and opens
  a browser (`open: true` in [rollup.config.js](rollup.config.js)).

All four crate paths in [rollup.config.js](rollup.config.js) (`input.index`,
`output.dir`, `serve.contentBase`, `livereload`) and both
[package.json](package.json) `rimraf` paths point at `crates/why-ui` — update
them together if the layout changes again.

### Verifying the UI without a browser

`yarnpkg start` never exits, so run it in the background, wait for the
`http://localhost:10001` line in its log, then check the three artifacts:

```sh
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:10001/
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:10001/js/index.js
# the wasm filename is content-hashed; read it out of index.js first
curl -s http://localhost:10001/js/index.js | grep -o 'assets/why_ui_rs-[a-z0-9]*\.wasm'
```

All three should return `200`. A `200` on `index.js` alone is not enough — the
WASM asset is fetched separately and is the part that usually breaks.

Stop the background process when done; do not leave a server holding port 10001.

## Rust checks

```sh
cargo build
cargo test
cargo clippy --all-targets
cargo test -p why-data     # single crate

# Ports of the Causal AI book's chapter 2 notebooks. Both assert every value
# against the number the Python original produces, so they fail loudly on a
# regression in `scm`; treat them as tests you can read.
cargo run -p why-data --example chapter2_part1   # SCM, d-separation
cargo run -p why-data --example chapter2_part2
cargo run -p why-data --example chapter4_part1   # back-door criterion
cargo run -p why-data --example chapter4_part2   # do-calculus + ID
cargo run -p why-data --example chapter4_cinfer  # C-INFER, q-trees and d-trees
cargo run -p why-data --example chapter6_fairness   # Tasks 1–2 (no notebook; see below)
cargo run -p why-data --example chapter6_decisions  # Task 3, checks the book's Ex. 6.32/6.35 numbers
cargo run -p why-data --example chapter6_explanations  # §6.2: SHAP vs do-Shapley vs L3, Ex. 6.37–6.43
cargo run -p why-data --example chapter8_crl  # Ex. 8.6, 8.16–8.18, 8.23–8.25, RCT vs UCB
cargo run -p why-data --example chapter9_crl  # Tables 9.2–9.3, 9.6–9.7, 9.15; UCB+, Ctf-UCB, CNE
cargo run -p why-parser --example notebook_graphs  # the book's file format
```

The root is a virtual manifest, so these cover all members by default — no
`--workspace` needed. As of the last verified run the workspace has 223 tests
(179 in `why-data` plus 21 doctests, 21 in `why-parser` plus 2 doctests), all
passing, and clippy is clean workspace-wide (the warnings this file used to
record were fixed in f9aee15). The tree is not rustfmt-clean, so do not run
`cargo fmt` over it as part of an unrelated change.

`why-data` has no Cargo features, and no `RUSTFLAGS` or `.cargo/config.toml` are
involved. The `scm` module was gated behind a feature until `rssn` 0.2.12 made
it buildable for WASM; the gate is gone and
[crates/why-data/src/scm.rs](crates/why-data/src/scm.rs) is always compiled.

### C-INFER and ID

[crates/why-data/src/graph/cinfer.rs](crates/why-data/src/graph/cinfer.rs)
(chapter 4.4: q-trees, d-trees, the `Σ`/`δ`/`σ`/`Γ` operators) and
[crates/why-data/src/graph/identification.rs](crates/why-data/src/graph/identification.rs)
(Shpitser-Pearl ID/IDC) share `Formula` and the private `Dist` helper. On
observational input they must identify exactly the same queries: both are
complete. `cinfer`'s `agrees_with_id_and_is_numerically_sound` test checks
this over random graphs. It also evaluates every estimand against the true
effect of a random latent-confounder model, so run it after touching either
file. Estimand *strings* can differ: C-INFER conditions each factor on its
district's blanket, and ID conditions on every predecessor.

[crates/why-data/src/graph/cinfer/counterfactual.rs](crates/why-data/src/graph/cinfer/counterfactual.rs)
extends C-INFER to counterfactual queries (chapter 5, Correa, Lee & Bareinboim
2021). It adds CUT unnesting, the Ξ collapse of consistent ctf-factors and a
counterfactual Γ, and reuses the layer-2 d-trees through `grow_input_tree`.
`CtfQuery::parse` reads textbook notation such as `P(y_x | x')`. Its
`estimands_are_numerically_sound` test builds random SCMs with response
functions and per-variable noise shared across worlds. It checks every
estimand against the exact counterfactual probability, and checks that
`P(y_x)` is identified exactly when layer 2 identifies it. It takes about 30s
in debug. The book's Python (`_counterfactual_engine.py`) is not a reliable
oracle: its `simplify` zeroes any variable with two values across worlds, and
its ancestral components never merge across bidirected edges.

### Causal fairness

[crates/why-data/src/fairness.rs](crates/why-data/src/fairness.rs) implements
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

### Causal explanations

[crates/why-data/src/explain.rs](crates/why-data/src/explain.rs) implements
section 6.2 (marked as a draft in the book): the NTE, the GDE and
counterfactual (L3) Shapley values, plus SHAP (L1) and do-Shapley (L2) for
comparison. Everything reduces to one mean per set `S`,
`m(S) = Σ_s P(S = s) E[Y_{S=s} | e]`, from which `NTE(S) = m(∅) − m(S)`,
`GDE_S(X) = m(S) − m(S ∪ {X})`, and `φ^L3` is the Shapley value of the game
`v(S) = NTE(S)`.

- **It works on an `scm::Model`, not on data.** `E[Y_s | e]` conditions on
  evidence that includes `Y`, e.g. `P(y_x | x', y')`. That is not identifiable
  from L1/L2 data even without confounding (Thm 6.1.13 (iv)), and
  `ctf_infer` rejects it as an inconsistent ctf-factor. The book's examples
  all assume a known SCM too.
- `E[Y_s | e]` uses `scm`'s shared exogenous draw. It intervenes, takes Y's
  qualified name in that world (`{y}_{x=1}`), and tabulates it with the
  evidence from the **observational** world. Evidence names must be resolved
  there: inside an intervened world, `"x"` resolves to that world's copy.
- Cost is `2^|X|` subsets times `|D_S|` interventions, each a full
  enumeration of `U`. Exact Shapley values are capped at 20 variables.
- The tests are the book's Ex. 6.37–6.43 with exact numbers, plus random
  SCMs checking efficiency (Cor. 6.2.2), Thm 6.2.1 along an ordering, and
  admissibility for a variable with no path to `Y`.

Possible follow-ups: identification of individual `m(S)` terms through
`ctf_infer` when the evidence excludes `Y`, bounds when it does not, and
permutation-sampled Shapley values for larger `X`.

### Causal reinforcement learning

[crates/why-data/src/crl.rs](crates/why-data/src/crl.rs) implements chapter 8,
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

### Causal RL tasks (chapter 9)

Chapter 9 lives in the same `crl` module, one file per task. All of it is
dependency-free and builds for WASM. [crates/why-data/src/crl/lp.rs](crates/why-data/src/crl/lp.rs)
is a small dense two-phase simplex (Bland's rule) used by the optimistic
planner, Causal MWAL and Nash support enumeration. The chapter 9 tests are in
`crl/tests9.rs`.

| File | Book |
| --- | --- |
| `cool.rs` | Task 1: `natural_bounds` (Thm 9.2.2), `causal_bounds` (Thms 9.2.5–9.2.6, `Γ` of Eq. 9.49), `policy_bounds` (Eq. 9.67), `UcbPlus` (Alg. 23), `UcbTransfer` (Alg. 22, UCB−), `UcbCdm` (Algs. 24/25) |
| `mixed.rs` | Task 2 without context: `latent_projection`, MIS (Prop. 9.3.1), `muct`, `interventional_border`, `pomis` (Thm 9.3.2) |
| `ctf.rs` | Task 3: `ett`, `CtfRct`/`CtfUcb` (Algs. 26/27) on a `CtfEnvironment`, `CtfMdp` (Eq. 9.174) with value iteration, `CtfUcbvi` (Algs. 28/29) |
| `imitation.rs` | Task 4: `imitation_backdoor` (Def. 9.5.3), minimal subspaces (Def. 9.5.5), `behavioral_cloning`, `imitation_distribution` (Thm 9.5.4), `causal_mwal` (Prop. 9.5.5, exact LP) |
| `games.rs` | Sec. 9.6: `Bimatrix` equilibria, `CausalGame` from an SCM, `learn_cne` (Alg. 30), `ctf_nash_learning` (Alg. 31, deterministic rewards) |

Facts that are easy to get wrong:

- **Thm 9.3.2 projects onto `X⋆ ∪ {Y}`**, not `X′ ∪ {Y}` as printed. The
  `some_pomis_is_optimal_on_random_scms` test checks completeness against
  brute force.
- **Thm 9.5.4 is false when a stage passes only condition 1** of the
  imitation backdoor (X_i not an ancestor of Y) while X_i and Y share a
  confounder. `ImitationCondition::Blocked` takes precedence in reports so
  callers can tell; `condition_1_breaks_theorem_9_5_4` is the counterexample.
- **The greedy casino needs `X ← U_D ⊕ U_B`** (Eq. 9.98 prints a negation
  that contradicts Table 9.6 and Ex. 9.12).
- **Book table errata**: Table 9.9's `S′ = 0, X′′ = 1` block is a copy of the
  `X′′ = 0` block; Table 9.11's `Q*(s, x′, x′)` should be 8.2, not 1; Ex. 9.15's
  best experimental policy is `X ← S` (Table 8.1), not `X ← ¬S`; Fig. 9.36 is
  truncated rather than rounded in places (compare Table 9.17).
- **Experiment 9.4's `r_π = 0.6283` is not reproducible.** Every entry of
  Table 9.3 is (to 4 decimals, by quadrature over U), and the exact bound for
  that policy is 0.6788.
- **Equilibrium selection** is the first pure NE in row-major order, which
  reproduces Tables 9.15 and 9.37(b); the book leaves it open.
- **Eq. 9.197's bonus constant 7** makes UCBVI stall at `Q = H` for any test-
  sized run; `CtfUcbvi::bonus` exposes it, and the test uses 0.3.
- Figures 9.12, 9.30 and 9.31 are not given as edge lists; the tests
  reconstruct them from the c-factorisation and the worked examples, and
  say so.

Not implemented: mixed policies with context (Sec. 9.3.2), hybrid policies
under an autonomy budget (Sec. 9.4.3), Causal GAIL (Prop. 9.5.6) and
Secs. 9.6.5–9.6.6.

### σ-calculus

[crates/why-data/src/graph/sigmacalculus.rs](crates/why-data/src/graph/sigmacalculus.rs)
ports the book's `SigmaCalculusInspector` (Correa & Bareinboim 2020), reusing
`docalculus`'s formatting helpers. It deliberately departs from the Python in
three places, listed in its module docs. The Python has a mutable-default-list
bug in `TransformNodes` that leaks cuts between queries. So when its answers
disagree with this module, check the Python first. Its test graphs were
cross-checked by running the Python inspector, which needs `networkx`,
`toposort` and `pydash`. There is no σ-calculus notebook, so there is no
`examples/` port. `why_parser::fusion` reads the book's `<QUERY>` and
`<INTERVENTIONS>` sections, and `Document::sigma_calculus` runs the rule they
describe. Watch the naming: `treatment:` is X in `<TASK>` but Z in `<QUERY>`.

### `rssn` and WASM

[crates/why-data/src/scm.rs](crates/why-data/src/scm.rs) depends on
[`rssn`](https://crates.io/crates/rssn) for symbolic expressions. It is an
unconditional dependency, so **`rssn` is in the browser bundle** — `why-ui`
depends on `why-data`.

It builds for `wasm32-unknown-unknown`, verified:

```sh
cargo build -p why-data --target wasm32-unknown-unknown
cargo build -p why-ui-rs --target wasm32-unknown-unknown
```

Three things make that work, and the third is why the `rssn` requirement is
pinned to an exact patch release rather than `0.2`:

1. **`getrandom`**, which `rssn` pulls in three generations of (`rand`
   0.8/0.9/0.10 → getrandom 0.2/0.3/0.4). All three are declared as
   target-gated dependencies of `why-data` with `js` (0.2) and
   `wasm_js` (0.3, 0.4). Features are enough: 0.3.4 and 0.4.3 pick the browser
   backend on `#[cfg(feature = "wasm_js")]` alone, despite 0.3.4's error text
   still insisting the feature is "insufficient" — that message is stale, and
   older 0.3.x really did also want `--cfg getrandom_backend="wasm_js"`.
2. **`uuid`** with its `js` feature, likewise.
3. **`rssn` 0.2.13 or newer.** Up to 0.2.11 it declared a bare
   `faer = "0.24"`, whose default features include `rayon` → `spindle` →
   `atomic-wait`, and `atomic-wait` 1.1 (latest, Jan 2023) has no wasm
   `platform` module, so the build died with `cannot find module platform`.

## Conventions

- All three crates are on edition 2024. Note the 2024 match-ergonomics rule:
  an explicit `ref` inside a pattern that already binds by reference (e.g.
  `if let Some(ref x) = &opt`) is a hard error — drop the `ref`, the binding
  type is unchanged.
- The release profile is tuned for size (`opt-level = "z"`, `lto`,
  `panic = "abort"`) because the output ships as WASM. Keep that in mind before
  adding heavy dependencies to `why-ui`.
- Do not edit `crates/why-ui/dist/js/`; it is gitignored and regenerated on
  every build. `crates/why-ui/dist/index.html` and `crates/why-ui/dist/images/`
  **are** tracked sources — edit those freely.
- `Cargo.lock` and `yarn.lock` are gitignored and untracked, so dependency
  resolution is not pinned across clones. If a build breaks after a fresh
  `yarnpkg install`, an upstream version bump is a plausible cause.
