# The hierarchy of graphical models (chapter 20)

[crates/why-data/src/hierarchy.rs](../../crates/why-data/src/hierarchy.rs)
implements the procedures in chapter 20. It adds no dependency and builds
for WASM.

| File | Book |
| --- | --- |
| [layers.rs](../../crates/why-data/src/hierarchy/layers.rs) | Sec. 20.1 and Sec. 20.3.2: `layer` (Lemma 20.1.4), `reachability_set` (Def. 20.1.16), `ancestor_set` (Def. 20.2.5), `swig` (Alg. 71), Prop. 20.3.4 and Def. 20.3.9 |
| [realize.rs](../../crates/why-data/src/hierarchy/realize.rs) | Sec. 20.2.2: `ctf_realize` (Algs. 68–69) and the action sets `fisher_actions`, `layer225_actions`, `maximal_actions` (`A†`) |
| [bandit.rs](../../crates/why-data/src/hierarchy/bandit.rs) | Sec. 20.2.3: `MabScm::strategies` (π_obs, π_int, π_ett, π_opt), `mab_opt` (Alg. 70), `mab_int` |
| [augmented.rs](../../crates/why-data/src/hierarchy/augmented.rs) | Sec. 20.3.3: `ag_construct` (Alg. 72), `Eci` |
| [tests.rs](../../crates/why-data/src/hierarchy/tests.rs) | book examples plus random-graph properties |

## Easy to get wrong

- **`‖W_t‖` everywhere.** Every term is first restricted to the
  subscript variables that are ancestors of `W` in `G_T̄`, without `W`
  itself. `Dag::restrict` does this, so `X_{x}` reads as `X` and `P(X_y)`
  as `P(X)`.
- **L2** is a single regime: the restricted subscripts must unite into one
  consistent `x` with `‖W_x‖ = W_t` for every term, and no term may be of
  an intervened variable.
- **The CRS merging rule** (Def. 20.1.16) is ambiguous in the book. Take
  the terms whose reachability sets hold an entry for `R`. Their entries
  merge into `‖R_{x∪}‖` when the union `x∪` of those terms' subscripts is
  consistent and changes none of the terms. Without merging, the chain's
  `P(Z_x, Y_z)` would wrongly leave L2.25.
- **The ancestor set leaves out the intervened variables.** Def. 20.2.5
  ranges over `An(Y)` in `G_X̄`, which contains `X` itself. The book's own
  examples leave `X` out.
- **COMPATIBLE (Alg. 69) misses a conflict.** It never stops a natural
  requirement from coexisting with `RAND(V)`. `ctf_realize` tags `RAND(V)`
  `Natural` whenever a child needs `V`'s natural value. The test
  `realizability_of_the_book_examples` covers the case: `P(Y_x, Z)` with
  `RAND(X)` alone.
- **Phase 2 writes the tagged values directly.** It does not draw them
  and reject mismatches; by Lemma 20.2.1 the result is the same. A child in
  several tagged `CTF-RAND` groups takes the smallest group's value.
- **The MAB template has no `D → Y` edge.** `MabUnit` therefore stores
  `Y_x` and `D_x` per exogenous type. MAB-OPT's hot start from `P(v)`
  (line 7) is left out.
- **The `Eci` oracle is deliberately weak.** It closes listed statements
  under symmetry, decomposition and weak union only, so a constraint FCI
  needs that follows only by contraction must be listed explicitly.
  Regime indicators are adjacent only to their target, and an F-pair with
  no separating set certifies no collider.

## Tests to rerun after touching the module

- `layers_match_realizability_on_random_graphs` (3000 random queries). On
  every query, CTF-REALIZE must succeed:
  - with no actions exactly on L1;
  - with `RAND` exactly on L2;
  - with one-value `CTF-RAND(V → Ch(V))` exactly on L2.25;
  - with `A†` exactly on L2.5, which is Cor. 20.2.6.

  The test also checks that Prop. 20.3.4 implies L2.25. These
  equivalences are not stated in the book; they are what ties the layer
  test to the realizability algorithm.
- `plans_read_the_potential_responses` runs every successful plan on
  units of random confounded SCMs. Each unit must return exactly
  `(W_t(u))_t`.
- `counterfactual_adjustment_is_the_back_door` checks that, with factual
  covariates, Def. 20.3.9 agrees with `graph::backdoor::satisfies`.
- `counterfactual_strategies_dominate` checks Cor. 20.2.9 on random
  models. It also checks a model that reproduces Table 20.5 exactly (0.65,
  0.70, 0.75, 0.80), on which MAB-OPT beats the L2 bandit.

## Reconstructions

- **Fig. 20.14:** G1 is `T → A → {Z, W}` with `X → Z`. G2 is
  `T → {Z, W}` with `X → Z`.
- **Experiment 1:** `X → {Y, Z}` with `Y ↔ Z`.
- **Experiments 2–3:** the book prints no models. The Experiment 3 model
  in the tests and the example is built to match Table 20.5.

There is no chapter 20 code in the book's companion Python.

## Not implemented

- The ctf-calculus restricted to L2.25 or L2.5 (Def. 20.1.15). The book
  gives no algorithm. Use `layer` for membership and C-INFER for
  identification.
- The SEM-RR and FFRCISTG constructions, and inference in augmented DAGs.
  These are arguments about frameworks rather than procedures.
