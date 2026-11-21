# Generalization across the PCH (chapter 12)

[crates/why-data/src/generalize.rs](../../crates/why-data/src/generalize.rs)
implements chapter 12. It transports causal (layer 2) and counterfactual
(layer 3) quantities, where chapter 11 transported statistical ones. It
reuses chapter 11's `SelectionDiagram` and `Estimand` and the C-INFER
engines, adds no dependency, and builds for WASM. Chapter 11's `Domain`
gained two variants: `Under(domain, Regime)` for `P^k(V; σ_Z)` and `Policy`
for the target policy's `P̂(x | pa*)`. A `Regime` is a map from a variable
to `Atomic` or `Policy { parents }`, and `Regime::apply` builds `G_σ`.
`Distributions::conditional` is a new provided method, and
`WithPolicy`/`TargetPolicy` override it to supply `P̂`.

| File | Book |
| --- | --- |
| `adjustment.rs` | e-adjustment (Def. 12.1.2) by `e_adjustment_order` = Alg. 40; tr-adjustment (Defs. 12.1.1, 12.1.3), Eq. 12.18, Thm 12.1.6 |
| `policy.rs` | σ-TR / g-TR: q-tree, `grow` = Alg. 41 (π- and σ-operators), `sigma_tr` = Alg. 43 |
| `counterfactual.rs` | ctf-TR: chapter 5's q-tree (Alg. 42) with Alg. 41's d-trees |

Facts that are easy to get wrong:

- **Condition (12.16) is rule 3.** `G_{X(Z≤i−1)}` is read as cutting the
  edges into the treatments that are not ancestors of `Z≤i−1`. The random
  test `e_adjustment_holds_exactly_when_the_formula_does` checks Thm 12.1.2
  in both directions on that reading. Def. 12.1.1 (i) names no graph and is
  read the same way, and Lemma 12.1.4 then holds on random diagrams.
- **Completeness tests need faithful models.** A random boolean response
  function of one bit is constant half the time, so the test `Model` uses
  continuous random conditionals (`k = 0`) for layer 2. Only the
  counterfactual tests use response functions (`Model::with_noise(.., 4)`).
- **Inputs keep their intervened variables.** A d-tree's root is
  `Q^k[V; σ_Z]` over all of `V`, grown by `cinfer::grow_input_tree` on
  `G_σZ` as if observational. An atomic input is evaluated as a trial that
  randomizes `Z`, so `do(z)` terms carry `z` as a conditioning variable,
  which `Estimand`'s display moves into `do(·)`.
- **Domains ride in `Formula::P::doing` as tags** (`"\0k"`) while
  `Formula` simplifies and avoids capture, so the chain rule never merges
  terms of different inputs. `policy::to_estimand` decodes them. Keep that
  in mind if `Formula::simplify` starts reading `doing`.
- `cinfer::counterfactual::compose` was split out of `ctf_infer` so ctf-TR
  can feed it factors from other domains. `build`, `Plan` and `Built` are
  `pub(crate)` for the same reason.
- **Conditional σ-TR queries are a plain ratio** over `An(Y ∪ W)` in
  `G_σX`. That is sound but can be incomplete, since it lacks layer 2's rule-2
  refinement.
- **Separation queries draw one S-node.** `SelectionDiagram::augmented_for(source)`
  (and the crate-internal `with_s_node`, which keeps an isolated node when
  `Δ` is empty) draws only that source's S-node. `augmented()` draws them
  all. There another source's S-node acts as a common cause of its
  children, which made chapter 11's `is_s_admissible`, `list_dtr_dist` and
  Thm 11.3.1 miss valid sets when sources share discrepancies. All three use
  `with_s_node` now. See the test
  `other_sources_s_nodes_do_not_connect_their_children`. `project()` still
  uses `augmented()`, which is correct: the S-nodes are kept roots there.

`sigma_tr_is_sound_and_extends_c_infer` and `ctf_tr_is_sound_and_extends_ctf_infer`
draw random diagrams, inputs (any domain, natural, atomic or soft) and
regimes. They check every estimand against the exact target quantity, and
check that target-only data decide exactly what C-INFER does. Exs. 12.11,
12.14 and 12.15 reproduce Eqs. 12.53, 12.75 and 12.77 verbatim.

Book notes: Ex. 12.13 fixes `x1` (Eq. 12.66's indicator) although `σ*`
says `σ_{X1} = P̂(X1)`, and Eq. 12.33 sums it. The tests sum it. Exercise
12.12's solution drops `z` from `P(y | x1, x2, z; σ*)`, but Eq. 12.67 keeps
it. Figs. 12.3, 12.5, 12.6 and 12.19 have no edge lists, so the tests
reconstruct them from the derivations. The loan study's bidirected edges
come from Fig. 12.14's trees. Fig. 12.4 (Ex. 12.4) and Fig. 12.11 (Ex. 12.12)
are not reconstructed. There is no chapter 12 code in the book's companion
Python.
