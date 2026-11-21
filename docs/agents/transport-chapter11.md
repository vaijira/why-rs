# Transportability (chapter 11)

[crates/why-data/src/transport.rs](../../crates/why-data/src/transport.rs)
implements chapter 11, layer-1 generalization across domains. It adds no
dependency and builds for WASM. A `SelectionDiagram` is an `Admg` plus one
discrepancy set `Δ^{k,*}` per source. Its S-nodes are implicit;
`augmented()` materialises them as `S_<source>`. Estimands are
`transport::Estimand`, not `Formula`, because each factor names its domain
(`P^1(y | x)`, `P^*(z)`). `Estimand::evaluate` runs on any `Distributions`.
`EmpiricalDistributions` (one `fairness::Dataset` per domain) is the plug-in
estimator.

| File | Book |
| --- | --- |
| `direct.rs` | Thm 11.2.1, `list_dtr_dist` (Alg. 32), structural covariate shift (Def. 11.2.3) |
| `score.rs` | `score_tr` (Alg. 33), `transportable_feature_sets` (maximal sets, Def. 11.2.5) |
| `partial.rs` | Thms 11.3.1–11.3.3, `statistical_transport` (Thm 11.3.4) |
| `representation.rs` | determined/constrained/free features (Def. 11.2.8); `SelectionDiagram::augment` is Def. 11.2.9 |
| `invariance.rs` | CMS (Assumption 11.2.12), source invariance with a χ² test (Def. 11.2.11) |
| `adaptation.rs` | Algs. 34–38: structure-informed/agnostic DA, multi-cause, sequential |

Facts that are easy to get wrong:

- **Thm 11.3.4's `pa⁺` must be Tian's set.** `pa⁺` should be
  `Pa(T_i) \ {V_i}`, which adds the parents of the earlier members of the
  c-component. The printed Eq. 11.241 is constant in `x2` while the truth is
  not; `example_11_34_needs_tians_conditioning_set` shows it.
- **Score-TR gets each c-factor from ID**, on the latent projection onto
  `W ∪ Z̃`. Line 11 of Alg. 33 is only the Markovian formula. Without the
  projection, ID mentions ancestors of `Z̃` that nothing sums out. Line 16's
  denominator sums over `W \ (Z \ Z̃)`, not `W \ {y}`.
- **ID primes captured indices (`v0'`)**, and `Estimand::evaluate` reads a
  primed name as its unprimed variable. Keep that if you touch either.
- `score_tr_and_the_product_agree_and_are_sound` runs Score-TR and Thm 11.3.4
  with `W = ∅` over 150 random multi-domain models. Both claim completeness,
  so they must decide the same queries, and every estimand must equal the
  true `P*(y | z)`. `the_product_is_sound_and_subsumes_the_scenarios` adds
  random target measurements. Run both after touching `score.rs`,
  `partial.rs` or `estimand.rs`.
- Book errata: Ex. 11.30 misses `Z = {Z, R}`, so Thm 11.3.3 does apply.
  Ex. 11.4's 3/8 is a joint; the conditional is 3/4. Ex. 11.40's equations
  contradict each other. Figs. 11.12–11.15, 11.20, 11.33 and 11.34 have no
  edge lists, so the tests reconstruct them from the text. The Fig. 11.13
  reconstruction matches Ex. 11.12's four maximal sets but has 25
  transportable sets, not Fig. 11.14's 15.

Not implemented: Alg. 39 (enumerates every partition of positions and every
set of graphs), the optimisation approach of Sec. 11.4.4, and learning
representations by UIRM (Def. 11.2.13). There is no chapter 11 code in the
book's companion Python.
