# Causal explanations

[crates/why-data/src/explain.rs](../../crates/why-data/src/explain.rs) implements
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
