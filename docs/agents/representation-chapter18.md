# Causal representation learning (chapter 18)

[crates/why-data/src/representation.rs](../../crates/why-data/src/representation.rs)
implements the identifiability side of chapter 18. It adds no dependency
and builds for WASM.

| Item | Book |
| --- | --- |
| `delta_v` | the changed variable set ΔV (Def. 18.3.1) |
| `delta_q` | the ΔQ set (Def. 18.3.2): `C(ΔV) ∪ Pa(C(ΔV))` in `G_T` |
| `crid`, `Cdm` | Algorithm 65 and the causal disentanglement map (Def. 18.5.1) |

Facts that are easy to get wrong:

- **Regime syntax.** `(domain, targets)` with targets such as `"V1"`
  (fresh mechanism), `"V1:2"` (shared mechanism 2, even across domains),
  or `"V2!"` / `"V2!:1"` (marginal `σ(∅)`, cutting parents and
  confounders). Unlabelled mechanisms always differ.
- **ΔV across domains.** An S-node child is in ΔV unless both regimes
  intervene on it with the same labelled mechanism (Def. 18.3.1 (2),
  footnote 8).
- **`Pa^{T+}` over all orders is `Pa`.** ΔQ must hold for every topological
  order. The union over orders of the extended parents of a union of
  c-components is its parent set.
- **The CDM reads as in Def. 18.5.1**: `cdm.depends(rep, on)` is the edge
  `V_on → V̂_rep`, and `identifiable(tar, en)` means no `V̂_t` depends on any
  `V_e`. Thm 18.5.1's sentence states the direction the other way;
  Ex. 18.38 agrees with the definition.
- **CRID's sub-procedures are reconstructions** (the book prints
  "Alg. ??"):
  - Prop. 18.3.3 becomes a matching argument on the Jacobian equations of
    each canceled variable, whose unknowns are the CDM edges still present.
    That is how one comparison suffices in Ex. 18.38's second epoch.
  - Prop. 18.3.4's condition (3′) is a matching of "each variable twice,
    each edge once" onto distinct comparisons.
  - Prop. 18.3.6 is applied as stated, for single targets.

  Every set of targets with marginal targets `T` is tried as a baseline,
  as Alg. 65 does. Subsets of comparisons are enumerated, so `crid` caps
  the input at 16 regimes.
- **Ex. 18.24's second set of targets.** The book says it fails (3′); that
  is only true for the baseline it picks. With a `{V2, V3}` distribution
  as the baseline, CRID disentangles `V3` from `V1` there too. The test
  asserts both that and the failure with one `{V2, V3}` fewer.
- `crid_is_monotone_and_keeps_the_diagonal` checks two properties on
  random graphs: more regimes never put an edge back, and `V_i → V̂_i` always
  stays.

Reconstructions: Fig. 18.12(a) is `V1 → V2 → V3` with `V1 ↔ V2`, the
domains differing at `V3`. Fig. 18.12(c) is `V1 → V2 → V3 → V4` with
`V2 ↔ V3`. Both are consistent with Exs. 18.11–18.26 and 18.38.

There is no chapter 18 code in the book's companion Python.

Not implemented:

- Indirect generative factors (Sec. 18.4: marginalized LSDs, Prop. 18.4.1).
- The neural proxy models and MCC scores of Sec. 18.6. The example checks
  CRID's output for every setting that section reports.
