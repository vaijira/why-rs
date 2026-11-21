# Causal abstractions (chapter 15)

[crates/why-data/src/abstraction.rs](../../crates/why-data/src/abstraction.rs)
implements the exact part of chapter 15. It adds no dependency and builds for
WASM. Algorithm 52 is `neural_abstract_id` in
[crates/why-ncm/src/abstraction.rs](../../crates/why-ncm/src/abstraction.rs).

| File | Book |
| --- | --- |
| `clusters.rs` | `Abstraction`: inter/intravariable clusters and `τ` (Defs. 15.2.1, 15.2.2), `transform` = `τ(P(V_L))`, admissibility, `cdag` (Def. 15.3.1) |
| `consistency.rs` | `Term`, `low_level` (`Q`, Eq. 15.19), the `HighLevel` trait (implemented for `scm::Model` and `Constructed`), `q_consistency` and `layer_consistency` (Def. 15.2.3) |
| `construct.rs` | `abstract_invariance` (Thm 15.2.2), `construct` = Algorithm 51 |
| `invariance.rs` | `conditional_invariance` (Def. 15.2.5, local form), `observational_partition`/`interventional_partition` (Eqs. 15.67, 15.68; Thm 15.2.5) |
| `identify.rs` | `abstract_identify` (Thm 15.3.2: C-INFER on the C-DAG), `abstract_estimate` (evaluates the estimand on `τ(P(V_L))`) |

Facts that are easy to get wrong:

- **A cluster is a member list plus a map** from the members' values to a
  level. The map names the intravariable block; levels are strings, and
  structural-model values enter as integers. `keep(v)` is the identity
  cluster.
- **A cluster's inputs** (Alg. 51 and the AIC): the other clusters are
  fixed to a preimage. The cluster's own members are computed in order, and
  projected-away variables are computed from `u`. Ex. 15.8's Eq. 15.60 does
  this (`r = u_RY`). The literal Eq. 15.50 would also fix `R` and reject
  Ex. 15.5's own clustering. `Constructed` evaluates a high-level
  intervention as `M_L` under one preimage of it.
- **C-DAGs project first.** With variables left out, `cdag` takes the latent
  projection onto the clustered variables (`crl::latent_projection`) before
  the quotient. Ex. 15.9's `X ↔ Y` comes from dropping `R`.
- **The conditional AIC is the local form.** Def. 15.2.5 compares all of
  `τ(V)` given any union of clusters. Conditioning shifts the ancestors'
  posterior, so it rejects even Ex. 15.7's invariant clustering. The code
  checks `P(τ_{C_i}(c_i) | pa¹) = P(τ_{C_i}(c_i) | pa²)` per cluster of the
  C-DAG, the refinement the text gives after the definition. It therefore
  takes the low-level graph.
- **`layer_consistency` compares joints**, over one world (layers 1, 2) or
  two (layer 3), for every union of clusters and every value its members
  take. It caps at 4096 interventions. Queries over three or more worlds are
  not enumerated.
- `the_aic_check_is_sound_on_random_models` checks Prop. 15.4.1 (the finest
  clustering always passes) and Prop. 15.2.4 (whenever the AIC check
  accepts, Alg. 51's model is τ-consistent). Consecutive topological blocks
  are always admissible, so they serve as random clusterings. Run it after
  touching `construct.rs` or `consistency.rs`.
- **Alg. 52 needs training to close the gap.** On Ex. 15.10 the default 1500
  epochs leave `[0.497, 0.550]`, which is above `τ = 0.05`. 3000 epochs give
  `[0.490, 0.524]`. Ex. 15.9's non-ID models spread to about
  `[0.30, 0.92]`, toward the natural bounds `[0.29, 0.95]`, which `M1` and
  `M2` attain.

Book notes:

- Alg. 51's output line and Prop. 15.3.1 write `τ: D_{V_H} → D_{V_L}`.
- Eq. 15.47 writes `τ_{C1}` for `τ_{C2}`.
- Eq. 15.12 writes `D` for `B`.
- Ex. 15.9's 0.95 and 0.29 are rounded; exactly, they are
  `0.66 + 0.34 · 0.852941` and `0.34 · 0.852941`.
- Fig. 15.12(a)'s `R ↔ Y` comes from the shared `U_RY`.

The nutrition study (Sec. 15.5.1) has no published model. The example builds
a synthetic one in which restaurants confound dish and BMI. It satisfies the
AIC by construction: nutrients read the dish only through "unhealthy", and
BMI reads the calories only through "≥ 58". The C-DAG is the front door
`DH → Z → BH` with `DH ↔ BH`.

There is no chapter 15 code in the book's companion Python.

Not implemented:

- Representational NCMs (Def. 15.4.1, the GAN-RNCM of Sec. 15.5.2). They
  need continuous high-level variables, which the NCMs here lack.
- Non-constructive τ-abstractions, beyond the declarative definitions
  (Defs. 15.1.1–15.1.3).
- The interventional AIC check (Def. 15.2.6).
