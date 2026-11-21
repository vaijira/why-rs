# Parametric identification (chapter 19)

[crates/why-data/src/parametric.rs](../../crates/why-data/src/parametric.rs)
covers both halves of chapter 19. It adds no dependency and builds for WASM.

| File | Book |
| --- | --- |
| [linear.rs](../../crates/why-data/src/parametric/linear.rs) | Secs. 19.2–19.5: `LinearScm`, `Covariance`, `single_door`, `back_door`, `conditional_instrument`, `generic_rank`, `instrumental_set` (Alg. 66) |
| [monotone.rs](../../crates/why-data/src/parametric/monotone.rs) | Sec. 19.7: `Cgma`, `CtfTerm`, `m_reduce` (Alg. 67), `m_id` |
| [tests.rs](../../crates/why-data/src/parametric/tests.rs) | book examples plus random-model soundness tests |

## Linear part

- **Λ is indexed `[from][to]`.** `V = ΛᵀV + U`, so
  `Σ = (I − Λ)⁻ᵀ Ω (I − Λ)⁻¹`. `LinearScm::new` takes coefficients for
  directed edges and covariances for bidirected edges only. Anything else
  is `NotAnEdge`.
- **`trek_covariance` enumerates directed paths on purpose.** It does not
  reuse the matrix inverse. That keeps the trek-rule test (Exs. 19.4–19.7)
  an independent check of `covariance()`.
- The criteria are purely graphical. They return the smallest qualifying
  set, or `None`. `Covariance` does the estimation:
  - `regression(y, x, z)` gives `β_{yx.z}`.
  - `instrumental_set(y, xs, zs)` solves `Σ_{Z,X} λ = Σ_{Z,y}`.
- **Flow graph** (Def. 19.5.1): top copies are `0..n`, bottom copies
  `n..2n`. The rank (Thm 19.5.2) is a vertex-capacitated max-flow, with
  each vertex split in two.
- **Match-Block candidates** (`instrumental_set`) are the variables whose
  every trek to `y` ends in a directed edge into `y`. The sinks are `y`'s
  parents. Ex. 19.21 has rank 1, so its instrumental set is empty. Adding
  `Z2 ↔ X2` (Ex. 19.28) brings the rank back to 2.

## Monotone part

- **Binary only.** Levels are 0 and 1, and the data columns hold `"0"` and
  `"1"`.
- **`m_id` pipeline:**
  1. CTF-FACTOR expands each joint over the counterfactual parents.
  2. `m_reduce` (Alg. 67) reduces every factor.
  3. `ctf_infer` (chapter 5) identifies each reduced factor.
  4. `transport::Estimand::from_formula` evaluates it on the data.

  `given` is handled as the ratio of two joints.
- **Naming in the generated ctf queries:** variables are lowercased, and
  level 1 is written `v` while level 0 is written `v'`. Evaluation binds
  `(V, "1")` and `(V', "0")`. Variable names must therefore stay distinct
  after lowercasing.
- **Alg. 67 needs full-parent ctf-factors.** That is CTF-FACTOR's output,
  and the book applies it the same way. `m_reduce` returns `InvalidInput`
  for a term that is missing a parent.
- **Rule 2 runs in reverse topological order.** For each variable, it picks
  Eq. 19.168 or Eq. 19.169 by the value that a child's subscript uses for
  that variable. The `m_reduce` doctest (LATE compliers) pins this down.
- **Failures** (Lemma 19.7.7) come back as `NotIdentifiable`. They cover
  terms differing in a non-monotone parent, and terms that are
  incomparable under the constraints.

## Book errata

- **Thm 19.4.1 prints `δ_xy = σ_zx / σ_zy`.** Eq. 19.100 and the text give
  `σ_zy / σ_zx`, which is correct.
- **Sec. 19.7.6.2's total effect.** The book reports 0.175. The printed
  mechanisms and Table 19.1 give exactly 0.205, so 0.175 is presumably the
  estimate from the book's 10 000 samples. The test and the example both
  assert 0.205, against the adjustment formula.

## Tests to rerun after touching the module

- `every_criterion_is_sound_on_random_linear_scms` builds random ADMGs with
  random coefficients. Every set a criterion returns must recover the true
  coefficient or total effect.
- `m_id_is_sound_on_random_monotone_scms` uses random threshold SCMs with
  monotone edges. Every value `m_id` returns must match the truth, computed
  by enumerating the exogenous units.

The other monotone tests check LATE, Sec. 19.7.6.2 and PN against the same
simulator.

There is no chapter 19 code in the book's companion Python.

## Not implemented

- The graph decompositions and recursive total-effect identification of
  Sec. 19.6. The book gives no algorithm for the decomposed covariance
  matrices.
- The 401k data experiment.
