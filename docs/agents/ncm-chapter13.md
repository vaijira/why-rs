# Neural causal models (chapter 13)

Chapter 13 is split in two crates. The tensor-free part is
[crates/why-data/src/ncm.rs](../../crates/why-data/src/ncm.rs):

- `c2_components` (Def. 13.1.10, maximal bidirected cliques, which may
  overlap) and `NcmStructure` (Def. 13.1.11).
- `expand`, which lays a `CtfQuery` out over worlds. `Fix::Copy` is a
  nested subscript, as in `Y_{x, W_{x'}}`.
- `StepNcm`, built from the book's `f̂_AND`, `f̂_OR`, `f̂_NOT` and `f̂_≥z`
  units. It is evaluated exactly by splitting `[0, 1]^k` at the thresholds
  of the units that read a noise. `parent()` returns an identity unit, so
  arrays of units need no `.into()`. Its tests reproduce Exs. 13.5, 13.6,
  13.10, 13.15 and 13.16 to the last digit. `StepNcm::sample` is
  Algorithm 48.

Training is in [crates/why-ncm/](../../crates/why-ncm/), which depends on `burn`
0.22 (`flex` CPU backend, `autodiff`, `optim`). It builds for
`wasm32-unknown-unknown` with no extra configuration, using the getrandom
features `why-data` already selects. burn 0.22 has no backend type
parameter: tensors are `Tensor<D>`, and the device is
`Device::flex().autodiff()`.

| File | Book |
| --- | --- |
| `model.rs` | `Ncm`: one MLP per variable over one-hot parents and its C²-components' uniforms (Def. 13.1.11, Eq. 13.141) |
| `explicit.rs` | MLE-NCM likelihood (Eqs. 13.142, 13.144) and exact enumeration of the query's worlds (Eqs. 13.145–13.147) |
| `implicit.rs` | sampling, `Ncm::sample` (Algorithm 48), MMD and the Table 13.6 log losses |
| `identify.rs` | `neural_id` (Algorithms 44/46, trained as 47 or 49), `fit`, `symbolic_then_neural` (Algorithm 45, deciding with `ctf_infer`) |

Facts that are easy to get wrong:

- **Eq. 13.146 is unsound.** It multiplies `σ` across worlds, as if each
  world drew fresh sampling noise. That violates consistency. The explicit
  model shares `ε_V` (inverse CDF) and `δ_V` (an order gate `g_V`) across
  worlds, so the joint is an exact, differentiable sum of overlaps.
  `worlds_share_each_variables_noise` checks consistency on an untrained
  model.
- **The gate is not decoration.** Without it, binary variables are
  comonotone across worlds given `û`, and the maximizing model on Ex. 13.16
  stalls at PS ≈ 0.04 instead of 1.
- **Empty overlaps have no gradient.** Training uses a straight-through leak
  and an unclamped `log(q + 1e-6)`; evaluation is exact.
- **Use `Kind::Explicit` for identification.** `Kind::Implicit` (Gumbel-max
  sampling, MMD) is right on Markovian models. With bidirected edges it does
  not learn to use `Û_C`: it returns about 0.40 on Ex. 13.10's front door
  (truth 0.469), whatever `λ`, epochs or width. The data term is MMD, not
  MMD²; the square is flat at the optimum and lets the query term win.
  The xx term of the MMD sums rows before summing the row means, because a
  single f32 sum over 4096² entries came out at 0.13 instead of 0.0002.
- **burn's Flex backend has one global RNG** (`static SEED` in
  `burn-flex`), which `device.seed` resets and `Linear` initialization
  draws from. Models initialized concurrently, as parallel tests are,
  interleave their draws and lose reproducibility; that made
  `algorithm_49_agrees_on_a_markovian_effect` fail about one run in four.
  `model::seeded` holds a crate-wide lock across seeding and
  initialization. Route every new network through it.
- `λ` decays geometrically from 1 to 0.001 (Sec. 13.3.3.1), and the
  learning rate to a tenth. The bow graph's min/max land near the natural
  bounds [54/256, 166/256], which Ex. 13.6's M̂4 and M̂3 attain.

There is no chapter 13 code in the book's companion Python. The book's
simulation code is at `CausalAILab/NCMCounterfactuals` on GitHub (PyTorch),
but nothing here was ported from it.

Not implemented: continuous variables, the GAN discriminator, choosing `τ`
empirically, and Thm 13.1.4's general canonical-SCM-to-NCM construction
(only Ex. 13.5's instance appears, in the tests).
