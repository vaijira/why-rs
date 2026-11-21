# Causal RL tasks (chapter 9)

Chapter 9 lives in the same `crl` module, one file per task. All of it is
dependency-free and builds for WASM. [crates/why-data/src/crl/lp.rs](../../crates/why-data/src/crl/lp.rs)
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
