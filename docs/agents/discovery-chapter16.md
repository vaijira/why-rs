# Structural learning (chapter 16)

[crates/why-data/src/discovery.rs](../../crates/why-data/src/discovery.rs)
implements chapter 16: MAGs and PAGs, FCI, and the interventional learners.
It adds no dependency and builds for WASM.

| File | Book |
| --- | --- |
| `pag.rs` | `Pag`/`Mark` (Def. 16.1.7; a MAG is a PAG with no circles), `edges()` text form, `is_visible` (Lemma 16.1.4) |
| `mag.rs` | `mag` (Algs. 53/55), `markov_equivalent` (Prop. 16.1.1), discriminating paths (Def. 16.1.6) |
| `fci.rs` | the shared engine: Phase I skeleton, Phase II colliders, rules R1-R4 and R8-R10, R11/R+ and R12; Ψ-PC's Meek rules; `fci` (Alg. 54) |
| `interventional.rs` | `augment` (Defs. 16.2.5, 16.2.11), `i_fci` (Algs. 56-58), `psi_fci` (Alg. 59), `psi_pc` (Alg. 60), `i_markov_equivalent` (Thm 16.2.6), `psi_markov_equivalent` (Thm 16.2.8) |
| `oracle.rs` | `Independence`/`Invariance` traits; `GraphOracle` and `AugmentedOracle` (d-separation, Prop. 16.2.5); `DataTests` (G-tests or exact tables) |

Facts that are easy to get wrong:

- **Edge text is canonical.** `Pag::edges()` writes endpoints in name order
  with the marks as they sit: `A o-> B`, `B <-- D` (an arrow into `B`
  from `D`), `X o-o Y`. Tests compare these strings, so keep the format.
- **Bidirected edges are latent confounders.** `mag` turns each one into an
  explicit latent parent before searching for inducing paths. Variables of
  the graph outside `observed` are latent too. Inducing paths are found by a
  depth-first search over simple paths, which is fine at test sizes.
- **Rules only overwrite circles.** That is what the theory does, and it
  keeps the loop from oscillating when finite-sample tests conflict.
- **F-nodes are named `F1`, `F2`, …** in pair order `(0,1), (0,2), …,
  (1,2), …`. I-FCI keeps one per distinct non-empty symmetric difference;
  Ψ-FCI and Ψ-PC keep one per pair. `augment` uses the same order, so its
  MAG can be compared with a learned graph name for name. A variable named
  `F1` is rejected (`Reserved`).
- **Targets may carry mechanisms**: `"X:2"`. The symmetric difference keeps
  a variable that both targets share under different mechanisms (Ex. 16.9).
- **Phase I tries every subset**, smallest first, as Alg. 54 prints it. It is
  exponential, but complete without the possible-d-sep step.
- **Invariance from data** is a G-test of `Y ⊥ label | W` on the two data
  sets pooled. In exact mode each data set is normalized first.
- `fci_i_fci_and_psi_fci_are_sound_on_random_diagrams` and
  `psi_pc_is_sound_on_markovian_diagrams` check every learned adjacency and
  non-circle mark against the true (augmented) MAG.
  `markov_equivalence_matches_m_separation` checks Prop. 16.1.1 against
  brute-force m-separation. Run all three after touching `fci.rs` or
  `mag.rs`.

Reconstructions (the figures give no edge lists):

- Fig. 16.5(a): `A → B ← D`, `B → C`, `D ↔ C`.
- Fig. 16.12(a): `X → Y`, `X ↔ Y`, `Z → Y`, `X → W`.
- Fig. 16.14(a): `Z → X → Y`, `Z → Y`, `Y → W`, `X ↔ W`, with targets
  `⟨∅, {X}⟩`. Ex. 16.24's narration needs both `Z *-> X` and `X` as an
  ancestor of `W`. Together they force `Z o-> W` in the Ψ-PAG, an edge the
  text does not mention.

There is no chapter 16 code in the book's companion Python.

Not implemented:

- Rules R5-R7, which only apply under selection bias.
- Score-based and parametric learners.
- The completeness proofs' MAG enumeration, which would check that every
  circle is open in some member of the class.
