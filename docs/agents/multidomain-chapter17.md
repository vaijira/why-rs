# Multi-domain structural learning (chapter 17)

Chapter 17 extends the chapter 16 learners in
[crates/why-data/src/discovery/multidomain.rs](../../crates/why-data/src/discovery/multidomain.rs).
It reuses the FCI engine in `fci.rs`, and reads [discovery-chapter16.md](discovery-chapter16.md)
first for the shared conventions (edge text, F-node names, the oracles).

| Item | Book |
| --- | --- |
| `Regime` | one distribution: `(domain, targets, known)`; an empty known target is the domain's observational distribution |
| `augment_selection` | the augmented selection diagram (Def. 17.2.4), with F-nodes per Alg. 62 and S-nodes `S12`, … |
| `md_markov_equivalent` | Thm 17.2.5 (MAG of the augmented diagram + Prop. 16.1.1) |
| `SelectionOracle` | Prop. 17.2.4: invariances by d-separation in the augmented selection diagram |
| `md_fci`, `md_pc` | Algorithms 64 and 61, Phase I per Alg. 63 |

Facts that are easy to get wrong:

- **F-nodes**: one per pair of distributions `(m, l)`, `m < l`, except
  pairs of *known* observational distributions from different domains. Within
  a domain, `K` is the symmetric difference by mechanism. Across domains it
  is the union of the two targets (Assumption 17.2.3). **S-nodes**: one per
  pair of domains, pointing to the selection diagram's children plus `D_ij`,
  what every cross-domain pair of targets changes. That is empty when both
  domains have observations.
- **The oracle conditions on every augmented node outside the tested set.**
  Within a pair of distributions all other F- and S-nodes are constant, and
  Alg. 63's separating sets include them. Conditioning only on the other
  F-nodes misses separations that go through a collider whose parents are
  an F-node and an S-node. The random test found that case.
- **What MD-FCI learns.** By Eq. 17.7 an invariance across two domains
  separates the pair's F-node and S-node from `Y` together. So MD-FCI learns
  the PAG of the augmented diagram in which each cross-domain F-node also
  points to its domains' S-node children (`learning_target`, test-only).
- **MD-PC uses knowledge for known-target F-nodes**: without latents an
  F-node's children are exactly `K`. When no invariance separates such a
  node from a variable outside `K`, the edge is removed with no separating set
  (`fci::UNKNOWN_SEPSET`). Phase II, R4, and every rule that infers a
  non-collider from non-adjacency (FCI R1/R3/R9/R10, uncovered paths, Meek
  R1/R3) treat such pairs as undecided. An earlier version let Meek R1 orient
  through them and got edges backwards. This reproduces Ex. 17.23's MD-PDAG
  exactly.
- Rule 9′ (R12) applies only to F-nodes whose two targets are known and in
  one domain. Across domains an F-node's extra adjacency may come from the
  S-node, which is Ex. 17.27's point: pooling domains under MDIX makes I-FCI's
  R12 orient `X ← Y` wrongly.
- `md_fci_and_md_pc_are_sound_on_random_selection_diagrams` (120 seeds,
  with and without latents) and `theorem_17_4_1_domains_as_interventions`
  are the checks to rerun after touching `multidomain.rs` or `fci.rs`.

Reconstructions:

- Fig. 17.4(a) and Fig. 17.5(a): `X → Y → Z`, the domains differing at `X`
  and at `Y` respectively.
- Ex. 17.23: `X → Z → Y`, differing at `X`.
- Ex. 17.27: `X → Y`, differing at `X`, with regimes `⟨{}¹, {Y}¹, {}²⟩`. The
  text's `⟨{}¹, {Y}²⟩` contradicts its own figure, which has `F_y^1` and
  `F_y^{1,2}`.

There is no chapter 17 code in the book's companion Python.

Not implemented: the empirical comparisons of Sec. 17.5 (ICP, JCI, NSC,
MDLS), and structural changes across domains (Assumption 17.2.1 holds).
