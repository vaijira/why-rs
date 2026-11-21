# C-INFER and ID

[crates/why-data/src/graph/cinfer.rs](../../crates/why-data/src/graph/cinfer.rs)
(chapter 4.4: q-trees, d-trees, the `Σ`/`δ`/`σ`/`Γ` operators) and
[crates/why-data/src/graph/identification.rs](../../crates/why-data/src/graph/identification.rs)
(Shpitser-Pearl ID/IDC) share `Formula` and the private `Dist` helper. On
observational input they must identify exactly the same queries: both are
complete. `cinfer`'s `agrees_with_id_and_is_numerically_sound` test checks
this over random graphs. It also evaluates every estimand against the true
effect of a random latent-confounder model, so run it after touching either
file. Estimand *strings* can differ: C-INFER conditions each factor on its
district's blanket, and ID conditions on every predecessor.

[crates/why-data/src/graph/cinfer/counterfactual.rs](../../crates/why-data/src/graph/cinfer/counterfactual.rs)
extends C-INFER to counterfactual queries (chapter 5, Correa, Lee & Bareinboim
2021). It adds CUT unnesting, the Ξ collapse of consistent ctf-factors and a
counterfactual Γ, and reuses the layer-2 d-trees through `grow_input_tree`.
`CtfQuery::parse` reads textbook notation such as `P(y_x | x')`. Its
`estimands_are_numerically_sound` test builds random SCMs with response
functions and per-variable noise shared across worlds. It checks every
estimand against the exact counterfactual probability, and checks that
`P(y_x)` is identified exactly when layer 2 identifies it. It takes about 30s
in debug. The book's Python (`_counterfactual_engine.py`) is not a reliable
oracle: its `simplify` zeroes any variable with two values across worlds, and
its ancestral components never merge across bidirected edges.
