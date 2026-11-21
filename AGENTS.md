# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup docs
live in [README.md](README.md). Per-module notes live in
[docs/agents/](docs/agents/). Read the one for a module before changing it.

## What this is

A Cargo workspace of Rust causal-inference tools, ported from the Causal AI
book, plus a WASM web UI. The root [Cargo.toml](Cargo.toml) is a **virtual
manifest** (`[workspace]` and profiles only, no `[package]`). Every crate
lives under [crates/](crates/).

| Crate | Role |
| --- | --- |
| [crates/why-data/](crates/why-data/) | Graphs, d-separation, back-door, do-/σ-calculus, ID, C-INFER, symbolic SCMs, and the book's chapters 6, 8, 9, 11, 12, 14, 15, 16 and the tensor-free part of 13 |
| [crates/why-ncm/](crates/why-ncm/) | Neural causal models trained with `burn` (chapter 13), the ANCM image generator (chapter 14) and NeuralAbstractID (chapter 15). A separate crate so `burn` stays out of the `why-ui` bundle |
| [crates/why-parser/](crates/why-parser/) | Parsers: dagitty (pest) and the book's section format (`<NODES>`, `<EDGES>`, `<TASK>`, `<QUERY>`, `<INTERVENTIONS>`) |
| [crates/why-ui/](crates/why-ui/) | `cdylib`+`rlib` WASM front end built with `dominator` |

The browser entry point is [crates/why-ui/src/lib.rs](crates/why-ui/src/lib.rs);
static assets (`index.html`, images) live in
[crates/why-ui/dist/](crates/why-ui/dist/) and generated output goes to
`crates/why-ui/dist/js/` (gitignored via the `**/dist/js/` pattern — note the
`**`, a single-level `*/dist/js/` would no longer match at this depth).

## Module notes

| Module | Notes |
| --- | --- |
| `graph/cinfer.rs`, `graph/identification.rs`, `cinfer/counterfactual.rs` | [cinfer-id.md](docs/agents/cinfer-id.md) |
| `graph/sigmacalculus.rs`, `why_parser::fusion` | [sigma-calculus.md](docs/agents/sigma-calculus.md) |
| `fairness` (§6.1) | [fairness.md](docs/agents/fairness.md) |
| `explain` (§6.2) | [explanations.md](docs/agents/explanations.md) |
| `crl`, chapter 8 | [crl-chapter8.md](docs/agents/crl-chapter8.md) |
| `crl`, chapter 9 | [crl-chapter9.md](docs/agents/crl-chapter9.md) |
| `transport` (chapter 11) | [transport-chapter11.md](docs/agents/transport-chapter11.md) |
| `generalize` (chapter 12) | [generalize-chapter12.md](docs/agents/generalize-chapter12.md) |
| `ncm`, `why-ncm` (chapter 13) | [ncm-chapter13.md](docs/agents/ncm-chapter13.md) |
| `ascm`, `why_ncm::ancm` (chapter 14) | [ascm-chapter14.md](docs/agents/ascm-chapter14.md) |
| `abstraction`, `why_ncm::neural_abstract_id` (chapter 15) | [abstraction-chapter15.md](docs/agents/abstraction-chapter15.md) |
| `discovery` (chapter 16) | [discovery-chapter16.md](docs/agents/discovery-chapter16.md) |
| `scm`, `rssn`, WASM builds | [rssn-wasm.md](docs/agents/rssn-wasm.md) |

Each file lists the deliberate departures from the book, the errata found, and
the random-model tests to rerun after touching the module.

## The `yarn` vs `yarnpkg` gotcha

On Ubuntu/Debian the Yarn binary is installed as **`yarnpkg`** — `yarn` is taken
by the unrelated `cmdtest` package. The maintainer has a shell alias, but that
alias is **not** visible to non-interactive tool shells, so `yarn` fails with
`command not found`.

**Always invoke `yarnpkg` directly.** If it is missing, fall back to `yarn`
before concluding the environment is broken.

## Running the web UI

Run all three from the repository root:

```sh
yarnpkg install    # first time only, or after package.json changes
yarnpkg build      # release build -> crates/why-ui/dist/js/
yarnpkg start      # dev server + watch + live reload
```

- `build` = `rimraf crates/why-ui/dist/js && rollup --config`. Compiles
  `why-ui` to WASM in the release profile, runs `wasm-opt -Oz`, minifies with
  terser.
- `start` = same, plus `--watch`. Serves `crates/why-ui/dist` on
  **<http://localhost:10001>**, uses the `dev` profile, skips terser, and opens
  a browser (`open: true` in [rollup.config.js](rollup.config.js)).

All four crate paths in [rollup.config.js](rollup.config.js) (`input.index`,
`output.dir`, `serve.contentBase`, `livereload`) and both
[package.json](package.json) `rimraf` paths point at `crates/why-ui` — update
them together if the layout changes again.

### Verifying the UI without a browser

`yarnpkg start` never exits, so run it in the background, wait for the
`http://localhost:10001` line in its log, then check the three artifacts:

```sh
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:10001/
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:10001/js/index.js
# the wasm filename is content-hashed; read it out of index.js first
curl -s http://localhost:10001/js/index.js | grep -o 'assets/why_ui_rs-[a-z0-9]*\.wasm'
```

All three should return `200`. A `200` on `index.js` alone is not enough — the
WASM asset is fetched separately and is the part that usually breaks.

Stop the background process when done; do not leave a server holding port 10001.

## Rust checks

```sh
cargo build
cargo test
cargo clippy --all-targets
cargo test -p why-data     # single crate

# Ports of the Causal AI book's chapter 2 notebooks. Both assert every value
# against the number the Python original produces, so they fail loudly on a
# regression in `scm`; treat them as tests you can read.
cargo run -p why-data --example chapter2_part1   # SCM, d-separation
cargo run -p why-data --example chapter2_part2
cargo run -p why-data --example chapter4_part1   # back-door criterion
cargo run -p why-data --example chapter4_part2   # do-calculus + ID
cargo run -p why-data --example chapter4_cinfer  # C-INFER, q-trees and d-trees
cargo run -p why-data --example chapter6_fairness   # Tasks 1–2 (no notebook; see below)
cargo run -p why-data --example chapter6_decisions  # Task 3, checks the book's Ex. 6.32/6.35 numbers
cargo run -p why-data --example chapter6_explanations  # §6.2: SHAP vs do-Shapley vs L3, Ex. 6.37–6.43
cargo run -p why-data --example chapter8_crl  # Ex. 8.6, 8.16–8.18, 8.23–8.25, RCT vs UCB
cargo run -p why-data --example chapter9_crl  # Tables 9.2–9.3, 9.6–9.7, 9.15; UCB+, Ctf-UCB, CNE
cargo run -p why-data --example chapter11_transport  # Exs. 11.5–11.34, Algs. 32–37
cargo run -p why-data --example chapter12_generalize  # Exs. 12.1–12.20, Algs. 40–43
cargo run -p why-data --example chapter16_discovery  # Exs. 16.4–16.24, Algs. 53–59, FCI on samples
cargo run -p why-ncm --example chapter13_ncm  # Exs. 13.6–13.16, Algs. 44–49 (trains; ~1 min)
cargo run -p why-ncm --example chapter14_images  # Exs. 14.2–14.9, MNIST bounds, Alg. 50 (trains; ~75 s)
cargo run -p why-ncm --example chapter15_abstractions  # Exs. 15.5–15.10, Algs. 51–52, nutrition study (trains; ~50 s)
cargo run -p why-parser --example notebook_graphs  # the book's file format
```

The root is a virtual manifest, so these cover all members by default — no
`--workspace` needed. As of the last verified run the workspace has 330 tests
(251 in `why-data` plus 45 doctests, 10 in `why-ncm` plus 1 doctest, 21 in
`why-parser` plus 2 doctests), all passing. The `why-ncm` tests train
networks and take about 31 s. Clippy 1.100 reports one lint, `needless_range_loop` in
`fairness/estimate.rs`, which older toolchains did not flag; the warnings this
file used to record were fixed in f9aee15. The tree is not rustfmt-clean, so do not run
`cargo fmt` over it as part of an unrelated change.

`why-data` has no Cargo features, and no `RUSTFLAGS` or `.cargo/config.toml` are
involved. The `scm` module was gated behind a feature until `rssn` 0.2.12 made
it buildable for WASM; the gate is gone and
[crates/why-data/src/scm.rs](crates/why-data/src/scm.rs) is always compiled.


## `rssn` and WASM

`rssn` is an unconditional dependency of `why-data`, so it ships in the
browser bundle. Keep its requirement pinned to an exact patch release, 0.2.13
or newer, never a bare `0.2`: older releases pull in `atomic-wait`, which does
not build for wasm. Verify with:

```sh
cargo build -p why-data --target wasm32-unknown-unknown
cargo build -p why-ui-rs --target wasm32-unknown-unknown
```

The getrandom/uuid feature setup is in [rssn-wasm.md](docs/agents/rssn-wasm.md).

## Conventions

- All three crates are on edition 2024. Note the 2024 match-ergonomics rule:
  an explicit `ref` inside a pattern that already binds by reference (e.g.
  `if let Some(ref x) = &opt`) is a hard error — drop the `ref`, the binding
  type is unchanged.
- The release profile is tuned for size (`opt-level = "z"`, `lto`,
  `panic = "abort"`) because the output ships as WASM. Keep that in mind before
  adding heavy dependencies to `why-ui`.
- Do not edit `crates/why-ui/dist/js/`; it is gitignored and regenerated on
  every build. `crates/why-ui/dist/index.html` and `crates/why-ui/dist/images/`
  **are** tracked sources — edit those freely.
- `Cargo.lock` and `yarn.lock` are gitignored and untracked, so dependency
  resolution is not pinned across clones. If a build breaks after a fresh
  `yarnpkg install`, an upstream version bump is a plausible cause.
