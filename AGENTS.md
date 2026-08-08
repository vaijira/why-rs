# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup docs
live in [README.md](README.md).

## What this is

A Cargo workspace of Rust causal-inference tools plus a WASM web UI. The root
[Cargo.toml](Cargo.toml) is a **virtual manifest** — `[workspace]` and profiles
only, no `[package]`. Every crate lives under [crates/](crates/).

| Crate | Role |
| --- | --- |
| [crates/why-data/](crates/why-data/) | Graphs, d-separation, back-door, do-calculus, ID, symbolic SCMs |
| [crates/why-parser/](crates/why-parser/) | Parsers: dagitty (pest) and the book's `<NODES>/<EDGES>/<TASK>` format |
| [crates/why-ui/](crates/why-ui/) | `cdylib`+`rlib` WASM front end built with `dominator` |

The browser entry point is [crates/why-ui/src/lib.rs](crates/why-ui/src/lib.rs);
static assets (`index.html`, images) live in
[crates/why-ui/dist/](crates/why-ui/dist/) and generated output goes to
`crates/why-ui/dist/js/` (gitignored via the `**/dist/js/` pattern — note the
`**`, a single-level `*/dist/js/` would no longer match at this depth).

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
cargo run -p why-parser --example notebook_graphs  # the book's file format
```

The root is a virtual manifest, so these cover all members by default — no
`--workspace` needed. As of the last verified run the workspace has 86 tests
(64 in `why-data` plus 5 doctests, 16 in `why-parser` plus 1 doctest), all
passing, and clippy is clean workspace-wide (the warnings this file used to
record were fixed in f9aee15).

`why-data` has no Cargo features, and no `RUSTFLAGS` or `.cargo/config.toml` are
involved. The `scm` module was gated behind a feature until `rssn` 0.2.12 made
it buildable for WASM; the gate is gone and
[crates/why-data/src/scm.rs](crates/why-data/src/scm.rs) is always compiled.

### `rssn` and WASM

[crates/why-data/src/scm.rs](crates/why-data/src/scm.rs) depends on
[`rssn`](https://crates.io/crates/rssn) for symbolic expressions. It is an
unconditional dependency, so **`rssn` is in the browser bundle** — `why-ui`
depends on `why-data`.

It builds for `wasm32-unknown-unknown`, verified:

```sh
cargo build -p why-data --target wasm32-unknown-unknown
cargo build -p why-ui-rs --target wasm32-unknown-unknown
```

Three things make that work, and the third is why the `rssn` requirement is
pinned to an exact patch release rather than `0.2`:

1. **`getrandom`**, which `rssn` pulls in three generations of (`rand`
   0.8/0.9/0.10 → getrandom 0.2/0.3/0.4). All three are declared as
   target-gated dependencies of `why-data` with `js` (0.2) and
   `wasm_js` (0.3, 0.4). Features are enough: 0.3.4 and 0.4.3 pick the browser
   backend on `#[cfg(feature = "wasm_js")]` alone, despite 0.3.4's error text
   still insisting the feature is "insufficient" — that message is stale, and
   older 0.3.x really did also want `--cfg getrandom_backend="wasm_js"`.
2. **`uuid`** with its `js` feature, likewise.
3. **`rssn` 0.2.13 or newer.** Up to 0.2.11 it declared a bare
   `faer = "0.24"`, whose default features include `rayon` → `spindle` →
   `atomic-wait`, and `atomic-wait` 1.1 (latest, Jan 2023) has no wasm
   `platform` module, so the build died with `cannot find module platform`.

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
