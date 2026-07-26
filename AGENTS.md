# AGENTS.md

Guidance for coding agents working in this repository. Human-facing setup docs
live in [README.md](README.md).

## What this is

A Cargo workspace of Rust causal-inference tools plus a WASM web UI. The root
[Cargo.toml](Cargo.toml) is a **virtual manifest** — `[workspace]` and profiles
only, no `[package]`. Every crate lives under [crates/](crates/).

| Crate | Role |
| --- | --- |
| [crates/why-data/](crates/why-data/) | Graph/DAG data structures and algorithms |
| [crates/why-parser/](crates/why-parser/) | Parsers for causal model formats |
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
```

The root is a virtual manifest, so these cover all members by default — no
`--workspace` needed. As of the last verified run the workspace has 6 tests
(1 in `why-data`, 5 in `why-parser`), all passing; clippy reports pre-existing
warnings in `why-data` (6) and `why-parser` (4) and no errors.

Building `why-ui` for the browser needs the `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`). `wasm-bindgen` and `wasm-opt` come
from the `@wasm-tool/rollup-plugin-rust` and `binaryen` npm dev dependencies —
do not install them separately.

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
