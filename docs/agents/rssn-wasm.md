# `rssn` and WASM

[crates/why-data/src/scm.rs](../../crates/why-data/src/scm.rs) depends on
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
