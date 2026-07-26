# why-rs

Rust causal related tools. Inspired by daggity and pywhy.

## Workspace layout

The repository root is a *virtual* Cargo workspace — it declares members only,
it is not a package itself. Every crate lives under [crates/](crates/):

| Crate / directory | Purpose |
| --- | --- |
| [crates/why-data/](crates/why-data/) | Causal data structures (graphs, DAG algorithms) |
| [crates/why-parser/](crates/why-parser/) | Parsers for causal model formats |
| [crates/why-ui/](crates/why-ui/) | Web UI, a WASM crate built with [dominator](https://crates.io/crates/dominator) |
| [crates/why-ui/dist/](crates/why-ui/dist/) | Static assets served to the browser (`index.html`, images) |
| [examples/](examples/) | Sample datasets |

## Prerequisites

- **Rust** with the `wasm32-unknown-unknown` target:

  ```sh
  rustup target add wasm32-unknown-unknown
  ```

- **Node.js** (tested with v22) and **Yarn 4**.

  > **Ubuntu/Debian note:** the `yarn` name is taken by the unrelated `cmdtest`
  > package, so the Yarn binary installed from the distro repositories is called
  > **`yarnpkg`**. Use `yarnpkg` in place of `yarn` in every command below, or
  > define an alias:
  >
  > ```sh
  > alias yarn=yarnpkg
  > ```

`wasm-bindgen` and `wasm-opt` are **not** installed manually — the
`@wasm-tool/rollup-plugin-rust` and `binaryen` dev dependencies provide them.

## Running the web UI

1. **Install the JavaScript dependencies** (first time only, or after
   `package.json` changes):

   ```sh
   yarn install     # Ubuntu: yarnpkg install
   ```

2. **Build** a production bundle:

   ```sh
   yarn build       # Ubuntu: yarnpkg build
   ```

   This wipes `crates/why-ui/dist/js`, compiles `why-ui` to WASM in release
   mode, runs `wasm-opt -Oz`, and minifies the JS glue. Output:

   ```text
   crates/why-ui/dist/js/index.js
   crates/why-ui/dist/js/index.js.map
   crates/why-ui/dist/js/assets/why_ui_rs-<hash>.wasm
   ```

3. **Test locally** with the watching dev server:

   ```sh
   yarn start       # Ubuntu: yarnpkg start
   ```

   This serves `crates/why-ui/dist` at **<http://localhost:10001>**, opens your
   default browser automatically, and rebuilds plus live-reloads whenever a Rust
   or JS source file changes. Unlike `yarn build`, it compiles in the `dev`
   profile and skips minification. Press `Ctrl+C` to stop.

## Rust-only workflow

The Rust crates can be checked without going through Yarn:

```sh
cargo build
cargo test
cargo clippy --all-targets
```

Because the root manifest is a virtual workspace, these commands apply to every
member crate by default — no `--workspace` flag needed. To work on a single
crate, use `-p`:

```sh
cargo test -p why-data
```

## License

See [LICENSE](LICENSE).
