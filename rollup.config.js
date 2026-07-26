import nodeResolve from '@rollup/plugin-node-resolve';
import rust from "@wasm-tool/rollup-plugin-rust";
import serve from "rollup-plugin-serve";
import livereload from "rollup-plugin-livereload";
import { terser } from "rollup-plugin-terser";
import copy from 'rollup-plugin-copy';
import commonjs from '@rollup/plugin-commonjs';

const is_watch = !!process.env.ROLLUP_WATCH;

export default {
    input: {
        index: "./crates/why-ui/Cargo.toml",
    },
    output: {
        dir: "crates/why-ui/dist/js",
        format: "es",
        sourcemap: true,
    },
    plugins: [
        nodeResolve(),

        rust({
            extraArgs: {
              wasmOpt: [ "-Oz", "--enable-bulk-memory-opt", "--enable-nontrapping-float-to-int" ],
            },
        }),


        copy({
            targets: [
            ]
        }),

        commonjs(),

        is_watch && serve({
            contentBase: "crates/why-ui/dist",
            open: true,
        }),

        is_watch && livereload("crates/why-ui/dist"),

        !is_watch && terser(),
    ],
};
