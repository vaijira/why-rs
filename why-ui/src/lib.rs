#![warn(
    rust_2018_idioms,
    missing_docs,
    missing_debug_implementations,
    unused_extern_crates,
    warnings
)]

//! WASM app to show causal information

mod app;
mod bounds;
mod css;
mod graph;
mod svgedge;
mod svggraph;
mod svgvertex;

use app::App;

use graph::{EdgeInfo, NodeInfo};
use wasm_bindgen::prelude::*;
use why_data::graph::CausalGraph;

/// Default type for graphs
pub type ADMG = CausalGraph<NodeInfo, EdgeInfo>;

#[wasm_bindgen(start)]
/// Main entry point for why-rs app
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    wasm_logger::init(
        wasm_logger::Config::new(log::Level::Debug), //.module_prefix(env!("CARGO_PKG_NAME")),
    );

    let app = App::new();

    dominator::append_dom(&dominator::body(), App::render(app));

    Ok(())
}
