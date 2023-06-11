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
mod svggraph;
mod svgvertex;

use app::App;
use futures_signals::signal::Mutable;
use svgvertex::VertexType;
use wasm_bindgen::prelude::*;

/// Node information to represent a vertex.
#[derive(Debug)]
pub struct NodeInfo {
    id: String,
    _weight: u32,
    layout_pos_x: f64,
    layout_pos_y: f64,
    vertex_type: Mutable<VertexType>,
}

impl NodeInfo {
    /// Create a new vertex.
    pub fn new(id: &str, layout_pos_x: f64, layout_pos_y: f64, vertex_type: VertexType) -> Self {
        NodeInfo {
            id: id.to_string(),
            _weight: 1,
            layout_pos_x,
            layout_pos_y,
            vertex_type: Mutable::new(vertex_type),
        }
    }
}

/// Default type for graphs
pub type ADMG = daggy::Dag<NodeInfo, u32, u32>;

#[wasm_bindgen(start)]
/// Main entry point for why-rs app
pub fn main_js() -> Result<(), JsValue> {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();

    wasm_logger::init(
        wasm_logger::Config::new(log::Level::Debug).module_prefix(env!("CARGO_PKG_NAME")),
    );

    let app = App::new();

    dominator::append_dom(&dominator::body(), App::render(app));

    Ok(())
}
