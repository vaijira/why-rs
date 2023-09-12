use std::sync::Arc;

use dominator::{clone, svg, Dom};
use futures_signals::signal::SignalExt;
use futures_signals::{
    signal::Mutable,
    signal_vec::{MutableVec, SignalVecExt},
};
use web_sys::HtmlElement;

use crate::{
    bounds::{Bounds, VIEWBOX_HEIGHT, VIEWBOX_WIDTH},
    svgedge::SvgEdge,
    svgvertex::SvgVertex,
    ADMG,
};

pub struct SvgGraph {
    pub(crate) admg: ADMG,
    pub(crate) container: Mutable<Option<HtmlElement>>,
    pub(crate) vertexes: MutableVec<Arc<SvgVertex>>,
    pub(crate) edges: MutableVec<Arc<SvgEdge>>,
    pub(crate) bounds: Mutable<Bounds>,
}

impl SvgGraph {
    pub fn new(admg: ADMG) -> Arc<Self> {
        let vertexes = MutableVec::new();
        for idx in admg.node_indices() {
            vertexes.lock_mut().push_cloned(SvgVertex::new(idx))
        }

        let edges = MutableVec::new();
        for idx in admg.edge_indices() {
            edges.lock_mut().push_cloned(SvgEdge::new(idx))
        }

        let bounds = Bounds::calculate_bounds(&admg, VIEWBOX_HEIGHT as i32, VIEWBOX_WIDTH as i32);

        Arc::new(Self {
            admg,
            container: Mutable::new(None),
            vertexes,
            edges,
            bounds: Mutable::new(bounds),
        })
    }

    pub fn render(g: Arc<Self>) -> Dom {
        svg!("svg", {
            .attr("alt", "ADMG graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr_signal("style", g.bounds.signal().map(
                clone!(g => move |_| {
                    log::debug!("setting svg style height: {} width: {}",
                                g.bounds.get().height, g.bounds.get().width);
                    format!("font-family: Arial, sans-serif; height: {}; width: {};",
                            g.bounds.get().height, g.bounds.get().width)
                 })
            ))
            .attr_signal("height", g.bounds.signal().map(
                clone!(g => move |_| {
                     log::debug!("setting svg height: {}", g.bounds.get().height);
                     g.bounds.get().height.to_string()
                 })
            ))
            .attr_signal("width", g.bounds.signal().map(
                clone!(g => move |_| {
                     log::debug!("setting svg width: {}", g.bounds.get().height);
                     g.bounds.get().height.to_string()
                 })
            ))
            .children_signal_vec(
                g.vertexes.signal_vec_cloned()
                .map(clone!(g => move |vertex| {
                    SvgVertex::render(vertex, g.clone())
                })
            ))
            .children_signal_vec(
                g.edges.signal_vec_cloned()
                .map(clone!(g => move |edge| {
                    SvgEdge::render(edge, g.clone())
                }))
            )
        })
    }
}
