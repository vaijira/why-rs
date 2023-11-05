use std::rc::Rc;
use std::sync::Arc;

use dominator::{clone, svg, Dom};
use futures_signals::signal::SignalExt;
use futures_signals::{
    signal::Mutable,
    signal_vec::{MutableVec, SignalVecExt},
};
use web_sys::HtmlElement;
use why_data::graph::dagitty::{EdgeInfo, NodeInfo};
use why_data::graph::CausalGraph;

use crate::{
    bounds::{Bounds, VIEWBOX_HEIGHT, VIEWBOX_WIDTH},
    svgedge::SvgEdge,
    svgvertex::SvgVertex,
};

pub struct SvgGraph {
    pub(crate) graph: CausalGraph<NodeInfo, EdgeInfo>,
    pub(crate) container: Mutable<Option<HtmlElement>>,
    pub(crate) vertexes: MutableVec<Arc<SvgVertex>>,
    pub(crate) edges: MutableVec<Arc<SvgEdge>>,
    pub(crate) bounds: Mutable<Bounds>,
}

impl SvgGraph {
    pub fn new(graph: CausalGraph<NodeInfo, EdgeInfo>) -> Rc<Self> {
        let vertexes = MutableVec::new();
        let g = match &graph {
            CausalGraph::Dag(g) => g,
            _ => unimplemented!("Not implemented yet"),
        };
        for idx in g.node_indices() {
            vertexes.lock_mut().push_cloned(SvgVertex::new(idx))
        }

        let edges = MutableVec::new();
        for idx in g.edge_indices() {
            edges.lock_mut().push_cloned(SvgEdge::new(idx))
        }

        let bounds = Bounds::calculate_bounds(&graph, VIEWBOX_HEIGHT as i32, VIEWBOX_WIDTH as i32);

        Rc::new(Self {
            graph,
            container: Mutable::new(None),
            vertexes,
            edges,
            bounds: Mutable::new(bounds),
        })
    }

    pub fn render(svg_graph: Rc<Self>) -> Dom {
        svg!("svg", {
            .attr("alt", "Causal graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr_signal("style", svg_graph.bounds.signal().map(
                clone!(svg_graph => move |_| {
                    log::debug!("setting svg style height: {} width: {}",
                                svg_graph.bounds.get().height,
                                svg_graph.bounds.get().width);
                    format!("font-family: Arial, sans-serif; height: {}; width: {};",
                            svg_graph.bounds.get().height,
                            svg_graph.bounds.get().width)
                 })
            ))
            .attr_signal("height", svg_graph.bounds.signal().map(
                clone!(svg_graph => move |_| {
                     log::debug!("setting svg height: {}", svg_graph.bounds.get().height);
                     svg_graph.bounds.get().height.to_string()
                 })
            ))
            .attr_signal("width", svg_graph.bounds.signal().map(
                clone!(svg_graph => move |_| {
                     log::debug!("setting svg width: {}", svg_graph.bounds.get().height);
                     svg_graph.bounds.get().height.to_string()
                 })
            ))
            .children_signal_vec(
                svg_graph.vertexes.signal_vec_cloned()
                .map(clone!(svg_graph => move |vertex| {
                    SvgVertex::render(vertex, svg_graph.clone())
                })
            ))
            .children_signal_vec(
                svg_graph.edges.signal_vec_cloned()
                .map(clone!(svg_graph => move |edge| {
                    SvgEdge::render(edge, svg_graph.clone())
                }))
            )
        })
    }
}
