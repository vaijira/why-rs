use std::sync::Arc;

use dominator::{clone, svg, Dom};
use futures_signals::signal::SignalExt;
use futures_signals::{
    signal::Mutable,
    signal_vec::{MutableVec, SignalVecExt},
};
use why_data::graph::dagitty::{EdgeInfo, NodeInfo};
use why_data::graph::CausalGraph;

use crate::bounds::ContainerCoordinates;
use crate::{
    bounds::{Bounds, VIEWBOX_HEIGHT, VIEWBOX_WIDTH},
    svgedge::SvgEdge,
    svgvertex::SvgVertex,
};

pub struct SvgGraph {
    pub(crate) graph: Mutable<CausalGraph<Arc<NodeInfo>, Arc<EdgeInfo>>>,
    pub(crate) container: Mutable<Option<ContainerCoordinates>>,
    pub(crate) vertexes: MutableVec<Arc<SvgVertex>>,
    pub(crate) edges: MutableVec<Arc<SvgEdge>>,
    pub(crate) bounds: Mutable<Bounds>,
    pub(crate) current_variable: Mutable<Option<Arc<NodeInfo>>>,
}

impl SvgGraph {
    pub fn new(graph: CausalGraph<Arc<NodeInfo>, Arc<EdgeInfo>>) -> Arc<Self> {
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

        Arc::new(Self {
            graph: Mutable::new(graph),
            container: Mutable::new(None),
            vertexes,
            edges,
            bounds: Mutable::new(bounds),
            current_variable: Mutable::new(None),
        })
    }

    pub fn render(this: &Arc<Self>) -> Dom {
        svg!("svg", {
            .attr("alt", "Causal graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr_signal("style", this.bounds.signal().map(
                clone!(this => move |_| {
                    log::debug!("setting svg style height: {} width: {}",
                                this.bounds.get().height,
                                this.bounds.get().width);
                    format!("font-family: Arial, sans-serif; height: {}; width: {};",
                            this.bounds.get().height,
                            this.bounds.get().width)
                 })
            ))
            .attr_signal("height", this.bounds.signal().map(
                clone!(this => move |_| {
                     log::debug!("setting svg height: {}", this.bounds.get().height);
                     this.bounds.get().height.to_string()
                 })
            ))
            .attr_signal("width", this.bounds.signal().map(
                clone!(this => move |_| {
                     log::debug!("setting svg width: {}", this.bounds.get().height);
                     this.bounds.get().height.to_string()
                 })
            ))
            .children_signal_vec(
                this.vertexes.signal_vec_cloned()
                .map(clone!(this => move |vertex| {
                    SvgVertex::render(&vertex, &this)
                })
            ))
            .children_signal_vec(
                this.edges.signal_vec_cloned()
                .map(clone!(this => move |edge| {
                    SvgEdge::render(&edge, &this)
                }))
            )
        })
    }
}
