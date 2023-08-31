use std::sync::Arc;

use dominator::{clone, svg, Dom};
use futures_signals::signal_vec::{MutableVec, SignalVecExt};
use once_cell::sync::Lazy;

use crate::{
    bounds::{Bounds, VIEWBOX_HEIGHT, VIEWBOX_WIDTH},
    svgedge::SvgEdge,
    svgvertex::SvgVertex,
    ADMG,
};

pub struct SvgGraph {
    pub(crate) admg: ADMG,
    pub(crate) vertexes: MutableVec<Arc<SvgVertex>>,
    pub(crate) edges: MutableVec<Arc<SvgEdge>>,
}

static VIEWBOX_STR: Lazy<String> =
    Lazy::new(|| format!("0 0 {} {}", VIEWBOX_WIDTH, VIEWBOX_HEIGHT));

impl SvgGraph {
    pub fn new(admg: ADMG) -> Arc<Self> {
        let vertexes = MutableVec::new();
        for idx in admg.graph().node_indices() {
            vertexes.lock_mut().push_cloned(SvgVertex::new(idx))
        }

        let edges = MutableVec::new();
        for idx in admg.graph().edge_indices() {
            edges.lock_mut().push_cloned(SvgEdge::new(idx))
        }

        Arc::new(Self {
            admg,
            vertexes,
            edges,
        })
    }

    pub fn render(g: Arc<Self>) -> Dom {
        let bounds = Bounds::calculate_bounds(&g.admg);

        svg!("svg", {
            .attr("alt", "ADMG graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr("viewBox", &VIEWBOX_STR)
            .children_signal_vec(
                g.vertexes.signal_vec_cloned()
                .map(clone!(g, bounds => move |vertex| {
                    SvgVertex::render(vertex, g.clone(), bounds.clone())
                })
            ))
            .children_signal_vec(
                g.edges.signal_vec_cloned()
                .map(clone!(g => move |edge| {
                    SvgEdge::render(edge, g.clone(), bounds.clone())
                }))
            )
        })
    }
}
