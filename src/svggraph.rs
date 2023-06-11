use std::sync::Arc;

use dominator::{clone, svg, Dom};
use futures_signals::signal_vec::{MutableVec, SignalVecExt};
use once_cell::sync::Lazy;

use crate::{
    bounds::{Bounds, VIEWBOX_HEIGHT, VIEWBOX_WIDTH},
    svgvertex::SVGVertex,
    ADMG,
};

pub struct SVGGraph {
    pub(crate) admg: ADMG,
    pub(crate) vertexes: MutableVec<Arc<SVGVertex>>,
}

static VIEWBOX_STR: Lazy<String> =
    Lazy::new(|| format!("0 0 {} {}", VIEWBOX_WIDTH, VIEWBOX_HEIGHT));

impl SVGGraph {
    pub fn new(admg: ADMG) -> Arc<Self> {
        let vertexes = MutableVec::new();
        for idx in admg.graph().node_indices() {
            vertexes.lock_mut().push_cloned(SVGVertex::new(idx))
        }
        Arc::new(Self { admg, vertexes })
    }

    pub fn render(g: Arc<Self>) -> Dom {
        let bounds = Bounds::calculate_bounds(&g.admg);

        svg!("svg", {
            .attr("alt", "ADMG graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr("viewBox", &VIEWBOX_STR)
            .children_signal_vec(
                g.vertexes.signal_vec_cloned()
                .map(clone!(g => move |vertex| {
                    SVGVertex::render(vertex, g.clone(), &bounds)
                })
            ))
        })
    }
}
