use std::sync::Arc;

use daggy::NodeIndex;
use dominator::{clone, events, svg, Dom};
use futures_signals::signal::Mutable;

use crate::{bounds::Bounds, svggraph::SVGGraph};

/// vertex type
#[derive(Debug)]
pub enum VertexType {
    /// Default vertex type
    None,
    /// Outcome vertex
    Outcome,
    /// Exposure vertex
    Exposure,
}
#[derive(Clone, Debug)]
pub struct SVGVertex {
    id: NodeIndex,
    marked: Mutable<bool>,
}

const CSS_VERTEX_TYPE_NONE_FILL_COLOR: &str = "#aaaaaa";
const CSS_VERTEX_TYPE_EXPOSURE_FILL_COLOR: &str = "#bed403";
const CSS_VERTEX_TYPE_OUTCOME_FILL_COLOR: &str = "#00a2e0";

const CSS_VERTEX_TYPE_NONE_STROKE_COLOR: &str = "#666666";
const CSS_VERTEX_TYPE_EXPOSURE_STROKE_COLOR: &str = "#000000";
const CSS_VERTEX_TYPE_OUTCOME_STROKE_COLOR: &str = "#000000";

impl SVGVertex {
    pub fn new(id: NodeIndex) -> Arc<Self> {
        Arc::new(Self {
            id,
            marked: Mutable::new(false),
        })
    }

    pub fn render(v: Arc<SVGVertex>, g: Arc<SVGGraph>, bounds: &Bounds) -> Dom {
        let info = g.admg.node_weight(v.id).unwrap();
        let (x, y) = bounds.to_svg_coordinates(info.layout_pos_x, info.layout_pos_y);

        let translate = format!("translate({}, {})", x, y);
        let children = vec![
            svg!("path", {
                .attr("fill-opacity", "0.7")
                .attr("z-index", "1")
                .attr_signal("stroke-width", v.marked.signal_ref({|marked|
                    if *marked {
                        "4.5"
                    } else {
                        "1.5"
                    }
                }))
                .attr_signal("fill", info.vertex_type.signal_ref({|v_type|
                    match v_type {
                        VertexType::None => CSS_VERTEX_TYPE_NONE_FILL_COLOR,
                        VertexType::Exposure => CSS_VERTEX_TYPE_EXPOSURE_FILL_COLOR,
                        VertexType::Outcome => CSS_VERTEX_TYPE_OUTCOME_FILL_COLOR,
                    }
                }))
                .attr_signal("stroke", info.vertex_type.signal_ref({|v_type|
                    match v_type {
                        VertexType::None => CSS_VERTEX_TYPE_NONE_STROKE_COLOR,
                        VertexType::Exposure => CSS_VERTEX_TYPE_EXPOSURE_STROKE_COLOR,
                        VertexType::Outcome => CSS_VERTEX_TYPE_OUTCOME_STROKE_COLOR,
                    }
                }))
                .attr("d", "M 0 0 m 20, 0 a 20,15 0 1,1 -40,0 a 20,15 0 1,1 40,0")
            }),
            svg!("rect", {
                .attr("fill", "#ffffff")
            }),
            svg!("text", {
                .attr("text-anchor", "middle")
                .attr("y", "35")
                .text(&info.id)
            }),
        ];

        svg!("g", {
            .attr("transform", &translate)
            .attr("style", "cursor: move; touch-action: none;" )
            .children(children)
            .event(clone!(v => move |e: events::TouchStart| {
                v.marked.set(!v.marked.get());
                e.prevent_default();
            }))
            .event(|_: events::TouchEnd| {
            })
            .event(clone!(v => move |_: events::MouseDown| {
                v.marked.set(!v.marked.get());
            }))
            .event(|_: events::MouseMove| {

            })
            .event(|_: events::MouseLeave| {
            })
        })
    }
}
