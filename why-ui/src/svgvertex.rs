use std::sync::Arc;

use daggy::NodeIndex;
use dominator::{clone, events, svg, with_node, Dom};
use futures_signals::signal::Mutable;
use web_sys::{SvgGraphicsElement, SvgPathElement};

use crate::{graph::Point, svggraph::SvgGraph};

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
pub struct SvgVertex {
    id: NodeIndex,
    marked: Mutable<bool>,
    dragging: Mutable<bool>,
}

const CSS_VERTEX_TYPE_NONE_FILL_COLOR: &str = "#aaaaaa";
const CSS_VERTEX_TYPE_EXPOSURE_FILL_COLOR: &str = "#bed403";
const CSS_VERTEX_TYPE_OUTCOME_FILL_COLOR: &str = "#00a2e0";

const CSS_VERTEX_TYPE_NONE_STROKE_COLOR: &str = "#666666";
const CSS_VERTEX_TYPE_EXPOSURE_STROKE_COLOR: &str = "#000000";
const CSS_VERTEX_TYPE_OUTCOME_STROKE_COLOR: &str = "#000000";

impl SvgVertex {
    pub fn new(id: NodeIndex) -> Arc<Self> {
        Arc::new(Self {
            id,
            marked: Mutable::new(false),
            dragging: Mutable::new(false),
        })
    }

    pub fn render(v: Arc<SvgVertex>, g: Arc<SvgGraph>) -> Dom {
        let info = g.admg.node_weight(v.id).unwrap();

        let children = vec![
            svg!("path" => SvgPathElement, {
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
                .with_node!(path_element => {
                    .after_inserted(clone!(g, v  => move |_| {
                         *g.admg.node_weight(v.id).unwrap().vertex_path_element.lock_mut() = Some(path_element);
                    }))
                })
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

        svg!("g" => SvgGraphicsElement, {
            .attr_signal("transform", info.layout_pos.signal_ref(clone!(g => move |layout_pos| {
                let point = g.bounds.lock_ref().to_svg_coordinates(layout_pos);
                log::trace!("Vertex transform translate to x:{} y:{}", point.x(), point.y());
                format!("translate({}, {})", point.x(), point.y())
            })))
            .attr("style", "cursor: move; touch-action: none;" )
            .children(children)
            .event(|e: events::DragStart| {
                e.prevent_default();
            })
            .with_node!(graph_element => {
                .event(clone!(v => move |e: events::PointerDown| {
                    v.marked.set(!v.marked.get());
                    v.dragging.set_neq(true);
                    if graph_element.set_pointer_capture(e.pointer_id()).is_err() {
                        log::error!("Unable to capture pointer id for vertex");
                    }
                }))
            })
            .event(clone!(g, v => move |e: events::PointerMove| {
                if v.dragging.get() {
                    let info = g.admg.node_weight(v.id).unwrap();

                    log::trace!("Vertex PointerMove event x:{} y:{}", e.x() , e.y());
                    log::trace!("Vertex PointerMove event page_x:{} page_y:{}", e.page_x() , e.page_y());

                    let ptr_x = e.page_x() - g.container.lock_ref().as_ref().map(|element| element.client_left()).unwrap_or(0);
                    let ptr_y = e.page_y() - g.container.lock_ref().as_ref().map(|element| element.client_top()).unwrap_or(0);

                    log::trace!("Vertex PointerMove event ptr_x:{} ptr_y:{}", ptr_x , ptr_y);

                    *info.layout_pos.lock_mut() = g.bounds.lock_ref().to_graph_coordinates(&Point::new(ptr_x as f64, ptr_y as f64));

                    log::trace!("Vertex PointerMove after graph_coordinates x:{} y:{}", info.layout_pos.lock_ref().x() , info.layout_pos.lock_ref().y());
                }
            }))
            .event(clone!(v => move |_: events::PointerUp| {
                v.dragging.set_neq(false);
            }))
        })
    }
}
