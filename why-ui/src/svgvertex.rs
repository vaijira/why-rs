use dominator::{clone, events, svg, with_node, Dom};
use futures_signals::signal::Mutable;
use std::sync::Arc;
use web_sys::{SvgGraphicsElement, SvgPathElement};
use why_data::{
    graph::{dagitty::VertexType, CausalGraph, NodeIndex},
    types::Point,
};

use crate::svggraph::SvgGraph;

#[derive(Clone, Debug)]
pub struct SvgVertex {
    pub(crate) id: NodeIndex,
    marked: Mutable<bool>,
    dragging: Mutable<bool>,
}

const CSS_VERTEX_TYPE_NONE_FILL_COLOR: &str = "#aaaaaa";
const CSS_VERTEX_TYPE_EXPOSURE_FILL_COLOR: &str = "#bed403";
const CSS_VERTEX_TYPE_OUTCOME_FILL_COLOR: &str = "#00a2e0";
const CSS_VERTEX_TYPE_SELECTED_FILL_COLOR: &str = "#aaaaaa";
const CSS_VERTEX_TYPE_ADJUSTED_FILL_COLOR: &str = "#ffffff";
const CSS_VERTEX_TYPE_UNOBSERVED_FILL_COLOR: &str = "#00a2e0";

const CSS_VERTEX_TYPE_NONE_STROKE_COLOR: &str = "#666666";
const CSS_VERTEX_TYPE_EXPOSURE_STROKE_COLOR: &str = "#000000";
const CSS_VERTEX_TYPE_OUTCOME_STROKE_COLOR: &str = "#000000";
const CSS_VERTEX_TYPE_SELECTED_STROKE_COLOR: &str = "#666666";
const CSS_VERTEX_TYPE_ADJUSTED_STROKE_COLOR: &str = "#000000";
const CSS_VERTEX_TYPE_UNOBSERVED_STROKE_COLOR: &str = "#000000";

impl SvgVertex {
    pub fn new(id: NodeIndex) -> Arc<Self> {
        Arc::new(Self {
            id,
            marked: Mutable::new(false),
            dragging: Mutable::new(false),
        })
    }

    pub fn render(this: &Arc<Self>, svg_graph: &Arc<SvgGraph>) -> Dom {
        let info = match &*svg_graph.graph.lock_ref() {
            CausalGraph::Dag(g) => g.node_weight(this.id).unwrap().clone(),
            _ => unimplemented!(),
        };

        let children = vec![
            svg!("path" => SvgPathElement, {
                .attr("id", &*info.vertex_path_id.lock_ref())
                .attr("fill-opacity", "0.7")
                .attr("z-index", "1")
                .attr_signal("stroke-width", this.marked.signal_ref({|marked|
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
                        VertexType::Selected => CSS_VERTEX_TYPE_SELECTED_FILL_COLOR,
                        VertexType::Adjusted => CSS_VERTEX_TYPE_ADJUSTED_FILL_COLOR,
                        VertexType::Unobserved => CSS_VERTEX_TYPE_UNOBSERVED_FILL_COLOR,
                    }
                }))
                .attr_signal("stroke", info.vertex_type.signal_ref({|v_type|
                    match v_type {
                        VertexType::None => CSS_VERTEX_TYPE_NONE_STROKE_COLOR,
                        VertexType::Exposure => CSS_VERTEX_TYPE_EXPOSURE_STROKE_COLOR,
                        VertexType::Outcome => CSS_VERTEX_TYPE_OUTCOME_STROKE_COLOR,
                        VertexType::Selected => CSS_VERTEX_TYPE_SELECTED_STROKE_COLOR,
                        VertexType::Adjusted => CSS_VERTEX_TYPE_ADJUSTED_STROKE_COLOR,
                        VertexType::Unobserved => CSS_VERTEX_TYPE_UNOBSERVED_STROKE_COLOR,
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

        svg!("g" => SvgGraphicsElement, {
            .attr_signal("transform", info.layout_pos.signal_ref(clone!(svg_graph => move |layout_pos| {
                let point = svg_graph.bounds.lock_ref().to_svg_coordinates(layout_pos);
                log::trace!("Vertex transform translate to x:{} y:{}", point.x(), point.y());
                format!("translate({}, {})", point.x(), point.y())
            })))
            .attr("style", "cursor: move; touch-action: none;" )
            .children(children)
            .event(|e: events::DragStart| {
                e.prevent_default();
            })
            .with_node!(graph_element => {
                .event(clone!(this, svg_graph => move |e: events::PointerDown| {
                    this.marked.set(!this.marked.get());
                    if this.marked.get() {
                        svg_graph.vertexes.lock_mut().iter().filter(|v| v.id != this.id).for_each(|v| v.marked.set_neq(false));
                        svg_graph.current_variable.set(Some(info.clone()));
                    } else {
                        svg_graph.current_variable.set(None);
                    }
                    this.dragging.set_neq(true);
                    if graph_element.set_pointer_capture(e.pointer_id()).is_err() {
                        log::error!("Unable to capture pointer id for vertex");
                    }
                }))
            })
            .event(clone!(svg_graph, this => move |e: events::PointerMove| {
                if this.dragging.get() {
                    let info = match &*svg_graph.graph.lock_ref() {
                        CausalGraph::Dag(g) => g.node_weight(this.id).unwrap().clone(),
                        _ => unimplemented!(),
                    };

                    log::trace!("Vertex PointerMove event x:{} y:{}", e.x() , e.y());
                    log::trace!("Vertex PointerMove event page_x:{} page_y:{}", e.page_x() , e.page_y());

                    let ptr_x = e.page_x() - svg_graph.container.lock_ref().as_ref().map(|container| container.left()).unwrap_or(0);
                    let ptr_y = e.page_y() - svg_graph.container.lock_ref().as_ref().map(|container| container.top()).unwrap_or(0);

                    log::trace!("Vertex PointerMove event ptr_x:{} ptr_y:{}", ptr_x , ptr_y);

                    *info.layout_pos.lock_mut() = svg_graph.bounds.lock_ref().to_graph_coordinates(&Point::new(ptr_x as f64, ptr_y as f64));

                    log::trace!("Vertex PointerMove after graph_coordinates x:{} y:{}", info.layout_pos.lock_ref().x() , info.layout_pos.lock_ref().y());
                }
            }))
            .event(clone!(this => move |_: events::PointerUp| {
                this.dragging.set_neq(false);
            }))
        })
    }
}
