use crate::{css::PATH_CLASS, graph::Point, svggraph::SvgGraph};
use dominator::{clone, events, svg, with_node, Dom};
use futures_signals::{map_ref, signal::Mutable};
use std::sync::Arc;
use why_data::graph::{EdgeIndex, NodeIndex};

/// vertex type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeType {
    /// Directed
    Directed,
    /// Bidirected
    Bidirected,
    /// Undirected
    Undirected,
}

#[derive(Clone, Debug)]
pub struct SvgEdge {
    id: EdgeIndex,
    marked: Mutable<bool>,
    dragging: Mutable<bool>,
}

impl SvgEdge {
    pub fn new(id: EdgeIndex) -> Arc<Self> {
        Arc::new(Self {
            id,
            marked: Mutable::new(false),
            dragging: Mutable::new(false),
        })
    }

    fn svg_edge_anchor(
        g: Arc<SvgGraph>,
        v1: NodeIndex,
        point_v1: &Point<f64>,
        point_v2: &Point<f64>,
        arrow_head: bool,
    ) -> Point<f64> {
        let svg_length = g
            .admg
            .node_weight(v1)
            .unwrap()
            .vertex_path_element
            .lock_ref()
            .as_ref()
            .map(|path| path.get_total_length())
            .unwrap_or(0.0);

        let dx = point_v2.x() - point_v1.x();
        let dy = point_v2.y() - point_v1.y();
        let length = f64::sqrt((dx * dx) + (dy * dy));
        let length = if length < 0.01 { 0.01 } else { length };

        let svg_point = if dy > 0.0 {
            g.admg
                .node_weight(v1)
                .unwrap()
                .vertex_path_element
                .lock_ref()
                .as_ref()
                .map(|path| {
                    path.get_point_at_length(
                        (f64::acos(dx / length) / 2.0 / std::f64::consts::PI * svg_length as f64)
                            as f32,
                    )
                })
        } else {
            g.admg
                .node_weight(v1)
                .unwrap()
                .vertex_path_element
                .lock_ref()
                .as_ref()
                .map(|path| {
                    path.get_point_at_length(
                        (1.0 - f64::acos(dx / length) / 2.0 / std::f64::consts::PI) as f32
                            * svg_length,
                    )
                })
        }
        .unwrap()
        .ok()
        .map(|p| Point::new(p.x() as f64, p.y() as f64))
        .unwrap_or(Point::new(0.0, 0.0));

        let lp = f64::sqrt(svg_point.x() * svg_point.x() + svg_point.y() * svg_point.y());

        let elongate = if arrow_head { 1.0 } else { (lp + 5.0) / lp };

        Point::new(
            svg_point.x() * elongate + point_v1.x(),
            svg_point.y() * elongate + point_v1.y(),
        )
    }

    fn svg_edge_anchors(
        edge: Arc<SvgEdge>,
        g: Arc<SvgGraph>,
        point_v1: &Point<f64>,
        point_v2: &Point<f64>,
    ) -> (Point<f64>, Point<f64>) {
        let edge_info = g.admg.edge_weight(edge.id).unwrap();
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let edge_type = *edge_info.edge_type.lock_ref();
        let edge_point = edge_info.layout_pos.get();

        let p2 = edge_point
            .map(|p| g.bounds.lock_ref().to_svg_coordinates(&p))
            .unwrap_or(*point_v2);
        let arrow_head = edge_type == EdgeType::Undirected || edge_type == EdgeType::Directed;
        let v1_anchor = SvgEdge::svg_edge_anchor(g.clone(), v1, point_v1, &p2, arrow_head);

        let p1 = edge_point
            .map(|p| g.bounds.lock_ref().to_svg_coordinates(&p))
            .unwrap_or(*point_v1);
        let arrow_head = edge_type == EdgeType::Undirected;
        let v2_anchor = SvgEdge::svg_edge_anchor(g, v2, point_v2, &p1, arrow_head);

        (v1_anchor, v2_anchor)
    }

    fn calculate_arrow(edge: Arc<SvgEdge>, g: Arc<SvgGraph>) -> String {
        let layout_pos = g.admg.edge_weight(edge.id).unwrap().layout_pos.get();
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let info_v1 = g.admg.node_weight(v1).unwrap();
        let info_v2 = g.admg.node_weight(v2).unwrap();

        let point_v1 = g
            .bounds
            .lock_ref()
            .to_svg_coordinates(&info_v1.layout_pos.get());
        let point_v2 = g
            .bounds
            .lock_ref()
            .to_svg_coordinates(&info_v2.layout_pos.get());

        let (_anchor_back, anchor_front) =
            SvgEdge::svg_edge_anchors(edge.clone(), g.clone(), &point_v1, &point_v2);

        let sxy = if let Some(p) = layout_pos {
            g.bounds.lock_ref().to_svg_coordinates(&p)
        } else {
            point_v1
        };

        let mut afront = 360.0
            * f64::atan((anchor_front.y() - sxy.y()) / (anchor_front.x() - sxy.x()))
            / 2.0
            / std::f64::consts::PI;
        if sxy.x() < anchor_front.x() {
            afront += 180.0;
        }
        if sxy.x() == anchor_front.x() {
            afront = if anchor_front.y() > sxy.y() {
                -90.0
            } else {
                90.0
            }
        };

        format!(
            "translate({}, {}) rotate({})",
            anchor_front.x(),
            anchor_front.y(),
            afront
        )
    }

    fn calculate_edge(edge: Arc<SvgEdge>, g: Arc<SvgGraph>) -> String {
        let layout_pos = g.admg.edge_weight(edge.id).unwrap().layout_pos.get();
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let info_v1 = g.admg.node_weight(v1).unwrap();
        let info_v2 = g.admg.node_weight(v2).unwrap();

        let point_v1 = g
            .bounds
            .lock_ref()
            .to_svg_coordinates(&info_v1.layout_pos.get());
        let point_v2 = g
            .bounds
            .lock_ref()
            .to_svg_coordinates(&info_v2.layout_pos.get());

        let (anchor_back, anchor_front) =
            SvgEdge::svg_edge_anchors(edge.clone(), g.clone(), &point_v1, &point_v2);
        let line_path = if let Some(p) = layout_pos {
            let p = g.bounds.lock_ref().to_svg_coordinates(&p);
            format!(
                "M{:.2},{:.2}Q{:.2},{:.2},{:.2},{:.2}",
                anchor_back.x(),
                anchor_back.y(),
                p.x(),
                p.y(),
                anchor_front.x(),
                anchor_front.y()
            )
        } else {
            format!(
                "M{:.2},{:.2}L{:.2},{:.2}",
                anchor_back.x(),
                anchor_back.y(),
                anchor_front.x(),
                anchor_front.y()
            )
        };

        line_path
    }

    pub fn render(edge: Arc<SvgEdge>, g: Arc<SvgGraph>) -> Dom {
        let edge_info = g.admg.edge_weight(edge.id).unwrap();
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let info_v1 = g.admg.node_weight(v1).unwrap();
        let info_v2 = g.admg.node_weight(v2).unwrap();

        let children = vec![
            svg!("path", {
                .attr("stroke-width", "1.5")
                .attr("fill", "none")
                .attr("stroke", "black")
                .attr_signal("d", clone!(edge, g => {
                    map_ref! {
                    let _v1 = info_v1.layout_pos.signal_cloned(),
                    let _v2 = info_v2.layout_pos.signal_cloned(),
                    let _edge = edge_info.layout_pos.signal_cloned() => move {
                        SvgEdge::calculate_edge(edge.clone(), g.clone())
                    }
                }}))
            }),
            svg!("path", {
                .attr("stroke-width", "1.5")
                .attr("fill", "white")
                .attr("stroke", "black")
                .attr("d", "M-1,0L15,5L15,-5Z")
                .attr_signal("transform", clone!(edge, g => {
                    map_ref! {
                    let _v1 = info_v1.layout_pos.signal_cloned(),
                    let _v2 = info_v2.layout_pos.signal_cloned(),
                    let _edge = edge_info.layout_pos.signal_cloned() => move {
                        SvgEdge::calculate_arrow(edge.clone(), g.clone())
                    }
                }}))
            }),
        ];

        svg!("g", {
            .class(&*PATH_CLASS)
            .attr("style", "cursor: move; touch-action: none;" )
            .children(children)
            .event(|e: events::DragStart| {
                e.prevent_default();
            })
            .with_node!(graph_element => {
                .event(clone!(edge => move |e: events::PointerDown| {
                    edge.marked.set(!edge.marked.get());
                    edge.dragging.set_neq(true);
                    if graph_element.set_pointer_capture(e.pointer_id()).is_err() {
                        log::error!("Unable to capture pointer id for edge");
                    }
                }))
            })
            .event(clone!(g, edge => move |e: events::PointerMove| {
                if edge.dragging.get() {
                    let info = g.admg.edge_weight(edge.id).unwrap();
                    log::debug!("Edge PointerMove event x:{} y:{}", e.x() , e.y());
                    log::debug!("Edge PointerMove event page_x:{} page_y:{}", e.page_x() , e.page_y());
                    let ptr_x = e.page_x() - g.container.lock_ref().as_ref().map(|element| element.client_left()).unwrap_or(0);
                    let ptr_y = e.page_y() - g.container.lock_ref().as_ref().map(|element| element.client_top()).unwrap_or(0);
                    log::debug!("Edge PointerMove event ptr_x:{} ptr_y:{}", ptr_x , ptr_y);
                    *info.layout_pos.lock_mut() = Some(g.bounds.lock_ref().to_graph_coordinates(&Point::new(ptr_x as f64, ptr_y as f64)));
                    log::debug!("Edge PointerMove after graph_coordinates x:{} y:{}",
                                info.layout_pos.lock_ref().unwrap().x() ,
                                info.layout_pos.lock_ref().unwrap().y());
                }
            }))
            .event(clone!(edge => move |_: events::PointerUp| {
                edge.dragging.set_neq(false);
            }))
        })
    }
}
