use daggy::{EdgeIndex, NodeIndex};
use dominator::{clone, events, svg, Dom};
use futures_signals::signal::Mutable;
use std::sync::Arc;

use crate::{bounds::Bounds, css::PATH_CLASS, graph::Point, svggraph::SvgGraph};

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
}

impl SvgEdge {
    pub fn new(id: EdgeIndex) -> Arc<Self> {
        Arc::new(Self {
            id,
            marked: Mutable::new(false),
        })
    }

    fn svg_edge_anchor(
        g: Arc<SvgGraph>,
        v1: NodeIndex,
        point_v1: &Point,
        point_v2: &Point,
        arrow_head: bool,
    ) -> Point {
        let svg_length = g
            .admg
            .node_weight(v1)
            .unwrap()
            .vertex_path_element
            .lock_ref()
            .as_ref()
            .and_then(|path| Some(path.get_total_length()))
            .unwrap();

        let dx = point_v2.x() - point_v1.x();
        let dy = point_v2.x() - point_v1.x();
        let length = f64::sqrt((dx * dx) + (dy * dy));
        let length = if length < 0.01 { 0.01 } else { length };

        let svg_point = if dy > 0.0 {
            g.admg
                .node_weight(v1)
                .unwrap()
                .vertex_path_element
                .lock_ref()
                .as_ref()
                .and_then(|path| {
                    Some(path.get_point_at_length(
                        (f64::acos(dx / length) / 2.0 / std::f64::consts::PI) as f32 * svg_length,
                    ))
                })
        } else {
            g.admg
                .node_weight(v1)
                .unwrap()
                .vertex_path_element
                .lock_ref()
                .as_ref()
                .and_then(|path| {
                    Some(path.get_point_at_length(
                        (1.0 - f64::acos(dx / length) / 2.0 / std::f64::consts::PI) as f32
                            * svg_length,
                    ))
                })
        }
        .unwrap()
        .ok()
        .and_then(|p| Some(Point::new(p.x() as f64, p.y() as f64)))
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
        point_v1: &Point,
        point_v2: &Point,
    ) -> (Point, Point) {
        let edge_info = g.admg.edge_weight(edge.id).unwrap();
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let edge_type = *edge_info.edge_type.lock_ref();

        let arrow_head = edge_type == EdgeType::Undirected || edge_type == EdgeType::Directed;
        let v1_anchor = SvgEdge::svg_edge_anchor(g.clone(), v1, point_v1, point_v2, arrow_head);

        let arrow_head = edge_type == EdgeType::Undirected;
        let v2_anchor = SvgEdge::svg_edge_anchor(g, v2, point_v2, point_v1, arrow_head);

        (v1_anchor, v2_anchor)
    }

    pub fn render(edge: Arc<SvgEdge>, g: Arc<SvgGraph>, bounds: Bounds) -> Dom {
        let (v1, v2) = g.admg.edge_endpoints(edge.id).unwrap();
        let info_v1 = g.admg.node_weight(v1).unwrap();
        let info_v2 = g.admg.node_weight(v2).unwrap();

        let point_v1 = bounds.to_svg_coordinates(&info_v1.layout_pos);
        let point_v2 = bounds.to_svg_coordinates(&info_v2.layout_pos);

        let (anchor_back, anchor_front) =
            SvgEdge::svg_edge_anchors(edge.clone(), g, &point_v1, &point_v2);
        let line_path = format!(
            "M{:.2},{:.2}L{:.2},{:.2}",
            anchor_back.x(),
            anchor_back.y(),
            anchor_front.x(),
            anchor_front.y()
        );

        let mut afront = 360.0
            * f64::atan(
                (anchor_front.y() - anchor_back.y()) / (anchor_front.x() - anchor_back.x()),
            )
            / 2.0
            / std::f64::consts::PI;
        if anchor_back.x() < anchor_front.x() {
            afront += 180.0;
        }
        if anchor_back.x() == anchor_front.x() {
            afront = if anchor_front.y() > anchor_back.y() {
                -90.0
            } else {
                90.0
            }
        };

        let translate = format!(
            "translate({}, {}) rotate({})",
            anchor_front.x(),
            anchor_front.y(),
            afront
        );

        let children = vec![
            svg!("path", {
                .attr("stroke-width", "1.5")
                .attr("fill", "none")
                .attr("stroke", "black")
                .attr("d", &line_path)
            }),
            svg!("path", {
                .attr("stroke-width", "1.5")
                .attr("fill", "white")
                .attr("stroke", "black")
                .attr("d", "M-1,0L15,5L15,-5Z")
                .attr("transform", &translate)
            }),
        ];

        svg!("g", {
            .class(&*PATH_CLASS)
            .attr("style", "cursor: move;" )
            .children(children)
            .event(clone!(edge => move |e: events::TouchStart| {
                edge.marked.set(!edge.marked.get());
                e.prevent_default();
            }))
            .event(|_: events::TouchEnd| {
            })
            .event(clone!(edge => move |_: events::MouseDown| {
                edge.marked.set(!edge.marked.get());
            }))
            .event(|_: events::MouseMove| {

            })
            .event(|_: events::MouseLeave| {
            })
        })
    }
}
