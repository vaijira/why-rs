use futures_signals::signal::Mutable;
use why_data::types::Point;

use crate::svgedge::EdgeType;
use crate::svgvertex::VertexType;

/// Node information to represent a vertex.
#[derive(Debug)]
pub struct NodeInfo {
    pub(crate) id: String,
    pub(crate) _weight: u32,
    pub(crate) layout_pos: Mutable<Point<f64>>,
    pub(crate) vertex_type: Mutable<VertexType>,
    pub(crate) vertex_path_element: Mutable<Option<web_sys::SvgPathElement>>,
}

impl NodeInfo {
    /// Create a new vertex.
    pub fn new(id: &str, layout_pos_x: f64, layout_pos_y: f64, vertex_type: VertexType) -> Self {
        NodeInfo {
            id: id.to_string(),
            _weight: 1,
            layout_pos: Mutable::new(Point::new(layout_pos_x, layout_pos_y)),
            vertex_type: Mutable::new(vertex_type),
            vertex_path_element: Mutable::new(None),
        }
    }
}

/// Edge information to represent a edge.
#[derive(Debug)]
pub struct EdgeInfo {
    pub(crate) _id: String,
    pub(crate) _weight: u32,
    pub(crate) layout_pos: Mutable<Option<Point<f64>>>,
    pub(crate) edge_type: Mutable<EdgeType>,
}

impl EdgeInfo {
    /// Create a new vertex.
    pub fn new(id: &str, layout_pos: Option<Point<f64>>, edge_type: EdgeType) -> Self {
        EdgeInfo {
            _id: id.to_string(),
            _weight: 1,
            layout_pos: Mutable::new(layout_pos),
            edge_type: Mutable::new(edge_type),
        }
    }
}
