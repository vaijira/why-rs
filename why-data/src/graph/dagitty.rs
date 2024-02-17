use crate::types::Point;
use futures_signals::signal::Mutable;

/// vertex type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VertexType {
    /// Default vertex type
    None,
    /// Adjusted vertex,
    Adjusted,
    /// Exposure vertex
    Exposure,
    /// Outcome vertex
    Outcome,
    /// Selected vertex
    Selected,
    /// Unobserved vertex
    Unobserved,
}

/// edge type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeType {
    /// Directed
    Directed,
    /// Bidirected
    _Bidirected,
    /// Undirected
    Undirected,
}

/// Node information to represent a vertex.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Node identifier.
    pub id: String,
    /// Node layout position.
    pub layout_pos: Mutable<Point<f64>>,
    /// Vertex type.
    pub vertex_type: Mutable<VertexType>,
    /// Vertex path html id
    pub vertex_path_id: Mutable<String>,
}

impl NodeInfo {
    const DEFAULT_HTML_PATH_ID_PREFIX: &'static str = "node_path_id_";
    /// Create a new vertex.
    pub fn new(id: &str, layout_pos_x: f64, layout_pos_y: f64, vertex_type: VertexType) -> Self {
        NodeInfo {
            id: id.to_string(),
            layout_pos: Mutable::new(Point::new(layout_pos_x, layout_pos_y)),
            vertex_type: Mutable::new(vertex_type),
            vertex_path_id: Mutable::new(format!(
                "{}{}",
                NodeInfo::DEFAULT_HTML_PATH_ID_PREFIX,
                id
            )),
        }
    }
}

/// Edge information to represent a edge.
#[derive(Debug)]
pub struct EdgeInfo {
    /// Edge identifier.
    pub _id: String,
    /// Edge layout position.
    pub layout_pos: Mutable<Option<Point<f64>>>,
    /// Edge type.
    pub edge_type: Mutable<EdgeType>,
}

impl EdgeInfo {
    /// Create a new vertex.
    pub fn new(id: &str, layout_pos: Option<Point<f64>>, edge_type: EdgeType) -> Self {
        EdgeInfo {
            _id: id.to_string(),
            layout_pos: Mutable::new(layout_pos),
            edge_type: Mutable::new(edge_type),
        }
    }
}
