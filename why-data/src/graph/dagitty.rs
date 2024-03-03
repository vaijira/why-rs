use crate::types::Point;
use futures_signals::signal::Mutable;
use std::sync::Arc;

use super::CausalGraph;

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

impl ToString for VertexType {
    fn to_string(&self) -> String {
        let result = match &self {
            VertexType::None => "",
            VertexType::Adjusted => "adjusted",
            VertexType::Exposure => "exposure",
            VertexType::Outcome => "outcome",
            VertexType::Selected => "selected",
            VertexType::Unobserved => "unobserved",
        };

        result.to_string()
    }
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

impl ToString for NodeInfo {
    fn to_string(&self) -> String {
        if *self.vertex_type.lock_ref() == VertexType::None {
            format!(
                r#"{} [pos="{}"]"#,
                self.id,
                self.layout_pos.lock_ref().to_string()
            )
        } else {
            format!(
                r#"{} [{},pos="{}"]"#,
                self.id,
                self.vertex_type.lock_ref().to_string(),
                self.layout_pos.lock_ref().to_string()
            )
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

impl ToString for EdgeInfo {
    fn to_string(&self) -> String {
        if self.layout_pos.lock_ref().is_some() {
            format!(" [pos={}]", self.layout_pos.lock_ref().unwrap().to_string())
        } else {
            "".to_string()
        }
    }
}

impl ToString for CausalGraph<Arc<NodeInfo>, Arc<EdgeInfo>> {
    fn to_string(&self) -> String {
        let mut result = match self {
            CausalGraph::Dag(_dag) => "dag {\n".to_string(),
            CausalGraph::Ungraph(_g) => "graph {\n".to_string(),
        };

        match self {
            CausalGraph::Dag(g) => {
                for node_index in g.node_indices() {
                    let node = g.node_weight(node_index).unwrap();
                    result.push_str(&node.to_string());
                    result.push('\n');
                }

                for edge_index in g.edge_indices() {
                    let edge = g.edge_weight(edge_index).unwrap();
                    let (source, dst) = g.edge_endpoints(edge_index).unwrap();
                    result.push_str(&format!(
                        "{} -> {}",
                        g.node_weight(source).unwrap().id,
                        g.node_weight(dst).unwrap().id
                    ));
                    result.push_str(&edge.to_string());
                    result.push('\n');
                }
            }
            CausalGraph::Ungraph(g) => {
                for node_index in g.node_indices() {
                    let node = g.node_weight(node_index).unwrap();
                    result.push_str(&node.to_string());
                    result.push('\n');
                }

                for edge_index in g.edge_indices() {
                    let edge = g.edge_weight(edge_index).unwrap();
                    let (source, dst) = g.edge_endpoints(edge_index).unwrap();
                    result.push_str(&format!(
                        "{} -- {}",
                        g.node_weight(source).unwrap().id,
                        g.node_weight(dst).unwrap().id
                    ));
                    result.push_str(&edge.to_string());
                    result.push('\n');
                }
            }
        };

        result.push_str("}");
        result
    }
}
