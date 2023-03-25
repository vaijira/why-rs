use std::{iter::once, sync::Arc};

use dominator::{html, Dom};

use crate::css::{MAIN_CLASS, SVG_DIV_CLASS};
use crate::VertexType;
use crate::{svggraph::SVGGraph, NodeInfo, ADMG};

pub struct App {
    g: Arc<SVGGraph>,
}

impl App {
    pub fn new() -> Arc<Self> {
        let mut example_graph = ADMG::new();

        let a = example_graph.add_node(NodeInfo::new("A", -2.2, -1.52, VertexType::None));
        let b = example_graph.add_node(NodeInfo::new("B", 1.4, -1.46, VertexType::None));
        let d = example_graph.add_node(NodeInfo::new("D", 1.4, 1.621, VertexType::Outcome));
        let e = example_graph.add_node(NodeInfo::new("E", -2.2, 1.597, VertexType::Exposure));
        let z = example_graph.add_node(NodeInfo::new("Z", -0.3, -0.082, VertexType::None));

        let edges = once((a, e, 0))
            .chain(once((a, z, 0)))
            .chain(once((b, d, 0)))
            .chain(once((b, z, 0)))
            .chain(once((e, d, 0)));

        example_graph.add_edges(edges).unwrap();

        Arc::new(Self {
            g: SVGGraph::new(example_graph),
        })
    }

    pub fn render(app: Arc<Self>) -> Dom {
        html!("main", {
            .class(&*MAIN_CLASS)
            .children(&mut [
                html!("div", {
                    .class(&*SVG_DIV_CLASS)
                    .children(&mut [
                        SVGGraph::render(app.g.clone()),
                    ])
                })
            ])
        })
    }
}
