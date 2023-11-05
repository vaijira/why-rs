use std::iter::once;
use std::rc::Rc;

use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::SignalExt;
use web_sys::HtmlElement;
use why_data::graph::dagitty::{EdgeInfo, EdgeType, NodeInfo, VertexType};
use why_data::graph::{CausalGraph, CausalGraphExt, DiGraph};
use why_data::types::Point;
use why_parser::dagitty::DagittyParser;

use crate::bounds::Bounds;
use crate::css::{MAIN_CLASS, SVG_DIV_CLASS};
use crate::svggraph::SvgGraph;

pub struct App {
    svg_graph: Rc<SvgGraph>,
}

impl App {
    pub fn new() -> Rc<Self> {
        let mut example_graph = DiGraph::<NodeInfo, EdgeInfo>::new();

        let a = example_graph.add_node(NodeInfo::new("A", -2.2, -1.52, VertexType::None));
        let b = example_graph.add_node(NodeInfo::new("B", 1.4, -1.46, VertexType::None));
        let d = example_graph.add_node(NodeInfo::new("D", 1.4, 1.621, VertexType::Outcome));
        let e = example_graph.add_node(NodeInfo::new("E", -2.2, 1.597, VertexType::Exposure));
        let z = example_graph.add_node(NodeInfo::new("Z", -0.3, -0.082, VertexType::None));

        let edges = once((a, e, EdgeInfo::new("", None, EdgeType::Directed)))
            .chain(once((
                a,
                z,
                EdgeInfo::new("", Some(Point::new(-0.791, -1.045)), EdgeType::Directed),
            )))
            .chain(once((b, d, EdgeInfo::new("", None, EdgeType::Directed))))
            .chain(once((
                b,
                z,
                EdgeInfo::new("", Some(Point::new(0.680, -0.496)), EdgeType::Directed),
            )))
            .chain(once((e, d, EdgeInfo::new("", None, EdgeType::Directed))));

        example_graph.add_edges(edges);

        Rc::new(Self {
            svg_graph: SvgGraph::new(CausalGraph::Dag(example_graph)),
        })
    }

    pub fn render(app: Rc<Self>) -> Dom {
        html!("main", {
            .class(&*MAIN_CLASS)
            .children(&mut [
                html!("div" => HtmlElement, {
                    .class(&*SVG_DIV_CLASS)
                    .with_node!(element => {
                        .after_inserted(clone!(app  => move |_| {
                            *app.svg_graph.container.lock_mut() = Some(element);
                        }))
                    })
                    .child_signal(app.svg_graph.bounds.signal().map(
                        clone!(app => move |_| {
                             log::debug!("Rerendering graph");
                             Some(SvgGraph::render(app.svg_graph.clone()))
                         })
                    ))
                    .with_node!(element => {
                        .global_event(clone!(app => move |_: events::Resize| {
                            let h = element.offset_height() - 4;
                            let w = element.offset_width() - 4;
                            log::debug!("Resizing new height:{} width:{}", h, w);
                            *app.svg_graph.bounds.lock_mut() = Bounds::calculate_bounds(&app.svg_graph.graph, h, w);
                        }))

                    })
                })
            ])

        })
    }
}
