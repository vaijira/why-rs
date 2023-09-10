use std::{iter::once, sync::Arc};

use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::SignalExt;
use web_sys::HtmlElement;

use crate::bounds::Bounds;
use crate::css::{MAIN_CLASS, SVG_DIV_CLASS};
use crate::graph::{EdgeInfo, Point};
use crate::svgedge::EdgeType;
use crate::svgvertex::VertexType;
use crate::{svggraph::SvgGraph, NodeInfo, ADMG};

pub struct App {
    svg_graph: Arc<SvgGraph>,
}

impl App {
    pub fn new() -> Arc<Self> {
        let mut example_graph = ADMG::new();

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

        example_graph.add_edges(edges).unwrap();

        Arc::new(Self {
            svg_graph: SvgGraph::new(example_graph),
        })
    }

    pub fn render(app: Arc<Self>) -> Dom {
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
                            *app.svg_graph.bounds.lock_mut() = Bounds::calculate_bounds(&app.svg_graph.admg, h, w);
                        }))

                    })
                })
            ])

        })
    }
}
