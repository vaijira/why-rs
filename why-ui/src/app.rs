use std::rc::Rc;

use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use web_sys::HtmlElement;
use why_parser::dagitty::DagittyParser;

use crate::bounds::Bounds;
use crate::css::{MAIN_CLASS, SVG_DIV_CLASS};
use crate::svggraph::SvgGraph;

const DEFAULT_GRAPH: &str = r#"
dag {
A [selected,pos="-2.200,-1.520"]
B [pos="1.400,-1.460"]
D [outcome,pos="1.400,1.621"]
E [exposure,pos="-2.200,1.597"]
Z [adjusted,pos="-0.300,-0.082"]
A -> E
A -> Z [pos="-0.791,-1.045"]
B -> D
B -> Z [pos="0.680,-0.496"]
E -> D
}
"#;

pub struct App {
    graph_text: Mutable<String>,
    svg_graph: Rc<SvgGraph>,
}

impl App {
    pub fn new() -> Rc<Self> {
        let g = match DagittyParser::parse_str(DEFAULT_GRAPH) {
            Ok(g) => g,
            Err(err) => panic!("Unable to parse default graph, error: {}", err),
        };

        Rc::new(Self {
            graph_text: Mutable::new(DEFAULT_GRAPH.into()),
            svg_graph: SvgGraph::new(g),
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
