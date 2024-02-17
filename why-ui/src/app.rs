use std::sync::Arc;

use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use web_sys::HtmlElement;
use why_data::graph::dagitty::{NodeInfo, VertexType};
use why_parser::dagitty::DagittyParser;

use crate::bounds::{Bounds, ContainerCoordinates};
use crate::css::{
    LEFT_LEGEND_DIV_CLASS, MAIN_CLASS, MENU_DIV_CLASS, RIGHT_LEGEND_DIV_CLASS, SVG_DIV_CLASS,
    TITLE_LEGEND_DIV_CLASS,
};
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
    svg_graph: Arc<SvgGraph>,
}

impl App {
    pub fn new() -> Arc<Self> {
        let g = match DagittyParser::parse_str(DEFAULT_GRAPH) {
            Ok(g) => g,
            Err(err) => panic!("Unable to parse default graph, error: {}", err),
        };

        Arc::new(Self {
            graph_text: Mutable::new(DEFAULT_GRAPH.into()),
            svg_graph: SvgGraph::new(g),
        })
    }

    fn get_variable_name(vertex_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("p", {
            .child(html!("span", {
                .attr("id", "variable_label")
                .style("font-weight", "bold")
                .text(vertex_info.as_ref().map_or("", |v| &v.id ))
            }))
        })
    }

    fn variable_div(_this: &Arc<Self>, vertex_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("form", {
          .attr("autocomplete", "off")
          .child(Self::get_variable_name(vertex_info))
          .child(html!("input", {
            .attr("id", "selected_checkbox")
            .attr("type", "checkbox")
            .attr("alt", "Vertex selected")
            .apply(|dom| {
                if vertex_info.is_some() && *vertex_info.as_ref().unwrap().vertex_type.lock_ref() == VertexType::Selected {
                    dom.attr("checked", "checked")
                } else {
                    dom
                }
            })
            .attr("value", "selected")
           }))
           .child(html!("label", {
            .attr("for", "selected_checkbox")
            .text("selected")
           }))
        })
    }

    fn left_side_tag(this: &Arc<Self>) -> Dom {
        html!("div", {
            .class(&*LEFT_LEGEND_DIV_CLASS)
            .child(html!("h3", {
              .class(&*TITLE_LEGEND_DIV_CLASS)
              .text("Variable")
            }))
            .child(html!("div", {
              .child_signal(this.svg_graph.current_variable.signal_cloned().map(
                clone!(this => move |variable| {
                  Some(Self::variable_div(&this, &variable))
                })))
            }))
        })
    }

    fn right_side_tag(this: &Arc<Self>) -> Dom {
        html!("div", {
            .class(&*RIGHT_LEGEND_DIV_CLASS)
            .children(&mut [
                html!("h3", {
                    .class(&*TITLE_LEGEND_DIV_CLASS)
                    .text("Model code")
                }),
                html!("div", {
                    .child_signal(this.graph_text.signal_cloned().map(
                        clone!(this => move |_| {
                            Some(html!("textarea", {
                                .text(&this.graph_text.get_cloned())
                            }))
                        })
                    ))
                }),
            ])
        })
    }

    fn aside_tag(this: &Arc<Self>) -> Dom {
        html!(
            "aside", {
            .children(&mut [
                Self::left_side_tag(&this.clone()),
                Self::right_side_tag(this),
            ])
            }
        )
    }

    fn main_tag(this: &Arc<Self>) -> Dom {
        html!("main", {
            .class(&*MAIN_CLASS)
            .children(&mut [
                html!("div", {
                    .class(&*MENU_DIV_CLASS)
                }),
                html!("div" => HtmlElement, {
                    .class(&*SVG_DIV_CLASS)
                    .with_node!(element => {
                        .after_inserted(clone!(this  => move |_| {
                            *this.svg_graph.container.lock_mut() = Some(ContainerCoordinates::new(element.client_top(), element.client_left()));
                        }))
                    })
                    .child_signal(this.svg_graph.bounds.signal().map(
                        clone!(this => move |_| {
                             log::debug!("Rerendering graph");
                             Some(SvgGraph::render(&this.svg_graph))
                         })
                    ))
                    .with_node!(element => {
                        .global_event(clone!(this => move |_: events::Resize| {
                            let h = element.offset_height() - 4;
                            let w = element.offset_width() - 4;
                            log::debug!("Resizing new height:{} width:{}", h, w);
                            *this.svg_graph.bounds.lock_mut() = Bounds::calculate_bounds(&this.svg_graph.graph.lock_ref(), h, w);
                        }))

                    })
                })
            ])

        })
    }

    pub fn render(this: &Arc<Self>) -> Dom {
        html!("body", {
            .children(&mut [
                Self::main_tag(&this.clone()),
                Self::aside_tag(this),
            ])
        })
    }
}
