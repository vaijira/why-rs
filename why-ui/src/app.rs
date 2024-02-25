use std::sync::Arc;

use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::SignalExt;
use web_sys::HtmlElement;
use why_parser::dagitty::DagittyParser;

use crate::bounds::{Bounds, ContainerCoordinates};
use crate::css::{
    LEFT_LEGEND_DIV_CLASS, MAIN_CLASS, MENU_DIV_CLASS, RIGHT_LEGEND_DIV_CLASS, SVG_DIV_CLASS,
};
use crate::model_data_section::ModelDataSection;
use crate::svggraph::{SvgGraph, DEFAULT_GRAPH};
use crate::variable_section::VariableSection;

pub struct App {
    svg_graph: Arc<SvgGraph>,
}

impl App {
    pub fn new() -> Arc<Self> {
        let g = match DagittyParser::parse_str(DEFAULT_GRAPH) {
            Ok(g) => g,
            Err(err) => panic!("Unable to parse default graph, error: {}", err),
        };

        Arc::new(Self {
            svg_graph: SvgGraph::new(g),
        })
    }

    fn left_side_tag(this: &Arc<Self>) -> Dom {
        let variable_section = VariableSection::new();
        html!("div", {
            .class(&*LEFT_LEGEND_DIV_CLASS)
            .child(VariableSection::render(&variable_section, &this.svg_graph))
        })
    }

    fn right_side_tag(this: &Arc<Self>) -> Dom {
        let model_data_section = ModelDataSection::new();
        html!("div", {
            .class(&*RIGHT_LEGEND_DIV_CLASS)
            .child(ModelDataSection::render(&model_data_section, &this.svg_graph))
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

    fn resize(this: &Arc<Self>, element: &HtmlElement) {
        let h = element.offset_height() - 4;
        let w = element.offset_width() - 4;
        log::debug!("Resizing new height:{} width:{}", h, w);
        *this.svg_graph.bounds.lock_mut() =
            Bounds::calculate_bounds(&this.svg_graph.graph.lock_ref(), h, w);
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
                   .child_signal(this.svg_graph.bounds.signal().map(
                        clone!(this => move |_| {
                             log::debug!("Rerendering graph");
                             Some(SvgGraph::render(&this.svg_graph))
                         })
                    ))
                    .with_node!(element => {
                        .after_inserted(clone!(this  => move |_| {
                            *this.svg_graph.container.lock_mut() = Some(ContainerCoordinates::new(element.client_top(), element.client_left()));
                            Self::resize(&this, &element)
                        }))
                    })
                    .with_node!(element => {
                        .global_event(clone!(this => move |_: events::Resize| {
                            Self::resize(&this, &element)
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
