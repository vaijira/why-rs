use crate::svggraph::SvgGraph;
use crate::{css::BUTTON_CLASS, section_header::SectionHeader};
use dominator::{clone, events, html, with_node, Dom, DomBuilder};
use futures_signals::signal::{Mutable, SignalExt};
use std::sync::Arc;
use web_sys::HtmlInputElement;
use why_data::graph::dagitty::{NodeInfo, VertexType};
use why_data::graph::CausalGraph;

pub struct VariableSection {
    header: Arc<SectionHeader>,
    displayed: Mutable<bool>,
}

impl VariableSection {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            header: SectionHeader::new(" Variable".to_string(), "a_variable".to_string()),
            displayed: Mutable::new(true),
        })
    }

    fn get_variable_name(node_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("p", {
            .child(html!("span", {
                .attr("id", "variable_label")
                .style("font-weight", "bold")
                .text(node_info.as_ref().map_or("", |v| &v.id ))
            }))
        })
    }

    fn check_vertex_type(
        dom: DomBuilder<HtmlInputElement>,
        node_info: &Option<Arc<NodeInfo>>,
        vertex_type: VertexType,
    ) -> DomBuilder<HtmlInputElement> {
        if node_info.is_some() && *node_info.as_ref().unwrap().vertex_type.lock_ref() == vertex_type
        {
            dom.attr("checked", "checked")
        } else {
            dom
        }
    }

    fn update_vertex_type(
        svg_graph: &Arc<SvgGraph>,
        node_info: &Option<Arc<NodeInfo>>,
        vertex_type: VertexType,
    ) {
        if let Some(ref node) = node_info {
            *node.vertex_type.lock_mut() = vertex_type;
            *svg_graph.current_variable.lock_mut() = Some(node.clone());
        }
    }

    fn remove_vertex(svg_graph: &Arc<SvgGraph>, node_info: &Option<Arc<NodeInfo>>) {
        if let Some(ref node) = node_info {
            let node_index = {
                match &*svg_graph.graph.lock_ref() {
                    CausalGraph::Dag(g) => g.node_indices().find(|i| g[*i].id == node.id).unwrap(),
                    _ => unimplemented!(),
                }
            };
            let node_edges = svg_graph.graph.lock_ref().edges(node_index);
            svg_graph.vertexes.lock_mut().retain(|v| node_index != v.id);
            svg_graph
                .edges
                .lock_mut()
                .retain(|v| !node_edges.contains(&v.id));
            (*svg_graph.graph.lock_mut()).remove_node(node_index);
            *svg_graph.current_variable.lock_mut() = None;
        }
    }

    fn div(svg_graph: &Arc<SvgGraph>, node_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("form", {
            .attr("autocomplete", "off")
            .child(Self::get_variable_name(node_info))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "none_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "None variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::None))
                    .attr("value", "exposure")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::None);
                        }))
                    })
                }))
                .child(html!("label", {
                    .attr("for", "none_radio")
                    .text("none")
                }))
            }))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "exposure_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "Exposure variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::Exposure))
                    .attr("value", "exposure")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::Exposure);
                        }))
                    })
                }))
                .child(html!("label", {
                    .attr("for", "exposure_radio")
                    .text("exposure")
                }))
            }))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "outcome_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "Outcome variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::Outcome))
                    .attr("value", "outcome")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::Outcome);
                        }))
                    })
                 }))
                .child(html!("label", {
                    .attr("for", "outcome_radio")
                    .text("outcome")
                }))
            }))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "adjusted_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "Adjusted variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::Adjusted))
                    .attr("value", "adjusted")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::Adjusted);
                        }))
                    })
                 }))
                .child(html!("label", {
                    .attr("for", "adjusted_radio")
                    .text("adjusted")
                }))
            }))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "selected_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "Selected variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::Selected))
                    .attr("value", "selected")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::Selected);
                        }))
                    })
                 }))
                .child(html!("label", {
                    .attr("for", "selected_radio")
                    .text("selected")
                }))
            }))
            .child(html!("p", {
                .child(html!("input" => HtmlInputElement, {
                    .attr("id", "unobserved_radio")
                    .attr("name", "variable_type")
                    .attr("type", "radio")
                    .attr("alt", "Unobserved variable")
                    .apply(|dom| Self::check_vertex_type(dom, node_info, VertexType::Unobserved))
                    .attr("value", "unobserved")
                    .with_node!(_input_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::update_vertex_type(&svg_graph, &node_info, VertexType::Unobserved);
                        }))
                    })
                 }))
                .child(html!("label", {
                    .attr("for", "unobserved_radio")
                    .text("unobserved")
                }))
            }))
            .child(html!("p", {
                .child(html!("button", {
                    .class(&*BUTTON_CLASS)
                    .attr("type", "button")
                    .text("delete")
                    .with_node!(_element => {
                        .event(clone!(svg_graph, node_info => move |_: events::Click| {
                            Self::remove_vertex(&svg_graph, &node_info);
                        }))
                    })
                }))
                .child(html!("button", {
                    .class(&*BUTTON_CLASS)
                    .attr("type", "button")
                    .text("rename")
                }))

            }))
        })
    }

    pub fn render(this: &Arc<Self>, svg_graph: &Arc<SvgGraph>) -> Dom {
        html!("section", {
            .child(SectionHeader::render(&this.header, &this.displayed))
            .child(html!("div", {
                .visible_signal(this.displayed.signal())
                .child_signal(svg_graph.current_variable.signal_cloned().map(
                    clone!(svg_graph => move |variable| {
                    Some(Self::div(&svg_graph, &variable))
                })))
            }))
        })
    }
}
