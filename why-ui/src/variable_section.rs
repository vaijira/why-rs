use crate::section_header::SectionHeader;
use crate::svggraph::SvgGraph;
use dominator::{clone, html, Dom, DomBuilder};
use futures_signals::signal::{Mutable, SignalExt};
use std::sync::Arc;
use web_sys::HtmlElement;
use why_data::graph::dagitty::{NodeInfo, VertexType};

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

    fn get_variable_name(vertex_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("p", {
            .child(html!("span", {
                .attr("id", "variable_label")
                .style("font-weight", "bold")
                .text(vertex_info.as_ref().map_or("", |v| &v.id ))
            }))
        })
    }

    fn check_vertex_type(
        dom: DomBuilder<HtmlElement>,
        vertex_info: &Option<Arc<NodeInfo>>,
        vertex_type: VertexType,
    ) -> DomBuilder<HtmlElement> {
        if vertex_info.is_some()
            && *vertex_info.as_ref().unwrap().vertex_type.lock_ref() == vertex_type
        {
            dom.attr("checked", "checked")
        } else {
            dom
        }
    }

    fn div(_this: &Arc<Self>, vertex_info: &Option<Arc<NodeInfo>>) -> Dom {
        html!("form", {
          .attr("autocomplete", "off")
          .child(Self::get_variable_name(vertex_info))
          .child(
            html!("p", {
              .child(
                html!("input", {
                  .attr("id", "exposure_radio")
                  .attr("type", "radio")
                  .attr("alt", "Exposure variable")
                  .apply(|dom| Self::check_vertex_type(dom, vertex_info, VertexType::Exposure))
                  .attr("value", "exposure")
                }))
              .child(
                html!("label", {
                  .attr("for", "exposure_radio")
                  .text("exposure")
                }))
           }))
          .child(
             html!("p", {
              .child(
                html!("input", {
                  .attr("id", "outcome_radio")
                  .attr("type", "radio")
                  .attr("alt", "Outcome variable")
                  .apply(|dom| Self::check_vertex_type(dom, vertex_info, VertexType::Outcome))
                  .attr("value", "outcome")
                }))
              .child(
                html!("label", {
                  .attr("for", "outcome_radio")
                  .text("outcome")
                }))
           }))
          .child(
             html!("p", {
              .child(
                html!("input", {
                  .attr("id", "adjusted_radio")
                  .attr("type", "radio")
                  .attr("alt", "Adjusted variable")
                  .apply(|dom| Self::check_vertex_type(dom, vertex_info, VertexType::Adjusted))
                  .attr("value", "adjusted")
                }))
              .child(
                html!("label", {
                  .attr("for", "adjusted_radio")
                  .text("adjusted")
                }))
           }))
           .child(
             html!("p", {
              .child(
                html!("input", {
                  .attr("id", "selected_radio")
                  .attr("type", "radio")
                  .attr("alt", "Selected variable")
                  .apply(|dom| Self::check_vertex_type(dom, vertex_info, VertexType::Selected))
                  .attr("value", "selected")
                }))
              .child(
                html!("label", {
                  .attr("for", "selected_radio")
                  .text("selected")
                }))
           }))
           .child(
             html!("p", {
              .child(
                html!("input", {
                  .attr("id", "unobserved_radio")
                  .attr("type", "radio")
                  .attr("alt", "Unobserved variable")
                  .apply(|dom| Self::check_vertex_type(dom, vertex_info, VertexType::Unobserved))
                  .attr("value", "unobserved")
                }))
              .child(
                html!("label", {
                  .attr("for", "unobserved_radio")
                  .text("unobserved")
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
                  clone!(this => move |variable| {
                  Some(Self::div(&this, &variable))
                })))
            }))
        })
    }
}
