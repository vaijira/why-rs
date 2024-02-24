use crate::section_header::SectionHeader;
use crate::{css::TEXTAREA_CLASS, svggraph::SvgGraph};
use dominator::{html, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use std::sync::Arc;

pub struct ModelDataSection {
    header: Arc<SectionHeader>,
    displayed: Mutable<bool>,
}

impl ModelDataSection {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            header: SectionHeader::new(" Model code".to_string(), "model_data".to_string()),
            displayed: Mutable::new(true),
        })
    }

    pub fn render(this: &Arc<Self>, svg_graph: &Arc<SvgGraph>) -> Dom {
        html!("section", {
            .child(SectionHeader::render(&this.header, &this.displayed))
            .child(html!("div", {
                .visible_signal(this.displayed.signal())
                .child(html!("form", {
                  .child_signal(svg_graph.model_data.signal_cloned().map(
                    |data| {
                        Some(html!("textarea", {
                            .class(&*TEXTAREA_CLASS)
                            .attr("rows", "10")
                            .attr("cols", "35")
                            .text(&data)
                        }))
                    }))
                }))
            }))
        })
    }
}
