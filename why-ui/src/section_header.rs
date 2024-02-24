use crate::css::TITLE_LEGEND_DIV_CLASS;
use dominator::{clone, events, html, with_node, Dom};
use futures_signals::signal::{Mutable, SignalExt};
use std::sync::Arc;

pub struct SectionHeader {
    title: String,
    id: String,
}

impl SectionHeader {
    pub fn new(title: String, id: String) -> Arc<Self> {
        Arc::new(Self { title, id })
    }

    pub fn render(this: &Arc<Self>, displayed: &Mutable<bool>) -> Dom {
        html!("h3", {
            .class(&*TITLE_LEGEND_DIV_CLASS)
            .child(html!("img" , {
                .attr("id", &this.id)
                .attr_signal("src", displayed.signal().map(
                    |displayed| if displayed {
                        "images/arrow-down.png"
                    } else {
                        "images/arrow-right.png"
                    }
                    ))
                .attr_signal("alt", displayed.signal().map(
                    |displayed| if displayed {
                        "arrow pointing down"
                    } else {
                        "arrow pointing right"
                    }
                ))
                .with_node!(_image_element => {
                .event(clone!(displayed => move |_: events::Click| {
                    displayed.set(!displayed.get());
                }))
                })
            }))
            .text(&this.title)
        })
    }
}
