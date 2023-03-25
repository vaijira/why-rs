use std::sync::Arc;

use dominator::{svg, Dom};
use once_cell::sync::Lazy;

use crate::{NodeInfo, VertexType, ADMG};

pub struct SVGGraph {
    admg: ADMG,
}

struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Bounds {
    fn calculate_bounds(admg: &ADMG) -> Self {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for idx in admg.graph().node_indices() {
            min_x = min_x.min(admg.node_weight(idx).unwrap().layout_pos_x);
            max_x = max_x.max(admg.node_weight(idx).unwrap().layout_pos_x);
            min_y = min_y.min(admg.node_weight(idx).unwrap().layout_pos_y);
            max_y = max_y.max(admg.node_weight(idx).unwrap().layout_pos_y);
        }
        if max_x == min_x {
            max_x = min_x + 1.0
        }
        if max_y == min_y {
            max_y = min_y + 1.0
        }
        let xpad = 50.0 / VIEWBOX_WIDTH as f64 * (max_x - min_x);
        let ypad = 80.0 / VIEWBOX_HEIGHT as f64 * (max_y - min_y);
        min_x -= xpad;
        max_x += xpad;
        min_y -= ypad;
        max_y += ypad;

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    fn to_svg_coordinates(&self, pos_x: f64, pos_y: f64) -> (f64, f64) {
        let x = (pos_x - self.min_x) / (self.max_x - self.min_x) * VIEWBOX_WIDTH as f64;
        let y = (pos_y - self.min_y) / (self.max_y - self.min_y) * VIEWBOX_HEIGHT as f64;
        (x, y)
    }
}

const VIEWBOX_WIDTH: u32 = 800;
const VIEWBOX_HEIGHT: u32 = 600;

static VIEWBOX_STR: Lazy<String> =
    Lazy::new(|| format!("0 0 {} {}", VIEWBOX_WIDTH, VIEWBOX_HEIGHT));

impl SVGGraph {
    pub fn new(admg: ADMG) -> Arc<Self> {
        Arc::new(Self { admg })
    }

    fn create_vertex_shape(info: &NodeInfo, x: f64, y: f64) -> Dom {
        let translate = format!("translate({}, {})", x, y);
        let v = match info.vertex_type {
            VertexType::None => vec![
                svg!("path", {
                    .attr("fill-opacity", "0.7")
                    .attr("z-index", "1")
                    .attr("stroke-width", "1.5")
                    .attr("fill", "#aaaaaa")
                    .attr("stroke", "#666666")
                    .attr("d", "M 0 0 m 20, 0 a 20,15 0 1,1 -40,0 a 20,15 0 1,1 40,0")
                }),
                svg!("rect", {
                    .attr("fill", "#ffffff")
                }),
                svg!("text", {
                    .attr("text-anchor", "middle")
                    .attr("y", "35")
                    .text(&info.id)
                }),
            ],
            VertexType::Exposure => vec![
                svg!("path", {
                    .attr("fill-opacity", "0.7")
                    .attr("z-index", "1")
                    .attr("stroke-width", "1.5")
                    .attr("fill", "#bed403")
                    .attr("stroke", "#000000")
                    .attr("d", "M 0 0 m 20, 0 a 20,15 0 1,1 -40,0 a 20,15 0 1,1 40,0")
                }),
                svg!("rect", {
                    .attr("fill", "#ffffff")
                }),
                svg!("text", {
                    .attr("text-anchor", "middle")
                    .attr("y", "35")
                    .text(&info.id)
                }),
            ],
            VertexType::Outcome => vec![
                svg!("path", {
                    .attr("fill-opacity", "0.7")
                    .attr("z-index", "1")
                    .attr("stroke-width", "1.5")
                    .attr("fill", "#00a2e0")
                    .attr("stroke", "#000000")
                    .attr("d", "M 0 0 m 20, 0 a 20,15 0 1,1 -40,0 a 20,15 0 1,1 40,0")
                }),
                svg!("rect", {
                    .attr("fill", "#ffffff")
                }),
                svg!("text", {
                    .attr("text-anchor", "middle")
                    .attr("y", "35")
                    .text(&info.id)
                }),
            ],
        };
        svg!("g", {
            .attr("transform", &translate)
            .attr("style", "cursor: move; touch-action: none;" )
            .children(v)
        })
    }

    fn create_vertexes(g: Arc<Self>, bounds: &Bounds) -> Vec<Dom> {
        let mut vertexes = vec![];

        for idx in g.admg.graph().node_indices() {
            let info = g.admg.node_weight(idx).unwrap();
            let (x, y) = bounds.to_svg_coordinates(info.layout_pos_x, info.layout_pos_y);
            vertexes.push(SVGGraph::create_vertex_shape(info, x, y));
        }

        vertexes
    }

    pub fn render(g: Arc<Self>) -> Dom {
        let bounds = Bounds::calculate_bounds(&g.admg);

        svg!("svg", {
            .attr("alt", "ADMG graph")
            .attr("style", "font-family: Arial, sans-serif" )
            .attr("viewBox", &VIEWBOX_STR)
            .children(SVGGraph::create_vertexes(g, &bounds))
        })
    }
}
