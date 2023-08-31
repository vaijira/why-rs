use crate::{graph::Point, ADMG};

pub(crate) const VIEWBOX_WIDTH: u32 = 800;
pub(crate) const VIEWBOX_HEIGHT: u32 = 600;

#[derive(Clone, Debug)]
pub struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Bounds {
    pub(crate) fn calculate_bounds(admg: &ADMG) -> Self {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for idx in admg.graph().node_indices() {
            min_x = min_x.min(admg.node_weight(idx).unwrap().layout_pos.x());
            max_x = max_x.max(admg.node_weight(idx).unwrap().layout_pos.x());
            min_y = min_y.min(admg.node_weight(idx).unwrap().layout_pos.y());
            max_y = max_y.max(admg.node_weight(idx).unwrap().layout_pos.y());
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

    pub(crate) fn to_svg_coordinates(&self, pos: &Point) -> Point {
        let x = (pos.x() - self.min_x) / (self.max_x - self.min_x) * VIEWBOX_WIDTH as f64;
        let y = (pos.y() - self.min_y) / (self.max_y - self.min_y) * VIEWBOX_HEIGHT as f64;
        Point::new(x, y)
    }
}
