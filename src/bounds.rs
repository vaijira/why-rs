use crate::{graph::Point, ADMG};

pub(crate) const VIEWBOX_WIDTH: u32 = 764;
pub(crate) const VIEWBOX_HEIGHT: u32 = 764;

#[derive(Copy, Clone, Debug)]
pub struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    height: f64,
    width: f64,
}

impl Bounds {
    pub(crate) fn calculate_bounds(admg: &ADMG, height: i32, width: i32) -> Self {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let height = if height > 0 { height as f64 } else { 0.0 };

        let width = if width > 0 { width as f64 } else { 0.0 };

        for idx in admg.graph().node_indices() {
            min_x = min_x.min(admg.node_weight(idx).unwrap().layout_pos.get().x());
            max_x = max_x.max(admg.node_weight(idx).unwrap().layout_pos.get().x());
            min_y = min_y.min(admg.node_weight(idx).unwrap().layout_pos.get().y());
            max_y = max_y.max(admg.node_weight(idx).unwrap().layout_pos.get().y());
        }
        if max_x == min_x {
            max_x = min_x + 1.0
        }
        if max_y == min_y {
            max_y = min_y + 1.0
        }
        let xpad = 50.0 / width * (max_x - min_x);
        let ypad = 80.0 / height * (max_y - min_y);
        min_x -= xpad;
        max_x += xpad;
        min_y -= ypad;
        max_y += ypad;

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            height,
            width,
        }
    }

    pub(crate) fn to_svg_coordinates(self, pos: &Point<f64>) -> Point<f64> {
        let x = (pos.x() - self.min_x) / (self.max_x - self.min_x) * self.width;
        let y = (pos.y() - self.min_y) / (self.max_y - self.min_y) * self.height;
        Point::new(x, y)
    }

    pub(crate) fn to_graph_coordinates(self, pos: &Point<f64>) -> Point<f64> {
        let x = pos.x() / self.width * (self.max_x - self.min_x) + self.min_x;
        let y = pos.y() / self.height * (self.max_y - self.min_y) + self.min_y;
        Point::new(x, y)
    }
}
