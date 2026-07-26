use std::fmt::Display;

/// Store 2D Point information
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point<T: Copy> {
    x: T,
    y: T,
}

impl<T: Copy> Point<T> {
    /// Create new instance of Point
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Returns x coordinate.
    pub fn x(&self) -> T {
        self.x
    }

    /// Returns y coordinate.
    pub fn y(&self) -> T {
        self.y
    }
}

impl<T: Copy + Display> ToString for Point<T> {
    fn to_string(&self) -> String {
        format!("{},{}", self.x, self.y)
    }
}
