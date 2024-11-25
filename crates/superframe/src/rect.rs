use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.w == 0 && self.h == 0
    }

    pub fn contains(&self, other: &Self) -> bool {
        other.x < self.w
            && other.x >= self.x
            && other.y < self.h
            && other.y >= self.y
            && other.x + other.w <= self.x + self.w
            && other.y + other.h <= self.y + self.h
    }

    pub fn relative_to_self(&self) -> Self {
        Self {
            x: 0,
            y: 0,
            ..*self
        }
    }

    pub fn with_base(&self, other: &Rect) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            ..*self
        }
    }

    pub fn rows(&self) -> Range<usize> {
        self.y..self.y + self.h
    }

    pub fn cols(&self) -> Range<usize> {
        self.x..self.x + self.w
    }
}
