use crate::const_algebra::Vec2;

pub const D65: Vec2 = Vec2::new(0.3127, 0.3290);

pub const BT709_R: Vec2 = Vec2::new(0.640, 0.330);
pub const BT709_G: Vec2 = Vec2::new(0.300, 0.600);
pub const BT709_B: Vec2 = Vec2::new(0.150, 0.060);
pub const BT709_W: Vec2 = D65;

pub const BT601_525_R: Vec2 = Vec2::new(0.630, 0.340);
pub const BT601_525_G: Vec2 = Vec2::new(0.310, 0.595);
pub const BT601_525_B: Vec2 = Vec2::new(0.155, 0.070);
pub const BT601_525_W: Vec2 = D65;

pub const BT601_625_R: Vec2 = Vec2::new(0.640, 0.330);
pub const BT601_625_G: Vec2 = Vec2::new(0.290, 0.600);
pub const BT601_625_B: Vec2 = Vec2::new(0.150, 0.060);
pub const BT601_625_W: Vec2 = D65;

// const BT709_KR: f32 = 0.2126;
// const BT709_KG: f32 = 0.7152;
// const BT709_KB: f32 = 0.0722;
