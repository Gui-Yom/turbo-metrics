#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use crate::const_algebra::{Vec2, Vec3};
use nvptx_core::prelude::*;

mod biplanar;
mod const_algebra;
mod sample_conv;

trait Sample {
    type Type: Into<u32> + RoundFromf32;
    const MAX_VALUE: u32;
}

struct Bitdepth<const N: usize>;
impl Sample for Bitdepth<8> {
    type Type = u8;
    const MAX_VALUE: u32 = (1 << 8) - 1;
}
impl Sample for Bitdepth<10> {
    type Type = u16;
    const MAX_VALUE: u32 = (1 << 10) - 1;
}
impl Sample for Bitdepth<12> {
    type Type = u16;
    const MAX_VALUE: u32 = (1 << 12) - 1;
}

trait ColorRange {
    fn min<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample;

    fn chroma_neutral<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample;

    fn luma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample;

    fn chroma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample;

    fn luma_range<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        Self::luma_max() - Self::min()
    }

    fn chroma_range<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        Self::chroma_max() - Self::min()
    }
}
struct Full;
impl ColorRange for Full {
    fn min<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        0
    }

    fn chroma_neutral<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        1 << (N - 1)
    }

    fn luma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        (1 << N) - 1
    }

    fn chroma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        (1 << N) - 1
    }
}
struct Limited;
impl ColorRange for Limited {
    fn min<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        16 << (N - 8)
    }

    fn chroma_neutral<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        1 << (N - 1)
    }

    fn luma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        235 << (N - 8)
    }

    fn chroma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        240 << (N - 8)
    }
}
struct Special;
impl ColorRange for Special {
    fn min<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        0
    }

    fn chroma_neutral<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        1 << (N - 1)
    }

    fn luma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        235 << (N - 8)
    }

    fn chroma_max<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        240 << (N - 8)
    }

    fn chroma_range<const N: usize>() -> u32
    where
        Bitdepth<N>: Sample,
    {
        Self::chroma_max() - 16
    }
}

trait TransferCharacteristics {
    fn eotf(value: f32) -> f32;
}

trait ColorPrimaries {
    const CONSTANTS: Vec2;
    fn coefficients<LumaCR: ColorRange, ChromaCR: ColorRange, const N: usize>(
    ) -> (f32, f32, f32, f32, f32)
    where
        Bitdepth<N>: Sample;
}

struct BT709;
impl TransferCharacteristics for BT709 {
    fn eotf(value: f32) -> f32 {
        const BETA: f32 = 0.018053968510807;
        const ALPHA: f32 = 1.0 + 5.5 * BETA;
        /// threshold = bt709_oetf(BETA)
        const THRESHOLD: f32 = 0.08124285829863521110029445797874;
        if value >= THRESHOLD {
            ((value + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
        } else {
            value / 4.5
        }
        // const GAMMA: f32 = 2.4;
        // value.max(0.0).powf(GAMMA).min(1.0)
        // value.max(0.0).min(1.0)
    }
}
impl ColorPrimaries for BT709 {
    const CONSTANTS: Vec2 = constants_from_primaries(BT709_R, BT709_G, BT709_B, BT709_W);

    fn coefficients<LumaCR: ColorRange, ChromaCR: ColorRange, const N: usize>(
    ) -> (f32, f32, f32, f32, f32)
    where
        Bitdepth<N>: Sample,
    {
        //let rgb_max = <Bitdepth<N> as Sample>::MAX_VALUE as f32;
        let Vec2 { x: kr, y: kb } = Self::CONSTANTS;
        let y_coeff = 1.0 / LumaCR::luma_range() as f32;
        let r_coeff = 2.0 * (1.0 - kr) * 1.0 / ChromaCR::chroma_range() as f32;
        let b_coeff = 2.0 * (1.0 - kb) * 1.0 / ChromaCR::chroma_range() as f32;
        let kg = 1.0 - kr - kb;
        let g_coeff1 = -2.0 * (1.0 - kb) * kb / kg * 1.0 / ChromaCR::chroma_range() as f32;
        let g_coeff2 = -2.0 * (1.0 - kr) * kr / kg * 1.0 / ChromaCR::chroma_range() as f32;
        (y_coeff, r_coeff, b_coeff, g_coeff1, g_coeff2)
    }
}

const D65: Vec2 = Vec2::new(0.3127, 0.3290);

const BT709_R: Vec2 = Vec2::new(0.640, 0.330);
const BT709_G: Vec2 = Vec2::new(0.300, 0.600);
const BT709_B: Vec2 = Vec2::new(0.150, 0.060);
const BT709_W: Vec2 = D65;
// const BT709_KR: f32 = 0.2126;
// const BT709_KG: f32 = 0.7152;
// const BT709_KB: f32 = 0.0722;

const fn constants_from_primaries(r: Vec2, g: Vec2, b: Vec2, w: Vec2) -> Vec2 {
    let r_xyz = r.xy_to_xyz();
    let g_xyz = g.xy_to_xyz();
    let b_xyz = b.xy_to_xyz();
    let w_xyz = w.xy_to_xyz();

    let x_rgb = Vec3::new(r_xyz.x, g_xyz.x, b_xyz.x);
    let y_rgb = Vec3::new(r_xyz.y, g_xyz.y, b_xyz.y);
    let z_rgb = Vec3::new(r_xyz.z, g_xyz.z, b_xyz.z);

    let mul = 1.0 / x_rgb.dot(y_rgb.cross(z_rgb));
    Vec2::new(
        w_xyz.dot(g_xyz.cross(b_xyz)) * mul,
        w_xyz.dot(r_xyz.cross(g_xyz)) * mul,
    )
}
