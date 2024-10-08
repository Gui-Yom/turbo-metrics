use crate::{
    Bitdepth, ColorRange, Limited, MatrixCoefficients, Sample, TransferCharacteristics, BT601_525,
    BT601_625, BT709,
};
use nvptx_core::prelude::*;

#[inline]
unsafe fn biplanaryuv420_to_linearrgb_generic<const BITDEPTH: usize, LumaCR, ChromaCR, Tr, CP>(
    src_y: *const <Bitdepth<BITDEPTH> as Sample>::Type,
    src_uv: *const <Bitdepth<BITDEPTH> as Sample>::Type,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) where
    Bitdepth<BITDEPTH>: Sample,
    LumaCR: ColorRange,
    ChromaCR: ColorRange,
    Tr: TransferCharacteristics,
    CP: MatrixCoefficients,
{
    let (x, y) = coords_2d();
    if x >= width || y >= height {
        return;
    }

    let (y_coeff, r_coeff, b_coeff, g_coeff1, g_coeff2) =
        CP::coefficients::<LumaCR, ChromaCR, BITDEPTH>();

    // Each UV pair can be used for 4 Y samples
    let src_uv = src_uv.byte_add(y * src_pitch).add(2 * x);
    let cb = (src_uv.read().into() as i32 - ChromaCR::chroma_neutral() as i32) as f32;
    let cr = (src_uv.add(1).read().into() as i32 - ChromaCR::chroma_neutral() as i32) as f32;
    let r_ = r_coeff * cr;
    let g_ = fmaf(g_coeff1, cb, g_coeff2 * cr);
    let b_ = b_coeff * cb;

    // printf!(c"cb: %i, cr: %i, r_: %i, g_: %i, b_: %i\n", i32, i32, i32, i32, i32; (cb*1000.0) as i32, (cr*1000.0) as i32, (r_*1000.0) as i32, (g_*1000.0) as i32, (b_*1000.0) as i32);

    for iy in 0..=1 {
        let y = (2 * y) + iy;
        let dst = dst.byte_add(y * dst_pitch);
        let src_y = src_y.byte_add(y * src_pitch);
        for ix in 0..=1 {
            let x = (2 * x) + ix;
            let luma = (src_y
                .add(x)
                .read()
                .into()
                // .min(LumaCR::luma_max())
                .max(LumaCR::min())
                - LumaCR::min()) as f32
                * y_coeff;

            // printf!(c"min: %d, range: %d, value: %d\n", u32, u32, u32; LumaCR::min(), LumaCR::luma_range(), (luma*1000.0) as u32);

            let r = luma + r_;
            let g = luma + g_;
            let b = luma + b_;

            // printf!(c"r: %d, g: %d, b: %d\n", u32, u32, u32; (r*1000.0) as u32, (g*1000.0) as u32, (b*1000.0) as u32);

            let dst = dst.add(3 * x);
            *dst = Tr::eotf(r).max(0.0).min(1.0);
            *dst.add(1) = Tr::eotf(g).max(0.0).min(1.0);
            *dst.add(2) = Tr::eotf(b).max(0.0).min(1.0);
        }
    }
}

// Manual generic monomorphisation
#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_8_L_BT709(
    src_y: *const u8,
    src_uv: *const u8,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<8, Limited, Limited, BT709, BT709>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_16_L_BT709(
    src_y: *const u16,
    src_uv: *const u16,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<16, Limited, Limited, BT709, BT709>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_8_L_BT601_525(
    src_y: *const u8,
    src_uv: *const u8,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<8, Limited, Limited, BT601_525, BT601_525>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_16_L_BT601_525(
    src_y: *const u16,
    src_uv: *const u16,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<16, Limited, Limited, BT601_525, BT601_525>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_8_L_BT601_625(
    src_y: *const u8,
    src_uv: *const u8,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<8, Limited, Limited, BT601_625, BT601_625>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_16_L_BT601_625(
    src_y: *const u16,
    src_uv: *const u16,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<16, Limited, Limited, BT601_625, BT601_625>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

/// This directly maps YCbCr values to RGB for inspection.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_debug(
    src_y: *const u8,
    src_uv: *const u8,
    src_pitch: usize,
    dst: *mut u8,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    let (x, y) = coords_2d();
    if x >= width || y >= height {
        return;
    }

    // Each UV pair can be used for 4 Y samples
    let src_uv = src_uv.byte_add(y * src_pitch).add(2 * x);
    let cb = src_uv.read();
    let cr = src_uv.add(1).read();

    for iy in 0..=1 {
        let y = (2 * y) + iy;
        let dst = dst.byte_add(y * dst_pitch);
        let src_y = src_y.byte_add(y * src_pitch);
        for ix in 0..=1 {
            let x = (2 * x) + ix;
            let luma = src_y.add(x).read();

            let dst = dst.add(3 * x);
            *dst = luma;
            *dst.add(1) = cb;
            *dst.add(2) = cr;

            // *dst = if Y < Luma::min() { Y * 17 } else { 0 } as u8;
            // *dst.add(1) = if U < C::min() { U * 17 } else { 0 } as u8;
            // *dst.add(2) = if V < C::min() { V * 17 } else { 0 } as u8;
        }
    }
}
