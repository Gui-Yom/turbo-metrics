use crate::{
    Bitdepth, ColorPrimaries, ColorRange, Full, Limited, Sample, TransferCharacteristics, BT709,
};
use nvptx_core::prelude::coords_2d;

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
    CP: ColorPrimaries,
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
    let r_coeff = r_coeff * cr;
    let g_coeff = g_coeff1 * cb + g_coeff2 * cr;
    let b_coeff = b_coeff * cb;

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

            let r = luma + r_coeff;
            let b = luma + b_coeff;
            let g = luma + g_coeff;

            let dst = dst.add(3 * x);
            *dst = Tr::eotf(r);
            *dst.add(1) = Tr::eotf(g);
            *dst.add(2) = Tr::eotf(b);
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
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_8_F_BT709(
    src_y: *const u8,
    src_uv: *const u8,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<8, Full, Full, BT709, BT709>(
        src_y, src_uv, src_pitch, dst, dst_pitch, width, height,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn biplanaryuv420_to_linearrgb_10_L_BT709(
    src_y: *const u16,
    src_uv: *const u16,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    biplanaryuv420_to_linearrgb_generic::<10, Limited, Limited, BT709, BT709>(
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
