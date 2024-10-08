use crate::{Bitdepth, Sample};
use nvptx_core::prelude::*;

/// This kernel can be called with 3*width, because channels can be processed independently.
#[inline]
unsafe fn f32_to_generic<const BITDEPTH: usize>(
    src: *const f32,
    src_pitch: usize,
    dst: *mut <Bitdepth<BITDEPTH> as Sample>::Type,
    dst_pitch: usize,
    width: usize,
    height: usize,
) where
    Bitdepth<BITDEPTH>: Sample,
{
    let (x, y) = coords_2d();
    if x >= width || y >= height {
        return;
    }
    let v = src.byte_add(y * src_pitch).add(x).read();
    *dst.byte_add(y * dst_pitch).add(x) =
        RoundFromf32::round(v * <Bitdepth<BITDEPTH>>::MAX_VALUE as f32);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_to_8bit(
    src: *const f32,
    src_pitch: usize,
    dst: *mut u8,
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    f32_to_generic::<8>(src, src_pitch, dst, dst_pitch, width, height)
}
