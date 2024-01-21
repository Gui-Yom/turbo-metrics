use cuda_npp_sys::{NppStreamContext, NppiInterpolationMode, NppiRect, NppiSize};

use crate::{
    generic_lut, ChannelLayoutPacked, ChannelLayoutResizePacked, NppResult, Sample, SampleResize,
};

/// returned device pointer is guaranteed to not be null.
/// Step is in bytes
pub fn malloc<S: Sample + 'static, C: ChannelLayoutPacked + 'static>(
    width: i32,
    height: i32,
) -> NppResult<(*const S, i32)> {
    let mut step = 0;
    unsafe { generic_lut::malloc::<S, C>(width, height, &mut step) }.map(|ptr| (ptr, step))
}

pub fn resize<S: SampleResize, C: ChannelLayoutResizePacked>(
    src: *const S,
    src_step: i32,
    src_size: NppiSize,
    src_roi: NppiRect,
    dst: *mut S,
    dst_step: i32,
    dst_size: NppiSize,
    dst_roi: NppiRect,
    interpolation: NppiInterpolationMode,
    ctx: NppStreamContext,
) -> NppResult<()> {
    unsafe {
        generic_lut::resize::<S, C>(
            src,
            src_step,
            src_size,
            src_roi,
            dst,
            dst_step,
            dst_size,
            dst_roi,
            interpolation,
            ctx,
        )
    }
}
