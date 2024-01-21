use cuda_npp_sys::{NppStreamContext, NppiInterpolationMode, NppiRect, NppiSize};

use crate::{
    generic_lut, ChannelPacked, ChannelResize, ChannelSet, NppResult, Sample, SampleResize,
};

/// returned device pointer is guaranteed to not be null.
/// Step is in bytes
pub fn malloc<S: Sample + 'static, C: ChannelPacked + 'static>(
    width: i32,
    height: i32,
) -> NppResult<(*const S, i32)> {
    let mut step = 0;
    unsafe { generic_lut::malloc::<S, C>(width, height, &mut step) }.map(|ptr| (ptr, step))
}

pub fn resize_packed<S: SampleResize, C: ChannelResize>(
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
        generic_lut::resize_packed::<S, C>(
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

pub fn set_many_channel<S: Sample, C: ChannelSet>(
    value: *const S,
    dst: *mut S,
    dst_step: i32,
    dst_roi: NppiSize,
    ctx: NppStreamContext,
) -> NppResult<()> {
    unsafe { generic_lut::set_many_channel::<S, C>(value, dst, dst_step, dst_roi, ctx) }
}
