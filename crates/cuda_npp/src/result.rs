use crate::{generic, ChannelLayout, Sample};
use cuda_npp_sys::{NppStatus, NppStreamContext, NppiInterpolationMode, NppiRect, NppiSize};

pub type Result<T> = std::result::Result<T, NppStatus>;

pub unsafe fn resize<S: Sample + 'static, C: ChannelLayout + 'static>(
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
) -> Result<()> {
    let status = generic::resize::<S, C>(
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
    );
    if status == NppStatus::NPP_NO_ERROR {
        Ok(())
    } else {
        Err(status)
    }
}
