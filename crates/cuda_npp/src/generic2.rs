use core::ffi::c_void;

use crate::sys::*;
use crate::{ChannelLayout, ChannelLayoutPacked, Sample};

pub unsafe fn resize2<S: Sample, L: ChannelLayoutPacked>(
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
) -> NppStatus {
    const LUT: [[unsafe extern "C" fn(
        src: *const c_void,
        src_step: i32,
        src_size: NppiSize,
        src_roi: NppiRect,
        dst: *mut c_void,
        dst_step: i32,
        dst_size: NppiSize,
        dst_roi: NppiRect,
        interpolation: i32,
        ctx: NppStreamContext,
    ) -> NppStatus; 2]; 3] = [
        [nppiResize_8u_C1R_Ctx, nppiResize_32f_C1R_Ctx],
        [nppiResize_8u_C3R_Ctx, nppiResize_32f_C3R_Ctx],
        [nppiResize_8u_C4R_Ctx, nppiResize_32f_C4R_Ctx],
    ];

    LUT[L::LUT_INDEX][S::LUT_INDEX](
        src as _,
        src_step,
        src_size,
        src_roi,
        dst as _,
        dst_step,
        dst_size,
        dst_roi,
        interpolation as i32,
        ctx,
    )
}
