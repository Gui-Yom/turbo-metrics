use core::ffi::c_void;

use crate::sys::*;
use crate::{
    ChannelPacked, ChannelResize, ChannelSet, NppResult, Sample, SampleResize,
    CHANNEL_PACKED_LUT_SIZE, CHANNEL_RESIZE_LUT_SIZE, CHANNEL_SET_LUT_SIZE, SAMPLE_LUT_SIZE,
    SAMPLE_RESIZE_LUT_SIZE,
};

// flut::flut!(nppiMalloc("8u" "16u" "16s" "32s" "32f")(C1 C2 C3 C4));

pub unsafe fn malloc<S: Sample, L: ChannelPacked>(
    width: i32,
    height: i32,
    step: *mut i32,
) -> NppResult<*const S> {
    use crate::emulated::{nppiMalloc_16s_C3, nppiMalloc_32s_C2};
    const LUT: [[unsafe extern "C" fn(i32, i32, *mut i32) -> *mut c_void;
        CHANNEL_PACKED_LUT_SIZE]; SAMPLE_LUT_SIZE] = [
        [
            nppiMalloc_8u_C1,
            nppiMalloc_8u_C2,
            nppiMalloc_8u_C3,
            nppiMalloc_8u_C4,
        ],
        [
            nppiMalloc_16u_C1,
            nppiMalloc_16u_C2,
            nppiMalloc_16u_C3,
            nppiMalloc_16u_C4,
        ],
        [
            nppiMalloc_16s_C1,
            nppiMalloc_16s_C2,
            nppiMalloc_16s_C3,
            nppiMalloc_16s_C4,
        ],
        [
            nppiMalloc_32s_C1,
            nppiMalloc_32s_C2,
            nppiMalloc_32s_C3,
            nppiMalloc_32s_C4,
        ],
        [
            nppiMalloc_32f_C1,
            nppiMalloc_32f_C2,
            nppiMalloc_32f_C3,
            nppiMalloc_32f_C4,
        ],
    ];
    let ptr = LUT[S::LUT_INDEX][<L as ChannelPacked>::LUT_INDEX](width, height, step);
    if ptr.is_null() {
        Err(NppStatus::NPP_MEMORY_ALLOCATION_ERR)
    } else {
        Ok(ptr as _)
    }
}

pub unsafe fn resize_packed<S: SampleResize, L: ChannelResize>(
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
    ) -> NppStatus; CHANNEL_RESIZE_LUT_SIZE]; SAMPLE_RESIZE_LUT_SIZE] = [
        [
            nppiResize_8u_C1R_Ctx,
            nppiResize_8u_C3R_Ctx,
            nppiResize_8u_C4R_Ctx,
        ],
        [
            nppiResize_16u_C1R_Ctx,
            nppiResize_16u_C3R_Ctx,
            nppiResize_16u_C4R_Ctx,
        ],
        [
            nppiResize_16s_C1R_Ctx,
            nppiResize_16s_C3R_Ctx,
            nppiResize_16s_C4R_Ctx,
        ],
        [
            nppiResize_32f_C1R_Ctx,
            nppiResize_32f_C3R_Ctx,
            nppiResize_32f_C4R_Ctx,
        ],
    ];

    let status = LUT[<S as SampleResize>::LUT_INDEX][<L as ChannelResize>::LUT_INDEX](
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
    );
    if status == NppStatus::NPP_NO_ERROR {
        Ok(())
    } else {
        Err(status)
    }
}

pub unsafe fn set_many_channel<S: Sample, L: ChannelSet>(
    value: *const S,
    dst: *mut S,
    dst_step: i32,
    dst_roi: NppiSize,
    ctx: NppStreamContext,
) -> NppResult<()> {
    const LUT: [[unsafe extern "C" fn(
        value: *const c_void,
        dst: *mut c_void,
        dst_step: i32,
        dst_roi: NppiSize,
        ctx: NppStreamContext,
    ) -> NppStatus; CHANNEL_SET_LUT_SIZE]; SAMPLE_LUT_SIZE] = [
        [nppiSet_8u_C2R_Ctx, nppiSet_8u_C3R_Ctx, nppiSet_8u_C4R_Ctx],
        [
            nppiSet_16u_C2R_Ctx,
            nppiSet_16u_C3R_Ctx,
            nppiSet_16u_C4R_Ctx,
        ],
        [
            nppiSet_16s_C2R_Ctx,
            nppiSet_16s_C3R_Ctx,
            nppiSet_16s_C4R_Ctx,
        ],
        [
            nppiSet_32s_C2R_Ctx,
            nppiSet_32s_C3R_Ctx,
            nppiSet_32s_C4R_Ctx,
        ],
        [
            nppiSet_32f_C2R_Ctx,
            nppiSet_32f_C3R_Ctx,
            nppiSet_32f_C4R_Ctx,
        ],
    ];

    let status = LUT[<S as Sample>::LUT_INDEX][<L as ChannelSet>::LUT_INDEX](
        value as _, dst as _, dst_step, dst_roi, ctx,
    );
    if status == NppStatus::NPP_NO_ERROR {
        Ok(())
    } else {
        Err(status)
    }
}
