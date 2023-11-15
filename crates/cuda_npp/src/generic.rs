use std::any::TypeId;

use cuda_npp_sys::*;

use crate::{ChannelLayout, Sample, C};

pub type Result = std::result::Result<(), NppStatus>;

macro_rules! generic_dispatch {
    ($name:ident $($R:ident)?, $($advanced:ident)?, $($ctx:ident)? ($($arg:expr),+)) => {
        duplicate::duplicate! {
            [
                dummy;
                [dummy];
            ]
            if false { unreachable!() }
            duplicate! {
                [
                    ty_sample sample_type;
                    [ u8 ] [ 8u ];
                    [ u16 ] [ 16u ];
                    [ i16 ] [ 16s ];
                    [ f32 ] [ 32f ];
                ]
                else if TypeId::of::<S>() == TypeId::of::<ty_sample>() {
                    if false { unreachable!() }
                    duplicate! {
                        [
                            ty_channel channel_count;
                            [ C<1> ] [ C1 ];
                            [ C<3> ] [ C3 ];
                            [ C<4> ] [ C4 ];
                        ]
                        else if TypeId::of::<L>() == TypeId::of::<ty_channel>() {
                            paste::paste!([<nppi $name _ sample_type _ channel_count $($R)? $(_ $advanced)? $(_ $ctx)?>])($($arg),+)
                        }
                    }
                    else { unreachable!("Type constraint should have prevented this") }
                }
            }
            else { unreachable!("Type constraint should have prevented this") }
        }
    };
}

pub unsafe fn malloc<S: Sample, L: ChannelLayout>(
    width: i32,
    height: i32,
    step: *mut i32,
) -> *const S {
    let ret = generic_dispatch!(Malloc,, (width, height, step as _));
    ret as _
}

pub unsafe fn resize<S: Sample + 'static, L: ChannelLayout + 'static>(
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
    generic_dispatch!(Resize R, , Ctx (
            src as _,
            src_step,
            src_size,
            src_roi,
            dst as _,
            dst_step,
            dst_size,
            dst_roi,
            interpolation as i32,
            ctx
        )
    )
}

pub unsafe fn resize_batch<S: Sample + 'static, L: ChannelLayout + 'static>(
    smallest_src_size: NppiSize,
    src_roi: NppiRect,
    smallest_dst_size: NppiSize,
    dst_roi: NppiRect,
    interpolation: NppiInterpolationMode,
    resize_batch: *mut NppiResizeBatchCXR,
    batch_size: u32,
    ctx: NppStreamContext,
) -> NppStatus {
    generic_dispatch!(ResizeBatch R, , Ctx (
            smallest_src_size,
            src_roi,
            smallest_dst_size,
            dst_roi,
            interpolation as i32,
            resize_batch as _,
            batch_size,
            ctx
        )
    )
}

pub unsafe fn resize_batch_advanced<S: Sample + 'static, L: ChannelLayout + 'static>(
    max_width: i32,
    max_height: i32,
    batch_src: *const NppiImageDescriptor,
    batch_dst: *mut NppiImageDescriptor,
    batch_roi: *const NppiResizeBatchROI_Advanced,
    batch_size: u32,
    interpolation: NppiInterpolationMode,
    ctx: NppStreamContext,
) -> NppStatus {
    generic_dispatch!(ResizeBatch R, Advanced, Ctx (
        max_width,
        max_height,
        batch_src as _,
        batch_dst as _,
        batch_roi as _,
        batch_size,
        interpolation as i32,
        ctx
    ))
}

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use cuda_npp_sys::{NppStreamContext, NppiInterpolationMode, NppiRect, NppiSize};

    use crate::generic::resize;
    use crate::C;

    #[test]
    fn test_resize() {
        unsafe {
            let result = resize::<f32, C<1>>(
                null(),
                0,
                NppiSize::default(),
                NppiRect::default(),
                null_mut(),
                0,
                NppiSize::default(),
                NppiRect::default(),
                NppiInterpolationMode::NPPI_INTER_LINEAR,
                NppStreamContext::default(),
            );
            dbg!(result);
        }
    }
}
