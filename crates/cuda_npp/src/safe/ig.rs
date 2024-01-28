//! Geometric transforms (resize, transforms)

use cuda_npp_sys::*;

use crate::{Channel, Sample, C};

use super::{Image, Result};

pub trait Resize {
    fn resize(
        &self,
        dst: &mut Self,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<()>;
}

macro_rules! impl_resize {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Resize for Image<$sample_ty, $channel_ty> {
            fn resize(
                &self,
                dst: &mut Self,
                interpolation: NppiInterpolationMode,
                ctx: NppStreamContext,
            ) -> Result<()> {
                unsafe { paste::paste!([<nppi Resize $sample_id _ $channel_id R_Ctx>])(
                    self.data as _,
                    self.line_step,
                    self.size(),
                    NppiRect {
                        x: 0,
                        y: 0,
                        width: self.width as i32,
                        height: self.height as i32,
                    },
                    dst.data as _,
                    dst.line_step,
                    dst.size(),
                    NppiRect {
                        x: 0,
                        y: 0,
                        width: dst.width as i32,
                        height: dst.height as i32,
                    },
                    interpolation as _,
                    ctx,
                )}.result()?;
                Ok(())
            }
        }
    };
}

impl_resize!(u8, C<1>, _8u, C1);
impl_resize!(u8, C<3>, _8u, C3);
impl_resize!(u8, C<4>, _8u, C4);

impl_resize!(u16, C<1>, _16u, C1);
impl_resize!(u16, C<3>, _16u, C3);
impl_resize!(u16, C<4>, _16u, C4);

impl_resize!(i16, C<1>, _16s, C1);
impl_resize!(i16, C<4>, _16s, C4);

impl_resize!(f32, C<1>, _32f, C1);
impl_resize!(f32, C<3>, _32f, C3);
impl_resize!(f32, C<4>, _32f, C4);

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel> Image<S, C>
where
    Image<S, C>: Resize + crate::safe::isu::Malloc,
{
    pub fn resize_new(
        &self,
        new_width: u32,
        new_height: u32,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<Self> {
        let mut dst = <Self as crate::safe::isu::Malloc>::malloc(new_width, new_height)?;
        self.resize(&mut dst, interpolation, ctx)?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use cuda_npp_sys::{NppiInterpolationMode, NppiSize};

    use crate::safe::isu::Malloc;
    use crate::safe::Image;
    use crate::{get_stream_ctx, C};

    #[test]
    fn resize() -> crate::safe::Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let resized = img.resize_new(2048, 2048, NppiInterpolationMode::NPPI_INTER_LANCZOS, ctx)?;
        dev.synchronize().unwrap();

        assert_eq!(
            resized.size(),
            NppiSize {
                width: 2048,
                height: 2048,
            }
        );
        Ok(())
    }
}
