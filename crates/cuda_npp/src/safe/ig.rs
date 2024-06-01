//! Geometric transforms (resize, transforms)

use cuda_npp_sys::*;

use crate::{Channels, Sample, C};

use super::{Image, Img, ImgMut, Result};

pub trait Resize<S: Sample, C: Channels> {
    fn resize(
        &self,
        dst: impl ImgMut<S, C>,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn resize_new(
        &self,
        new_width: u32,
        new_height: u32,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<Image<S, C>>
    where
        Image<S, C>: super::isu::Malloc,
    {
        let mut dst = super::isu::Malloc::malloc(new_width, new_height)?;
        self.resize(&mut dst, interpolation, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_resize {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Resize<$sample_ty, $channel_ty> for T {
            fn resize(
                &self,
                mut dst: impl ImgMut<$sample_ty, $channel_ty>,
                interpolation: NppiInterpolationMode,
                ctx: NppStreamContext,
            ) -> Result<()> {
                unsafe { paste::paste!([<nppi Resize $sample_id _ $channel_id R_Ctx>])(
                    self.device_ptr(),
                    self.pitch(),
                    self.size(),
                    self.rect(),
                    dst.device_ptr_mut(),
                    dst.pitch(),
                    dst.size(),
                    dst.rect(),
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

pub trait ResizeSqrPixel<S: Sample, C: Channels> {
    fn resize_sqr_pixel(
        &self,
        dst: impl ImgMut<S, C>,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn resize_sqr_pixel_new(
        &self,
        new_width: u32,
        new_height: u32,
        interpolation: NppiInterpolationMode,
        ctx: NppStreamContext,
    ) -> Result<Image<S, C>>
    where
        Image<S, C>: super::isu::Malloc,
    {
        let mut dst = super::isu::Malloc::malloc(new_width, new_height)?;
        self.resize_sqr_pixel(&mut dst, interpolation, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_resize_sqr_pixel {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> ResizeSqrPixel<$sample_ty, $channel_ty> for T {
            fn resize_sqr_pixel(
                &self,
                mut dst: impl ImgMut<$sample_ty, $channel_ty>,
                interpolation: NppiInterpolationMode,
                ctx: NppStreamContext,
            ) -> Result<()> {
                unsafe { paste::paste!([<nppi ResizeSqrPixel $sample_id _ $channel_id R_Ctx>])(
                    self.device_ptr(),
                    self.size(),
                    self.pitch(),
                    self.rect(),
                    dst.device_ptr_mut(),
                    dst.pitch(),
                    dst.rect(),
                    0.5, 0.5, 0.0, 0.0,
                    interpolation as _,
                    ctx,
                )}.result()?;
                Ok(())
            }
        }
    };
}

impl_resize_sqr_pixel!(f32, C<3>, _32f, C3);

#[cfg(test)]
mod tests {
    use cuda_npp_sys::{NppiInterpolationMode, NppiSize};

    use crate::safe::ig::Resize;
    use crate::safe::isu::Malloc;
    use crate::safe::{Image, Img};
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
