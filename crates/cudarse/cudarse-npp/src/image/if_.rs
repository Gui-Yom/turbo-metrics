//! Image filtering

use crate::sys::*;
use crate::ScratchBuffer;

use super::{Channels, Image, Img, ImgMut, Sample, C};

pub trait FilterGaussBorder<S: Sample, C: Channels> {
    /// Filters the image using a Gaussian filter kernel with border control.
    /// Use filter_gauss_advanced_border if you want to supply your own filter coefficients.
    ///
    /// If any portion of the mask overlaps the source image boundary the requested border type operation is applied to
    /// all mask pixels which fall outside the source image.
    ///
    /// Currently only the NPP_BORDER_REPLICATE border type operation is supported.
    ///
    /// Note that all FilterGaussBorder functions currently support mask sizes up to 15x15. Filter kernels for these
    /// functions are calculated using a sigma value of 0.4F + (mask width / 2) * 0.6F.
    fn filter_gauss_border(
        &self,
        dst: impl ImgMut<S, C>,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn filter_gauss_border_new(
        &self,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<Image<S, C>>
    where
        Self: Img<S, C>,
        Image<S, C>: super::isu::Malloc,
    {
        let mut dst = self.malloc_same_size()?;
        self.filter_gauss_border(&mut dst, filter_size, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_filtergaussborder {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> FilterGaussBorder<$sample_ty, $channel_ty> for T {
            fn filter_gauss_border(&self, mut dst: impl ImgMut<$sample_ty, $channel_ty>, filter_size: NppiMaskSize, ctx: NppStreamContext) -> Result<()>
            where
                Self: Sized,
            {
                let status = unsafe {
                    paste::paste!([<nppi FilterGaussBorder $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        self.size(),
                        NppiPoint::default(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        dst.size(),
                        filter_size,
                        NppiBorderType::NPP_BORDER_REPLICATE,
                        ctx,
                    )
                };
                if status == NppStatus::NPP_NO_ERROR {
                    Ok(())
                } else {
                    Err(Error::from(status))
                }
            }
        }
    };
}

impl_filtergaussborder!(f32, C<3>, _32f, C3);

pub trait Filter<S: Sample, C: Channels> {
    /// Filters the image using a Gaussian filter kernel with border control.
    /// Use filter_gauss_advanced_border if you want to supply your own filter coefficients.
    ///
    /// If any portion of the mask overlaps the source image boundary the requested border type operation is applied to
    /// all mask pixels which fall outside the source image.
    ///
    /// Currently only the NPP_BORDER_REPLICATE border type operation is supported.
    ///
    /// Note that all FilterGaussBorder functions currently support mask sizes up to 15x15. Filter kernels for these
    /// functions are calculated using a sigma value of 0.4F + (mask width / 2) * 0.6F.
    fn filter(
        &self,
        dst: impl ImgMut<S, C>,
        coeffs: &ScratchBuffer,
        coeffs_size: NppiSize,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn filter_new(
        &self,
        coeffs: &ScratchBuffer,
        coeffs_size: NppiSize,
        ctx: NppStreamContext,
    ) -> Result<Image<S, C>>
    where
        Self: Img<S, C>,
        Image<S, C>: super::isu::Malloc,
    {
        let mut dst = self.malloc_same_size()?;
        self.filter(&mut dst, coeffs, coeffs_size, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_filter {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Filter<$sample_ty, $channel_ty> for T {
            fn filter(&self, mut dst: impl ImgMut<$sample_ty, $channel_ty>, coeffs: &ScratchBuffer, coeffs_size: NppiSize, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Filter $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        self.size(),
                        coeffs.ptr as _,
                        coeffs_size,
                        NppiPoint { x: 0, y: 0 },
                        1,
                        ctx,
                    )
                }.result()
            }
        }
    };
}

impl_filter!(u8, C<1>, _8u, C1);
impl_filter!(u16, C<1>, _16u, C1);
impl_filter!(i16, C<1>, _16s, C1);
