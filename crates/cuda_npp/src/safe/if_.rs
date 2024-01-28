//! Image filtering

use cuda_npp_sys::*;

use crate::{Channel, Sample, C};

use super::{Image, Result, E};

pub trait FilterGaussBorder {
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
        dst: &mut Self,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_filtergaussborder {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl FilterGaussBorder for Image<$sample_ty, $channel_ty> {
            fn filter_gauss_border(&self, dst: &mut Self, filter_size: NppiMaskSize, ctx: NppStreamContext) -> Result<()>
            where
                Self: Sized,
            {
                let status = unsafe {
                    paste::paste!([<nppi FilterGaussBorder $sample_id _ $channel_id R_Ctx>])(
                        self.data as _,
                        self.line_step,
                        self.size(),
                        NppiPoint::default(),
                        dst.data as _,
                        dst.line_step,
                        dst.size(),
                        filter_size,
                        NppiBorderType::NPP_BORDER_REPLICATE,
                        ctx,
                    )
                };
                if status == NppStatus::NPP_NO_ERROR {
                    Ok(())
                } else {
                    Err(E::from(status))
                }
            }
        }
    };
}

impl_filtergaussborder!(f32, C<3>, _32f, C3);

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel> Image<S, C>
where
    Image<S, C>: FilterGaussBorder + crate::safe::isu::Malloc,
{
    /// See [FilterGaussBorder::filter_gauss_border].
    ///
    /// This wrapper function takes care of allocating a new destination image.
    pub fn filter_gauss_border_new(
        &self,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<Self> {
        let mut dst = self.malloc_same()?;
        self.filter_gauss_border(&mut dst, filter_size, ctx)?;
        Ok(dst)
    }
}
