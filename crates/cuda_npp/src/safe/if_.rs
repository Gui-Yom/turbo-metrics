//! Image filtering

use cuda_npp_sys::*;

use crate::{Channel, Sample, C};

use super::{Image, Result, E};

pub trait FilterGauss {
    fn filter_gauss(
        &self,
        dst: &mut Self,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_filtergauss {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl FilterGauss for Image<$sample_ty, $channel_ty> {
            fn filter_gauss(&self, dst: &mut Self, filter_size: NppiMaskSize, ctx: NppStreamContext) -> Result<()>
            where
                Self: Sized,
            {
                let status = unsafe {
                    paste::paste!([<nppi FilterGauss $sample_id _ $channel_id R_Ctx>])(
                        self.data as _,
                        self.line_step,
                        dst.data as _,
                        dst.line_step,
                        self.size(),
                        filter_size,
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

impl_filtergauss!(f32, C<3>, _32f, C3);

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel> Image<S, C>
where
    Image<S, C>: FilterGauss + crate::safe::isu::Malloc,
{
    pub fn filter_gauss_new(
        &self,
        filter_size: NppiMaskSize,
        ctx: NppStreamContext,
    ) -> Result<Self> {
        let mut dst = self.malloc_same()?;
        self.filter_gauss(&mut dst, filter_size, ctx)?;
        Ok(dst)
    }
}
