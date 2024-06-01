//! Image color conversions

use cuda_npp_sys::*;

use crate::{Channels, Sample, __priv, C};

use super::{ImgMut, Result};

pub trait GammaFwdIP<S: Sample, C: Channels>: __priv::Sealed {
    /// Forward gamma correction in place with gamma = 2.2
    fn gamma_fwd_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_gammafwdip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: ImgMut<$sample_ty, $channel_ty>> GammaFwdIP<$sample_ty, $channel_ty> for T {
            fn gamma_fwd_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi GammaFwd $sample_id _ $channel_id IR_Ctx>])(self.device_ptr_mut(), self.pitch(), self.size(), ctx)
                }.result()?;
                Ok(())
            }
        }
    };
}

impl_gammafwdip!(u8, C<3>, _8u, C3);

pub trait GammaInvIP<S: Sample, C: Channels>: __priv::Sealed {
    /// Inverse gamma correction in place with gamma = 2.2
    fn gamma_inv_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_gammainvip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: ImgMut<$sample_ty, $channel_ty>> GammaInvIP<$sample_ty, $channel_ty> for T {
            fn gamma_inv_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi GammaInv $sample_id _ $channel_id IR_Ctx>])(self.device_ptr_mut(), self.pitch(), self.size(), ctx)
                }.result()?;
                Ok(())
            }
        }
    };
}

impl_gammainvip!(u8, C<3>, _8u, C3);
