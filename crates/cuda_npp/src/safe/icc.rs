use cuda_npp_sys::*;

use crate::C;

use super::{Image, Result, E};

pub trait GammaFwdIP {
    fn gamma_fwd_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_gammafwdip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl GammaFwdIP for Image<$sample_ty, $channel_ty> {
            fn gamma_fwd_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                let status = unsafe {
                    paste::paste!([<nppi GammaFwd $sample_id _ $channel_id IR_Ctx>])(self.data as _, self.line_step, self.size(), ctx)
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

impl_gammafwdip!(u8, C<3>, _8u, C3);

pub trait GammaInvIP {
    fn gamma_inv_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_gammainvip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl GammaInvIP for Image<$sample_ty, $channel_ty> {
            fn gamma_inv_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                let status = unsafe {
                    paste::paste!([<nppi GammaInv $sample_id _ $channel_id IR_Ctx>])(self.data as _, self.line_step, self.size(), ctx)
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

impl_gammainvip!(u8, C<3>, _8u, C3);
