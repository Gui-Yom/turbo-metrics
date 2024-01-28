//! Image Arithmetic and Logical operations

use cuda_npp_sys::*;

use crate::{Channel, Sample, C};

use super::{Image, Result};

pub trait Mul {
    /// Pixel by pixel multiply of two images.
    fn mul(&self, other: &Self, dst: &mut Self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_mul {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Mul for Image<$sample_ty, $channel_ty> {
            fn mul(&self, other: &Self, dst: &mut Self, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Mul $sample_id _ $channel_id R_Ctx>])(
                        self.data as _,
                        self.line_step,
                        other.data as _,
                        other.line_step,
                        dst.data as _,
                        dst.line_step,
                        self.size(),
                        ctx,
                    )
                }.result()?;
                Ok(())
            }
        }
    };
}

impl_mul!(f32, C<3>, _32f, C3);

pub trait SqrIP {
    fn sqr_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_sqrip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl SqrIP for Image<$sample_ty, $channel_ty> {
            fn sqr_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Sqr $sample_id _ $channel_id IR_Ctx>])(
                        self.data as _,
                        self.line_step,
                        self.size(),
                        ctx,
                    )
                }.result()?;
                Ok(())
            }
        }
    };
}

impl_sqrip!(f32, C<3>, _32f, C3);

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel> Image<S, C>
where
    Image<S, C>: Mul + crate::safe::isu::Malloc,
{
    /// Pixel by pixel multiply of two images into a new image.
    pub fn mul_new(&self, other: &Self, ctx: NppStreamContext) -> Result<Self> {
        let mut dst = self.malloc_same()?;
        self.mul(other, &mut dst, ctx)?;
        Ok(dst)
    }
}
