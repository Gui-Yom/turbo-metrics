//! Image Arithmetic and Logical operations

use cuda_npp_sys::*;

use crate::{__priv, assert_same_size, C, Channels, Sample};

use super::{Image, Img, ImgMut, Result};

pub trait Mul<S: Sample, C: Channels>: __priv::Sealed {
    /// Pixel by pixel multiply of two images.
    fn mul(&self, other: impl Img<S, C>, dst: impl ImgMut<S, C>, ctx: NppStreamContext) -> Result<()>;

    /// Pixel by pixel multiply of two images into a new image.
    #[cfg(feature = "isu")]
    fn mul_new(&self, other: impl Img<S, C>, ctx: NppStreamContext) -> Result<Image<S, C>>
        where Self: Img<S, C>, Image<S, C>: super::isu::Malloc
    {
        let mut dst = self.malloc_same_size()?;
        self.mul(other, &mut dst, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_mul {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Mul<$sample_ty, $channel_ty> for T {
            fn mul(&self, other: impl Img<$sample_ty, $channel_ty>, mut dst: impl ImgMut<$sample_ty, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                assert_same_size!(self, other);
                assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Mul $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        other.device_ptr(),
                        other.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
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
impl_mul!(f32, C<1>, _32f, C1);

pub trait SqrIP<S: Sample, C: Channels>: __priv::Sealed {
    fn sqr_ip(&mut self, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_sqrip {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: ImgMut<$sample_ty, $channel_ty>> SqrIP<$sample_ty, $channel_ty> for T {
            fn sqr_ip(&mut self, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Sqr $sample_id _ $channel_id IR_Ctx>])(
                        self.device_ptr_mut(),
                        self.pitch(),
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
impl_sqrip!(f32, C<1>, _32f, C1);

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaDevice;

    use crate::{C, get_stream_ctx};
    use crate::safe::ial::SqrIP;
    use crate::safe::Image;
    use crate::safe::isu::Malloc;
    use crate::safe::Result;

    #[test]
    fn mul() {}

    #[test]
    fn sqrip() -> Result<()> {
        let dev = CudaDevice::new(0).unwrap();
        let ctx = get_stream_ctx()?;
        let mut img = Image::<f32, C<3>>::malloc(16, 16)?;
        img.sqr_ip(ctx)?;
        Ok(())
    }
}
