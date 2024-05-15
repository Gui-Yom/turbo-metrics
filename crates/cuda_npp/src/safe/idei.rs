//! Image data exchange and initialization

use std::ops::Range;

use cuda_npp_sys::*;

use crate::{Channels, Sample, C, __priv, assert_same_size};

use super::{Image, Result, ImgMut, Img};

pub trait Set<S: Sample, C: Channels>: __priv::Sealed {
    fn set(&mut self, value: S, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_set_single {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: ImgMut<$sample_ty, $channel_ty>> Set<$sample_ty, $channel_ty> for T {
            fn set(&mut self, value: $sample_ty, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value,
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

impl_set_single!(u8, C<1>, _8u, C1);
impl_set_single!(u16, C<1>, _16u, C1);
impl_set_single!(i16, C<1>, _16s, C1);
impl_set_single!(f32, C<1>, _32f, C1);

pub trait SetMany<S: Sample, C: Channels, const N: usize>: __priv::Sealed {
    fn set(&mut self, value: [S; N], ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_set {
    ($sample_ty:ty, $channel_ty:ty, $channel_count:literal, $sample_id:ident, $channel_id:ident) => {
        impl<T: ImgMut<$sample_ty, $channel_ty>> SetMany<$sample_ty, $channel_ty, $channel_count> for T {
            fn set(&mut self, value: [$sample_ty; <$channel_ty>::NUM_SAMPLES], ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value.as_ptr(),
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

impl_set!(u8, C<2>, 2, _8u, C2);
impl_set!(u8, C<3>, 3, _8u, C3);
impl_set!(u8, C<4>, 4, _8u, C4);

impl_set!(f32, C<2>, 2, _32f, C2);
impl_set!(f32, C<3>, 3, _32f, C3);
impl_set!(f32, C<4>, 4, _32f, C4);

pub trait Convert<S: Sample, C: Channels>: __priv::Sealed {
    // Associated type because we want to be able to use T in return position only
    type Target: Sample;

    fn convert(&self, dst: impl ImgMut<Self::Target, C>, ctx: NppStreamContext) -> Result<()>;

    #[cfg(feature = "isu")]
    fn convert_new(&self, ctx: NppStreamContext) -> Result<Image<Self::Target, C>>
        where Self: Img<S, C>, Image<Self::Target, C>: crate::safe::isu::Malloc
    {
        let mut dst = self.malloc_same_size()?;
        self.convert(&mut dst, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_convert {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Convert<$sample_ty, $channel_ty> for T {
            type Target = $tsample_ty;

            fn convert(&self, mut dst: impl ImgMut<Self::Target, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Convert $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
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

impl_convert!(u8, C<3>, f32, _8u32f, C3);

pub trait Scale<S: Sample, C: Channels>: __priv::Sealed {
    // Associated type because we want to be able to use T in return position only
    type Target: Sample;

    fn scale_float(
        &self,
        dst: impl ImgMut<Self::Target, C>,
        bounds: Range<f32>,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn scale_float_new(
        &self,
        bounds: Range<f32>,
        ctx: NppStreamContext,
    ) -> Result<Image<Self::Target, C>>
        where Self: Img<S, C>, Image<Self::Target, C>: crate::safe::isu::Malloc
    {
        let mut dst = self.malloc_same_size()?;
        self.scale_float(&mut dst, bounds, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_scale {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Scale<$sample_ty, $channel_ty> for T {
            type Target = $tsample_ty;

            fn scale_float(&self, mut dst: impl ImgMut<Self::Target, $channel_ty>, bounds: Range<f32>, ctx: NppStreamContext) -> Result<()> {
                assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Scale $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        self.size(),
                        bounds.start,
                        bounds.end,
                        ctx,
                    )
                }.result()?;
                Ok(())
            }
        }
    };
}

impl_scale!(u8, C<3>, f32, _8u32f, C3);
impl_scale!(f32, C<3>, u8, _32f8u, C3);

#[cfg(test)]
mod tests {
    use crate::safe::idei::{Convert, Scale, Set, SetMany};
    use crate::safe::isu::Malloc;
    use crate::safe::Image;
    use crate::safe::Result;
    use crate::{get_stream_ctx, C};

    #[test]
    fn set_single() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let mut img = Image::<u8, C<1>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        img.set(127, ctx)?;
        dev.synchronize().unwrap();
        Ok(())
    }

    #[test]
    fn set() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let mut img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        img.set([1.0, 1.0, 0.0], ctx)?;
        dev.synchronize().unwrap();
        Ok(())
    }

    #[test]
    fn convert() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.convert_new(ctx)?;
        dev.synchronize().unwrap();
        Ok(())
    }

    #[test]
    fn into_f32() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        dev.synchronize().unwrap();
        Ok(())
    }

    #[test]
    fn into_u8() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        dev.synchronize().unwrap();
        Ok(())
    }
}
