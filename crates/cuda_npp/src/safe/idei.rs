//! Image data exchange and initialization

use std::ops::Range;

use cuda_npp_sys::*;

use crate::{Channel, Sample, C};

use super::{Image, Result};

pub trait Set<S> {
    fn set(&mut self, value: S, ctx: NppStreamContext) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_set_single {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Set<$sample_ty> for Image<$sample_ty, $channel_ty> {
            fn set(&mut self, value: $sample_ty, ctx: NppStreamContext) -> Result<()>
            where
                Self: Sized,
            {
                unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value,
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

impl_set_single!(u8, C<1>, _8u, C1);
impl_set_single!(u16, C<1>, _16u, C1);
impl_set_single!(i16, C<1>, _16s, C1);
impl_set_single!(f32, C<1>, _32f, C1);

trait SetMany<S, const N: usize> {
    fn set(&mut self, value: [S; N], ctx: NppStreamContext) -> Result<()>
    where
        Self: Sized;
}

macro_rules! impl_set {
    ($sample_ty:ty, $channel_ty:ty, $channel_count:literal, $sample_id:ident, $channel_id:ident) => {
        impl SetMany<$sample_ty, $channel_count> for Image<$sample_ty, $channel_ty> {
            fn set(&mut self, value: [$sample_ty; $channel_count], ctx: NppStreamContext) -> Result<()>
            where
                Self: Sized,
            {
                unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value.as_ptr(),
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

impl_set!(u8, C<2>, 2, _8u, C2);
impl_set!(u8, C<3>, 3, _8u, C3);
impl_set!(u8, C<4>, 4, _8u, C4);

impl_set!(f32, C<2>, 2, _32f, C2);
impl_set!(f32, C<3>, 3, _32f, C3);
impl_set!(f32, C<4>, 4, _32f, C4);

pub trait Convert<C: Channel> {
    // Associated type because we want to be able to use T in return position only
    type T: Sample;

    fn convert(&self, dst: &mut Image<Self::T, C>, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_convert {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Convert<$channel_ty> for Image<$sample_ty, $channel_ty> {
            type T = $tsample_ty;

            fn convert(&self, dst: &mut Image<$tsample_ty, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Convert $sample_id _ $channel_id R_Ctx>])(
                        self.data,
                        self.line_step,
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

impl_convert!(u8, C<3>, f32, _8u32f, C3);

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel, T: Sample> Image<S, C>
where
    Image<S, C>: Convert<C, T = T>,
    Image<T, C>: crate::safe::isu::Malloc,
{
    pub fn convert_new(&self, ctx: NppStreamContext) -> Result<Image<T, C>> {
        let mut dst = self.malloc_same_size()?;
        self.convert(&mut dst, ctx)?;
        Ok(dst)
    }
}

pub trait Scale<C: Channel> {
    // Associated type because we want to be able to use T in return position only
    type T: Sample;

    fn scale_float(
        &self,
        dst: &mut Image<Self::T, C>,
        bounds: Range<f32>,
        ctx: NppStreamContext,
    ) -> Result<()>;
}

macro_rules! impl_scale {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Scale<$channel_ty> for Image<$sample_ty, $channel_ty> {
            type T = $tsample_ty;

            fn scale_float(&self, dst: &mut Image<$tsample_ty, $channel_ty>, bounds: Range<f32>, ctx: NppStreamContext) -> Result<()> {
                unsafe {
                    paste::paste!([<nppi Scale $sample_id _ $channel_id R_Ctx>])(
                        self.data,
                        self.line_step,
                        dst.data as _,
                        dst.line_step,
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

#[cfg(feature = "isu")]
impl<S: Sample, C: Channel, T: Sample> Image<S, C>
where
    Image<S, C>: Scale<C, T = T>,
    Image<T, C>: crate::safe::isu::Malloc,
{
    pub fn scale_float_new(
        &self,
        bounds: Range<f32>,
        ctx: NppStreamContext,
    ) -> Result<Image<T, C>> {
        let mut dst = self.malloc_same_size()?;
        self.scale_float(&mut dst, bounds, ctx)?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use cuda_npp_sys::NppStatus;

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
        Ok(())
    }

    #[test]
    fn set() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let mut img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        img.set([1.0, 1.0, 0.0], ctx)?;
        Ok(())
    }

    #[test]
    fn convert() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.convert_new(ctx)?;
        Ok(())
    }

    #[test]
    fn into_f32() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        Ok(())
    }

    #[test]
    fn into_u8() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        NppStatus::NPP_NO_ERROR.result()?;
        Ok(())
    }
}
