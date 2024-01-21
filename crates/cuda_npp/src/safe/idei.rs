use std::ops::Range;

use cuda_npp_sys::*;

use crate::safe::isu::Malloc;
use crate::{Channel, Sample, C};

use super::{Image, Result, E};

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
                let status = unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value,
                        self.data as _,
                        self.line_step,
                        NppiSize {
                            width: self.width as i32,
                            height: self.height as i32,
                        },
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
                let status = unsafe {
                    paste::paste!([<nppi Set $sample_id _ $channel_id R_Ctx>])(
                        value.as_ptr(),
                        self.data as _,
                        self.line_step,
                        NppiSize {
                            width: self.width as i32,
                            height: self.height as i32,
                        },
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

impl_set!(u8, C<2>, 2, _8u, C2);
impl_set!(u8, C<3>, 3, _8u, C3);
impl_set!(u8, C<4>, 4, _8u, C4);

impl_set!(f32, C<2>, 2, _32f, C2);
impl_set!(f32, C<3>, 3, _32f, C3);
impl_set!(f32, C<4>, 4, _32f, C4);

pub trait Convert<T: Sample, C: Channel> {
    fn convert(&self, ctx: NppStreamContext) -> Result<Image<T, C>>;
}

macro_rules! impl_convert {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Convert<$tsample_ty, $channel_ty> for Image<$sample_ty, $channel_ty> {
            fn convert(&self, ctx: NppStreamContext) -> Result<Image<$tsample_ty, $channel_ty>> {
                let mut dst = Image::malloc(self.width, self.height)?;
                let status = unsafe {
                    paste::paste!([<nppi Convert $sample_id _ $channel_id R_Ctx>])(
                        self.data,
                        self.line_step,
                        dst.data as _,
                        dst.line_step,
                        self.size(),
                        ctx,
                    )
                };
                if status == NppStatus::NPP_NO_ERROR {
                    Ok(dst)
                } else {
                    Err(E::from(status))
                }
            }
        }
    };
}

impl_convert!(u8, C<3>, f32, _8u32f, C3);

pub trait Scale<T: Sample, C: Channel> {
    fn scale_float(&self, bounds: Range<f32>, ctx: NppStreamContext) -> Result<Image<T, C>>;
}

macro_rules! impl_scale {
    ($sample_ty:ty, $channel_ty:ty, $tsample_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Scale<$tsample_ty, $channel_ty> for Image<$sample_ty, $channel_ty> {
            fn scale_float(&self, bounds: Range<f32>, ctx: NppStreamContext) -> Result<Image<$tsample_ty, $channel_ty>> {
                let mut dst = Image::malloc(self.width, self.height)?;
                let status = unsafe {
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
                };
                if status == NppStatus::NPP_NO_ERROR {
                    Ok(dst)
                } else {
                    Err(E::from(status))
                }
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
        let img2 = img.convert(ctx)?;
        Ok(())
    }

    #[test]
    fn into_f32() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float(0.0..1.0, ctx)?;
        Ok(())
    }

    #[test]
    fn into_u8() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float(0.0..1.0, ctx)?;
        Ok(())
    }
}
