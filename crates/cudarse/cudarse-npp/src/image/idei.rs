//! Image data exchange and initialization

use std::ops::Range;

use crate::sys::*;
use crate::{debug_assert_same_size, Result};

use super::{Channels, Image, Img, ImgMut, Sample, C, P};

pub trait Set<S: Sample, C: Channels> {
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
                }.result()
            }
        }
    };
}

impl_set_single!(u8, C<1>, _8u, C1);
impl_set_single!(u16, C<1>, _16u, C1);
impl_set_single!(i16, C<1>, _16s, C1);
impl_set_single!(f32, C<1>, _32f, C1);

pub trait SetMany<S: Sample, C: Channels, const N: usize> {
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
                }.result()
            }
        }
    };
}

impl_set!(u8, C<2>, 2, _8u, C2);
impl_set!(u8, C<3>, 3, _8u, C3);
impl_set!(u8, C<4>, 4, _8u, C4);

impl_set!(u16, C<2>, 2, _16u, C2);

impl_set!(f32, C<2>, 2, _32f, C2);
impl_set!(f32, C<3>, 3, _32f, C3);
impl_set!(f32, C<4>, 4, _32f, C4);

pub trait Copy<S: Sample, C: Channels> {
    fn copy(&self, dst: impl ImgMut<S, C>, ctx: NppStreamContext) -> Result<()>;
}

macro_rules! impl_copy {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Copy<$sample_ty, $channel_ty> for T {
            fn copy(&self, mut dst: impl ImgMut<$sample_ty, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                debug_assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Copy $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        self.size(),
                        ctx,
                    )
                }.result()
            }
        }
    };
}

impl_copy!(i8, C<1>, _8s, C1);
impl_copy!(i8, C<2>, _8s, C2);
impl_copy!(i8, C<3>, _8s, C3);
impl_copy!(i8, C<4>, _8s, C4);

impl_copy!(u8, C<1>, _8u, C1);
impl_copy!(u8, C<3>, _8u, C3);
impl_copy!(u8, C<4>, _8u, C4);

impl_copy!(i16, C<1>, _16s, C1);
impl_copy!(i16, C<3>, _16s, C3);
impl_copy!(i16, C<4>, _16s, C4);

impl_copy!(u16, C<1>, _16u, C1);
impl_copy!(u16, C<3>, _16u, C3);
impl_copy!(u16, C<4>, _16u, C4);

impl_copy!(f32, C<1>, _32f, C1);
impl_copy!(f32, C<3>, _32f, C3);
impl_copy!(f32, C<4>, _32f, C4);

pub trait ConvertChannel<S: Sample, C: Channels> {
    type Target: Channels;

    fn convert_channel(
        &self,
        dst: impl ImgMut<S, Self::Target>,
        ctx: NppStreamContext,
    ) -> Result<()>;

    #[cfg(feature = "isu")]
    fn convert_channel_new(&self, ctx: NppStreamContext) -> Result<Image<S, Self::Target>>
    where
        Self: Img<S, C>,
        Image<S, Self::Target>: crate::image::isu::Malloc,
    {
        let mut dst = self.malloc_same_size()?;
        self.convert_channel(&mut dst, ctx).map(|_| dst)
    }
}

macro_rules! impl_convert_channel {
    ($sample_ty:ty, $channel_ty:ty, $tchannel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> ConvertChannel<$sample_ty, $channel_ty> for T {
            type Target = $tchannel_ty;

            fn convert_channel(&self, mut dst: impl ImgMut<$sample_ty, $tchannel_ty>, ctx: NppStreamContext) -> Result<()> {
                debug_assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Copy $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        self.size(),
                        ctx,
                    )
                }.result()
            }
        }
    };
}

impl_convert_channel!(f32, C<3>, P<3>, _32f, C3P3);
impl_convert_channel!(f32, C<4>, P<4>, _32f, C4P4);
impl_convert_channel!(f32, P<3>, C<3>, _32f, P3C3);
impl_convert_channel!(f32, P<4>, C<4>, _32f, P4C4);

pub trait Convert<S: Sample, C: Channels> {
    // Associated type because we want to be able to use T in return position only
    type Target: Sample;

    /// Convert sample type
    fn convert(&self, dst: impl ImgMut<Self::Target, C>, ctx: NppStreamContext) -> Result<()>;

    #[cfg(feature = "isu")]
    fn convert_new(&self, ctx: NppStreamContext) -> Result<Image<Self::Target, C>>
    where
        Self: Img<S, C>,
        Image<Self::Target, C>: crate::image::isu::Malloc,
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
                debug_assert_same_size!(self, dst);
                unsafe {
                    paste::paste!([<nppi Convert $sample_id _ $channel_id R_Ctx>])(
                        self.device_ptr(),
                        self.pitch(),
                        dst.device_ptr_mut(),
                        dst.pitch(),
                        self.size(),
                        ctx,
                    )
                }.result()
            }
        }
    };
}

impl_convert!(u8, C<1>, i16, _8u16s, C1);
impl_convert!(u8, C<3>, f32, _8u32f, C3);

pub trait Scale<S: Sample, C: Channels> {
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
    where
        Self: Img<S, C>,
        Image<Self::Target, C>: crate::image::isu::Malloc,
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
                debug_assert_same_size!(self, dst);
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

pub trait Transpose<S: Sample, C: Channels> {
    fn transpose(&self, dst: impl ImgMut<S, C>, ctx: NppStreamContext) -> Result<()>;

    #[cfg(feature = "isu")]
    fn transpose_new(&self, ctx: NppStreamContext) -> Result<Image<S, C>>
    where
        Self: Img<S, C>,
        Image<S, C>: crate::image::isu::Malloc,
    {
        let mut dst = crate::image::isu::Malloc::malloc(self.height(), self.width())?;
        self.transpose(&mut dst, ctx)?;
        Ok(dst)
    }
}

macro_rules! impl_transpose {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Transpose<$sample_ty, $channel_ty> for T {
            fn transpose(&self, mut dst: impl ImgMut<$sample_ty, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                assert_eq!(self.width(), dst.height());
                assert_eq!(self.height(), dst.width());
                unsafe {
                    paste::paste!([<nppi Transpose $sample_id _ $channel_id R_Ctx>])(
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

impl_transpose!(f32, C<3>, _32f, C3);

macro_rules! impl_transpose_planar {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl<T: Img<$sample_ty, $channel_ty>> Transpose<$sample_ty, $channel_ty> for T {
            fn transpose(&self, mut dst: impl ImgMut<$sample_ty, $channel_ty>, ctx: NppStreamContext) -> Result<()> {
                assert_eq!(self.width(), dst.height());
                assert_eq!(self.height(), dst.width());
                unsafe {
                    let dst_pitch = dst.pitch();
                    for (r, w) in self.alloc_ptrs().zip(dst.alloc_ptrs_mut()) {
                        paste::paste!([<nppi Transpose $sample_id _ C1 R_Ctx>])(
                            r,
                            self.pitch(),
                            w,
                            dst_pitch,
                            self.size(),
                            ctx,
                        ).result()?;
                    }
                }
                Ok(())
            }
        }
    };
}

impl_transpose_planar!(f32, P<3>, _32f, P3);

#[cfg(test)]
mod tests {
    use crate::get_stream_ctx;
    use crate::image::idei::{Convert, Scale, Set, SetMany};
    use crate::image::isu::Malloc;
    use crate::image::{Image, C};
    use crate::Result;
    use cudarse_driver::{init_cuda_and_primary_ctx, sync_ctx};

    #[test]
    fn set_single() -> Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let mut img = Image::<u8, C<1>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        img.set(127, ctx)?;
        sync_ctx().unwrap();
        Ok(())
    }

    #[test]
    fn set() -> Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let mut img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        img.set([1.0, 1.0, 0.0], ctx)?;
        sync_ctx().unwrap();
        Ok(())
    }

    #[test]
    fn convert() -> Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.convert_new(ctx)?;
        sync_ctx().unwrap();
        Ok(())
    }

    #[test]
    fn into_f32() -> Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let img = Image::<u8, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        sync_ctx().unwrap();
        Ok(())
    }

    #[test]
    fn into_u8() -> Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let img2 = img.scale_float_new(0.0..1.0, ctx)?;
        sync_ctx().unwrap();
        Ok(())
    }
}
