//! Image color conversions

use crate::sys::*;

use crate::Result;

use super::{Channels, Img, ImgMut, Sample, C, P};

pub trait GammaFwdIP<S: Sample, C: Channels> {
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

pub trait GammaInvIP<S: Sample, C: Channels> {
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

pub trait NV12toRGB {
    fn nv12_to_rgb(&self, dst: impl ImgMut<u8, C<3>>, ctx: NppStreamContext) -> Result<()>;
    fn nv12_to_rgb_bt709_limited(
        &self,
        dst: impl ImgMut<u8, C<3>>,
        ctx: NppStreamContext,
    ) -> Result<()>;
    fn nv12_to_rgb_bt709_full(
        &self,
        dst: impl ImgMut<u8, C<3>>,
        ctx: NppStreamContext,
    ) -> Result<()>;
}

impl<T: Img<u8, P<2>>> NV12toRGB for T {
    fn nv12_to_rgb(&self, mut dst: impl ImgMut<u8, C<3>>, ctx: NppStreamContext) -> Result<()> {
        unsafe {
            nppiNV12ToRGB_8u_P2C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                dst.device_ptr_mut(),
                dst.pitch(),
                self.size(),
                ctx,
            )
            .result()
        }
    }

    fn nv12_to_rgb_bt709_limited(
        &self,
        mut dst: impl ImgMut<u8, C<3>>,
        ctx: NppStreamContext,
    ) -> Result<()> {
        unsafe {
            nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                dst.device_ptr_mut(),
                dst.pitch(),
                self.size(),
                ctx,
            )
            .result()
        }
    }

    fn nv12_to_rgb_bt709_full(
        &self,
        mut dst: impl ImgMut<u8, C<3>>,
        ctx: NppStreamContext,
    ) -> Result<()> {
        unsafe {
            nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                dst.device_ptr_mut(),
                dst.pitch(),
                self.size(),
                ctx,
            )
            .result()
        }
    }
}

pub trait NV12ToRGBColorTwist {
    fn nv12_to_rgb(
        &self,
        dst: impl ImgMut<u8, C<3>>,
        twist: &[[f32; 4]; 3],
        ctx: NppStreamContext,
    ) -> Result<()>;
}

impl<T: Img<u8, P<2>>> NV12ToRGBColorTwist for T {
    fn nv12_to_rgb(
        &self,
        mut dst: impl ImgMut<u8, C<3>>,
        twist: &[[f32; 4]; 3],
        ctx: NppStreamContext,
    ) -> Result<()> {
        unsafe {
            nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
                self.device_ptr(),
                [self.pitch(), self.pitch()].as_mut_ptr(),
                dst.device_ptr_mut(),
                dst.pitch(),
                self.size(),
                twist.as_ptr(),
                ctx,
            )
            .result()
        }
    }
}

pub trait YUVtoRGB {
    fn yuv_to_rgb(&self, dst: impl ImgMut<u8, C<3>>, ctx: NppStreamContext) -> Result<()>;
}

impl<T: Img<u8, C<3>>> YUVtoRGB for T {
    fn yuv_to_rgb(&self, mut dst: impl ImgMut<u8, C<3>>, ctx: NppStreamContext) -> Result<()> {
        unsafe {
            nppiYUVToRGB_8u_C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                dst.device_ptr_mut(),
                dst.pitch(),
                self.size(),
                ctx,
            )
            .result()
        }
    }
}
