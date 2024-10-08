use crate::kernel::Kernel;
use cudarse_driver::sys::CuResult;
use cudarse_driver::CuStream;
use cudarse_npp::image::{Img, ImgMut, C, P};

pub mod kernel;

#[derive(Debug, Copy, Clone)]
pub enum ColorMatrix {
    BT709,
    BT601_525,
    BT601_625,
}

#[derive(Debug, Copy, Clone)]
pub enum Transfer {
    /// This is the same for BT601
    BT709,
    Passthrough,
}

pub struct ColorspaceConversion {
    kernel: Kernel,
}

impl ColorspaceConversion {
    pub fn new() -> Self {
        Self {
            kernel: Kernel::load().unwrap(),
        }
    }

    pub fn biplanaryuv420_to_linearrgb_8(
        &self,
        matrix: ColorMatrix,
        transfer: Transfer,
        full_color_range: bool,
        src: impl Img<u8, P<2>>,
        dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        match matrix {
            ColorMatrix::BT709 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_8_L_BT709(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
            ColorMatrix::BT601_525 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_8_L_BT601_525(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
            ColorMatrix::BT601_625 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_8_L_BT601_625(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
        }
    }

    pub fn biplanaryuv420_to_linearrgb_16(
        &self,
        matrix: ColorMatrix,
        transfer: Transfer,
        full_color_range: bool,
        src: impl Img<u16, P<2>>,
        dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        match matrix {
            ColorMatrix::BT709 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_16_L_BT709(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
            ColorMatrix::BT601_525 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_16_L_BT601_525(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
            ColorMatrix::BT601_625 => match transfer {
                Transfer::BT709 => {
                    if full_color_range {
                        todo!()
                    } else {
                        self.kernel
                            .biplanaryuv420_to_linearrgb_16_L_BT601_625(src, dst, stream)
                    }
                }
                _ => todo!(),
            },
        }
    }

    pub fn biplanaryuv420_to_linearrgb_debug_8(
        &self,
        src: impl Img<u8, P<2>>,
        dst: impl ImgMut<u8, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        self.kernel
            .biplanaryuv420_to_linearrgb_debug(src, dst, stream)
    }

    pub fn f32_to_8bit(
        &self,
        src: impl Img<f32, C<3>>,
        dst: impl ImgMut<u8, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        self.kernel.f32_to_8bit(src, dst, stream)
    }

    pub fn srgb_to_linear_u8(
        &self,
        src: impl Img<u8, C<3>>,
        dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        self.kernel.srgb_to_linear_u8_lookup(src, dst, stream)
    }

    pub fn srgb_to_linear_u16(
        &self,
        src: impl Img<u16, C<3>>,
        dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        self.kernel.srgb_to_linear_u16(src, dst, stream)
    }

    pub fn srgb_to_linear_f32(
        &self,
        src: impl Img<f32, C<3>>,
        dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        self.kernel.srgb_to_linear_f32(src, dst, stream)
    }
}
