use crate::color::ColorRange;
use crate::img::{reinterpret_slice, reinterpret_vec, ColorRepr, CpuImg, SampleType};
use crate::{FormatIdentifier, FrameSource, HwFrame};
use codec_bitstream::{
    ColorCharacteristics, ColourPrimaries, MatrixCoefficients, TransferCharacteristic,
};
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, ImgMut, C};
use image::DynamicImage;
use std::collections::VecDeque;
use std::error::Error;
use std::io::{BufRead, Read};
use tracing::instrument;
use zune_core::bit_depth::BitType;
use zune_core::options::DecoderOptions;

pub const PROBE_LEN: usize = 64;

#[derive(Debug, Copy, Clone)]
pub enum ImageProbe {
    /// Can be decoded using zune-image
    Zune(zune_image::codecs::ImageFormat),
    /// Can be decoded with image-rs
    Image(image::ImageFormat),
}

impl ImageProbe {
    pub fn can_decode(&self) -> bool {
        match self {
            ImageProbe::Zune(f) => f.has_decoder(),
            ImageProbe::Image(f) => match f {
                #[cfg(feature = "gif")]
                image::ImageFormat::Gif => true,
                #[cfg(feature = "tiff")]
                image::ImageFormat::Tiff => true,
                #[cfg(feature = "webp")]
                image::ImageFormat::WebP => true,
                #[cfg(feature = "avif")]
                image::ImageFormat::Avif => true,
                _ => false,
            },
        }
    }

    /// `None` if we cannot even recognize the image, `Some` if we can recognize the format.
    /// This will peek at the first bytes on the stream.
    pub fn probe_image(mut r: impl BufRead) -> Option<Self> {
        let start = r.fill_buf().unwrap();
        if start.len() < PROBE_LEN {
            panic!("unexpected eof");
        }
        // First try zune_image
        if let Some((f, _)) = zune_image::codecs::ImageFormat::guess_format(start) {
            Some(Self::Zune(f))
        } else if let Ok(f) = image::guess_format(start) {
            Some(Self::Image(f))
        } else {
            None
        }
    }

    fn codec(&self) -> String {
        match self {
            ImageProbe::Zune(f) => {
                format!("{f:?}")
            }
            ImageProbe::Image(f) => {
                format!("{f:?}")
            }
        }
    }

    fn decoder(&self) -> String {
        match self {
            ImageProbe::Zune(_) => "zune".to_string(),
            ImageProbe::Image(_) => "image".to_string(),
        }
    }
}

enum InnerHwFrame {
    U8(Image<u8, C<3>>),
    U16(Image<u16, C<3>>),
    F32(Image<f32, C<3>>),
}

pub struct ImageFrameSource {
    frames: VecDeque<CpuImg>,
    probe: ImageProbe,
    img_dev: InnerHwFrame,
    width: u32,
    height: u32,
}

impl ImageFrameSource {
    #[instrument(level = "debug", name = "image_cpu_decode", skip(r))]
    pub fn new(mut r: impl Read, probe: ImageProbe) -> Result<Self, Box<dyn Error>> {
        let mut data = Vec::with_capacity(4 * 1024 * 1024);
        r.read_to_end(&mut data)?;
        let frames = match probe {
            ImageProbe::Zune(_) => {
                let img = zune_image::image::Image::read(&data, DecoderOptions::new_fast())?;
                let sample_type = match img.depth().bit_type() {
                    BitType::U8 => SampleType::U8,
                    BitType::U16 => SampleType::U16,
                    BitType::F32 => SampleType::F32,
                    _ => todo!("Unsupported sample type"),
                };
                let (width, height) = img.dimensions();
                let colorspace = img.colorspace();
                img.frames_ref()
                    .iter()
                    .map(|f| CpuImg {
                        sample_type,
                        colortype: colorspace.into(),
                        width: width as _,
                        height: height as _,
                        data: match sample_type {
                            SampleType::U8 => f.flatten::<u8>(colorspace),
                            SampleType::U16 => reinterpret_vec(f.flatten::<u16>(colorspace)),
                            SampleType::F32 => reinterpret_vec(f.flatten::<f32>(colorspace)),
                        },
                    })
                    .collect()
            }
            ImageProbe::Image(f) => {
                let img = image::load_from_memory_with_format(&data, f)?;
                let width = img.width();
                let height = img.height();
                let (sample_type, data) = match img {
                    DynamicImage::ImageRgb8(i) => (SampleType::U8, i.into_vec()),
                    DynamicImage::ImageRgb16(i) => (SampleType::U16, reinterpret_vec(i.into_vec())),
                    DynamicImage::ImageRgb32F(i) => {
                        (SampleType::F32, reinterpret_vec(i.into_vec()))
                    }
                    _ => todo!("Unsupported image layout"),
                };
                VecDeque::from(vec![CpuImg {
                    sample_type,
                    colortype: ColorRepr::RGB,
                    width,
                    height,
                    data,
                }])
            }
        };
        let img_dev = match frames[0].sample_type {
            SampleType::U8 => InnerHwFrame::U8(Image::malloc(frames[0].width, frames[0].height)?),
            SampleType::U16 => InnerHwFrame::U16(Image::malloc(frames[0].width, frames[0].height)?),
            SampleType::F32 => InnerHwFrame::F32(Image::malloc(frames[0].width, frames[0].height)?),
        };
        Ok(Self {
            width: frames[0].width,
            height: frames[0].height,
            img_dev,
            frames,
            probe,
        })
    }
}

impl FrameSource for ImageFrameSource {
    fn format_id(&self) -> FormatIdentifier {
        FormatIdentifier {
            container: None,
            codec: self.probe.codec(),
            decoder: self.probe.decoder(),
        }
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn color_characteristics(&self) -> (ColorCharacteristics, ColorRange) {
        // TODO image color characteristics
        //  This is unused in the image path
        (
            ColorCharacteristics {
                cp: ColourPrimaries::BT709,
                mc: MatrixCoefficients::BT709,
                tc: TransferCharacteristic::BT709,
            },
            ColorRange::Full,
        )
    }

    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn skip_frames(&mut self, n: u32) {
        for _ in 0..n {
            self.frames.pop_front();
        }
    }

    fn next_frame<'a>(
        &'a mut self,
        stream: &CuStream,
    ) -> Result<Option<HwFrame<'a>>, Box<dyn Error>> {
        if let Some(f) = self.frames.pop_front() {
            Ok(Some(match &mut self.img_dev {
                InnerHwFrame::U8(img) => {
                    img.copy_from_cpu(&f.data, stream.inner() as _)?;
                    HwFrame::Npp8(img.full_view())
                }
                InnerHwFrame::U16(img) => {
                    img.copy_from_cpu(reinterpret_slice(&f.data), stream.inner() as _)?;
                    HwFrame::Npp16(img.full_view())
                }
                InnerHwFrame::F32(img) => {
                    img.copy_from_cpu(reinterpret_slice(&f.data), stream.inner() as _)?;
                    HwFrame::Npp32(img.full_view())
                }
            }))
        } else {
            Ok(None)
        }
    }
}
