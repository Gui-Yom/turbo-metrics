use crate::img::{reinterpret_vec, ColorRepr, CpuImg, SampleType};
use image::DynamicImage;
use std::io::{BufRead, Read};
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
}

/// `None` if we cannot even recognize the image, `Some` if we can recognize the format.
/// This will peek at the first bytes on the stream.
pub fn probe_image(mut r: impl BufRead) -> Option<ImageProbe> {
    let start = r.fill_buf().unwrap();
    if start.len() < PROBE_LEN {
        panic!("unexpected eof");
    }
    // First try zune_image
    if let Some((f, _)) = zune_image::codecs::ImageFormat::guess_format(start) {
        Some(ImageProbe::Zune(f))
    } else if let Ok(f) = image::guess_format(start) {
        Some(ImageProbe::Image(f))
    } else {
        None
    }
}

pub fn decode_image_frames(mut r: impl Read, probe: ImageProbe) -> Vec<CpuImg> {
    let mut data = Vec::with_capacity(4 * 1024 * 1024);
    r.read_to_end(&mut data).unwrap();
    match probe {
        ImageProbe::Zune(_) => {
            let img = zune_image::image::Image::read(&data, DecoderOptions::new_fast()).unwrap();
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
                    colortype: dbg!(colorspace).into(),
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
            let img = image::load_from_memory_with_format(&data, f).unwrap();
            let width = img.width();
            let height = img.height();
            let (sample_type, data) = match img {
                DynamicImage::ImageRgb8(i) => (SampleType::U8, i.into_vec()),
                DynamicImage::ImageRgb16(i) => (SampleType::U16, reinterpret_vec(i.into_vec())),
                DynamicImage::ImageRgb32F(i) => (SampleType::F32, reinterpret_vec(i.into_vec())),
                _ => todo!("Unsupported image layout"),
            };
            vec![CpuImg {
                sample_type,
                colortype: ColorRepr::RGB,
                width,
                height,
                data,
            }]
        }
    }
}
