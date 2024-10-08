use crate::img::{reinterpret_vec, ColorRepr, CpuImg, SampleType};
use codec_bitstream::{av1, h264};
use cudarse_video::dec::{CuVideoParser, CuvidParserCallbacks};
use cudarse_video::sys::{cudaVideoCodec, cudaVideoCodec_enum};
use image::DynamicImage;
use matroska_demuxer::{Frame, MatroskaFile};
pub use peekable;
use peekable::Peekable;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
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
pub fn probe_image(r: &mut Peekable<impl Read>) -> Option<ImageProbe> {
    let mut start = [0; PROBE_LEN];
    let len = r.peek(&mut start).unwrap();
    // First try zune_image
    if let Some((f, _)) = zune_image::codecs::ImageFormat::guess_format(&start[..len]) {
        Some(ImageProbe::Zune(f))
    } else if let Ok(f) = image::guess_format(&start[..len]) {
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

pub fn can_decode_video(mut r: impl Read) -> Option<bool> {
    None
}

pub struct DemuxerParser<'dec> {
    parser: CuVideoParser<'dec>,
    mkv: MatroskaFile<BufReader<File>>,
    nal_length_size: usize,
    frame: Frame,
    packet: Vec<u8>,
    track_id: u64,
    codec: cudaVideoCodec,
}

impl<'dec> DemuxerParser<'dec> {
    pub fn new(file: impl AsRef<Path>, dec: &'dec impl CuvidParserCallbacks) -> Self {
        let mkv = MatroskaFile::open(BufReader::new(File::open(file).unwrap())).unwrap();

        let (id, v_track) = mkv
            .tracks()
            .iter()
            .enumerate()
            .find(|(_, t)| t.video().is_some())
            .expect("No video track in mkv file");
        let codec =
            mkv_codec_id_to_nvdec(v_track.codec_id()).expect("Unsupported video codec in mkv");

        let mut parser = CuVideoParser::new(
            codec,
            dec,
            Some(mkv.info().timestamp_scale().get() as _),
            None,
        )
        .unwrap();

        let mut nal_length_size = 0;

        match codec {
            cudaVideoCodec_enum::cudaVideoCodec_MPEG2 => {
                dbg!(v_track.codec_private());
                if let Some(private) = v_track.codec_private() {
                    parser.parse_data(private, 0).unwrap();
                }
            }
            cudaVideoCodec_enum::cudaVideoCodec_H264 => {
                let (nls, sps_pps_bitstream) =
                    h264::avcc_extradata_to_annexb(v_track.codec_private().unwrap());
                // dbg!(nal_length_size);
                parser.parse_data(&sps_pps_bitstream, 0).unwrap();
                nal_length_size = nls;
            }
            cudaVideoCodec_enum::cudaVideoCodec_AV1 => {
                let extradata =
                    av1::extract_seq_hdr_from_mkv_codec_private(v_track.codec_private().unwrap())
                        .to_vec();
                parser.parse_data(&extradata, 0).unwrap();
            }
            _ => todo!("unsupported codec"),
        }

        Self {
            parser,
            mkv,
            nal_length_size,
            frame: Default::default(),
            packet: vec![],
            track_id: id as u64,
            codec,
        }
    }

    /// Demux a packet and schedule frame to be decoded and displayed.
    pub fn demux(&mut self) -> bool {
        loop {
            if let Ok(true) = self.mkv.next_frame(&mut self.frame) {
                if self.frame.track - 1 == self.track_id {
                    match self.codec {
                        cudaVideoCodec::cudaVideoCodec_H264 => {
                            h264::packet_to_annexb(
                                &mut self.packet,
                                &self.frame.data,
                                self.nal_length_size,
                            );
                            self.parser
                                .parse_data(&self.packet, self.frame.timestamp as i64)
                                .unwrap();
                        }
                        cudaVideoCodec::cudaVideoCodec_AV1
                        | cudaVideoCodec::cudaVideoCodec_MPEG2 => {
                            self.parser
                                .parse_data(&self.frame.data, self.frame.timestamp as i64)
                                .unwrap();
                        }
                        _ => todo!("Unsupported codec"),
                    }
                    return true;
                } else {
                    continue;
                }
            } else {
                self.parser.flush().unwrap();
                return false;
            }
        }
    }
}

fn mkv_codec_id_to_nvdec(id: &str) -> Option<cudaVideoCodec> {
    match id {
        "V_MPEG4/ISO/AVC" => Some(cudaVideoCodec::cudaVideoCodec_H264),
        "V_AV1" => Some(cudaVideoCodec::cudaVideoCodec_AV1),
        "V_MPEG2" => Some(cudaVideoCodec::cudaVideoCodec_MPEG2),
        // Unsupported
        _ => None,
    }
}
