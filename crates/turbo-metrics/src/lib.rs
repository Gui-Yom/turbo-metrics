use codec_bitstream::{
    av1, h264, Codec, ColorCharacteristics, ColourPrimaries, MatrixCoefficients,
    TransferCharacteristic,
};
use cuda_colorspace::{ColorMatrix, ColorspaceConversion, Transfer};
pub use cudarse_driver;
use cudarse_driver::CuStream;
pub use cudarse_npp;
use cudarse_npp::image::{Img, ImgMut, C};
pub use cudarse_video;
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::dec::{CuVideoParser, CuvidParserCallbacks};
use cudarse_video::sys::{cudaVideoCodec, cudaVideoCodec_enum, CUVIDEOFORMAT};
use matroska_demuxer::{Frame, MatroskaFile};
pub use stats;
use std::fmt::Display;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub fn color_characteristics_from_format(format: &CUVIDEOFORMAT) -> (ColorCharacteristics, bool) {
    (
        ColorCharacteristics::from_codec_bytes(
            cuda_codec_to_codec(format.codec),
            format.video_signal_description.color_primaries,
            format.video_signal_description.matrix_coefficients,
            format.video_signal_description.transfer_characteristics,
        )
        .or(color_characteristics_fallback(format)),
        format.video_signal_description.full_range(),
    )
}

fn cuda_codec_to_codec(codec: cudaVideoCodec) -> Codec {
    match codec {
        cudaVideoCodec_enum::cudaVideoCodec_MPEG2 => Codec::H262,
        cudaVideoCodec_enum::cudaVideoCodec_H264 => Codec::H264,
        cudaVideoCodec_enum::cudaVideoCodec_AV1 => Codec::AV1,
        _ => todo!(),
    }
}

fn color_characteristics_fallback(format: &CUVIDEOFORMAT) -> ColorCharacteristics {
    let height = format.display_height();
    if height <= 525 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT601_525,
            mc: MatrixCoefficients::BT601_525,
            tc: TransferCharacteristic::BT709,
        }
    } else if height <= 625 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT601_625,
            mc: MatrixCoefficients::BT601_625,
            tc: TransferCharacteristic::BT709,
        }
    } else if height <= 1080 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT709,
            mc: MatrixCoefficients::BT709,
            tc: TransferCharacteristic::BT709,
        }
    } else {
        ColorCharacteristics {
            cp: ColourPrimaries::BT709,
            mc: MatrixCoefficients::BT709,
            tc: TransferCharacteristic::BT709,
        }
    }
}

pub fn get_color_matrix(colors: &ColorCharacteristics) -> ColorMatrix {
    match (colors.cp, colors.mc) {
        (ColourPrimaries::BT709, MatrixCoefficients::BT709) => ColorMatrix::BT709,
        (ColourPrimaries::BT601_525, MatrixCoefficients::BT601_525) => ColorMatrix::BT601_525,
        (ColourPrimaries::BT601_625, MatrixCoefficients::BT601_625) => ColorMatrix::BT601_625,
        _ => todo!(),
    }
}

pub fn get_transfer(colors: &ColorCharacteristics) -> Transfer {
    match colors.tc {
        TransferCharacteristic::BT709 => Transfer::BT709,
        _ => todo!(),
    }
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

pub fn video_format_line(format: &CUVIDEOFORMAT) -> impl Display {
    let (colors, full_range) = color_characteristics_from_format(format);
    format!(
        "CP: {:?}, MC: {:?}, TC: {:?}, Full range: {}",
        colors.cp, colors.mc, colors.tc, full_range
    )
}

pub fn convert_frame_to_linearrgb(
    frame: NvDecFrame<'_>,
    colors: (ColorCharacteristics, bool),
    colorspace: &ColorspaceConversion,
    dst: impl ImgMut<f32, C<3>>,
    stream: &CuStream,
) {
    let color_matrix = get_color_matrix(&colors.0);
    let transfer = get_transfer(&colors.0);
    match frame {
        NvDecFrame::NV12(frame) => colorspace
            .biplanaryuv420_to_linearrgb_8(color_matrix, transfer, colors.1, frame, dst, stream)
            .unwrap(),
        NvDecFrame::P016(frame) => colorspace
            .biplanaryuv420_to_linearrgb_16(color_matrix, transfer, colors.1, frame, dst, stream)
            .unwrap(),
        other => todo!("Unsupported frame type in turbo metrics : {other:#?}"),
    };
}

pub fn save_img(img: impl Img<u8, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_u8(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}

pub fn save_img_u16(img: impl Img<u16, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_u16(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}

pub fn save_img_f32(img: impl Img<f32, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_f32(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}
