use std::fmt::Display;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use codec_bitstream::h264::{ColourPrimaries, MatrixCoefficients, TransferCharacteristic};
use codec_bitstream::{av1, h264};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, ImgMut, C};
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::sys::CUVIDEOFORMAT;
use matroska_demuxer::{Frame, MatroskaFile};
use turbo_metrics::cudarse_driver::{CuDevice, CuStream};
use turbo_metrics::cudarse_video::dec::npp::Img;
use turbo_metrics::cudarse_video::dec::{CuVideoParser, CuvidParserCallbacks};
use turbo_metrics::cudarse_video::dec_simple::NvDecoderSimple;
use turbo_metrics::cudarse_video::sys::{cudaVideoCodec, cudaVideoCodec_enum};

fn main() {
    cudarse_driver::init_cuda().expect("Could not initialize the CUDA API");
    let dev = CuDevice::get(0).unwrap();
    println!(
        "Using device {} with CUDA version {}",
        dev.name().unwrap(),
        cudarse_driver::cuda_driver_version().unwrap()
    );
    // Bind to main thread
    let ctx = dev.retain_primary_ctx().unwrap();
    ctx.set_current().unwrap();

    let colorspace = cuda_colorspace::Kernel::load().unwrap();

    {
        let cb_ref = NvDecoderSimple::new(3, None);
        let cb_dis = NvDecoderSimple::new(3, None);

        let mut demuxer_ref = DemuxerParser::new("data/friends_cut.mkv", &cb_ref);
        // let mut demuxer_dis = DemuxerParser::new("data/dummy_encode2.mkv", &cb_dis);
        let mut demuxer_dis = DemuxerParser::new("data/encode0.mkv", &cb_dis);

        let mut counter = 0;

        while cb_ref.format().is_none() {
            demuxer_ref.demux();
        }
        let format = cb_ref.format().unwrap();
        println!("ref: {}", video_format_line(&format));

        let size = format.size();

        while cb_dis.format().is_none() {
            demuxer_dis.demux();
        }
        let format_dis = cb_dis.format().unwrap();
        println!("dis: {}", video_format_line(&format_dis));

        assert_eq!(size, format_dis.size());

        let main_stream = CuStream::new().unwrap();
        let alt_stream = CuStream::new().unwrap();
        main_stream.sync().unwrap();
        alt_stream.sync().unwrap();
        cudarse_npp::set_stream(main_stream.inner() as _).unwrap();

        let mut linear_rgb_ref = Image::malloc(size.width as _, size.height as _).unwrap();
        let mut linear_rgb_dis = linear_rgb_ref.malloc_same_size().unwrap();

        println!("Initialized, now processing ...");
        let start = Instant::now();

        'main: loop {
            while !cb_ref.has_frames() {
                demuxer_ref.demux();
            }
            while !cb_dis.has_frames() {
                demuxer_dis.demux();
            }
            for (fref, fdis) in cb_ref.frames_sync(&cb_dis) {
                if let (Some(fref), Some(fdis)) = (fref, fdis) {
                    let format = cb_ref.format().unwrap();
                    convert_frame_to_linearrgb(
                        cb_ref.map_npp(&fref, &main_stream).unwrap(),
                        format,
                        &colorspace,
                        &mut linear_rgb_ref,
                        &main_stream,
                    );

                    // if counter == 130 || counter == 200 || counter == 250 || counter == 300 {
                    //     save_img(dst, &format!("dst{}", counter), &main_stream);
                    //     // break 'main;
                    // }

                    let format = cb_dis.format().unwrap();
                    convert_frame_to_linearrgb(
                        cb_dis.map_npp(&fdis, &alt_stream).unwrap(),
                        format,
                        &colorspace,
                        &mut linear_rgb_dis,
                        &alt_stream,
                    );

                    main_stream.wait_for_stream(&alt_stream).unwrap();
                    main_stream.sync().unwrap();

                    counter += 1;
                } else {
                    break 'main;
                }
            }
        }

        let total = start.elapsed().as_millis();
        let total_frames = counter;
        println!(
            "Done ! Processed {} frame pairs in {total} ms ({} fps)",
            total_frames,
            total_frames as u128 * 1000 / total
        );
        // let stats = turbo.stats();
        // println!("Stats : {stats:#?}");
    }

    dev.release_primary_ctx().unwrap();
}

struct DemuxerParser<'dec> {
    parser: CuVideoParser<'dec>,
    mkv: MatroskaFile<BufReader<File>>,
    nal_length_size: usize,
    frame: Frame,
    packet: Vec<u8>,
    track_id: u64,
    codec: cudaVideoCodec,
}

impl<'dec> DemuxerParser<'dec> {
    fn new(file: &str, dec: &'dec impl CuvidParserCallbacks) -> Self {
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
    fn demux(&mut self) -> bool {
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
                        cudaVideoCodec::cudaVideoCodec_AV1 => {
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

fn video_format_line(format: &CUVIDEOFORMAT) -> impl Display {
    format!(
        "CP {:?}, TC {:?}, MC {:?}, Full range: {}",
        ColourPrimaries::from_byte(format.video_signal_description.color_primaries,),
        TransferCharacteristic::from_byte(format.video_signal_description.transfer_characteristics,),
        MatrixCoefficients::from_byte(format.video_signal_description.matrix_coefficients,),
        format.video_signal_description.full_range()
    )
}

fn convert_frame_to_linearrgb(
    frame: NvDecFrame<'_>,
    format: &CUVIDEOFORMAT,
    colorspace: &cuda_colorspace::Kernel,
    mut dst: impl ImgMut<f32, C<3>>,
    stream: &CuStream,
) {
    match frame {
        NvDecFrame::NV12(frame) => {
            match MatrixCoefficients::from_byte(format.video_signal_description.matrix_coefficients)
            {
                MatrixCoefficients::BT709 | MatrixCoefficients::Unspecified => {
                    if format.video_signal_description.full_range() {
                        colorspace
                            .biplanaryuv420_to_linearrgb_8_F_BT709(frame, dst, stream)
                            .unwrap();
                        // frame.nv12_to_rgb_bt709_full();
                    } else {
                        colorspace
                            .biplanaryuv420_to_linearrgb_8_L_BT709(frame, dst, stream)
                            .unwrap();
                        // frame.nv12_to_rgb_bt709_limited();
                    }
                }
                _ => todo!("Unsupported matrix coefficients"),
            }
        }
        NvDecFrame::P010(frame) => {
            match MatrixCoefficients::from_byte(format.video_signal_description.matrix_coefficients)
            {
                MatrixCoefficients::BT709 | MatrixCoefficients::Unspecified => {
                    if format.video_signal_description.full_range() {
                        todo!();
                        // frame.nv12_to_rgb_bt709_full();
                    } else {
                        colorspace
                            .biplanaryuv420_to_linearrgb_10_L_BT709(frame, dst, stream)
                            .unwrap();
                        // frame.nv12_to_rgb_bt709_limited();
                    }
                }
                _ => todo!("Unsupported matrix coefficients"),
            }
        }
        other => todo!("Unsupported frame type in turbo metrics : {other:#?}"),
    };
}

fn save_img(img: impl Img<u8, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    // dev.synchronize().unwrap();
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
//
// fn save_img_f32(img: impl Img<f32, C<3>>, name: &str, stream: &CuStream) {
//     // dev.synchronize().unwrap();
//     let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
//     stream.sync().unwrap();
//     let mut img = zune_image::image::Image::from_f32(
//         &bytes,
//         img.width() as usize,
//         img.height() as usize,
//         ColorSpace::RGB,
//     );
//     img.metadata_mut()
//         .set_color_trc(ColorCharacteristics::Linear);
//     img.save(format!("frames/{name}.png")).unwrap()
// }
