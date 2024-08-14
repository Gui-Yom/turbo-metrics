use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::thread;
use std::time::{Duration, Instant};

use codec_bitstream::{av1, h264};
use cuda_driver::{CuDevice, CuStream};
use matroska_demuxer::{Frame, MatroskaFile, TrackType};
use nv_video_codec::dec::CuVideoParser;
use nv_video_codec::dec_mt::npp::icc::NV12toRGB;
use nv_video_codec::dec_mt::npp::{get_stream_ctx, Img, C};
use nv_video_codec::dec_mt::DecoderHolder;
use nv_video_codec::sys::cudaVideoCodec;
use ssimulacra2_cuda::Ssimulacra2;
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

fn main() {
    cuda_driver::init_cuda().expect("Could not initialize the CUDA API");
    let dev = CuDevice::get(0).unwrap();
    println!(
        "Using device {} with CUDA version {}",
        dev.name().unwrap(),
        cuda_driver::cuda_driver_version().unwrap()
    );
    // Bind to main thread
    let ctx = dev.retain_primary_ctx().unwrap();
    ctx.set_current().unwrap();

    let mut scores = Vec::with_capacity(4096);

    {
        let mut cb_ref = DecoderHolder::new(None);
        let mut cb_dis = DecoderHolder::new(None);
        thread::scope(|s| {
            thread::Builder::new()
                .name("demuxer-parser-ref".to_string())
                .spawn_scoped(s, || {
                    ctx.set_current().unwrap();
                    let mut mkv_ref = MatroskaFile::open(BufReader::new(
                        File::open("data/friends_cut.mkv").unwrap(),
                    ))
                        .unwrap();
                    demux_h264(&mut mkv_ref, &cb_ref);
                })
                .unwrap();

            thread::Builder::new()
                .name("demuxer-parser-dis".to_string())
                .spawn_scoped(s, || {
                    ctx.set_current().unwrap();
                    let mut mkv_dis = MatroskaFile::open(BufReader::new(
                        File::open("data/dummy_encode2.mkv").unwrap(),
                    ))
                        .unwrap();
                    demux_h264(&mut mkv_dis, &cb_dis);
                })
                .unwrap();

            let mut counter: i32 = -1;
            let mut rx_ref = cb_ref.wait_for_rx();
            let mut rx_dis = cb_dis.wait_for_rx();
            let format = cb_ref.format().unwrap();
            let mut ss = Ssimulacra2::new(format.display_width(), format.display_height()).unwrap();
            println!("Decoders initialized, now processing ...");
            let start = Instant::now();
            while let Ok(Some(disp_ref)) = rx_ref.peek() {
                counter += 1;

                {
                    let src = cb_ref.map_npp_nv12(&disp_ref, &ss.main_ref).unwrap();
                    src.nv12_to_rgb_bt709_limited(
                        &mut ss.ref_input,
                        get_stream_ctx()
                            .unwrap()
                            .with_stream(ss.main_ref.inner() as _),
                    )
                        .unwrap();

                    let disp_dis = rx_dis.peek().unwrap().unwrap();

                    let src = cb_dis.map_npp_nv12(&disp_dis, &ss.main_dis).unwrap();
                    src.nv12_to_rgb_bt709_limited(
                        &mut ss.dis_input,
                        get_stream_ctx()
                            .unwrap()
                            .with_stream(ss.main_dis.inner() as _),
                    )
                        .unwrap();

                    // println!(
                    //     "ref {}/dis {} : {}, {}",
                    //     disp_ref.picture_index,
                    //     disp_dis.picture_index,
                    //     Duration::from_millis(disp_ref.timestamp as u64).as_secs_f32(),
                    //     Duration::from_millis(disp_dis.timestamp as u64).as_secs_f32()
                    // );

                    // if counter < 406 {
                    //     continue;
                    // }
                    // if counter > 400 {
                    //     break;
                    // }

                    let score = ss.compute().unwrap();
                    scores.push(score);

                    // nvdec frames are unmapped at this point
                }

                // Free up a spot in the queue
                rx_ref.recv().unwrap();
                rx_dis.recv().unwrap();

                // if score < 0.0 {
                //     save_img(&ss.ref_input, &format!("ref{counter}"), &ss.main_ref);
                //     save_img(&ss.dis_input, &format!("dis{counter}"), &ss.main_dis);
                // }
            }
            let total = start.elapsed().as_millis();
            let total_frames = counter + 1;
            println!(
                "Done ! Processed {} frame pairs in {total} ms ({} fps)",
                total_frames,
                total_frames as u128 * 1000 / total
            );
        });
    }

    println!("all threads are done !");

    let mut stats = incr_stats::vec::descriptive(&scores).unwrap();
    println!("ssimulacra2 stats");
    println!("min: {}", stats.min().unwrap());
    println!("mean: {}", stats.mean().unwrap());

    dev.release_primary_ctx().unwrap();
}

fn demux_av1<R: Read + Seek>(mkv: &mut MatroskaFile<R>, cb: &DecoderHolder) {
    let extradata_dis =
        av1::extract_seq_hdr_from_mkv_codec_private(mkv.tracks()[0].codec_private().unwrap())
            .to_vec();
    let mut parser_dis = CuVideoParser::new(
        cudaVideoCodec::cudaVideoCodec_AV1,
        cb,
        None,
        // Some(&extradata_dis),
        None,
    )
        .unwrap();
    parser_dis.parse_data(&extradata_dis, 0).unwrap();
    thread::sleep(Duration::from_secs(1));
    let mut frame = Frame::default();
    let mut packet = vec![];
    while let Ok(remaining) = mkv.next_frame(&mut frame) {
        if !remaining || !cb.is_open() {
            break;
        }
        let track = &mkv.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            packet.clear();
            packet.extend_from_slice(&frame.data);
            parser_dis
                .parse_data(&packet, frame.timestamp as i64)
                .unwrap();
        }
    }
    parser_dis.flush().unwrap();
}

fn demux_h264<R: Read + Seek>(mkv: &mut MatroskaFile<R>, cb: &DecoderHolder) {
    let mut parser = CuVideoParser::new(
        cudaVideoCodec::cudaVideoCodec_H264,
        cb,
        Some(mkv.info().timestamp_scale().get() as _),
        None,
    )
        .unwrap();
    let (nal_length_size, sps_pps_bitstream) =
        h264::avcc_extradata_to_annexb(mkv.tracks()[0].codec_private().unwrap());
    dbg!(nal_length_size);
    parser.parse_data(&sps_pps_bitstream, 0).unwrap();
    let mut frame = Frame::default();
    let mut packet = Vec::new();
    while let Ok(remaining) = mkv.next_frame(&mut frame) {
        if !remaining || !cb.is_open() {
            break;
        }
        let track = &mkv.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            // dbg!(frame.timestamp);
            h264::packet_to_annexb(&mut packet, &frame.data, nal_length_size);
            parser.parse_data(&packet, frame.timestamp as i64).unwrap();
        }
    }
    parser.flush().unwrap();
}

fn save_img(img: impl Img<u8, C<3>>, name: &str, stream: &CuStream) {
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

fn save_img_f32(img: impl Img<f32, C<3>>, name: &str, stream: &CuStream) {
    // dev.synchronize().unwrap();
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
