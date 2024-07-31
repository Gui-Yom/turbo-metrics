use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::thread;
use std::time::{Duration, Instant};

use matroska_demuxer::{Frame, MatroskaFile, TrackType};

use codec_cfg_record::{av1, h264};
use cuda_driver::CuDevice;
use nv_video_codec::dec::{CuVideoCtxLock, CuVideoParser};
use nv_video_codec::dec_helper::npp::get_stream_ctx;
use nv_video_codec::dec_helper::npp::icc::NV12toRGB;
use nv_video_codec::dec_helper::DecoderHolder;
use nv_video_codec::sys::cudaVideoCodec;
use ssimulacra2_cuda::Ssimulacra2;

fn main() {
    let mut mkv_ref = matroska_demuxer::MatroskaFile::open(BufReader::new(
        File::open("data/friends_cut.mkv").unwrap(),
    ))
    .unwrap();

    let mut mkv_dis = matroska_demuxer::MatroskaFile::open(BufReader::new(
        File::open("data/dummy_encode2.mkv").unwrap(),
    ))
    .unwrap();

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

    let lock = CuVideoCtxLock::new(&ctx).unwrap();
    let lock2 = CuVideoCtxLock::new(&ctx).unwrap();

    {
        let mut cb_ref = DecoderHolder::new(lock);
        let mut cb_dis = DecoderHolder::new(lock2);
        thread::scope(|s| {
            thread::Builder::new()
                .name("demuxer-parser-ref".to_string())
                .spawn_scoped(s, || {
                    ctx.set_current().unwrap();
                    demux_h264(&mut mkv_ref, &cb_ref);
                })
                .unwrap();

            thread::Builder::new()
                .name("demuxer-parser-dis".to_string())
                .spawn_scoped(s, || {
                    ctx.set_current().unwrap();
                    demux_h264(&mut mkv_dis, &cb_dis);
                })
                .unwrap();

            let mut counter: i32 = -1;
            let mut rx_ref = cb_ref.wait_for_rx();
            let mut rx_dis = cb_dis.wait_for_rx();
            let format = cb_ref.format().unwrap();
            let mut ss = Ssimulacra2::new(format.display_width(), format.display_height()).unwrap();
            println!("Initialized all");
            let start = Instant::now();
            while let Ok(Some(disp_ref)) = rx_ref.peek() {
                counter += 1;
                {
                    let src = cb_ref.map_npp_nv12(&disp_ref, &ss.main_ref).unwrap();
                    src.nv12_to_rgb(
                        &mut ss.ref_input,
                        get_stream_ctx()
                            .unwrap()
                            .with_stream(ss.main_ref.inner() as _),
                    )
                    .unwrap();
                }
                let disp_dis = rx_dis.peek().unwrap().unwrap();
                {
                    let src = cb_dis.map_npp_nv12(&disp_dis, &ss.main_dis).unwrap();
                    src.nv12_to_rgb(
                        &mut ss.dis_input,
                        get_stream_ctx()
                            .unwrap()
                            .with_stream(ss.main_dis.inner() as _),
                    )
                    .unwrap();
                }
                rx_ref.recv().unwrap();
                rx_dis.recv().unwrap();
                if counter < 400 || counter > 405 {
                    continue;
                }

                dbg!(ss.compute().unwrap());
            }
            let total = start.elapsed().as_millis();
            let total_frames = (counter + 1) * 2;
            println!(
                "Decoded {} frames in {total} ms, {} fps",
                total_frames,
                total_frames as u128 * 1000 / total
            );
        });
    }

    println!("done !");

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
    parser_dis.feed_packet(&extradata_dis, 0).unwrap();
    thread::sleep(Duration::from_secs(1));
    let mut frame = Frame::default();
    let mut packet = vec![];
    while let Ok(remaining) = mkv.next_frame(&mut frame) {
        if !remaining {
            break;
        }
        let track = &mkv.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            packet.clear();
            packet.extend_from_slice(&frame.data);
            parser_dis
                .feed_packet(&packet, frame.timestamp as i64)
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
    parser.feed_packet(&sps_pps_bitstream, 0).unwrap();
    let mut frame = Frame::default();
    let mut packet = Vec::new();
    while let Ok(remaining) = mkv.next_frame(&mut frame) {
        if !remaining {
            break;
        }
        let track = &mkv.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            h264::packet_to_annexb(&mut packet, &frame.data, nal_length_size);
            parser.feed_packet(&packet, frame.timestamp as i64).unwrap();
        }
    }
    parser.flush().unwrap();
}
