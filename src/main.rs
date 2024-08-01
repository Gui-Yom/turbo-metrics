use std::fs::File;
use std::io::BufReader;
use std::thread;
use std::time::Instant;

use matroska_demuxer::{Frame, TrackType};
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use codec_cfg_record::h264;
use cuda_driver::{CuDevice, CuStream};
use nv_video_codec::dec::{CuVideoCtxLock, CuVideoParser};
use nv_video_codec::dec_helper::npp::icc::NV12toRGB;
use nv_video_codec::dec_helper::npp::{get_stream_ctx, Img, C};
use nv_video_codec::dec_helper::DecoderHolder;
use nv_video_codec::sys::cudaVideoCodec;

fn main() {
    let mut matroska = matroska_demuxer::MatroskaFile::open(BufReader::new(
        File::open("data/friends_cut.mkv").unwrap(),
    ))
    .unwrap();
    for (i, track) in matroska.tracks().iter().enumerate() {
        println!(
            "{i} {:?}({}): {}",
            track.track_type(),
            track.codec_id(),
            track.language().unwrap_or("en")
        );
    }

    dbg!(matroska.info().timestamp_scale());

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

    {
        let mut cb = DecoderHolder::new(lock);
        thread::scope(|s| {
            thread::Builder::new()
                .name("demuxer-parser-thread".to_string())
                .spawn_scoped(s, || {
                    // Bind to main thread
                    ctx.set_current().unwrap();
                    let mut parser = CuVideoParser::new(
                        cudaVideoCodec::cudaVideoCodec_H264,
                        &cb,
                        Some(matroska.info().timestamp_scale().get() as _),
                        None,
                    )
                    .unwrap();
                    let (nal_length_size, sps_pps_bitstream) = h264::avcc_extradata_to_annexb(
                        matroska.tracks()[0].codec_private().unwrap(),
                    );
                    dbg!(nal_length_size);
                    parser.feed_packet(&sps_pps_bitstream, 0).unwrap();
                    let mut frame = Frame::default();
                    let mut packet = Vec::new();
                    while let Ok(remaining) = matroska.next_frame(&mut frame) {
                        if !remaining {
                            break;
                        }
                        let track = &matroska.tracks()[frame.track as usize - 1];
                        if track.track_type() == TrackType::Video {
                            h264::packet_to_annexb(&mut packet, &frame.data, nal_length_size);
                            parser.feed_packet(&packet, frame.timestamp as i64).unwrap();
                        }
                    }
                    parser.flush().unwrap();
                })
                .unwrap();

            let stream = CuStream::new().unwrap();
            let mut counter = 0;
            let mut dst = None;
            let mut rx = cb.wait_for_rx();
            let start = Instant::now();
            while let Ok(Some(disp)) = rx.peek() {
                {
                    let src = cb.map_npp_nv12(&disp, &stream).unwrap();
                    let dst = dst.get_or_insert_with(|| src.malloc_same_size().unwrap());
                    src.nv12_to_rgb_bt709_limited(
                        dst,
                        get_stream_ctx().unwrap().with_stream(stream.inner() as _),
                    )
                    .unwrap();
                }
                // dbg!(disp.timestamp);

                rx.recv().unwrap();

                counter += 1;
                if counter < 420 || counter > 425 {
                    continue;
                }
                println!("saving");

                save_img(
                    dst.as_ref().unwrap(),
                    &format!("decoded_{}", counter),
                    &stream,
                );
            }
            let total = start.elapsed().as_millis();
            println!(
                "Decoded {counter} frames in {total} ms, {} fps",
                counter * 1000 / total
            );
        });
    }

    println!("done !");

    dev.release_primary_ctx().unwrap();
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
