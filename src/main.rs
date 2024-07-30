use std::fs::File;
use std::io::{BufReader, Cursor};
use std::iter::repeat;
use std::thread;
use std::time::{Duration, Instant};

use bitstream_io::{BigEndian, BitRead, BitReader};
use matroska_demuxer::{Frame, TrackType};
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use cuda_driver::{CuDevice, CuStream};
use nv_video_codec::dec::CuVideoParser;
use nv_video_codec::dec_helper::npp::icc::NV12toRGB;
use nv_video_codec::dec_helper::npp::{get_stream_ctx, Img, C};
use nv_video_codec::dec_helper::DecoderHolder;
use nv_video_codec::sys::cudaVideoCodec;

fn main() {
    let mut matroska = matroska_demuxer::MatroskaFile::open(BufReader::new(
        File::open("data/chainsaw_man_s01e01_v.mkv").unwrap(),
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

    let extradata = read_avc_config_record(matroska.tracks()[0].codec_private().unwrap());

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

    {
        let mut cb = DecoderHolder::new();
        thread::scope(|s| {
            thread::Builder::new()
                .name("decoder-thread".to_string())
                .spawn_scoped(s, || {
                    // Bind to main thread
                    ctx.set_current().unwrap();
                    let mut parser =
                        CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &cb).unwrap();
                    parser.feed_packet(&extradata, 0).unwrap();
                    thread::sleep(Duration::from_secs(2));
                    let mut frame = Frame::default();
                    let mut num_packets = 0;
                    let mut packet = vec![0, 0, 0, 1];
                    let start = Instant::now();
                    while let Ok(end) = matroska.next_frame(&mut frame) {
                        let track = &matroska.tracks()[frame.track as usize - 1];
                        if track.track_type() == TrackType::Video {
                            // Keep only NALU header
                            packet.truncate(4);
                            // Ignore first 4 bytes (slice len)
                            packet.extend_from_slice(&frame.data[4..]);
                            parser.feed_packet(&packet, frame.timestamp as i64).unwrap();
                            num_packets += 1;
                            if num_packets > 2600 {
                                break;
                            }
                        }
                        // if let Ok(frame) = frames.try_recv() {
                        //     // let fps = 1000000 / last_frame.elapsed().as_micros();
                        //     // println!("{} fps", fps);
                        //     last_frame = Instant::now();
                        //     num_frames += 1;
                        // }
                    }
                    parser.flush().unwrap();
                    let total = start.elapsed().as_millis();
                    println!("Done demuxing {num_packets} packets in {total} ms");
                })
                .unwrap();

            let stream = CuStream::new().unwrap();
            let mut counter = 0;
            let mut dst = None;
            let mut rx = cb.wait_for_rx();
            let start = Instant::now();
            while let Ok(Some(disp)) = rx.peek() {
                {
                    let src = cb.map_npp(&disp, &stream).unwrap();
                    let dst = dst.get_or_insert_with(|| src.malloc_same_size().unwrap());
                    src.nv12_to_rgb(
                        dst,
                        get_stream_ctx().unwrap().with_stream(stream.inner() as _),
                    )
                    .unwrap();
                }
                dbg!(disp.timestamp);

                rx.recv().unwrap();

                counter += 1;
                if counter < 2100 || counter > 2102 {
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

fn read_avc_config_record(codec_private: &[u8]) -> Vec<u8> {
    let mut reader = BitReader::endian(Cursor::new(codec_private), BigEndian);
    let version: u8 = reader.read(8).unwrap();
    let profile: u8 = reader.read(8).unwrap();
    let profile_compat: u8 = reader.read(8).unwrap();
    let level: u8 = reader.read(8).unwrap();
    reader.read::<u8>(6).unwrap(); // Reserved
    let nal_size: u8 = reader.read::<u8>(2).unwrap() + 1;
    reader.read::<u8>(3).unwrap(); // Reserved
    let num_sps: u8 = reader.read(5).unwrap();

    let mut nalus = Vec::new();
    for _ in 0..num_sps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }
    let num_pps: u8 = reader.read(8).unwrap();
    for _ in 0..num_pps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }

    nalus
}
