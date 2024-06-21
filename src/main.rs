use std::fs::File;
use std::io::{BufReader, Cursor};
use std::iter::repeat;
use std::time::{Duration, Instant};
use std::{mem, thread};

use bitstream_io::{BigEndian, BitRead, BitReader};
use matroska_demuxer::{Frame, TrackType};
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use cuda_driver::{CuDevice, CuStream};
use cuda_npp::get_stream_ctx;
use cuda_npp::image::icc::NV12toRGB;
use cuda_npp::image::{Image, Img, C, P};
use nv_video_codec::dec::CuVideoParser;
use nv_video_codec::dec_helper::DecoderState;
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

    let extradata = read_avc_config_record(matroska.tracks()[0].codec_private().unwrap());

    cuda_driver::init_cuda().expect("Could not initialize the CUDA API");
    let dev = CuDevice::get(0).unwrap();
    println!(
        "Using device {} with CUDA version {}",
        dev.name().unwrap(),
        cuda_driver::cuda_driver_version().unwrap()
    );
    // Bind to main thread
    dev.retain_primary_ctx().unwrap().set_current().unwrap();

    let stream = CuStream::new().unwrap();
    let mut counter = 0;
    let mut cb = DecoderState::new(|decoder, format, disp| {
        counter += 1;
        if counter < 2000 || counter > 2002 {
            return;
        }
        let mapping = decoder.map(disp, &stream).unwrap();

        let src = Image::<u8, P<2>>::from_raw(
            format.display_area.right as u32,
            format.display_area.bottom as u32,
            mapping.pitch as i32,
            [
                mapping.ptr as *mut u8,
                (mapping.ptr + mapping.pitch as u64 * format.coded_height as u64) as *mut u8,
            ],
        );
        let mut dst = src.malloc_same_size().unwrap();
        src.nv12_to_rgb(
            &mut dst,
            get_stream_ctx().unwrap().with_stream(stream.inner() as _),
        )
        .unwrap();
        mem::forget(src);

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

        save_img(&dst, &format!("decoded_{}", counter), &stream);
    });
    let mut parser = CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &mut cb).unwrap();
    parser.feed_packet(&extradata, 0).unwrap();
    let mut frame = Frame::default();
    let mut num_packets = 0;
    let mut packet = vec![0, 0, 0, 1];
    let mut num_frames = 0;
    let mut last_frame = Instant::now();
    while let Ok(end) = matroska.next_frame(&mut frame) {
        let track = &matroska.tracks()[frame.track as usize - 1];
        if track.track_type() == TrackType::Video {
            // Keep only NALU header
            packet.truncate(4);
            // Ignore first 4 bytes (slice len)
            packet.extend_from_slice(&frame.data[4..]);
            parser.feed_packet(&packet, frame.timestamp as i64).unwrap();
            num_packets += 1;
            if num_packets > 2500 {
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

    thread::sleep(Duration::from_secs_f32(5.0));
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
