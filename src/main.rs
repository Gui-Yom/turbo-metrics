use std::fs::File;
use std::io::{BufReader, Cursor};
use std::iter::repeat;
use std::thread;
use std::time::{Duration, Instant};

use bitstream_io::{BigEndian, BitRead, BitReader};
use matroska_demuxer::{Frame, TrackType};

use cuda_driver::{CuCtx, CuDevice, CuStream};
use nv_video_codec::sys::{
    cudaVideoCodec, cudaVideoSurfaceFormat, CUVIDEOFORMAT, CUVIDOPERATINGPOINTINFO,
    CUVIDPARSERDISPINFO, CUVIDPICPARAMS, CUVIDSEIMESSAGEINFO,
};
use nv_video_codec::{query_caps, CuVideoCtxLock, CuVideoDecoder, CuVideoParser, VideoParserCb};

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

    let mut cb = DecoderState {
        decoder: None,
        stream: CuStream::new().unwrap(),
    };
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
            if num_packets > 1000 {
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

    thread::sleep(Duration::from_secs_f32(2.0));
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

pub struct DecoderState {
    decoder: Option<CuVideoDecoder>,
    stream: CuStream,
}

impl VideoParserCb for DecoderState {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        let format = &dbg!(*format);

        let caps = query_caps(
            format.codec,
            format.chroma_format,
            format.bit_depth_luma_minus8 as u32 + 8,
        )
        .unwrap();
        if dbg!(caps).bIsSupported == 0 {
            println!("Unsupported codec/chroma/bitdepth");
            return 0;
        }

        assert!(
            caps.is_output_format_supported(cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12)
        );

        let lock = CuVideoCtxLock::new(&CuCtx::get_current().unwrap()).unwrap();

        self.decoder = Some(
            CuVideoDecoder::new(
                format,
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
                &lock,
            )
            .unwrap(),
        );

        let surfaces = (*format).min_num_decode_surfaces;
        dbg!(surfaces.max(1) as i32)
    }

    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32 {
        if let Some(decoder) = &self.decoder {
            decoder.decode(pic).unwrap();
            // dbg!(pic.CurrPicIdx);
        }
        1
    }

    fn display_picture(&mut self, disp: &CUVIDPARSERDISPINFO) -> i32 {
        if let Some(decoder) = &self.decoder {
            let mapping = decoder.map(disp, &self.stream).unwrap();
            dbg!(mapping);
        }
        // let mut buf = vec![0u8; 1920 * 1080 + 1920 * 540];
        // let mut copy = bindings::CUDA_MEMCPY2D {
        //     srcMemoryType: bindings::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        //     srcDevice: srcDev,
        //     srcPitch: srcPitch as usize,
        //     dstMemoryType: bindings::CUmemorytype::CU_MEMORYTYPE_HOST,
        //     dstHost: buf.as_mut_ptr() as *mut c_void,
        //     dstPitch: 1920,
        //     WidthInBytes: 1920,
        //     Height: 1080,
        //     ..Default::default()
        // };
        // bindings::cuMemcpy2DAsync_v2(&copy, inner.stream);
        // copy.srcDevice = srcDev + (srcPitch * 1080) as u64;
        // copy.dstHost = buf[copy.dstPitch * 1080..].as_mut_ptr() as *mut c_void;
        // copy.Height = 540;
        // bindings::cuStreamSynchronize(inner.stream);
        // bindings::cuvidUnmapVideoFrame64(inner.video_decoder, srcDev);
        //
        // inner
        //     .frames
        //     .send(FrameNV12 {
        //         width: 1920,
        //         height: 1080,
        //         buf,
        //     })
        //     .unwrap();
        1
    }
}
