extern crate core;

use std::ffi::{c_int, c_short, c_uint, c_ulong, c_void};
use std::ptr::null_mut;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

use video_common::{FrameNV12, VideoCodec};

pub mod bindings;

impl From<VideoCodec> for bindings::cudaVideoCodec {
    fn from(value: VideoCodec) -> Self {
        match value {
            VideoCodec::H264 => bindings::cudaVideoCodec::cudaVideoCodec_H264,
        }
    }
}

pub fn init() {
    unsafe {
        dbg!(bindings::cuInit(0));

        let mut device = 0;
        dbg!(bindings::cuDeviceGet(&mut device, 0));

        let mut ctx = null_mut();
        dbg!(bindings::cuCtxCreate_v2(&mut ctx, 0, device));

        dbg!(bindings::cuCtxSetCurrent(ctx));
    }
}

pub fn sync() {
    unsafe {
        dbg!(bindings::cuCtxSynchronize());
    }
}

struct NvDecoderData {
    video_decoder: bindings::CUvideodecoder,
    video_parser: bindings::CUvideoparser,
    vid_lock: bindings::CUvideoctxlock,
    stream: bindings::CUstream,
    frames: Sender<FrameNV12>,
}

pub struct NvDecoder {
    inner: Box<NvDecoderData>,
}

impl NvDecoder {
    pub fn new(expected_codec: VideoCodec) -> (Self, Receiver<FrameNV12>) {
        let (tx, rx) = mpsc::channel();

        let mut inner = Box::new(NvDecoderData {
            video_decoder: null_mut(),
            video_parser: null_mut(),
            vid_lock: null_mut(),
            stream: null_mut(),
            frames: tx,
        });

        unsafe {
            dbg!(bindings::cuStreamCreate(&mut inner.stream, 0));
        }

        let mut parser_params = bindings::CUVIDPARSERPARAMS {
            CodecType: expected_codec.into(),
            ulMaxNumDecodeSurfaces: 1,
            ulMaxDisplayDelay: 4,
            pUserData: inner.as_mut() as *mut NvDecoderData as *mut c_void,
            pfnSequenceCallback: Some(cb_video_seq),
            pfnDecodePicture: Some(cb_decode_pic),
            pfnDisplayPicture: Some(cb_display),
            pfnGetOperatingPoint: None,
            pfnGetSEIMsg: Some(cb_sei),
            ..Default::default()
        };
        unsafe {
            dbg!(bindings::cuvidCreateVideoParser(
                &mut inner.video_parser,
                &mut parser_params,
            ));
        }

        (Self { inner }, rx)
    }

    pub fn feed_packet(&mut self, packet: &[u8], timestamp: i64) {
        let mut flags = bindings::CUvideopacketflags::CUVID_PKT_TIMESTAMP;
        if timestamp == 0 {
            flags |= bindings::CUvideopacketflags::CUVID_PKT_DISCONTINUITY;
        }
        if packet.len() == 0 {
            flags |= bindings::CUvideopacketflags::CUVID_PKT_ENDOFSTREAM;
        }
        // dbg!(&packet[..4]);
        let mut packet = bindings::CUVIDSOURCEDATAPACKET {
            flags: flags.0 as c_ulong,
            payload_size: packet.len() as c_ulong,
            payload: packet.as_ptr(),
            timestamp,
        };
        unsafe {
            bindings::cuvidParseVideoData(self.inner.video_parser, &mut packet);
        }
    }
}

extern "C" fn cb_video_seq(user_data: *mut c_void, format: *mut bindings::CUVIDEOFORMAT) -> c_int {
    unsafe {
        let mut inner = &mut *(user_data as *mut NvDecoderData);

        let format = &dbg!(*format);

        let mut caps = bindings::CUVIDDECODECAPS {
            eCodecType: format.codec,
            eChromaFormat: format.chroma_format,
            nBitDepthMinus8: format.bit_depth_luma_minus8 as c_uint,
            reserved1: [0; 3],
            bIsSupported: 0,
            nNumNVDECs: 0,
            nOutputFormatMask: 0,
            nMaxWidth: 0,
            nMaxHeight: 0,
            nMaxMBCount: 0,
            nMinWidth: 0,
            nMinHeight: 0,
            bIsHistogramSupported: 0,
            nCounterBitDepth: 0,
            nMaxHistogramBins: 0,
            reserved3: [0; 10],
        };
        unsafe { dbg!(bindings::cuvidGetDecoderCaps(&mut caps)) };
        if dbg!(caps).bIsSupported == 0 {
            println!("Unsupported codec/chroma format");
            return 0;
        }

        unsafe {
            let mut ctx = null_mut();
            dbg!(bindings::cuCtxGetCurrent(&mut ctx));
            dbg!(bindings::cuvidCtxLockCreate(&mut inner.vid_lock, ctx));
        }

        let mut create_info = bindings::CUVIDDECODECREATEINFO {
            ulWidth: format.coded_width,
            ulHeight: format.coded_height,
            ulNumDecodeSurfaces: format.min_num_decode_surfaces.next_power_of_two() as c_ulong,
            CodecType: format.codec,
            ChromaFormat: format.chroma_format,
            ulCreationFlags: bindings::cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as c_ulong,
            bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: format.coded_width,
            ulMaxHeight: format.coded_height,
            Reserved1: 0,
            display_area: bindings::_CUVIDDECODECREATEINFO__bindgen_ty_1 {
                left: format.display_area.left as c_short,
                top: format.display_area.left as c_short,
                right: format.display_area.left as c_short,
                bottom: format.display_area.left as c_short,
            },
            OutputFormat: bindings::cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
            DeinterlaceMode: bindings::cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive,
            ulTargetWidth: format.coded_width,
            ulTargetHeight: format.coded_height,
            ulNumOutputSurfaces: 1,
            vidLock: inner.vid_lock,
            target_rect: bindings::_CUVIDDECODECREATEINFO__bindgen_ty_2 {
                left: format.display_area.left as c_short,
                top: format.display_area.left as c_short,
                right: format.display_area.left as c_short,
                bottom: format.display_area.left as c_short,
            },
            enableHistogram: 0,
            Reserved2: [0; 4],
        };
        unsafe {
            dbg!(bindings::cuvidCreateDecoder(
                &mut inner.video_decoder,
                &mut create_info,
            ));
        }

        let surfaces = (*format).min_num_decode_surfaces as c_int;
        surfaces.max(1)
    }
}

extern "C" fn cb_decode_pic(user_data: *mut c_void, pic: *mut bindings::CUVIDPICPARAMS) -> c_int {
    unsafe {
        bindings::cuvidDecodePicture((*(user_data as *mut NvDecoderData)).video_decoder, pic);
    }
    1
}

extern "C" fn cb_display(
    user_data: *mut c_void,
    disp: *mut bindings::CUVIDPARSERDISPINFO,
) -> c_int {
    unsafe {
        let mut inner = &mut *(user_data as *mut NvDecoderData);
        let disp = &*disp;
        let mut srcDev = 0;
        let mut srcPitch = 0;
        let mut params = bindings::CUVIDPROCPARAMS {
            progressive_frame: disp.progressive_frame,
            second_field: disp.repeat_first_field + 1,
            top_field_first: disp.top_field_first,
            unpaired_field: c_int::from(disp.repeat_first_field < 0),
            output_stream: inner.stream,
            ..Default::default()
        };
        bindings::cuvidMapVideoFrame64(
            inner.video_decoder,
            disp.picture_index,
            &mut srcDev,
            &mut srcPitch,
            &mut params,
        );
        let mut buf = vec![0u8; 1920 * 1080 + 1920 * 540];
        let mut copy = bindings::CUDA_MEMCPY2D {
            srcMemoryType: bindings::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcDevice: srcDev,
            srcPitch: srcPitch as usize,
            dstMemoryType: bindings::CUmemorytype::CU_MEMORYTYPE_HOST,
            dstHost: buf.as_mut_ptr() as *mut c_void,
            dstPitch: 1920,
            WidthInBytes: 1920,
            Height: 1080,
            ..Default::default()
        };
        bindings::cuMemcpy2DAsync_v2(&copy, inner.stream);
        copy.srcDevice = srcDev + (srcPitch * 1080) as u64;
        copy.dstHost = buf[copy.dstPitch * 1080..].as_mut_ptr() as *mut c_void;
        copy.Height = 540;
        bindings::cuStreamSynchronize(inner.stream);
        bindings::cuvidUnmapVideoFrame64(inner.video_decoder, srcDev);

        inner
            .frames
            .send(FrameNV12 {
                width: 1920,
                height: 1080,
                buf,
            })
            .unwrap();

        // image::GrayImage::from_raw(1920, 1080, buf)
        //     .unwrap()
        //     .save(format!("frames/frame{}.bmp", disp.timestamp))
        //     .unwrap();
    }
    1
}

extern "C" fn cb_sei(user_data: *mut c_void, sei: *mut bindings::CUVIDSEIMESSAGEINFO) -> c_int {
    // dbg!(sei);
    1
}

impl Drop for NvDecoderData {
    fn drop(&mut self) {
        unsafe {
            bindings::cuvidDestroyVideoParser(self.video_parser);
            bindings::cuvidDestroyDecoder(self.video_decoder);
            bindings::cuvidCtxLockDestroy(self.vid_lock);
        }
    }
}
