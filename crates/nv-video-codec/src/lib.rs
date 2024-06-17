use core::ffi::{c_short, c_uint, c_ulong, c_void};
use std::marker::PhantomData;
use std::ptr::{null_mut, NonNull};

use cuda_driver::sys::CuResult;
use cuda_driver::CuCtx;
pub use nv_video_codec_sys as sys;
use nv_video_codec_sys::{
    _CUVIDDECODECREATEINFO__bindgen_ty_1, _CUVIDDECODECREATEINFO__bindgen_ty_2, cudaVideoCodec,
    cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat, cuvidCreateDecoder, cuvidCreateVideoParser,
    cuvidCtxLockCreate, cuvidCtxLockDestroy, cuvidDecodePicture, cuvidDestroyDecoder,
    cuvidDestroyVideoParser, cuvidGetDecoderCaps, cuvidParseVideoData, CUvideopacketflags,
    CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDOPERATINGPOINTINFO,
    CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDSEIMESSAGEINFO,
    CUVIDSOURCEDATAPACKET,
};

trait VideoParserCb {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32;
    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32;
    fn display_picture(&mut self, disp: &CUVIDPARSERDISPINFO) -> i32;
    fn get_operating_point(&mut self, point: &CUVIDOPERATINGPOINTINFO) -> i32;
    fn get_sei_msg(&mut self, sei: &CUVIDSEIMESSAGEINFO) -> i32;
}

extern "C" fn sequence_callback<CB: VideoParserCb>(
    user: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> i32 {
    let s = unsafe { &mut *(user.cast::<CB>()) };
    s.sequence_callback(unsafe { &*format })
}

extern "C" fn decode_picture<CB: VideoParserCb>(
    user: *mut c_void,
    pic: *mut CUVIDPICPARAMS,
) -> i32 {
    let s = unsafe { &mut *(user.cast::<CB>()) };
    s.decode_picture(unsafe { &*pic })
}

extern "C" fn display_picture<CB: VideoParserCb>(
    user: *mut c_void,
    disp: *mut CUVIDPARSERDISPINFO,
) -> i32 {
    let s = unsafe { &mut *(user.cast::<CB>()) };
    s.display_picture(unsafe { &*disp })
}

extern "C" fn get_operating_point<CB: VideoParserCb>(
    user: *mut c_void,
    point: *mut CUVIDOPERATINGPOINTINFO,
) -> i32 {
    let s = unsafe { &mut *(user.cast::<CB>()) };
    s.get_operating_point(unsafe { &*point })
}

extern "C" fn get_sei_msg<CB: VideoParserCb>(
    user: *mut c_void,
    sei: *mut CUVIDSEIMESSAGEINFO,
) -> i32 {
    let s = unsafe { &mut *(user.cast::<CB>()) };
    s.get_sei_msg(unsafe { &*sei })
}

pub struct CuVideoParser<'a> {
    pub(crate) inner: sys::CUvideoparser,
    marker: PhantomData<&'a mut dyn VideoParserCb>,
}

impl<'a> CuVideoParser<'a> {
    pub fn new<CB: VideoParserCb>(codec: cudaVideoCodec, cb: &'a mut CB) -> CuResult<Self> {
        let mut ptr = null_mut();
        let mut params = CUVIDPARSERPARAMS {
            CodecType: codec,
            ulMaxNumDecodeSurfaces: 1,
            ulErrorThreshold: 0,
            ulMaxDisplayDelay: 0,
            pUserData: (cb as *mut CB).cast(),
            pfnSequenceCallback: Some(sequence_callback::<CB>),
            pfnDecodePicture: Some(decode_picture::<CB>),
            pfnDisplayPicture: Some(display_picture::<CB>),
            pfnGetOperatingPoint: Some(get_operating_point::<CB>),
            pfnGetSEIMsg: Some(get_sei_msg::<CB>),
            pExtVideoInfo: null_mut(),
            ..Default::default()
        };
        unsafe {
            cuvidCreateVideoParser(&mut ptr, &mut params).result()?;
        }
        Ok(Self {
            inner: ptr,
            marker: PhantomData,
        })
    }

    pub fn feed_packet(&mut self, packet: &[u8], timestamp: i64) -> CuResult<()> {
        let mut flags = CUvideopacketflags::CUVID_PKT_TIMESTAMP;
        if timestamp == 0 {
            flags |= CUvideopacketflags::CUVID_PKT_DISCONTINUITY;
        }
        if packet.len() == 0 {
            flags |= CUvideopacketflags::CUVID_PKT_ENDOFSTREAM;
        }
        // dbg!(&packet[..4]);
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags: flags.0 as c_ulong,
            payload_size: packet.len() as c_ulong,
            payload: packet.as_ptr(),
            timestamp,
        };
        unsafe { cuvidParseVideoData(self.inner, &mut packet).result() }
    }
}

impl<'a> Drop for CuVideoParser<'a> {
    fn drop(&mut self) {
        unsafe { cuvidDestroyVideoParser(self.inner).result().unwrap() }
    }
}

pub struct CuVideoCtxLock(pub(crate) sys::CUvideoctxlock);

impl CuVideoCtxLock {
    pub fn new(ctx: &CuCtx) -> CuResult<Self> {
        let mut lock = null_mut();
        unsafe {
            cuvidCtxLockCreate(&mut lock, ctx.inner()).result()?;
        }
        Ok(Self(lock))
    }
}

impl Drop for CuVideoCtxLock {
    fn drop(&mut self) {
        unsafe { cuvidCtxLockDestroy(self.0).result().unwrap() }
    }
}

pub struct CommonCallbacks {}

impl VideoParserCb for CommonCallbacks {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        let format = &dbg!(*format);

        let mut caps = CUVIDDECODECAPS {
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
        unsafe { dbg!(cuvidGetDecoderCaps(&mut caps)) };
        if dbg!(caps).bIsSupported == 0 {
            println!("Unsupported codec/chroma format");
            return 0;
        }

        let lock = CuVideoCtxLock::new(&CuCtx::get_current().unwrap()).unwrap();

        // let mut create_info = CUVIDDECODECREATEINFO {
        //     ulWidth: format.coded_width,
        //     ulHeight: format.coded_height,
        //     ulNumDecodeSurfaces: format.min_num_decode_surfaces.next_power_of_two() as c_ulong,
        //     CodecType: format.codec,
        //     ChromaFormat: format.chroma_format,
        //     ulCreationFlags: 0,
        //     bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
        //     ulIntraDecodeOnly: 0,
        //     ulMaxWidth: format.coded_width,
        //     ulMaxHeight: format.coded_height,
        //     Reserved1: 0,
        //     display_area: _CUVIDDECODECREATEINFO__bindgen_ty_1 {
        //         left: format.display_area.left as c_short,
        //         top: format.display_area.left as c_short,
        //         right: format.display_area.left as c_short,
        //         bottom: format.display_area.left as c_short,
        //     },
        //     OutputFormat: cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
        //     DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive,
        //     ulTargetWidth: format.coded_width,
        //     ulTargetHeight: format.coded_height,
        //     ulNumOutputSurfaces: 1,
        //     vidLock: lock.inner(),
        //     target_rect: _CUVIDDECODECREATEINFO__bindgen_ty_2 {
        //         left: format.display_area.left as c_short,
        //         top: format.display_area.left as c_short,
        //         right: format.display_area.left as c_short,
        //         bottom: format.display_area.left as c_short,
        //     },
        //     enableHistogram: 0,
        //     Reserved2: [0; 4],
        // };
        // unsafe {
        //     dbg!(cuvidCreateDecoder(
        //         &mut video_decoder,
        //         &mut create_info,
        //     ));
        // }
        //
        let surfaces = (*format).min_num_decode_surfaces;
        surfaces.max(1) as i32
    }

    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32 {
        unsafe {
            cuvidDecodePicture(video_decoder, pic).result().unwrap();
        }
        1
    }

    fn display_picture(&mut self, disp: &CUVIDPARSERDISPINFO) -> i32 {
        // dbg!(disp);
        // let mut inner = &mut *(user_data as *mut NvDecoderData);
        // let disp = &*disp;
        // let mut srcDev = 0;
        // let mut srcPitch = 0;
        // let mut params = bindings::CUVIDPROCPARAMS {
        //     progressive_frame: disp.progressive_frame,
        //     second_field: disp.repeat_first_field + 1,
        //     top_field_first: disp.top_field_first,
        //     unpaired_field: c_int::from(disp.repeat_first_field < 0),
        //     output_stream: inner.stream,
        //     ..Default::default()
        // };
        // bindings::cuvidMapVideoFrame64(
        //     inner.video_decoder,
        //     disp.picture_index,
        //     &mut srcDev,
        //     &mut srcPitch,
        //     &mut params,
        // );
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

    fn get_operating_point(&mut self, point: &CUVIDOPERATINGPOINTINFO) -> i32 {
        // dbg!(point);
        1
    }

    fn get_sei_msg(&mut self, sei: &CUVIDSEIMESSAGEINFO) -> i32 {
        if sei.sei_message_count > 0 {
            dbg!(sei.picIdx);
            dbg!(sei.sei_message_count);
            unsafe {
                dbg!(sei.pSEIMessage.read());
            }
        }
        1
    }
}

#[repr(transparent)]
pub struct CuVideoDecoder(pub(crate) NonNull<c_void>);

impl CuVideoDecoder {
    pub fn new(format: &CUVIDEOFORMAT, lock: &CuVideoCtxLock) -> CuResult<Self> {
        let mut ptr = null_mut();
        let mut create_info = CUVIDDECODECREATEINFO {
            ulWidth: format.coded_width,
            ulHeight: format.coded_height,
            ulNumDecodeSurfaces: format.min_num_decode_surfaces.next_power_of_two() as c_ulong,
            CodecType: format.codec,
            ChromaFormat: format.chroma_format,
            ulCreationFlags: 0,
            bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
            ulIntraDecodeOnly: 0,
            ulMaxWidth: format.coded_width,
            ulMaxHeight: format.coded_height,
            Reserved1: 0,
            display_area: _CUVIDDECODECREATEINFO__bindgen_ty_1 {
                left: format.display_area.left as c_short,
                top: format.display_area.left as c_short,
                right: format.display_area.left as c_short,
                bottom: format.display_area.left as c_short,
            },
            OutputFormat: cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
            DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive,
            ulTargetWidth: format.coded_width,
            ulTargetHeight: format.coded_height,
            ulNumOutputSurfaces: 1,
            vidLock: lock.0,
            target_rect: _CUVIDDECODECREATEINFO__bindgen_ty_2 {
                left: format.display_area.left as c_short,
                top: format.display_area.left as c_short,
                right: format.display_area.left as c_short,
                bottom: format.display_area.left as c_short,
            },
            enableHistogram: 0,
            Reserved2: [0; 4],
        };
        unsafe {
            cuvidCreateDecoder(&mut ptr, &mut create_info).result()?;
        }
        Ok(Self(NonNull::new(ptr).unwrap()))
    }
}

impl Drop for CuVideoDecoder {
    fn drop(&mut self) {
        unsafe {
            cuvidDestroyDecoder(self.0.as_ptr()).result().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use cuda_driver::CuDevice;
    use nv_video_codec_sys::cudaVideoCodec;

    use crate::{CommonCallbacks, CuVideoParser};

    #[test]
    fn test() {
        cuda_driver::init_cuda().expect("Could not initialize the CUDA API");
        let dev = CuDevice::get(0).unwrap();
        // Bind to main thread
        dev.retain_primary_ctx().unwrap().set_current().unwrap();

        let mut cb = CommonCallbacks {};
        let mut parser = CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &mut cb).unwrap();
    }
}
