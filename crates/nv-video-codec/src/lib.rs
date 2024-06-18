use core::ffi::{c_short, c_ulong, c_void};
use std::ffi::c_int;
use std::marker::PhantomData;
use std::mem;
use std::ptr::{null_mut, NonNull};

use cuda_driver::sys::CuResult;
use cuda_driver::{CuCtx, CuStream};
pub use nv_video_codec_sys as sys;
use nv_video_codec_sys::{
    _CUVIDDECODECREATEINFO__bindgen_ty_1, _CUVIDDECODECREATEINFO__bindgen_ty_2,
    cudaVideoChromaFormat, cudaVideoCodec, cudaVideoDeinterlaceMode, cudaVideoSurfaceFormat,
    cuvidCreateDecoder, cuvidCreateVideoParser, cuvidCtxLock, cuvidCtxLockCreate,
    cuvidCtxLockDestroy, cuvidCtxUnlock, cuvidDecodePicture, cuvidDecodeStatus,
    cuvidDestroyDecoder, cuvidDestroyVideoParser, cuvidGetDecodeStatus, cuvidGetDecoderCaps,
    cuvidMapVideoFrame64, cuvidParseVideoData, cuvidUnmapVideoFrame64, CUvideopacketflags,
    CUVIDDECODECAPS, CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDOPERATINGPOINTINFO,
    CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDSEIMESSAGEINFO,
    CUVIDSOURCEDATAPACKET,
};

pub trait VideoParserCb {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32;
    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32;
    fn display_picture(&mut self, disp: &CUVIDPARSERDISPINFO) -> i32;
    fn get_operating_point(&mut self, point: &CUVIDOPERATINGPOINTINFO) -> i32 {
        1
    }
    fn get_sei_msg(&mut self, sei: &CUVIDSEIMESSAGEINFO) -> i32 {
        1
    }
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
    pub(crate) inner: NonNull<c_void>,
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
            inner: NonNull::new(ptr).unwrap(),
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
        unsafe { cuvidParseVideoData(self.inner.as_ptr(), &mut packet).result() }
    }
}

impl<'a> Drop for CuVideoParser<'a> {
    fn drop(&mut self) {
        unsafe {
            cuvidDestroyVideoParser(self.inner.as_ptr())
                .result()
                .unwrap()
        }
    }
}

pub struct CuVideoCtxLock(pub(crate) NonNull<sys::_CUcontextlock_st>);

impl CuVideoCtxLock {
    pub fn new(ctx: &CuCtx) -> CuResult<Self> {
        let mut lock = null_mut();
        unsafe {
            cuvidCtxLockCreate(&mut lock, ctx.inner()).result()?;
        }
        Ok(Self(NonNull::new(lock).unwrap()))
    }

    pub fn lock(&self) -> CuResult<()> {
        unsafe { cuvidCtxLock(self.0.as_ptr(), 0).result() }
    }

    pub fn unlock(&self) -> CuResult<()> {
        unsafe { cuvidCtxUnlock(self.0.as_ptr(), 0).result() }
    }
}

impl Drop for CuVideoCtxLock {
    fn drop(&mut self) {
        unsafe { cuvidCtxLockDestroy(self.0.as_ptr()).result().unwrap() }
    }
}

pub fn query_caps(
    codec: cudaVideoCodec,
    chroma_format: cudaVideoChromaFormat,
    bit_depth: u32,
) -> CuResult<CUVIDDECODECAPS> {
    let mut caps = CUVIDDECODECAPS {
        eCodecType: codec,
        eChromaFormat: chroma_format,
        nBitDepthMinus8: bit_depth - 8,
        ..Default::default()
    };
    unsafe {
        cuvidGetDecoderCaps(&mut caps).result()?;
    }
    Ok(caps)
}

#[derive(Debug)]
#[repr(transparent)]
pub struct CuVideoDecoder(pub(crate) NonNull<c_void>);

impl CuVideoDecoder {
    pub fn new(
        format: &CUVIDEOFORMAT,
        surface_format: cudaVideoSurfaceFormat,
        lock: &CuVideoCtxLock,
    ) -> CuResult<Self> {
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
            OutputFormat: surface_format,
            DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Adaptive,
            ulTargetWidth: format.coded_width,
            ulTargetHeight: format.coded_height,
            ulNumOutputSurfaces: 1,
            vidLock: lock.0.as_ptr(),
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

    pub fn decode(&self, params: &CUVIDPICPARAMS) -> CuResult<()> {
        unsafe {
            cuvidDecodePicture(self.0.as_ptr(), params as *const CUVIDPICPARAMS as *mut _).result()
        }
    }

    pub fn status(&self) -> CuResult<cuvidDecodeStatus> {
        let mut status = Default::default();
        unsafe {
            cuvidGetDecodeStatus(self.0.as_ptr(), 0, &mut status).result()?;
        }
        Ok(status.decodeStatus)
    }

    pub fn map<'a>(
        &'a self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<FrameMapping<'a>> {
        let mut ptr = 0;
        let mut pitch = 0;
        let mut proc = CUVIDPROCPARAMS {
            progressive_frame: info.progressive_frame,
            second_field: info.repeat_first_field + 1,
            top_field_first: info.top_field_first,
            unpaired_field: c_int::from(info.repeat_first_field < 0),
            output_stream: stream.inner() as _,
            ..Default::default()
        };
        unsafe {
            cuvidMapVideoFrame64(
                self.0.as_ptr(),
                info.picture_index,
                &mut ptr,
                &mut pitch,
                &mut proc,
            )
            .result()?;
        }
        Ok(FrameMapping {
            ptr,
            pitch,
            decoder: self,
        })
    }

    pub fn unmap(&self, mapping: FrameMapping) -> CuResult<()> {
        self.unmap_inner(mapping.ptr)?;
        mem::forget(mapping);
        Ok(())
    }

    pub(crate) fn unmap_inner(&self, frame: u64) -> CuResult<()> {
        unsafe { cuvidUnmapVideoFrame64(self.0.as_ptr(), frame).result() }
    }
}

impl Drop for CuVideoDecoder {
    fn drop(&mut self) {
        unsafe {
            cuvidDestroyDecoder(self.0.as_ptr()).result().unwrap();
        }
    }
}

#[derive(Debug)]
pub struct FrameMapping<'a> {
    ptr: u64,
    pitch: u32,
    decoder: &'a CuVideoDecoder,
}

impl<'a> Drop for FrameMapping<'a> {
    fn drop(&mut self) {
        unsafe { self.decoder.unmap_inner(self.ptr).unwrap() }
    }
}
