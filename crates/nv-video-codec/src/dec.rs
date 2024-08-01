use std::ffi::{c_int, c_short, c_ulong, c_void};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{null, null_mut, NonNull};

use cuda_driver::sys::CuResult;
use cuda_driver::{CuCtx, CuStream};
use nv_video_codec_sys::{CUVIDEOFORMATEX__bindgen_ty_1, CUVIDEOFORMATEX};
use sys::{
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

use crate::sys;

pub trait VideoParserCb {
    /// Called when a new sequence is being parsed (parameters change)
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32;
    /// Called when a picture has been parsed and can be decoded
    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32;
    /// Called when a picture has been decoded and can be displayed
    fn display_picture(&mut self, disp: Option<&CUVIDPARSERDISPINFO>) -> i32;
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
    let disp = if disp.is_null() {
        None
    } else {
        Some(unsafe { &*disp })
    };
    s.display_picture(disp)
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
    pub fn new<CB: VideoParserCb>(
        codec: cudaVideoCodec,
        cb: &'a CB,
        clock_rate: Option<u32>,
        extra_data: Option<&[u8]>,
    ) -> CuResult<Self> {
        let mut ptr = null_mut();
        let mut ext = if let Some(extra) = extra_data {
            let mut raw = [0; 1024];
            raw[0..extra.len()].copy_from_slice(extra);
            Some(CUVIDEOFORMATEX {
                format: CUVIDEOFORMAT {
                    codec,
                    seqhdr_data_length: extra_data.map(|s| s.len()).unwrap_or(0) as _,
                    ..Default::default()
                },
                __bindgen_anon_1: CUVIDEOFORMATEX__bindgen_ty_1 {
                    raw_seqhdr_data: raw,
                },
            })
        } else {
            None
        };
        let mut params = CUVIDPARSERPARAMS {
            CodecType: codec,
            ulMaxNumDecodeSurfaces: 1,
            ulErrorThreshold: 0,
            ulMaxDisplayDelay: 4,
            ulClockRate: clock_rate.unwrap_or(0),
            pExtVideoInfo: ext.as_mut().map(|p| p as *mut _).unwrap_or(null_mut()),
            pUserData: (cb as *const CB as *mut CB).cast(),
            pfnSequenceCallback: Some(sequence_callback::<CB>),
            pfnDecodePicture: Some(decode_picture::<CB>),
            pfnDisplayPicture: Some(display_picture::<CB>),
            pfnGetOperatingPoint: Some(get_operating_point::<CB>),
            pfnGetSEIMsg: Some(get_sei_msg::<CB>),
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
        let mut flags = CUvideopacketflags(0);
        flags |= CUvideopacketflags::CUVID_PKT_TIMESTAMP;
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

    pub fn flush(&mut self) -> CuResult<()> {
        let mut packet = CUVIDSOURCEDATAPACKET {
            flags: (CUvideopacketflags::CUVID_PKT_ENDOFSTREAM
                | CUvideopacketflags::CUVID_PKT_NOTIFY_EOS)
                .0 as _,
            payload_size: 0,
            payload: null(),
            timestamp: 0,
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

unsafe impl Send for CuVideoCtxLock {}
unsafe impl Sync for CuVideoCtxLock {}

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

unsafe impl Sync for CuVideoDecoder {}
unsafe impl Send for CuVideoDecoder {}

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
            // 0 means same as ulWidth
            ulMaxWidth: 0,
            ulMaxHeight: 0,
            // Same as ulWidth means no scaling
            ulTargetWidth: format.coded_width,
            ulTargetHeight: format.coded_height,

            display_area: _CUVIDDECODECREATEINFO__bindgen_ty_1 {
                left: format.display_area.left as c_short,
                top: format.display_area.top as c_short,
                right: format.display_area.right as c_short,
                bottom: format.display_area.bottom as c_short,
            },
            target_rect: _CUVIDDECODECREATEINFO__bindgen_ty_2 {
                left: format.display_area.left as c_short,
                top: format.display_area.top as c_short,
                right: format.display_area.right as c_short,
                bottom: format.display_area.bottom as c_short,
            },

            CodecType: format.codec,
            ChromaFormat: format.chroma_format,
            bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
            OutputFormat: surface_format,
            DeinterlaceMode: cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,

            ulNumDecodeSurfaces: format.min_num_decode_surfaces.next_power_of_two() as c_ulong,
            ulNumOutputSurfaces: 1,

            ulCreationFlags: 0,
            ulIntraDecodeOnly: 0,
            enableHistogram: 0,
            vidLock: lock.0.as_ptr(),

            Reserved1: 0,
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
            progressive_frame: 1,
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

    /// FrameMapping has a drop impl, but you can manually destroy it to get any possible errors.
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

/// Handle to a frame mapped by nvdec in device memory.
///
/// The frame is unmapped when this handle is dropped.
/// You can also call [CuVideoDecoder::unmap] to unmap it manually (and handle any possible errors).
#[derive(Debug)]
pub struct FrameMapping<'a> {
    pub ptr: u64,
    pub pitch: u32,
    decoder: &'a CuVideoDecoder,
}

impl<'a> Drop for FrameMapping<'a> {
    fn drop(&mut self) {
        unsafe { self.decoder.unmap_inner(self.ptr).unwrap() }
    }
}
