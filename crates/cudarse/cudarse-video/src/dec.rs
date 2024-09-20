use std::ffi::{c_int, c_short, c_ulong, c_void};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{null, null_mut, NonNull};

use crate::sys::{
    _CUVIDDECODECREATEINFO__bindgen_ty_1, _CUVIDDECODECREATEINFO__bindgen_ty_2,
    cudaVideoChromaFormat, cudaVideoCodec, cudaVideoCreateFlags, cudaVideoDeinterlaceMode,
    cudaVideoSurfaceFormat, cuvidCreateDecoder, cuvidCreateVideoParser, cuvidCtxLock,
    cuvidCtxLockCreate, cuvidCtxLockDestroy, cuvidCtxUnlock, cuvidDecodePicture, cuvidDecodeStatus,
    cuvidDestroyDecoder, cuvidDestroyVideoParser, cuvidGetDecodeStatus, cuvidGetDecoderCaps,
    cuvidMapVideoFrame64, cuvidParseVideoData, cuvidUnmapVideoFrame64,
    CUVIDEOFORMATEX__bindgen_ty_1, CUvideopacketflags, _CUcontextlock_st, CUVIDDECODECAPS,
    CUVIDDECODECREATEINFO, CUVIDEOFORMAT, CUVIDEOFORMATEX, CUVIDOPERATINGPOINTINFO,
    CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS, CUVIDPROCPARAMS, CUVIDSEIMESSAGEINFO,
    CUVIDSOURCEDATAPACKET,
};
use cudarse_driver::sys::CuResult;
use cudarse_driver::{CuCtx, CuStream};

pub trait CuvidParserCallbacks {
    /// Called when a new sequence is being parsed (parameters change)
    fn sequence_callback(&self, format: &CUVIDEOFORMAT) -> Result<u32, ()>;
    /// Called when a picture has been parsed and can be decoded (decode order).
    fn decode_picture(&self, pic: &CUVIDPICPARAMS) -> Result<(), ()>;
    /// Called when a picture can be mapped (display order).
    fn display_picture(&self, disp: Option<&CUVIDPARSERDISPINFO>) -> Result<(), ()>;
    fn get_operating_point(&self, _point: &CUVIDOPERATINGPOINTINFO) -> i32 {
        1
    }
    fn get_sei_msg(&self, _sei: &CUVIDSEIMESSAGEINFO) -> i32 {
        1
    }
}

extern "C" fn sequence_callback<CB: CuvidParserCallbacks>(
    user: *mut c_void,
    format: *mut CUVIDEOFORMAT,
) -> i32 {
    let s = unsafe { &*(user.cast::<CB>()) };
    if let Ok(new_size) = s.sequence_callback(unsafe { &*format }) {
        new_size as i32
    } else {
        0
    }
}

extern "C" fn decode_picture<CB: CuvidParserCallbacks>(
    user: *mut c_void,
    pic: *mut CUVIDPICPARAMS,
) -> i32 {
    let s = unsafe { &*(user.cast::<CB>()) };
    if s.decode_picture(unsafe { &*pic }).is_ok() {
        1
    } else {
        0
    }
}

extern "C" fn display_picture<CB: CuvidParserCallbacks>(
    user: *mut c_void,
    disp: *mut CUVIDPARSERDISPINFO,
) -> i32 {
    let s = unsafe { &*(user.cast::<CB>()) };
    let disp = if disp.is_null() {
        None
    } else {
        Some(unsafe { &*disp })
    };
    if s.display_picture(disp).is_ok() {
        1
    } else {
        0
    }
}

extern "C" fn get_operating_point<CB: CuvidParserCallbacks>(
    user: *mut c_void,
    point: *mut CUVIDOPERATINGPOINTINFO,
) -> i32 {
    let s = unsafe { &*(user.cast::<CB>()) };
    s.get_operating_point(unsafe { &*point })
}

extern "C" fn get_sei_msg<CB: CuvidParserCallbacks>(
    user: *mut c_void,
    sei: *mut CUVIDSEIMESSAGEINFO,
) -> i32 {
    let s = unsafe { &*(user.cast::<CB>()) };
    s.get_sei_msg(unsafe { &*sei })
}

pub struct CuVideoParser<'a> {
    pub(crate) inner: NonNull<c_void>,
    marker: PhantomData<&'a dyn CuvidParserCallbacks>,
}

impl<'a> CuVideoParser<'a> {
    pub fn new<CB: CuvidParserCallbacks>(
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
            pUserData: cb as *const CB as *mut c_void,
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

    /// The parser expects annexb format for H264 (with 0001 nalu delimiters).
    pub fn parse_data(&mut self, packet: &[u8], timestamp: i64) -> CuResult<()> {
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

/// The video context lock is required when decoding with CUDA instead of the video engines.
pub struct CuVideoCtxLock(pub(crate) NonNull<_CUcontextlock_st>);

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

/// Choose a supported surface format based on bit depth and chroma format.
pub fn select_output_format(
    format: &CUVIDEOFORMAT,
    caps: &CUVIDDECODECAPS,
) -> cudaVideoSurfaceFormat {
    let high_bpp = format.bit_depth_luma_minus8 > 0;
    let mut surface_format = match format.chroma_format {
        cudaVideoChromaFormat::cudaVideoChromaFormat_420
        | cudaVideoChromaFormat::cudaVideoChromaFormat_Monochrome => {
            if high_bpp {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
            } else {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
            }
        }
        cudaVideoChromaFormat::cudaVideoChromaFormat_422 => {
            if high_bpp {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444_16Bit
            } else {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444
            }
        }
        cudaVideoChromaFormat::cudaVideoChromaFormat_444 => {
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
        }
    };

    if !caps.is_output_format_supported(surface_format) {
        for format in [
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016,
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444,
            cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444_16Bit,
        ] {
            // There should be at least one that works
            if caps.is_output_format_supported(format) {
                surface_format = format;
            }
        }
    }
    surface_format
}

#[derive(Debug)]
#[repr(transparent)]
pub struct CuVideoDecoder<'a> {
    pub(crate) inner: NonNull<c_void>,
    marker: PhantomData<&'a CuVideoCtxLock>,
}

unsafe impl Sync for CuVideoDecoder<'_> {}
unsafe impl Send for CuVideoDecoder<'_> {}

impl<'a> CuVideoDecoder<'a> {
    pub fn new(
        format: &CUVIDEOFORMAT,
        surface_format: cudaVideoSurfaceFormat,
        decode_surfaces: u32,
        lock_for_cuda: Option<&'a CuVideoCtxLock>,
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

            ulNumDecodeSurfaces: decode_surfaces,
            ulNumOutputSurfaces: 1,

            ulCreationFlags: if lock_for_cuda.is_some() {
                cudaVideoCreateFlags::cudaVideoCreate_PreferCUDA as _
            } else {
                cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as _
            },
            ulIntraDecodeOnly: 0,
            enableHistogram: 0,
            vidLock: lock_for_cuda.map(|l| l.0.as_ptr()).unwrap_or(null_mut()),

            Reserved1: 0,
            Reserved2: [0; 4],
        };
        unsafe {
            cuvidCreateDecoder(&mut ptr, &mut create_info).result()?;
        }
        Ok(Self {
            inner: NonNull::new(ptr).unwrap(),
            marker: PhantomData,
        })
    }

    pub fn decode(&self, params: &CUVIDPICPARAMS) -> CuResult<()> {
        unsafe {
            cuvidDecodePicture(
                self.inner.as_ptr(),
                params as *const CUVIDPICPARAMS as *mut _,
            )
            .result()
        }
    }

    pub fn status(&self) -> CuResult<cuvidDecodeStatus> {
        let mut status = Default::default();
        unsafe {
            cuvidGetDecodeStatus(self.inner.as_ptr(), 0, &mut status).result()?;
        }
        Ok(status.decodeStatus)
    }

    pub fn map<'b>(
        &'b self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<FrameMapping<'b>> {
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
                self.inner.as_ptr(),
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
        unsafe {
            self.unmap_inner(mapping.ptr)?;
        }
        mem::forget(mapping);
        Ok(())
    }

    /// SAFETY: Ensure the frame mapping is not used afterward.
    pub(crate) unsafe fn unmap_inner(&self, frame: u64) -> CuResult<()> {
        cuvidUnmapVideoFrame64(self.inner.as_ptr(), frame).result()
    }
}

impl Drop for CuVideoDecoder<'_> {
    fn drop(&mut self) {
        unsafe {
            cuvidDestroyDecoder(self.inner.as_ptr()).result().unwrap();
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
    decoder: &'a CuVideoDecoder<'a>,
}

impl Drop for FrameMapping<'_> {
    fn drop(&mut self) {
        unsafe { self.decoder.unmap_inner(self.ptr).unwrap() }
    }
}

#[cfg(feature = "npp")]
pub mod npp {
    use crate::dec::FrameMapping;
    pub use cudarse_npp::get_stream;
    pub use cudarse_npp::get_stream_ctx;
    pub use cudarse_npp::image::*;
    pub use cudarse_npp::set_stream;
    use cudarse_video_sys::CUVIDEOFORMAT;

    #[derive(Debug)]
    pub struct NvDecNV12<'dec> {
        pub(crate) frame: FrameMapping<'dec>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) planes: [*mut u8; 2],
    }

    impl<'dec> NvDecNV12<'dec> {
        pub fn from_mapping(mapping: FrameMapping<'dec>, format: &CUVIDEOFORMAT) -> Self {
            Self {
                width: format.display_width(),
                height: format.display_height(),
                planes: [
                    mapping.ptr as *mut u8,
                    (mapping.ptr + mapping.pitch as u64 * format.coded_height as u64) as *mut u8,
                ],
                frame: mapping,
            }
        }
    }

    impl<'dec> Img<u8, P<2>> for NvDecNV12<'dec> {
        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }

        fn pitch(&self) -> i32 {
            self.frame.pitch as i32
        }

        fn storage(&self) -> <P<2> as Channels>::Storage<u8> {
            self.planes
        }

        fn device_ptr(&self) -> <P<2> as Channels>::Ref<u8> {
            P::<2>::make_ref(&self.planes)
        }

        fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const u8> {
            P::<2>::iter_ptrs(&self.planes)
        }
    }

    #[derive(Debug)]
    pub struct NvDecP010<'dec> {
        pub(crate) frame: FrameMapping<'dec>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) planes: [*mut u16; 2],
    }

    impl<'dec> NvDecP010<'dec> {
        pub fn from_mapping(mapping: FrameMapping<'dec>, format: &CUVIDEOFORMAT) -> Self {
            Self {
                width: format.display_width(),
                height: format.display_height(),
                planes: [
                    mapping.ptr as *mut u16,
                    (mapping.ptr + mapping.pitch as u64 * format.coded_height as u64) as *mut u16,
                ],
                frame: mapping,
            }
        }
    }

    impl<'dec> Img<u16, P<2>> for NvDecP010<'dec> {
        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }

        fn pitch(&self) -> i32 {
            self.frame.pitch as i32
        }

        fn storage(&self) -> <P<2> as Channels>::Storage<u16> {
            self.planes
        }

        fn device_ptr(&self) -> <P<2> as Channels>::Ref<u16> {
            P::<2>::make_ref(&self.planes)
        }

        fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const u16> {
            P::<2>::iter_ptrs(&self.planes)
        }
    }

    #[derive(Debug)]
    pub enum NvDecFrame<'dec> {
        /// yuv420 : 8 bits Y plane + 8 bits interleaved UV plane
        NV12(NvDecNV12<'dec>),
        /// yuv420 :  16 bits Y plane + 16 bits interleaved UV plane
        /// https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats#p016-and-p010
        /// Values are on 10 bits
        P010(NvDecP010<'dec>),
        /// yuv420 :  16 bits Y plane + 16 bits interleaved UV plane
        /// https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats#p016-and-p010
        /// Values are on 12 bits
        P012,
    }
}
