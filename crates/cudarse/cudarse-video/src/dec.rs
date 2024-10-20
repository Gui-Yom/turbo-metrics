use crate::sys;
use cudarse_driver::sys::CuResult;
use cudarse_driver::{CuCtx, CuStream};
use sys::cudaVideoChromaFormat::*;
use sys::cudaVideoSurfaceFormat::*;

use std::ffi::{c_int, c_short, c_ulong, c_void};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{null_mut, NonNull};
use tracing::debug;

/// The video context lock is required when decoding with CUDA instead of the video engines.
pub struct CuVideoCtxLock(pub(crate) NonNull<sys::_CUcontextlock_st>);

unsafe impl Send for CuVideoCtxLock {}
unsafe impl Sync for CuVideoCtxLock {}

impl CuVideoCtxLock {
    pub fn new(ctx: &CuCtx) -> CuResult<Self> {
        let mut lock = null_mut();
        unsafe {
            sys::cuvidCtxLockCreate(&mut lock, ctx.inner()).result()?;
        }
        Ok(Self(NonNull::new(lock).unwrap()))
    }

    pub fn lock(&self) -> CuResult<()> {
        unsafe { sys::cuvidCtxLock(self.0.as_ptr(), 0).result() }
    }

    pub fn unlock(&self) -> CuResult<()> {
        unsafe { sys::cuvidCtxUnlock(self.0.as_ptr(), 0).result() }
    }
}

impl Drop for CuVideoCtxLock {
    fn drop(&mut self) {
        unsafe { sys::cuvidCtxLockDestroy(self.0.as_ptr()).result().unwrap() }
    }
}

pub fn query_caps(
    codec: sys::cudaVideoCodec,
    chroma_format: sys::cudaVideoChromaFormat,
    bit_depth: u32,
) -> CuResult<sys::CUVIDDECODECAPS> {
    let mut caps = sys::CUVIDDECODECAPS {
        eCodecType: codec,
        eChromaFormat: chroma_format,
        nBitDepthMinus8: bit_depth - 8,
        ..Default::default()
    };
    unsafe {
        sys::cuvidGetDecoderCaps(&mut caps).result()?;
    }
    Ok(caps)
}

/// Choose a supported surface format based on bit depth and chroma format.
pub fn select_output_format(
    format: &sys::CUVIDEOFORMAT,
    caps: &sys::CUVIDDECODECAPS,
) -> sys::cudaVideoSurfaceFormat {
    let high_bpp = format.bit_depth_luma_minus8 > 0;
    let mut surface_format = match format.chroma_format {
        cudaVideoChromaFormat_420 | cudaVideoChromaFormat_Monochrome => {
            if high_bpp {
                cudaVideoSurfaceFormat_P016
            } else {
                cudaVideoSurfaceFormat_NV12
            }
        }
        cudaVideoChromaFormat_422 => {
            if high_bpp {
                cudaVideoSurfaceFormat_YUV444_16Bit
            } else {
                cudaVideoSurfaceFormat_YUV444
            }
        }
        cudaVideoChromaFormat_444 => cudaVideoSurfaceFormat_NV12,
    };

    if !caps.is_output_format_supported(surface_format) {
        for format in [
            cudaVideoSurfaceFormat_NV12,
            cudaVideoSurfaceFormat_P016,
            cudaVideoSurfaceFormat_YUV444,
            cudaVideoSurfaceFormat_YUV444_16Bit,
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
pub struct CuVideoDecoder<'lock> {
    pub(crate) inner: NonNull<c_void>,
    marker: PhantomData<&'lock CuVideoCtxLock>,
}

unsafe impl Sync for CuVideoDecoder<'_> {}
unsafe impl Send for CuVideoDecoder<'_> {}

impl<'a> CuVideoDecoder<'a> {
    pub fn new(
        format: &sys::CUVIDEOFORMAT,
        surface_format: sys::cudaVideoSurfaceFormat,
        decode_surfaces: u32,
        lock_for_cuda: Option<&'a CuVideoCtxLock>,
    ) -> CuResult<Self> {
        let mut ptr = null_mut();
        let mut create_info = sys::CUVIDDECODECREATEINFO {
            ulWidth: format.coded_width as _,
            ulHeight: format.coded_height as _,
            // 0 means same as ulWidth
            ulMaxWidth: 0,
            ulMaxHeight: 0,
            // Same as ulWidth means no scaling
            ulTargetWidth: format.coded_width as _,
            ulTargetHeight: format.coded_height as _,

            display_area: sys::_CUVIDDECODECREATEINFO__bindgen_ty_1 {
                left: format.display_area.left as c_short,
                top: format.display_area.top as c_short,
                right: format.display_area.right as c_short,
                bottom: format.display_area.bottom as c_short,
            },
            target_rect: sys::_CUVIDDECODECREATEINFO__bindgen_ty_2 {
                left: format.display_area.left as c_short,
                top: format.display_area.top as c_short,
                right: format.display_area.right as c_short,
                bottom: format.display_area.bottom as c_short,
            },

            CodecType: format.codec,
            ChromaFormat: format.chroma_format,
            bitDepthMinus8: format.bit_depth_luma_minus8 as c_ulong,
            OutputFormat: surface_format,
            DeinterlaceMode: sys::cudaVideoDeinterlaceMode::cudaVideoDeinterlaceMode_Weave,

            ulNumDecodeSurfaces: decode_surfaces as _,
            ulNumOutputSurfaces: 1,

            ulCreationFlags: if lock_for_cuda.is_some() {
                sys::cudaVideoCreateFlags::cudaVideoCreate_PreferCUDA as _
            } else {
                sys::cudaVideoCreateFlags::cudaVideoCreate_PreferCUVID as _
            },
            ulIntraDecodeOnly: 0,
            enableHistogram: 0,
            vidLock: lock_for_cuda.map(|l| l.0.as_ptr()).unwrap_or(null_mut()),

            Reserved1: 0,
            Reserved2: [0; 4],
        };
        unsafe {
            sys::cuvidCreateDecoder(&mut ptr, &mut create_info).result()?;
        }
        Ok(Self {
            inner: NonNull::new(ptr).unwrap(),
            marker: PhantomData,
        })
    }

    /// Reconfigure the decoder to accommodate a smaller frame size or to resize the number of decode surfaces.
    pub fn reconfigure(&self, format: &sys::CUVIDEOFORMAT) -> CuResult<()> {
        let mut info = sys::CUVIDRECONFIGUREDECODERINFO {
            ulWidth: format.coded_width,
            ulHeight: format.coded_height,
            ulTargetWidth: format.display_width(),
            ulTargetHeight: format.display_height(),
            ulNumDecodeSurfaces: 0,
            display_area: sys::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_1 {
                left: format.display_area.left as _,
                top: format.display_area.top as _,
                right: format.display_area.right as _,
                bottom: format.display_area.bottom as _,
            },
            target_rect: sys::_CUVIDRECONFIGUREDECODERINFO__bindgen_ty_2 {
                left: format.display_area.left as _,
                top: format.display_area.top as _,
                right: format.display_area.right as _,
                bottom: format.display_area.bottom as _,
            },
            ..Default::default()
        };
        debug!("Reconfiguring decoder : {info:#?}");
        unsafe { sys::cuvidReconfigureDecoder(self.inner.as_ptr(), &mut info).result() }
    }

    pub fn decode(&self, params: &sys::CUVIDPICPARAMS) -> CuResult<()> {
        unsafe {
            sys::cuvidDecodePicture(
                self.inner.as_ptr(),
                params as *const sys::CUVIDPICPARAMS as *mut _,
            )
            .result()
        }
    }

    pub fn status(&self) -> CuResult<sys::cuvidDecodeStatus> {
        let mut status = Default::default();
        unsafe {
            sys::cuvidGetDecodeStatus(self.inner.as_ptr(), 0, &mut status).result()?;
        }
        Ok(status.decodeStatus)
    }

    pub fn map<'b>(
        &'b self,
        info: &sys::CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<FrameMapping<'b>> {
        let mut ptr = 0;
        let mut pitch = 0;
        let mut proc = sys::CUVIDPROCPARAMS {
            progressive_frame: 1,
            second_field: info.repeat_first_field + 1,
            top_field_first: info.top_field_first,
            unpaired_field: c_int::from(info.repeat_first_field < 0),
            output_stream: stream.inner() as _,
            ..Default::default()
        };
        unsafe {
            sys::cuvidMapVideoFrame64(
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
        sys::cuvidUnmapVideoFrame64(self.inner.as_ptr(), frame).result()
    }
}

impl Drop for CuVideoDecoder<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::cuvidDestroyDecoder(self.inner.as_ptr())
                .result()
                .unwrap();
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
    pub struct NvDecP016<'dec> {
        pub(crate) frame: FrameMapping<'dec>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) planes: [*mut u16; 2],
    }

    impl<'dec> NvDecP016<'dec> {
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

    impl<'dec> Img<u16, P<2>> for NvDecP016<'dec> {
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
        /// 8 bits Y plane + 8 bits interleaved UV plane
        NV12(NvDecNV12<'dec>),
        /// 16 bits Y plane + 16 bits interleaved UV plane
        /// https://learn.microsoft.com/en-us/windows/win32/medfound/10-bit-and-16-bit-yuv-video-formats#p016-and-p010
        /// Even if it's supposed to be yuv420p10, the values are using the full 16 bit range.
        P016(NvDecP016<'dec>),
    }
}
