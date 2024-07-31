use std::ops::{Deref, DerefMut};
use std::sync::{Condvar, Mutex};

use spsc::{Receiver, Sender};

use cuda_driver::sys::CuResult;
use cuda_driver::CuStream;
use nv_video_codec_sys::{
    cudaVideoChromaFormat_enum, cudaVideoSurfaceFormat, CUVIDEOFORMAT, CUVIDOPERATINGPOINTINFO,
    CUVIDPARSERDISPINFO, CUVIDPICPARAMS,
};

use crate::dec::{query_caps, CuVideoCtxLock, CuVideoDecoder, VideoParserCb};

pub type Msg = Option<CUVIDPARSERDISPINFO>;

pub struct Trust<T>(T);

unsafe impl<T> Sync for Trust<T> {}
impl<T> Deref for Trust<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Trust<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct DecoderHolder {
    lock: CuVideoCtxLock,
    cvar: Condvar,
    pub decoder: Option<(CuVideoDecoder, CUVIDEOFORMAT)>,
    tx: Trust<Option<Sender<Msg>>>,
    rx: Mutex<Option<Receiver<Msg>>>,
}

impl DecoderHolder {
    pub fn new(lock: CuVideoCtxLock) -> Self {
        Self {
            lock,
            cvar: Condvar::new(),
            decoder: None,
            tx: Trust(None),
            rx: Mutex::new(None),
        }
    }

    /// rx exists only when the new sequence callback has been called
    pub fn wait_for_rx(&self) -> Receiver<Msg> {
        let guard = self.rx.lock().unwrap();
        self.cvar
            .wait_while(guard, |v| v.is_none())
            .unwrap()
            .take()
            .unwrap()
    }

    pub fn format(&self) -> Option<&CUVIDEOFORMAT> {
        if let Some((_, format)) = &self.decoder {
            Some(format)
        } else {
            None
        }
    }

    #[cfg(feature = "cuda-npp")]
    pub fn map_npp_nv12<'a>(
        &'a self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<npp::NvDecNV12> {
        if let Some((decoder, format)) = &self.decoder {
            let mapping = decoder.map(info, stream)?;
            Ok(npp::NvDecNV12 {
                width: format.display_width(),
                height: format.display_height(),
                planes: [
                    mapping.ptr as *mut u8,
                    (mapping.ptr + mapping.pitch as u64 * format.coded_height as u64) as *mut u8,
                ],
                frame: mapping,
            })
        } else {
            panic!("map_npp called when instance was not initialized")
        }
    }

    #[cfg(feature = "cuda-npp")]
    pub fn map_npp_yuv444<'a>(
        &'a self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<npp::NvDecYUV444> {
        if let Some((decoder, format)) = &self.decoder {
            let mapping = decoder.map(info, stream)?;
            Ok(npp::NvDecYUV444 {
                width: format.display_width(),
                height: format.display_height(),
                data: mapping.ptr as *mut u8,
                frame: mapping,
            })
        } else {
            panic!("map_npp called when instance was not initialized")
        }
    }
}

impl VideoParserCb for DecoderHolder {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        println!("sequence_callback");
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

        let high_bpp = format.bit_depth_luma_minus8 > 0;
        let mut surface_format = match format.chroma_format {
            cudaVideoChromaFormat_enum::cudaVideoChromaFormat_420
            | cudaVideoChromaFormat_enum::cudaVideoChromaFormat_Monochrome => {
                if high_bpp {
                    cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016
                } else {
                    cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12
                }
            }
            cudaVideoChromaFormat_enum::cudaVideoChromaFormat_422 => {
                if high_bpp {
                    cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444_16Bit
                } else {
                    cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_YUV444
                }
            }
            cudaVideoChromaFormat_enum::cudaVideoChromaFormat_444 => {
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
                if caps.is_output_format_supported(format) {
                    surface_format = format;
                }
            }
        }

        let surfaces = format.min_num_decode_surfaces.next_power_of_two().max(1);

        let (tx, rx) = spsc::bounded_channel(surfaces as usize);

        self.decoder = Some((
            CuVideoDecoder::new(dbg!(format), dbg!(surface_format), &self.lock).unwrap(),
            format.clone(),
        ));
        self.tx = Trust(Some(tx));
        *self.rx.lock().unwrap() = Some(rx);
        self.cvar.notify_one();

        dbg!(surfaces) as i32
    }

    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32 {
        if let Some((decoder, _)) = &self.decoder {
            decoder.decode(pic).unwrap();
            1
        } else {
            eprintln!("Decoder isn't initialized but decode_picture has been called !");
            0
        }
    }

    fn display_picture(&mut self, disp: Option<&CUVIDPARSERDISPINFO>) -> i32 {
        // let codec = self.format().unwrap().codec;
        if let Some(tx) = &mut *self.tx {
            if let Ok(()) = tx.send(disp.cloned()) {
                // println!("successfully queued {:?} picture : {:?}", codec, disp);
                1
            } else {
                0
            }
        } else {
            eprintln!("Decoder isn't initialized but display_picture has been called !");
            0
        }
    }

    fn get_operating_point(&mut self, point: &CUVIDOPERATINGPOINTINFO) -> i32 {
        dbg!(point.codec);
        1
    }
}

#[cfg(feature = "cuda-npp")]
pub mod npp {
    pub use cuda_npp::get_stream_ctx;
    pub use cuda_npp::image::*;

    use crate::dec::FrameMapping;

    #[derive(Debug)]
    pub struct NvDecNV12<'a> {
        pub(crate) frame: FrameMapping<'a>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) planes: [*mut u8; 2],
    }

    impl<'a> Img<u8, P<2>> for NvDecNV12<'a> {
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
    pub struct NvDecYUV444<'a> {
        pub(crate) frame: FrameMapping<'a>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) data: *mut u8,
    }

    impl<'a> Img<u8, C<3>> for NvDecYUV444<'a> {
        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }

        fn pitch(&self) -> i32 {
            self.frame.pitch as i32
        }

        fn storage(&self) -> <C<3> as Channels>::Storage<u8> {
            self.data
        }

        fn device_ptr(&self) -> <C<3> as Channels>::Ref<u8> {
            C::<3>::make_ref(&self.data)
        }

        fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const u8> {
            C::<3>::iter_ptrs(&self.data)
        }
    }
}
