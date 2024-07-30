use spsc::{Receiver, Sender};
use std::ops::{Deref, DerefMut};
use std::sync::{Condvar, Mutex};

use cuda_driver::sys::CuResult;
use cuda_driver::{CuCtx, CuStream};
use nv_video_codec_sys::{
    cudaVideoSurfaceFormat, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPICPARAMS,
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
    cvar: Condvar,
    pub decoder: Option<(CuVideoDecoder, CUVIDEOFORMAT)>,
    tx: Trust<Option<Sender<Msg>>>,
    rx: Mutex<Option<Receiver<Msg>>>,
}

impl DecoderHolder {
    pub fn new() -> Self {
        Self {
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

    #[cfg(feature = "cuda-npp")]
    pub fn map_npp<'a>(
        &'a self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<npp::NvDecImg> {
        if let Some((decoder, format)) = &self.decoder {
            let mapping = decoder.map(info, stream)?;
            Ok(npp::NvDecImg {
                width: (format.display_area.right - format.display_area.left) as u32,
                height: (format.display_area.bottom - format.display_area.top) as u32,
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
}

impl VideoParserCb for DecoderHolder {
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        let caps = query_caps(
            format.codec,
            format.chroma_format,
            format.bit_depth_luma_minus8 as u32 + 8,
        )
        .unwrap();
        if caps.bIsSupported == 0 {
            println!("Unsupported codec/chroma/bitdepth");
            return 0;
        }

        assert!(
            caps.is_output_format_supported(cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12)
        );

        let lock = CuVideoCtxLock::new(&CuCtx::get_current().unwrap()).unwrap();
        let surfaces = format.min_num_decode_surfaces.next_power_of_two().max(1);

        let (tx, rx) = spsc::bounded_channel(surfaces as usize);

        self.decoder = Some((
            CuVideoDecoder::new(
                dbg!(format),
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
                &lock,
            )
            .unwrap(),
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
        if let Some(tx) = &mut *self.tx {
            // (self.on_decode)(&instance.decoder, &instance.format, disp);
            tx.send(disp.cloned()).unwrap();
            1
        } else {
            eprintln!("Decoder isn't initialized but display_picture has been called !");
            0
        }
    }
}

#[cfg(feature = "cuda-npp")]
pub mod npp {
    pub use cuda_npp::get_stream_ctx;
    pub use cuda_npp::image::*;

    use crate::dec::FrameMapping;

    #[derive(Debug)]
    pub struct NvDecImg<'a> {
        pub(crate) frame: FrameMapping<'a>,
        pub(crate) width: u32,
        pub(crate) height: u32,
        pub(crate) planes: [*mut u8; 2],
    }

    impl<'a> Img<u8, P<2>> for NvDecImg<'a> {
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
}
