use std::cell::{OnceCell, RefCell};
use std::ops::{Deref, DerefMut};
use std::sync::{Condvar, Mutex};

use spsc::{Receiver, Sender};

use cuda_driver::sys::CuResult;
use cuda_driver::CuStream;
use nv_video_codec_sys::{CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPICPARAMS};

use crate::dec::{
    npp, query_caps, select_output_format, CuVideoCtxLock, CuVideoDecoder, CuvidParserCallbacks,
};

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

pub struct DecoderHolder<'a> {
    lock: Option<&'a CuVideoCtxLock>,
    cvar: Condvar,
    pub decoder: Trust<OnceCell<(CuVideoDecoder<'a>, CUVIDEOFORMAT)>>,
    tx: Trust<RefCell<Option<Sender<Msg>>>>,
    rx: Mutex<Option<Receiver<Msg>>>,
}

impl<'a> DecoderHolder<'a> {
    pub fn new(lock: Option<&'a CuVideoCtxLock>) -> Self {
        Self {
            lock,
            cvar: Condvar::new(),
            decoder: Trust(OnceCell::new()),
            tx: Trust(RefCell::new(None)),
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
        if let Some((_, format)) = self.decoder.get() {
            Some(format)
        } else {
            None
        }
    }

    pub fn is_open(&self) -> bool {
        self.tx
            .borrow()
            .as_ref()
            .map(|tx| tx.is_open())
            .unwrap_or(true)
    }

    #[cfg(feature = "cuda-npp")]
    pub fn map_npp_nv12<'map>(
        &'map self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<npp::NvDecNV12<'map>> {
        if let Some((decoder, format)) = self.decoder.get() {
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
            panic!("map_npp_nv12 called when instance was not initialized")
        }
    }

    #[cfg(feature = "cuda-npp")]
    pub fn map_npp_yuv444<'map>(
        &'map self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<npp::NvDecYUV444<'map>> {
        if let Some((decoder, format)) = self.decoder.get() {
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

impl CuvidParserCallbacks for DecoderHolder<'_> {
    fn sequence_callback(&self, format: &CUVIDEOFORMAT) -> Result<u32, ()> {
        println!("sequence_callback");
        let caps = query_caps(
            format.codec,
            format.chroma_format,
            format.bit_depth_luma_minus8 as u32 + 8,
        )
        .unwrap();
        if dbg!(caps).bIsSupported == 0 {
            println!("Unsupported codec/chroma/bitdepth");
            return Err(());
        }

        let surface_format = select_output_format(&format, &caps);

        let surfaces = format.min_num_decode_surfaces.next_power_of_two().max(1);

        let (tx, rx) = spsc::bounded_channel(2);

        let _ = self.decoder.set((
            CuVideoDecoder::new(
                dbg!(format),
                dbg!(surface_format),
                surfaces as u32,
                self.lock,
            )
            .unwrap(),
            format.clone(),
        ));
        *self.tx.borrow_mut() = Some(tx);
        *self.rx.lock().unwrap() = Some(rx);
        self.cvar.notify_one();

        Ok(dbg!(surfaces) as _)
    }

    fn decode_picture(&self, pic: &CUVIDPICPARAMS) -> Result<(), ()> {
        if let Some((decoder, _)) = self.decoder.get() {
            if !self.tx.borrow_mut().as_ref().unwrap().is_open() {
                println!("Receiver closed, no need to keep decoding");
                return Err(());
            }
            // println!(
            //     "{}: decoding {:>2}, {} {}",
            //     thread::current().name().unwrap_or(""),
            //     pic.CurrPicIdx,
            //     if pic.intra_pic_flag > 0 { "I" } else { " " },
            //     if pic.ref_pic_flag > 0 { "ref" } else { "" }
            // );
            decoder.decode(pic).unwrap();
            Ok(())
        } else {
            eprintln!("Decoder isn't initialized but decode_picture has been called !");
            Err(())
        }
    }

    fn display_picture(&self, disp: Option<&CUVIDPARSERDISPINFO>) -> Result<(), ()> {
        if let Some(tx) = self.tx.borrow_mut().as_mut() {
            if let Ok(()) = tx.send(disp.cloned()) {
                // println!(
                //     "{:?}: queued picture : {:?}",
                //     thread::current().name().unwrap_or(""),
                //     disp
                // );
                Ok(())
            } else {
                Err(())
            }
        } else {
            eprintln!("Decoder isn't initialized but display_picture has been called !");
            Err(())
        }
    }
}
