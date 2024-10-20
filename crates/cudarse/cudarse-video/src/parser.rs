use cudarse_driver::sys::CuResult;
use cudarse_video_sys::{
    cudaVideoCodec, cuvidCreateVideoParser, cuvidDestroyVideoParser, cuvidParseVideoData,
    CUVIDEOFORMATEX__bindgen_ty_1, CUvideopacketflags, CUVIDEOFORMAT, CUVIDEOFORMATEX,
    CUVIDOPERATINGPOINTINFO, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS,
    CUVIDSEIMESSAGEINFO, CUVIDSOURCEDATAPACKET,
};
use std::ffi::{c_ulong, c_void};
use std::ops::Deref;
use std::ptr::{null, null_mut, NonNull};

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

/// Unsafe low level video parser that takes a raw pointer to the callbacks
pub struct CuVideoParser {
    pub(crate) inner: NonNull<c_void>,
}

impl CuVideoParser {
    pub unsafe fn new<CB: CuvidParserCallbacks>(
        codec: cudaVideoCodec,
        cb: *mut CB,
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
            pUserData: cb.cast(),
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

impl Drop for CuVideoParser {
    fn drop(&mut self) {
        unsafe {
            cuvidDestroyVideoParser(self.inner.as_ptr())
                .result()
                .unwrap()
        }
    }
}

/// A cuvid parser that stores the callback by holding a reference (you must deal with lifetimes) or by storing a smart pointer.
pub struct CuvidParser<Storage: Deref>
where
    <Storage as Deref>::Target: CuvidParserCallbacks,
{
    inner: CuVideoParser,
    _cb: Storage,
}

impl<Storage, CB> CuvidParser<Storage>
where
    CB: CuvidParserCallbacks,
    Storage: Deref<Target = CB>,
{
    pub fn new(
        codec: cudaVideoCodec,
        cb: Storage,
        clock_rate: Option<u32>,
        extra_data: Option<&[u8]>,
    ) -> CuResult<Self> {
        unsafe {
            Ok(Self {
                inner: CuVideoParser::new(
                    codec,
                    cb.deref() as *const CB as *mut CB,
                    clock_rate,
                    extra_data,
                )?,
                _cb: cb,
            })
        }
    }

    /// The parser expects annexb format for H264 (with 0001 nalu delimiters).
    pub fn parse_data(&mut self, packet: &[u8], timestamp: i64) -> CuResult<()> {
        self.inner.parse_data(packet, timestamp)
    }

    pub fn flush(&mut self) -> CuResult<()> {
        self.inner.flush()
    }
}

// struct CallbacksSink {
//     events: VecDeque<()>,
// }
//
// impl CallbacksSink {
//     fn new() -> Self {
//         Self {
//             events: Default::default(),
//         }
//     }
// }
//
// impl CuvidParserCallbacks for CallbacksSink {
//     fn sequence_callback(&self, format: &CUVIDEOFORMAT) -> Result<u32, ()> {
//         todo!()
//     }
//
//     fn decode_picture(&self, pic: &CUVIDPICPARAMS) -> Result<(), ()> {
//         todo!()
//     }
//
//     fn display_picture(&self, disp: Option<&CUVIDPARSERDISPINFO>) -> Result<(), ()> {
//         todo!()
//     }
// }
