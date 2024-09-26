#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;
pub use widestring;
use widestring::{widecstr, WideCStr};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(all(target_os = "windows", target_pointer_width = "64"))]
pub const AMF_DLL_NAME: &str = "amfrt64";
#[cfg(all(target_os = "windows", target_pointer_width = "32"))]
pub const AMF_DLL_NAME: &str = "amfrt32";
#[cfg(target_os = "android")]
pub const AMF_DLL_NAME: &str = "libamf";
#[cfg(target_vendor = "apple")]
pub const AMF_DLL_NAME: &str = "libamfrt.framework/libamfrt";
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub const AMF_DLL_NAME: &str = "libamfrt64.so.1";
#[cfg(all(target_os = "linux", target_pointer_width = "32"))]
pub const AMF_DLL_NAME: &str = "libamfrt32.so.1";

/*
#if defined(_WIN32)
    #if defined(_M_AMD64)
        #define AMF_DLL_NAME    L"amfrt64.dll"
        #define AMF_DLL_NAMEA   "amfrt64.dll"
    #else
        #define AMF_DLL_NAME    L"amfrt32.dll"
        #define AMF_DLL_NAMEA   "amfrt32.dll"
    #endif
#elif defined(__ANDROID__) && !defined(AMF_ANDROID_ENCODER)
    #define AMF_DLL_NAME    L"libamf.so"
    #define AMF_DLL_NAMEA    "libamf.so"
#elif defined(__APPLE__)
    #define AMF_DLL_NAME    L"libamfrt.framework/libamfrt"
    #define AMF_DLL_NAMEA   "libamfrt.framework/libamfrt"
#elif defined(__linux__)
    #if defined(__x86_64__) || defined(__aarch64__)
        #define AMF_DLL_NAME    L"libamfrt64.so.1"
        #define AMF_DLL_NAMEA   "libamfrt64.so.1"
    #else
        #define AMF_DLL_NAME    L"libamfrt32.so.1"
        #define AMF_DLL_NAMEA   "libamfrt32.so.1"
    #endif
#endif
 */

pub const AMF_INIT_FUNCTION_NAME: &CStr = c"AMFInit";
pub const AMF_QUERY_VERSION_FUNCTION_NAME: &CStr = c"AMFQueryVersion";

/*
#define AMF_INIT_FUNCTION_NAME             "AMFInit"
#define AMF_QUERY_VERSION_FUNCTION_NAME    "AMFQueryVersion"
 */

/*
#define AMFVideoDecoderUVD_MPEG2                     L"AMFVideoDecoderUVD_MPEG2"
#define AMFVideoDecoderUVD_MPEG4                     L"AMFVideoDecoderUVD_MPEG4"
#define AMFVideoDecoderUVD_WMV3                      L"AMFVideoDecoderUVD_WMV3"
#define AMFVideoDecoderUVD_VC1                       L"AMFVideoDecoderUVD_VC1"
#define AMFVideoDecoderUVD_H264_AVC                  L"AMFVideoDecoderUVD_H264_AVC"
#define AMFVideoDecoderUVD_H264_MVC                  L"AMFVideoDecoderUVD_H264_MVC"
#define AMFVideoDecoderUVD_H264_SVC                  L"AMFVideoDecoderUVD_H264_SVC"
#define AMFVideoDecoderUVD_MJPEG                     L"AMFVideoDecoderUVD_MJPEG"
#define AMFVideoDecoderHW_H265_HEVC                  L"AMFVideoDecoderHW_H265_HEVC"
#define AMFVideoDecoderHW_H265_MAIN10                L"AMFVideoDecoderHW_H265_MAIN10"
#define AMFVideoDecoderHW_VP9                        L"AMFVideoDecoderHW_VP9"
#define AMFVideoDecoderHW_VP9_10BIT                  L"AMFVideoDecoderHW_VP9_10BIT"
#define AMFVideoDecoderHW_AV1                        L"AMFVideoDecoderHW_AV1"
#define AMFVideoDecoderHW_AV1_12BIT                  L"AMFVideoDecoderHW_AV1_12BIT"
 */

pub const AMFVideoDecoderUVD_MPEG2: &WideCStr = widecstr!("AMFVideoDecoderUVD_MPEG2");
pub const AMFVideoDecoderUVD_MPEG4: &WideCStr = widecstr!("AMFVideoDecoderUVD_MPEG4");
pub const AMFVideoDecoderUVD_WMV3: &WideCStr = widecstr!("AMFVideoDecoderUVD_WMV3");
pub const AMFVideoDecoderUVD_VC1: &WideCStr = widecstr!("AMFVideoDecoderUVD_VC1");
pub const AMFVideoDecoderUVD_H264_AVC: &WideCStr = widecstr!("AMFVideoDecoderUVD_H264_AVC");
pub const AMFVideoDecoderUVD_H264_MVC: &WideCStr = widecstr!("AMFVideoDecoderUVD_H264_MVC");
pub const AMFVideoDecoderUVD_H264_SVC: &WideCStr = widecstr!("AMFVideoDecoderUVD_H264_SVC");
pub const AMFVideoDecoderUVD_MJPEG: &WideCStr = widecstr!("AMFVideoDecoderUVD_MJPEG");
pub const AMFVideoDecoderHW_H265_HEVC: &WideCStr = widecstr!("AMFVideoDecoderHW_H265_HEVC");
pub const AMFVideoDecoderHW_H265_MAIN10: &WideCStr = widecstr!("AMFVideoDecoderHW_H265_MAIN10");
pub const AMFVideoDecoderHW_VP9: &WideCStr = widecstr!("AMFVideoDecoderHW_VP9");
pub const AMFVideoDecoderHW_VP9_10BIT: &WideCStr = widecstr!("AMFVideoDecoderHW_VP9_10BIT");
pub const AMFVideoDecoderHW_AV1: &WideCStr = widecstr!("AMFVideoDecoderHW_AV1");
pub const AMFVideoDecoderHW_AV1_12BIT: &WideCStr = widecstr!("AMFVideoDecoderHW_AV1_12BIT");

pub type Result<T> = core::result::Result<T, AMF_RESULT>;

impl Into<Result<()>> for AMF_RESULT {
    fn into(self) -> Result<()> {
        self.result()
    }
}

impl AMF_RESULT {
    pub fn result_with<T>(self, t: T) -> Result<T> {
        match self {
            AMF_RESULT::AMF_OK => Ok(t),
            other => Err(other),
        }
    }

    pub fn result(self) -> Result<()> {
        match self {
            AMF_RESULT::AMF_OK => Ok(()),
            other => Err(other),
        }
    }
}
