#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use cuda_driver_sys::{CUcontext, CUresult};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl CUVIDDECODECAPS {
    pub fn is_output_format_supported(&self, format: cudaVideoSurfaceFormat) -> bool {
        self.nOutputFormatMask & (1 << format as u16) > 0
    }
}
