#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use cudarse_driver_sys::{CUcontext, CUresult};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl CUVIDDECODECAPS {
    pub fn is_output_format_supported(&self, format: cudaVideoSurfaceFormat) -> bool {
        self.nOutputFormatMask & (1 << format as u16) > 0
    }
}

impl CUVIDEOFORMAT {
    pub fn display_width(&self) -> u32 {
        (self.display_area.right - self.display_area.left) as u32
    }

    pub fn display_height(&self) -> u32 {
        (self.display_area.bottom - self.display_area.top) as u32
    }

    #[cfg(feature = "npp")]
    pub fn size(&self) -> cudarse_npp_sys::NppiSize {
        cudarse_npp_sys::NppiSize {
            width: self.display_width() as _,
            height: self.display_height() as _,
        }
    }

    #[cfg(feature = "npp")]
    pub fn rect(&self) -> cudarse_npp_sys::NppiRect {
        cudarse_npp_sys::NppiRect {
            x: 0,
            y: 0,
            width: self.display_width() as _,
            height: self.display_height() as _,
        }
    }
}

impl CUVIDEOFORMAT__bindgen_ty_4 {
    /// true if video use full range color 0-255, false if using limited color range 16-235.
    pub fn full_range(&self) -> bool {
        self._bitfield_1.get_bit(4)
    }
}
