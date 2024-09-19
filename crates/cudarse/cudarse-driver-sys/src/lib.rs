#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::error::Error;
use std::ffi::{c_char, CStr};
use std::fmt::{Debug, Display, Formatter};
use std::ptr::null;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[repr(transparent)]
pub struct CuError(pub CUresult);

impl CUresult {
    pub fn result(self) -> CuResult<()> {
        CuResult::from(self)
    }
}

pub type CuResult<T> = Result<T, CuError>;

impl From<CUresult> for CuResult<()> {
    fn from(value: CUresult) -> Self {
        if value == CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CuError(value))
        }
    }
}

impl Debug for CuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error: {:?}", self.0)
    }
}

impl Display for CuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut err_str: *const c_char = null();
        unsafe {
            let _ = cuGetErrorString(self.0, &mut err_str as _);
            if err_str.is_null() {
                write!(f, "Invalid error code: {:?}", self.0)
            } else {
                write!(
                    f,
                    "cuda error: {:?} ({:?})",
                    self.0,
                    CStr::from_ptr(err_str)
                )
            }
        }
    }
}

impl Error for CuError {}
