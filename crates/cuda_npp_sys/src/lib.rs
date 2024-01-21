#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub type Npp8u = core::ffi::c_void;
pub type Npp8s = core::ffi::c_void;
pub type Npp16u = core::ffi::c_void;
pub type Npp16s = core::ffi::c_void;
pub type Npp32s = core::ffi::c_void;
pub type Npp32f = core::ffi::c_void;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
