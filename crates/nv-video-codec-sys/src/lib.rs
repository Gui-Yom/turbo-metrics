#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub use cuda_driver_sys::{CUcontext, CUresult};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
