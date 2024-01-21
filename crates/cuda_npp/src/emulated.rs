use core::ffi::c_void;

use crate::sys;

/// Emulated with [sys::nppiMalloc_16u_C3] because it produces the same shape.
#[allow(non_snake_case)]
pub unsafe extern "C" fn nppiMalloc_16s_C3(width: i32, height: i32, step: *mut i32) -> *mut c_void {
    sys::nppiMalloc_16u_C3(width, height, step)
}

/// Emulated with [sys::nppiMalloc_16s_C4] because it produces the same shape. (16 * 4 = 32 * 2)
#[allow(non_snake_case)]
pub unsafe extern "C" fn nppiMalloc_32s_C2(width: i32, height: i32, step: *mut i32) -> *mut c_void {
    sys::nppiMalloc_16s_C4(width, height, step)
}
