use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::{null_mut, NonNull};

use sys::{
    cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDevicePrimaryCtxRelease_v2,
    cuDevicePrimaryCtxRetain, cuDeviceTotalMem_v2, CuResult,
};

use crate::{sys, CuCtx};

#[repr(transparent)]
pub struct CuDevice(sys::CUdevice);

impl CuDevice {
    /// Total number of devices visible by this cuda driver.
    pub fn count() -> CuResult<i32> {
        let mut count = 0;
        unsafe {
            cuDeviceGetCount(&mut count).result()?;
        }
        Ok(count)
    }

    /// Get a device by its index.
    pub fn get(index: i32) -> CuResult<Self> {
        let mut device = MaybeUninit::uninit();
        unsafe {
            cuDeviceGet(device.as_mut_ptr(), index).result()?;
            Ok(Self(device.assume_init()))
        }
    }

    /// Get the device name.
    pub fn name(&self) -> CuResult<String> {
        let mut name = [0; 64];
        let name_str = unsafe {
            cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, self.0).result()?;
            CStr::from_ptr(name.as_ptr())
        };
        Ok(name_str.to_string_lossy().to_string())
    }

    /// Get the device total memory in bytes.
    pub fn total_mem(&self) -> CuResult<usize> {
        let mut size = 0;
        unsafe {
            cuDeviceTotalMem_v2(&mut size, self.0).result()?;
        }
        Ok(size)
    }

    /// Get the device primary context. Required to use any more complex API calls.
    pub fn retain_primary_ctx(&self) -> CuResult<CuCtx> {
        let mut ctx = null_mut();
        unsafe {
            cuDevicePrimaryCtxRetain(&mut ctx, self.0).result()?;
        }
        Ok(CuCtx(NonNull::new(ctx).unwrap()))
    }

    /// Release the primary context on the GPU.
    pub fn release_primary_ctx(&self) -> CuResult<()> {
        unsafe { cuDevicePrimaryCtxRelease_v2(self.0).result() }
    }
}
