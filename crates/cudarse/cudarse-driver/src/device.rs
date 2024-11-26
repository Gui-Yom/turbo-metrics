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

    // pub fn create_ctx(&self) -> CuResult<CuCtx> {
    //     let mut ctx = null_mut();
    //     unsafe {
    //         sys::cuCtxCreate_v4(
    //             &mut ctx,
    //             &mut sys::CUctxCreateParams {
    //                 execAffinityParams: &mut sys::CUexecAffinityParam {
    //                     type_: CUexecAffinityType_enum::CU_EXEC_AFFINITY_TYPE_SM_COUNT,
    //                     param: Default::default(),
    //                 },
    //                 numExecAffinityParams: 0,
    //                 cigParams: &mut sys::CUctxCigParam {
    //                     sharedDataType: CUcigDataType_enum::CIG_DATA_TYPE_D3D12_COMMAND_QUEUE,
    //                     sharedData: (),
    //                 },
    //             },
    //             0,
    //             self.0,
    //         )
    //         .result()?;
    //     }
    //     Ok(CuCtx(NonNull::new(ctx).unwrap()))
    // }

    pub fn attr(&self, attr: sys::CUdevice_attribute) -> CuResult<i32> {
        let mut out = 0;
        unsafe {
            sys::cuDeviceGetAttribute(&mut out, attr, self.0).result()?;
        }
        Ok(out)
    }
}
