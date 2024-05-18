use core::ffi::c_char;
use std::error::Error;
use std::ffi::CStr;
use std::fmt::{Debug, Display, Formatter};
use std::mem::MaybeUninit;
use std::ptr::null;

use crate::sys::{cuDeviceGet, cuDeviceGetCount, cuDeviceGetName, cuDeviceTotalMem_v2, cuGetErrorString, cuInit};

mod sys {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[repr(transparent)]
struct CuError(sys::CUresult);

impl sys::CUresult {
    pub fn result(self) -> CuResult<()> {
        CuResult::from(self)
    }
}

type CuResult<T> = Result<T, CuError>;

impl From<sys::CUresult> for CuResult<()> {
    fn from(value: sys::CUresult) -> Self {
        if value == sys::CUresult::CUDA_SUCCESS {
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
            cuGetErrorString(self.0, &mut err_str as _);
            if err_str.is_null() {
                write!(f, "Invalid error code: {:?}", self.0)
            } else {
                Debug::fmt(CStr::from_ptr(err_str), f)
            }
        }
    }
}

impl Error for CuError {}

pub fn init_cuda() -> CuResult<()> {
    unsafe {
        cuInit(0).result()
    }
}

struct CudaDevice(sys::CUdevice);

impl CudaDevice {
    pub fn count() -> CuResult<i32> {
        let mut count = 0;
        unsafe {
            cuDeviceGetCount(&mut count).result()?;
        }
        Ok(count)
    }

    pub fn get(index: i32) -> CuResult<Self> {
        let mut device = MaybeUninit::uninit();
        unsafe {
            cuDeviceGet(device.as_mut_ptr(), index).result()?;
            Ok(Self(device.assume_init()))
        }
    }

    pub fn name(&self) -> CuResult<String> {
        let mut name = [0; 64];
        let name_str = unsafe {
            cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, self.0).result()?;
            CStr::from_ptr(name.as_ptr())
        };
        Ok(name_str.to_string_lossy().to_string())
    }

    pub fn total_mem(&self) -> CuResult<usize> {
        let mut size = 0;
        unsafe {
            cuDeviceTotalMem_v2(&mut size, self.0).result()?;
        }
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CudaDevice, CuResult, init_cuda};

    #[test]
    fn test() -> CuResult<()> {
        init_cuda()?;
        let dev = CudaDevice::get(0)?;
        dbg!(dev.total_mem()?);
        dbg!(dev.name()?);
        Ok(())
    }
}
