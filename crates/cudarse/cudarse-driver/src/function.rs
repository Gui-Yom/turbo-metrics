use cudarse_driver_sys::CUfunction;
use std::ffi::{c_void, CStr};
use std::ptr::{null, null_mut, NonNull};
use sys::{
    cuFuncGetName, cuFuncGetParamInfo, cuFuncIsLoaded, cuFuncLoad, cuFuncSetCacheConfig,
    cuLaunchKernel, CuResult,
};

use crate::{sys, CuStream};

#[derive(Clone)]
#[repr(transparent)]
pub struct CuFunction(pub(crate) NonNull<sys::CUfunc_st>);

impl CuFunction {
    pub fn from_raw(raw: CUfunction) -> Self {
        Self(NonNull::new(raw).expect("null pointer"))
    }

    pub fn name(&self) -> CuResult<String> {
        let mut ptr = null();
        unsafe {
            cuFuncGetName(&mut ptr, self.0.as_ptr()).result()?;
            Ok(CStr::from_ptr(ptr).to_string_lossy().to_string())
        }
    }

    pub fn is_loaded(&self) -> CuResult<bool> {
        let mut state = sys::CUfunctionLoadingState::CU_FUNCTION_LOADING_STATE_MAX;
        unsafe {
            cuFuncIsLoaded(&mut state, self.0.as_ptr()).result()?;
        }
        Ok(state == sys::CUfunctionLoadingState_enum::CU_FUNCTION_LOADING_STATE_LOADED)
    }

    pub fn load(&self) -> CuResult<()> {
        unsafe { cuFuncLoad(self.0.as_ptr()).result() }
    }

    pub fn set_cache_config(&self, cache: sys::CUfunc_cache) -> CuResult<()> {
        unsafe { cuFuncSetCacheConfig(self.0.as_ptr(), cache).result() }
    }

    pub fn param_count(&self) -> CuResult<usize> {
        let mut i = 0;
        unsafe {
            let mut offset = 0;
            let mut size = 0;
            while let Ok(()) =
                cuFuncGetParamInfo(self.0.as_ptr(), i, &mut offset, &mut size).result()
            {
                i += 1;
            }
        }
        Ok(i)
    }

    /// # Safety
    /// This method is heavily unsafe as it will not check if the parameters given here match the kernel signature.
    pub unsafe fn launch(
        &self,
        cfg: &LaunchConfig,
        stream: &CuStream,
        params: &[*mut c_void],
    ) -> CuResult<()> {
        assert_eq!(params.len(), self.param_count()?);
        unsafe {
            cuLaunchKernel(
                self.0.as_ptr(),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream.0,
                params.as_ptr().cast_mut(),
                null_mut(),
            )
            .result()
        }
    }
}

#[derive(Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

#[macro_export]
macro_rules! kernel_params {
    // (@single $p:expr) => {
    //     ::core::ptr::addr_of_mut!($p).cast()
    // };
    (@single $p:expr) => {
        (&($p) as *const _ as *const ::core::ffi::c_void).cast_mut()
    };
    ($($p:expr,)*) => {
        &[$(kernel_params!(@single $p)),*]
    };
    ($($p:expr),*) => {
        &[$(kernel_params!(@single $p)),*]
    };
}
