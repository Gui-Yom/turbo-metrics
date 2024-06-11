use core::ffi::c_char;
use core::ffi::c_void;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{Debug, Display, Formatter};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::{null, null_mut};

use crate::sys::{
    cuCtxSetCurrent, cuCtxSynchronize, cuDeviceGet, cuDeviceGetCount, cuDeviceGetName,
    cuDevicePrimaryCtxRetain, cuDeviceTotalMem_v2, cuDriverGetVersion, cuEventCreate,
    cuEventDestroy_v2, cuEventElapsedTime, cuEventQuery, cuEventRecord, cuEventSynchronize,
    cuFuncGetName, cuFuncGetParamInfo, cuFuncIsLoaded, cuFuncLoad, cuFuncSetCacheConfig,
    cuGetErrorString, cuInit, cuLaunchKernel, cuMemGetInfo_v2, cuModuleEnumerateFunctions,
    cuModuleGetFunction, cuModuleGetFunctionCount, cuModuleLoad, cuModuleUnload, cuProfilerStart,
    cuProfilerStop, cuStreamCreate, cuStreamDestroy_v2, cuStreamQuery, cuStreamSynchronize,
    cuStreamWaitEvent, CUstream_flags,
};

pub mod sys {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[repr(transparent)]
pub struct CuError(sys::CUresult);

impl sys::CUresult {
    pub fn result(self) -> CuResult<()> {
        CuResult::from(self)
    }
}

pub type CuResult<T> = Result<T, CuError>;

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

/// Initialize the global cuda context. This needs to be called before any API call.
pub fn init_cuda() -> CuResult<()> {
    unsafe { cuInit(0).result() }
}

pub fn cuda_driver_version() -> CuResult<u32> {
    let mut version = 0;
    unsafe {
        cuDriverGetVersion(&mut version).result()?;
    }
    Ok(version as u32)
}

/// Synchronize with the whole CUDA context.
pub fn sync_ctx() -> CuResult<()> {
    unsafe { cuCtxSynchronize().result() }
}

pub fn profiler_start() -> CuResult<()> {
    unsafe { cuProfilerStart().result() }
}

pub fn profiler_stop() -> CuResult<()> {
    unsafe { cuProfilerStop().result() }
}

pub fn mem_info() -> CuResult<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    unsafe {
        cuMemGetInfo_v2(&mut free, &mut total).result()?;
    }
    Ok((free, total))
}

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
        Ok(CuCtx(ctx))
    }
}

#[repr(transparent)]
pub struct CuCtx(sys::CUcontext);

impl CuCtx {
    /// Bind this context to the calling thread.
    pub fn set_current(&self) -> CuResult<()> {
        unsafe { cuCtxSetCurrent(self.0).result() }
    }
}

#[repr(transparent)]
pub struct CuStream(sys::CUstream);

impl CuStream {
    pub const DEFAULT: Self = CuStream(null_mut());

    /// Create a new CUDA stream.
    pub fn new() -> CuResult<Self> {
        let mut stream = null_mut();
        unsafe {
            cuStreamCreate(&mut stream, CUstream_flags::CU_STREAM_NON_BLOCKING as _).result()?;
        }
        Ok(Self(stream))
    }

    pub fn inner(&self) -> *mut c_void {
        self.0 as _
    }

    /// Wait for any work on this stream to complete.
    pub fn sync(&self) -> CuResult<()> {
        unsafe { cuStreamSynchronize(self.0).result() }
    }

    /// Return true if this stream has finished any submitted work.
    pub fn completed(&self) -> CuResult<bool> {
        unsafe {
            match cuStreamQuery(self.0) {
                sys::CUresult::CUDA_SUCCESS => Ok(true),
                sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
                other => Err(CuError(other)),
            }
        }
    }

    /// Make this stream wait for an event ot complete.
    pub fn wait_for_evt(&self, evt: &CuEvent) -> CuResult<()> {
        unsafe { cuStreamWaitEvent(self.0, evt.0, 0).result() }
    }

    /// Make this stream wait for the work in another stream to complete
    pub fn wait_for_stream(&self, other: &Self) -> CuResult<()> {
        let evt = CuEvent::new()?;
        evt.record(other)?;
        self.wait_for_evt(&evt)
    }

    /// Join two streams by making each one wait for the other.
    /// After this point, both streams are synchronized with each other.
    pub fn join(&self, other: &Self) -> CuResult<()> {
        self.wait_for_stream(other)?;
        other.wait_for_stream(self)
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                cuStreamDestroy_v2(self.0).result().unwrap();
            }
        }
    }
}

/// An event is a point in a stream.
#[repr(transparent)]
pub struct CuEvent(sys::CUevent);

impl CuEvent {
    pub fn new() -> CuResult<Self> {
        let mut ptr = null_mut();
        unsafe { cuEventCreate(&mut ptr, 0).result()? };
        Ok(Self(ptr))
    }

    /// Return true if this event has completed.
    pub fn done(&self) -> CuResult<bool> {
        unsafe {
            match cuEventQuery(self.0) {
                sys::CUresult::CUDA_SUCCESS => Ok(true),
                sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
                other => Err(CuError(other)),
            }
        }
    }

    /// Wait for this event to complete.
    pub fn sync(&self) -> CuResult<()> {
        unsafe { cuEventSynchronize(self.0).result() }
    }

    pub fn elapsed_since(&self, start: &Self) -> CuResult<f32> {
        let mut elapsed = 0.0;
        unsafe { cuEventElapsedTime(&mut elapsed, start.0, self.0).result()? }
        Ok(elapsed)
    }

    pub fn record(&self, stream: &CuStream) -> CuResult<()> {
        unsafe { cuEventRecord(self.0, stream.0).result() }
    }
}

impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { cuEventDestroy_v2(self.0).result().unwrap() }
    }
}

#[repr(transparent)]
pub struct CuModule(sys::CUmodule);

impl CuModule {
    pub fn load_from_file(path: impl AsRef<Path>) -> CuResult<Self> {
        let mut cu_module = null_mut();
        let fname = CString::new(path.as_ref().as_os_str().as_encoded_bytes()).unwrap();
        unsafe {
            cuModuleLoad(&mut cu_module, fname.as_ptr()).result()?;
        }
        Ok(Self(cu_module))
    }

    pub fn function_count(&self) -> CuResult<u32> {
        let mut count = 0;
        unsafe {
            cuModuleGetFunctionCount(&mut count, self.0).result()?;
        }
        Ok(count)
    }

    pub fn functions(&self) -> CuResult<Vec<CuFunction>> {
        let mut buf = vec![CuFunction(null_mut()); self.function_count()? as usize];
        unsafe {
            cuModuleEnumerateFunctions(buf.as_mut_ptr().cast(), buf.len() as _, self.0).result()?;
        }
        Ok(buf)
    }

    pub fn function_by_name(&self, name: &str) -> CuResult<CuFunction> {
        let mut func = CuFunction(null_mut());
        let name = CString::new(name).unwrap();
        unsafe {
            cuModuleGetFunction(&mut func.0, self.0, name.as_ptr()).result()?;
        }
        Ok(func)
    }
}

impl Drop for CuModule {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.0).result().unwrap();
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct CuFunction(sys::CUfunction);

impl CuFunction {
    pub fn name(&self) -> CuResult<String> {
        let mut ptr = null();
        unsafe {
            cuFuncGetName(&mut ptr, self.0).result()?;
            Ok(CStr::from_ptr(ptr).to_string_lossy().to_string())
        }
    }

    pub fn is_loaded(&self) -> CuResult<bool> {
        let mut state = sys::CUfunctionLoadingState::CU_FUNCTION_LOADING_STATE_MAX;
        unsafe {
            cuFuncIsLoaded(&mut state, self.0).result()?;
        }
        Ok(state == sys::CUfunctionLoadingState_enum::CU_FUNCTION_LOADING_STATE_LOADED)
    }

    pub fn load(&self) -> CuResult<()> {
        unsafe { cuFuncLoad(self.0).result() }
    }

    pub fn set_cache_config(&self, cache: sys::CUfunc_cache) -> CuResult<()> {
        unsafe { cuFuncSetCacheConfig(self.0, cache).result() }
    }

    pub fn param_count(&self) -> CuResult<usize> {
        let mut i = 0;
        unsafe {
            let mut offset = 0;
            let mut size = 0;
            while let Ok(()) = cuFuncGetParamInfo(self.0, i, &mut offset, &mut size).result() {
                i += 1;
            }
        }
        Ok(i)
    }

    pub fn launch(
        &self,
        cfg: &LaunchConfig,
        stream: &CuStream,
        params: &[*mut c_void],
    ) -> CuResult<()> {
        assert_eq!(params.len(), self.param_count()?);
        unsafe {
            cuLaunchKernel(
                self.0,
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
}

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use crate::{init_cuda, CuDevice, CuModule, CuResult, CuStream, LaunchConfig};

    #[test]
    fn test_device() -> CuResult<()> {
        init_cuda()?;
        let dev = CuDevice::get(0)?;
        dbg!(dev.total_mem()?);
        dbg!(dev.name()?);
        Ok(())
    }

    #[test]
    fn test_module() -> CuResult<()> {
        init_cuda()?;
        let dev = CuDevice::get(0)?;
        let ctx = dev.retain_primary_ctx()?;
        ctx.set_current()?;
        let module = CuModule::load_from_file(
            "../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx",
        )?;
        dbg!(module.function_count());
        for f in module.functions()? {
            dbg!(f.name());
            dbg!(f.is_loaded());
            dbg!(f.param_count());
        }
        Ok(())
    }

    #[test]
    fn test_launch() -> CuResult<()> {
        init_cuda()?;
        let dev = CuDevice::get(0)?;
        let ctx = dev.retain_primary_ctx()?;
        ctx.set_current()?;
        let module = CuModule::load_from_file(
            "../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx",
        )?;
        let r: *const u8 = null();
        let w: *mut u8 = null_mut();
        module.function_by_name("srgb_to_linear")?.launch(
            &LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &CuStream::DEFAULT,
            kernel_params!(r, 1, w, 1,),
        )?;
        Ok(())
    }
}
