use std::ptr::{null_mut, NonNull};

pub use cudarse_driver_sys as sys;
pub use device::*;
pub use event::*;
pub use function::*;
pub use graph::*;
pub use module::*;
pub use stream::*;
use sys::{
    cuCtxGetCurrent, cuCtxSetCurrent, cuCtxSynchronize, cuDriverGetVersion, cuInit,
    cuMemGetInfo_v2, cuMemcpy2DAsync_v2, cuProfilerStart, cuProfilerStop, CuResult, CUDA_MEMCPY2D,
};

mod device;
mod event;
mod function;
mod graph;
mod module;
mod stream;

/// Initialize the global cuda context. This needs to be called before any API call.
pub fn init_cuda() -> CuResult<()> {
    unsafe { cuInit(0).result() }
}

pub fn init_cuda_and_primary_ctx() -> CuResult<()> {
    init_cuda()?;
    let dev = CuDevice::get(0)?;
    dev.retain_primary_ctx()?.set_current()
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
pub struct CuCtx(pub(crate) NonNull<sys::CUctx_st>);

unsafe impl Send for CuCtx {}
unsafe impl Sync for CuCtx {}

impl CuCtx {
    /// Bind this context to the calling thread.
    pub fn set_current(&self) -> CuResult<()> {
        unsafe { cuCtxSetCurrent(self.0.as_ptr()).result() }
    }

    pub fn get_current() -> CuResult<Self> {
        let mut ctx = null_mut();
        unsafe {
            cuCtxGetCurrent(&mut ctx).result()?;
        }
        Ok(Self(NonNull::new(ctx).unwrap()))
    }

    pub fn inner(&self) -> sys::CUcontext {
        self.0.as_ptr()
    }
}

pub fn copy(copy: &CUDA_MEMCPY2D, stream: &CuStream) -> CuResult<()> {
    unsafe { cuMemcpy2DAsync_v2(copy, stream.0).result() }
}

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use crate::{init_cuda, kernel_params, CuDevice, CuModule, CuResult, CuStream, LaunchConfig};

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
            "../../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx",
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
            "../../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx",
        )?;
        let r: *const u8 = null();
        let w: *mut u8 = null_mut();
        unsafe {
            module.function_by_name("srgb_to_linear")?.launch(
                &LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &CuStream::DEFAULT,
                kernel_params!(r, 1, w, 1,),
            )?;
        }

        Ok(())
    }
}
