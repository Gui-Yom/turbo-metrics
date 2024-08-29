use std::ffi::{c_void, CStr};
use std::fmt::Debug;
use std::mem;
use std::ptr::null_mut;

pub use cudarse_npp_sys as sys;
use sys::{
    cudaFreeAsync, cudaMallocAsync, cudaStream_t, nppGetGpuName, nppGetGpuNumSMs, nppGetLibVersion,
    nppGetMaxThreadsPerBlock, nppGetMaxThreadsPerSM, nppGetStream, nppGetStreamContext,
    nppSetStream, NppStreamContext, Result,
};

pub mod image;

mod __priv {
    /// So people don't implement child trait for themselves.
    /// Hacky way of doing closed polymorphism with traits.
    pub trait Sealed {}

    impl<T: Sealed> Sealed for &T {}

    impl<T: Sealed> Sealed for &mut T {}

    impl Sealed for u8 {}

    impl Sealed for u16 {}

    impl Sealed for i16 {}

    impl Sealed for i32 {}

    impl Sealed for f32 {}
}

/// Return the NPP lib version
pub fn version() -> (i32, i32, i32) {
    let ver = unsafe { &*nppGetLibVersion() };
    (ver.major, ver.minor, ver.build)
}

/// Return the name of the device in the NPP context
pub fn gpu_name() -> &'static CStr {
    unsafe {
        let name = nppGetGpuName();
        CStr::from_ptr::<'static>(name)
    }
}

/// Return the number of multiprocessors of the device in the NPP context
pub fn gpu_num_sm() -> u32 {
    unsafe { nppGetGpuNumSMs() as _ }
}

pub fn max_threads_per_block() -> u32 {
    unsafe { nppGetMaxThreadsPerBlock() as _ }
}

pub fn max_threads_per_sm() -> u32 {
    unsafe { nppGetMaxThreadsPerSM() as _ }
}

pub fn get_stream_ctx() -> Result<NppStreamContext> {
    let mut ctx = Default::default();
    unsafe { nppGetStreamContext(&mut ctx) }.result_with(ctx)
}

/// Get the globally set cuda stream
pub fn get_stream() -> cudaStream_t {
    unsafe { nppGetStream() }
}

/// Set the stream to use globally, except for methods that take a [NppStreamContext] as argument.
pub fn set_stream(stream: cudaStream_t) -> Result<()> {
    unsafe { nppSetStream(stream) }.result()
}

/// An opaque scratch buffer on device needed by some npp routines
#[derive(Debug, Clone)]
pub struct ScratchBuffer {
    // Device ptr !
    pub ptr: *mut c_void,
    pub len: usize,
}

impl ScratchBuffer {
    /// Uses stream ordered cuda malloc and free
    pub fn alloc(len: usize, stream: cudaStream_t) -> Result<Self> {
        let mut ptr = null_mut();
        unsafe { cudaMallocAsync(&mut ptr, len, stream).result_with(Self { ptr, len }) }
    }

    /// Size of the allocation on device
    pub fn len(&self) -> usize {
        self.len
    }

    /// The drop impl will free memory on the currently set NPP global stream *at the time of drop*.
    /// I suppose this can be impractical to deal with, so this function here will drop the buffer on an explicit stream.
    pub fn manual_drop(self, stream: cudaStream_t) -> Result<()> {
        unsafe {
            cudaFreeAsync(self.ptr, stream).result()?;
        }
        // Do not free a second time (with the drop impl)
        mem::forget(self);
        Ok(())
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFreeAsync(self.ptr, get_stream())
                .result()
                .expect("Could not free a scratch buffer");
        }
    }
}
