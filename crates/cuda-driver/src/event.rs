use crate::stream::CuStream;
use crate::sys;
use std::ptr::null_mut;
use sys::{
    cuEventCreate, cuEventDestroy_v2, cuEventElapsedTime, cuEventQuery, cuEventRecord,
    cuEventSynchronize, CuError, CuResult,
};

/// An event is a point in a stream.
#[repr(transparent)]
pub struct CuEvent(pub(crate) sys::CUevent);

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
