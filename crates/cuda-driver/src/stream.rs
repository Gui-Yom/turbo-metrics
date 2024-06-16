use std::ffi::c_void;
use std::ptr::null_mut;

use sys::{
    cuStreamBeginCapture_v2, cuStreamCreate, cuStreamDestroy_v2, cuStreamEndCapture, cuStreamQuery,
    cuStreamSynchronize, cuStreamWaitEvent, CUstreamCaptureMode_enum, CUstream_flags, CuError,
    CuResult,
};

use crate::{sys, CuEvent, CuGraph};

#[repr(transparent)]
pub struct CuStream(pub(crate) sys::CUstream);

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

    pub fn begin_capture(&self) -> CuResult<()> {
        unsafe {
            cuStreamBeginCapture_v2(
                self.0,
                CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL,
            )
            .result()
        }
    }

    pub fn end_capture(&self) -> CuResult<CuGraph> {
        let mut graph = null_mut();
        unsafe { cuStreamEndCapture(self.0, &mut graph).result()? };
        if graph.is_null() {
            panic!("Invalid graph")
        } else {
            Ok(CuGraph(graph))
        }
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
