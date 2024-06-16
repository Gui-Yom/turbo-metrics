use crate::{sys, CuStream};
use std::ffi::CString;
use std::path::Path;
use std::ptr::null_mut;
use sys::{
    cuGraphDebugDotPrint, cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags,
    cuGraphLaunch, CuResult,
};

pub struct CuGraph(pub(crate) sys::CUgraph);

impl CuGraph {
    pub fn dot(&self, path: impl AsRef<Path>) -> CuResult<()> {
        let path = CString::new(path.as_ref().as_os_str().as_encoded_bytes()).unwrap();
        unsafe { cuGraphDebugDotPrint(self.0, path.as_ptr(), 0).result() }
    }

    pub fn instantiate(&self) -> CuResult<CuGraphExec> {
        let mut exec = null_mut();
        unsafe {
            cuGraphInstantiateWithFlags(&mut exec, self.0, 0).result()?;
        }
        Ok(CuGraphExec(exec))
    }
}

impl Drop for CuGraph {
    fn drop(&mut self) {
        unsafe { cuGraphDestroy(self.0).result().unwrap() }
    }
}

pub struct CuGraphExec(sys::CUgraphExec);

impl CuGraphExec {
    pub fn launch(&self, stream: &CuStream) -> CuResult<()> {
        unsafe { cuGraphLaunch(self.0, stream.0).result() }
    }
}

impl Drop for CuGraphExec {
    fn drop(&mut self) {
        unsafe { cuGraphExecDestroy(self.0).result().unwrap() }
    }
}
