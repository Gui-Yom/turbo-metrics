use crate::{sys, CuFunction};
use cudarse_driver_sys::cuModuleLoadData;
use std::ffi::CString;
use std::path::Path;
use std::ptr::null_mut;
use sys::{
    cuModuleEnumerateFunctions, cuModuleGetFunction, cuModuleGetFunctionCount, cuModuleLoad,
    cuModuleUnload, CuResult,
};

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

    pub fn load_ptx(ptx: &str) -> CuResult<Self> {
        let mut cu_module = null_mut();
        let null_terminated = CString::new(ptx.as_bytes()).unwrap();
        unsafe {
            cuModuleLoadData(&mut cu_module, null_terminated.as_ptr().cast()).result()?;
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
