pub use amf_sys as sys;
use libloading::{Library, Symbol};
use std::ptr::{null_mut, NonNull};
use std::sync::LazyLock;
use widestring::WideCStr;

pub use sys::Result;

mod core;

static AMF_LIB: LazyLock<Library, fn() -> Library> =
    LazyLock::new(|| unsafe { Library::new(sys::AMF_DLL_NAME).expect("Can't load AMF library") });

pub fn split_version(version: sys::amf_uint64) -> (u16, u16, u16, u16) {
    let build = (version & 0xFFFF) as u16;
    let subminor = ((version >> 16) & 0xFFFF) as u16;
    let minor = ((version >> 32) & 0xFFFF) as u16;
    let major = ((version >> 48) & 0xFFFF) as u16;
    (major, minor, subminor, build)
}

pub fn runtime_version() -> Result<u64> {
    unsafe {
        let query_version: Symbol<sys::AMFQueryVersion_Fn> = AMF_LIB
            .get(sys::AMF_QUERY_VERSION_FUNCTION_NAME.to_bytes_with_nul())
            .expect("Can't find query version symbol");
        let mut version = 0;
        query_version.unwrap()(&mut version).result_with(version)
    }
}

pub struct AMFFactory(NonNull<sys::AMFFactory>);

impl AMFFactory {
    pub fn init(version: u64) -> Result<Self> {
        unsafe {
            let init: Symbol<sys::AMFInit_Fn> = AMF_LIB
                .get(sys::AMF_INIT_FUNCTION_NAME.to_bytes_with_nul())
                .expect("Can't find init symbol");
            let mut factory = null_mut();
            init.unwrap()(version, &mut factory).result_with(AMFFactory(
                NonNull::new(factory).expect("init returned a null ptr"),
            ))
        }
    }

    fn vtbl(&self) -> &sys::AMFFactoryVtbl {
        unsafe { self.0.read().pVtbl.as_ref().unwrap() }
    }

    pub fn trace(&self) -> Result<AMFTrace> {
        unsafe {
            let mut trace = null_mut();
            self.vtbl().GetTrace.unwrap()(self.0.as_ptr(), &mut trace)
                .result_with(AMFTrace(NonNull::new(trace).expect("null ptr")))
        }
    }

    pub fn debug(&self) -> Result<AMFDebug> {
        unsafe {
            let mut debug = null_mut();
            self.vtbl().GetDebug.unwrap()(self.0.as_ptr(), &mut debug)
                .result_with(AMFDebug(NonNull::new(debug).expect("null ptr")))
        }
    }

    pub fn programs(&self) -> Result<AMFPrograms> {
        unsafe {
            let mut programs = null_mut();
            self.vtbl().GetPrograms.unwrap()(self.0.as_ptr(), &mut programs)
                .result_with(AMFPrograms(NonNull::new(programs).expect("null ptr")))
        }
    }

    pub fn create_context(&self) -> Result<AMFContext> {
        unsafe {
            let mut context = null_mut();
            self.vtbl().CreateContext.unwrap()(self.0.as_ptr(), &mut context)
                .result_with(AMFContext(NonNull::new(context).expect("null ptr")))
        }
    }

    pub fn create_component(&self, ctx: &AMFContext, id: &WideCStr) -> Result<AMFComponent> {
        unsafe {
            // let wid = WideCString::from_str(id).unwrap();
            let mut component = null_mut();
            self.vtbl().CreateComponent.unwrap()(
                self.0.as_ptr(),
                ctx.0.as_ptr(),
                id.as_ptr(),
                &mut component,
            )
            .result_with(AMFComponent(NonNull::new(component).expect("null ptr")))
        }
    }

    pub fn cache_folder(&self) -> Result<String> {
        unsafe {
            let ptr = self.vtbl().GetCacheFolder.unwrap()(self.0.as_ptr());
            Ok(WideCStr::from_ptr_str(ptr).to_string_lossy())
        }
    }
}

pub struct AMFTrace(NonNull<sys::AMFTrace>);
pub struct AMFDebug(NonNull<sys::AMFDebug>);
pub struct AMFPrograms(NonNull<sys::AMFPrograms>);
pub struct AMFContext(NonNull<sys::AMFContext>);
impl_amf_interface!(AMFContext);
impl_property_storage!(AMFContext);

impl AMFContext {
    fn vtbl(&self) -> &sys::AMFContextVtbl {
        unsafe { self.0.read().pVtbl.as_ref().unwrap() }
    }
}

impl Drop for AMFContext {
    fn drop(&mut self) {
        unsafe {
            self.vtbl().Terminate.unwrap()(self.0.as_ptr());
        }
    }
}
pub struct AMFComponent(NonNull<sys::AMFComponent>);
impl_amf_interface!(AMFComponent);
impl_property_storage!(AMFComponent);

impl AMFComponent {
    fn vtbl(&self) -> &sys::AMFComponentVtbl {
        unsafe { self.0.read().pVtbl.as_ref().unwrap() }
    }

    fn init(&self, format: sys::AMF_SURFACE_FORMAT, width: i32, height: i32) {
        unsafe {
            self.vtbl().Init.unwrap()(self.0.as_ptr(), format, width, height);
        }
    }
}

impl Drop for AMFComponent {
    fn drop(&mut self) {
        unsafe {
            self.vtbl().Terminate.unwrap()(self.0.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{runtime_version, split_version, sys, AMFFactory, Result};

    #[test]
    fn it_works() -> Result<()> {
        let version = runtime_version()?;
        dbg!(split_version(version));
        let factory = AMFFactory::init(version)?;
        dbg!(factory.cache_folder())?;
        let ctx = factory.create_context()?;
        let dec = factory.create_component(&ctx, sys::AMFVideoDecoderHW_AV1)?;
        Ok(())
    }
}
