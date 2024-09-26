use crate::sys;
use core::ffi::c_void;
use widestring::{WideCStr, WideCString};

pub trait AMFInterface {
    fn acquire(&self) -> sys::amf_long;
    fn release(&self) -> sys::amf_long;
    fn query_interface(
        &self,
        guid: &sys::AMFGuid,
        interface: &mut *mut c_void,
    ) -> amf_sys::Result<()>;
}

#[macro_export]
macro_rules! impl_amf_interface {
    ($t:ty) => {
        impl $crate::core::AMFInterface for $t {
            fn acquire(&self) -> $crate::sys::amf_long {
                unsafe { self.vtbl().Acquire.unwrap()(self.0.as_ptr()) }
            }

            fn release(&self) -> $crate::sys::amf_long {
                unsafe { self.vtbl().Release.unwrap()(self.0.as_ptr()) }
            }

            fn query_interface(
                &self,
                guid: &$crate::sys::AMFGuid,
                interface: &mut *mut ::core::ffi::c_void,
            ) -> amf_sys::Result<()> {
                unsafe {
                    self.vtbl().QueryInterface.unwrap()(self.0.as_ptr(), guid, interface).result()
                }
            }
        }
    };
}

pub trait PropertyStorage: AMFInterface {
    fn set_property(&self, name: &WideCStr, value: sys::AMFVariantStruct) -> amf_sys::Result<()>;
    fn get_property(&self, name: &WideCStr) -> amf_sys::Result<sys::AMFVariantStruct>;
    fn has_property(&self, name: &WideCStr) -> bool;
    fn get_property_count(&self) -> usize;
    fn get_property_at(
        &self,
        index: usize,
    ) -> amf_sys::Result<(WideCString, sys::AMFVariantStruct)>;
    fn clear(&self) -> amf_sys::Result<()>;
    // fn add_to(
    //     &self,
    //     other: &impl PropertyStorage,
    //     overwrite: bool,
    //     deep: bool,
    // ) -> amf_sys::Result<()>;
    // fn copy_to(&self, other: &impl PropertyStorage, deep: bool) -> amf_sys::Result<()>;
    fn add_observer(&self, observer: &mut sys::AMFPropertyStorageObserver);
    fn remove_observer(&self, observer: &mut sys::AMFPropertyStorageObserver);
}

#[macro_export]
macro_rules! impl_property_storage {
    ($t:ty) => {
        impl $crate::core::PropertyStorage for $t {
            fn set_property(
                &self,
                name: &widestring::WideCStr,
                value: $crate::sys::AMFVariantStruct,
            ) -> $crate::Result<()> {
                unsafe {
                    self.vtbl().SetProperty.unwrap()(self.0.as_ptr(), name.as_ptr(), value).result()
                }
            }

            fn get_property(
                &self,
                name: &widestring::WideCStr,
            ) -> $crate::Result<$crate::sys::AMFVariantStruct> {
                let mut value = $crate::sys::AMFVariantStruct::default();
                unsafe {
                    self.vtbl().GetProperty.unwrap()(self.0.as_ptr(), name.as_ptr(), &mut value)
                        .result_with(value)
                }
            }

            fn has_property(&self, name: &widestring::WideCStr) -> bool {
                unsafe { self.vtbl().HasProperty.unwrap()(self.0.as_ptr(), name.as_ptr()) != 0 }
            }

            fn get_property_count(&self) -> usize {
                unsafe { self.vtbl().GetPropertyCount.unwrap()(self.0.as_ptr()) }
            }

            fn get_property_at(
                &self,
                index: usize,
            ) -> $crate::Result<(widestring::WideCString, $crate::sys::AMFVariantStruct)> {
                let mut name = vec![0; 64];
                let mut value = $crate::sys::AMFVariantStruct::default();
                unsafe {
                    self.vtbl().GetPropertyAt.unwrap()(
                        self.0.as_ptr(),
                        index,
                        name.as_mut_ptr(),
                        64,
                        &mut value,
                    )
                    .result_with((widestring::WideCString::from_vec(name).unwrap(), value))
                }
            }

            fn clear(&self) -> $crate::Result<()> {
                unsafe { self.vtbl().Clear.unwrap()(self.0.as_ptr()).result() }
            }

            // fn add_to(
            //     &self,
            //     other: &impl PropertyStorage,
            //     overwrite: bool,
            //     deep: bool,
            // ) -> Result<()> {
            //     self.vtbl().AddTo.unwrap()(self.0.as_ptr(), other, overwrite as _, deep as _)
            //         .result()
            // }
            //
            // fn copy_to(&self, other: &impl PropertyStorage, deep: bool) -> Result<()> {
            //     self.vtbl().CopyTo.unwrap()(self.0.as_ptr(), other, deep as _).result()
            // }

            fn add_observer(&self, observer: &mut $crate::sys::AMFPropertyStorageObserver) {
                unsafe {
                    self.vtbl().AddObserver.unwrap()(self.0.as_ptr(), observer);
                }
            }

            fn remove_observer(&self, observer: &mut $crate::sys::AMFPropertyStorageObserver) {
                unsafe {
                    self.vtbl().RemoveObserver.unwrap()(self.0.as_ptr(), observer);
                }
            }
        }
    };
}
