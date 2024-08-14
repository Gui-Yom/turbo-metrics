use std::ffi::CStr;
use std::ops::RangeInclusive;
use std::ptr::{null_mut, NonNull};

pub use sys::VmafPoolingMethod;

mod sys {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct Vmaf(pub(crate) NonNull<sys::VmafContext>);

impl Drop for Vmaf {
    fn drop(&mut self) {
        unsafe {
            sys::vmaf_close(self.0.as_ptr());
        }
    }
}

impl Vmaf {
    pub fn new() -> Self {
        let mut ptr = null_mut();
        unsafe {
            sys::vmaf_init(
                &mut ptr,
                sys::VmafConfiguration {
                    log_level: sys::VmafLogLevel::VMAF_LOG_LEVEL_DEBUG,
                    n_threads: 0,
                    n_subsample: 0,
                    cpumask: 0,
                    gpumask: 0,
                },
            );
        }
        Self(NonNull::new(ptr).unwrap())
    }

    pub fn use_features_from_model(&mut self, model: &VmafModel) {
        unsafe {
            sys::vmaf_use_features_from_model(self.0.as_ptr(), model.0.as_ptr());
        }
    }

    pub fn use_feature(&mut self, name: &CStr) {
        let mut dict = null_mut();

        unsafe {
            sys::vmaf_feature_dictionary_set(&mut dict, c"".as_ptr(), c"".as_ptr());
            sys::vmaf_use_feature(self.0.as_ptr(), name.as_ptr(), dict);
        }
    }

    pub fn read_pictures(&mut self, ref_: &mut VmafPicture, dis: &mut VmafPicture, index: u32) {
        unsafe {
            sys::vmaf_read_pictures(self.0.as_ptr(), &mut ref_.0, &mut dis.0, index);
        }
    }

    pub fn flush(&mut self) {
        unsafe {
            sys::vmaf_read_pictures(self.0.as_ptr(), null_mut(), null_mut(), 0);
        }
    }

    pub fn score(&mut self, model: &VmafModel, index: u32) -> f64 {
        let mut score = 0.0;
        unsafe {
            sys::vmaf_score_at_index(self.0.as_ptr(), model.0.as_ptr(), &mut score, index);
        }
        score
    }

    pub fn score_pooled(
        &mut self,
        model: &VmafModel,
        method: VmafPoolingMethod,
        range: RangeInclusive<u32>,
    ) -> f64 {
        let mut score = 0.0;
        unsafe {
            sys::vmaf_score_pooled(
                self.0.as_ptr(),
                model.0.as_ptr(),
                method,
                &mut score,
                *range.start(),
                *range.end(),
            );
        }
        score
    }
}

pub struct VmafModel(pub(crate) NonNull<sys::VmafModel>);

impl Drop for VmafModel {
    fn drop(&mut self) {
        unsafe {
            sys::vmaf_model_destroy(self.0.as_ptr());
        }
    }
}

impl VmafModel {
    pub fn load() -> Self {
        let mut model = null_mut();
        let mut cfg = sys::VmafModelConfig {
            name: c"vmaf_v0.6.1".as_ptr(),
            flags: 0,
        };
        let version = c"vmaf_v0.6.1";
        unsafe {
            sys::vmaf_model_load(&mut model, &mut cfg, version.as_ptr());
        }
        Self(NonNull::new(model).unwrap())
    }

    pub fn available_builtin(&self) {}
}

#[derive(Debug)]
pub struct VmafPicture(pub(crate) sys::VmafPicture);

impl Drop for VmafPicture {
    fn drop(&mut self) {
        unsafe {
            sys::vmaf_picture_unref(&mut self.0);
        }
    }
}

impl VmafPicture {
    pub fn new() -> Self {
        let mut picture = sys::VmafPicture::default();
        unsafe {
            sys::vmaf_picture_alloc(
                &mut picture,
                sys::VmafPixelFormat::VMAF_PIX_FMT_YUV420P,
                8,
                1920,
                1080,
            );
        }
        Self(picture)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vmaf, VmafModel, VmafPicture};

    #[test]
    fn it_works() {
        let mut vmaf = Vmaf::new();
        let model = VmafModel::load();
        vmaf.use_features_from_model(&model);
        let mut ref_ = VmafPicture::new();
        let mut dis = VmafPicture::new();
        vmaf.read_pictures(&mut ref_, &mut dis, 0);
        vmaf.flush();
        dbg!(vmaf.score(&model, 0));
    }
}
