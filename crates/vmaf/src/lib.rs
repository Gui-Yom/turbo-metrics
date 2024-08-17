use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ops::RangeInclusive;
use std::ptr::{null_mut, NonNull};

mod sys {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub enum BuiltinModel {}

pub fn version() -> &'static CStr {
    unsafe { CStr::from_ptr(sys::vmaf_version()) }
}

pub struct Vmaf(pub(crate) NonNull<sys::VmafContext>);

impl Drop for Vmaf {
    fn drop(&mut self) {
        unsafe {
            assert_eq!(sys::vmaf_close(self.0.as_ptr()), 0);
        }
    }
}

impl Vmaf {
    pub fn new() -> Self {
        let mut ptr = null_mut();
        unsafe {
            assert_eq!(
                sys::vmaf_init(
                    &mut ptr,
                    sys::VmafConfiguration {
                        log_level: sys::VmafLogLevel::VMAF_LOG_LEVEL_DEBUG,
                        n_threads: 0,
                        n_subsample: 0,
                        cpumask: 0,
                        gpumask: 0,
                    },
                ),
                0
            );
        }
        Self(NonNull::new(ptr).unwrap())
    }

    /// Initialize the vmaf context with CUDA computing.
    pub fn with_cuda(self) -> Self {
        let mut state = null_mut();
        unsafe {
            assert_eq!(
                sys::vmaf_cuda_state_init(
                    &mut state,
                    sys::VmafCudaConfiguration { cu_ctx: null_mut() },
                ),
                0
            );
            assert_eq!(sys::vmaf_cuda_import_state(self.0.as_ptr(), state), 0);
        }
        self
    }

    /// Preallocate cuda pictures.
    pub fn cuda_preallocate_pictures(
        &mut self,
        width: u32,
        height: u32,
        bpc: u32,
        pix_fmt: sys::VmafPixelFormat,
    ) {
        unsafe {
            assert_eq!(sys::vmaf_cuda_preallocate_pictures(self.0.as_ptr(), sys::VmafCudaPictureConfiguration {
                pic_params: sys::VmafCudaPictureConfiguration__bindgen_ty_1 {
                    w: width,
                    h: height,
                    bpc,
                    pix_fmt,
                },
                pic_prealloc_method: sys::VmafCudaPicturePreallocationMethod::VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
            }), 0);
        }
    }

    /// Fetch a preallocated cuda picture.
    pub fn cuda_fetch_picture(&mut self) -> VmafPictureCuda {
        let mut pic = sys::VmafPicture::default();
        unsafe {
            assert_eq!(
                sys::vmaf_cuda_fetch_preallocated_picture(self.0.as_ptr(), &mut pic),
                0
            );
        }
        VmafPictureCuda(pic)
    }

    pub fn use_features_from_model(&mut self, model: &VmafModel) {
        unsafe {
            assert_eq!(
                sys::vmaf_use_features_from_model(self.0.as_ptr(), model.0.as_ptr()),
                0
            );
        }
    }

    pub fn use_feature(&mut self, name: &CStr, options: &HashMap<CString, CString>) {
        let mut dict = null_mut();
        for (k, v) in options {
            unsafe {
                assert_eq!(
                    sys::vmaf_feature_dictionary_set(&mut dict, k.as_ptr(), v.as_ptr()),
                    0
                );
            }
        }

        unsafe {
            assert_eq!(
                sys::vmaf_use_feature(self.0.as_ptr(), name.as_ptr(), dict),
                0
            );
        }
    }

    pub fn read_pictures(&mut self, ref_: &mut VmafPicture, dis: &mut VmafPicture, index: u32) {
        unsafe {
            assert_eq!(
                sys::vmaf_read_pictures(self.0.as_ptr(), &mut ref_.0, &mut dis.0, index),
                0
            );
        }
    }

    pub fn read_pictures_cuda(
        &mut self,
        ref_: &mut VmafPictureCuda,
        dis: &mut VmafPictureCuda,
        index: u32,
    ) {
        unsafe {
            assert_eq!(
                sys::vmaf_read_pictures(self.0.as_ptr(), &mut ref_.0, &mut dis.0, index),
                0
            );
        }
    }

    pub fn flush(&mut self) {
        unsafe {
            assert_eq!(
                sys::vmaf_read_pictures(self.0.as_ptr(), null_mut(), null_mut(), 0),
                0
            );
        }
    }

    pub fn score(&mut self, model: &VmafModel, index: u32) -> f64 {
        let mut score = 0.0;
        unsafe {
            assert_eq!(
                sys::vmaf_score_at_index(self.0.as_ptr(), model.0.as_ptr(), &mut score, index),
                0
            );
        }
        score
    }

    pub fn score_pooled(
        &mut self,
        model: &VmafModel,
        method: sys::VmafPoolingMethod,
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

    pub fn feature_score(&mut self, feature: &CStr, index: u32) -> f64 {
        let mut score = 0.0;
        unsafe {
            sys::vmaf_feature_score_at_index(self.0.as_ptr(), feature.as_ptr(), &mut score, index);
        }
        score
    }

    pub fn feature_score_pooled(
        &mut self,
        feature: &CStr,
        method: sys::VmafPoolingMethod,
        range: RangeInclusive<u32>,
    ) -> f64 {
        let mut score = 0.0;
        unsafe {
            sys::vmaf_feature_score_pooled(
                self.0.as_ptr(),
                feature.as_ptr(),
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
            assert_eq!(
                sys::vmaf_model_load(&mut model, &mut cfg, version.as_ptr()),
                0
            );
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
            assert_eq!(sys::vmaf_picture_unref(&mut self.0), 0);
        }
    }
}

impl VmafPicture {
    pub fn new() -> Self {
        let mut picture = sys::VmafPicture::default();
        unsafe {
            assert_eq!(
                sys::vmaf_picture_alloc(
                    &mut picture,
                    sys::VmafPixelFormat::VMAF_PIX_FMT_YUV420P,
                    8,
                    1920,
                    1080,
                ),
                0
            );
        }
        Self(picture)
    }
}

struct VmafPictureCuda(pub(crate) sys::VmafPicture);

#[cfg(test)]
mod tests {
    use crate::sys::VmafPixelFormat;
    use crate::{version, Vmaf, VmafModel};
    use std::collections::HashMap;

    #[test]
    fn it_works() {
        println!("VMAF version : {:?}", version());
        let mut vmaf = Vmaf::new().with_cuda();
        vmaf.cuda_preallocate_pictures(1920, 1080, 8, VmafPixelFormat::VMAF_PIX_FMT_YUV420P);
        let model = VmafModel::load();
        vmaf.use_features_from_model(&model);
        vmaf.use_feature(c"cambi", &HashMap::new());
        let mut ref_ = vmaf.cuda_fetch_picture();
        let mut dis = vmaf.cuda_fetch_picture();
        vmaf.read_pictures_cuda(&mut ref_, &mut dis, 0);
        vmaf.flush();
        dbg!(vmaf.score(&model, 0));
        dbg!(vmaf.feature_score(c"cambi", 0));
    }
}
