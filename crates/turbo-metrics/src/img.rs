use std::fmt::{Display, Formatter};
use std::{mem, slice};

#[derive(Debug, Copy, Clone)]
pub enum SampleType {
    U8,
    U16,
    F32,
}

#[derive(Debug, Copy, Clone)]
pub enum ColorRepr {
    RGB,
    YCbCr,
}

impl From<zune_core::colorspace::ColorSpace> for ColorRepr {
    fn from(value: zune_core::colorspace::ColorSpace) -> Self {
        use zune_core::colorspace::ColorSpace;
        match value {
            ColorSpace::RGB => ColorRepr::RGB,
            _ => todo!(),
        }
    }
}

impl From<image::ColorType> for ColorRepr {
    fn from(value: image::ColorType) -> Self {
        use image::ColorType;
        match value {
            ColorType::Rgb8 => ColorRepr::RGB,
            ColorType::Rgb16 => ColorRepr::RGB,
            ColorType::Rgb32F => ColorRepr::RGB,
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct CpuImg {
    pub sample_type: SampleType,
    pub colortype: ColorRepr,
    pub width: u32,
    pub height: u32,
    /// Type erased, packed rgb channels
    pub data: Vec<u8>,
}

impl Display for CpuImg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{}, {:?}, {:?}, sRGB",
            self.width, self.height, self.sample_type, self.colortype
        )
    }
}

pub fn reinterpret_vec<SourceTy, TargetTy>(mut vec: Vec<SourceTy>) -> Vec<TargetTy> {
    let size_t = size_of::<SourceTy>();
    let size_s = size_of::<TargetTy>();
    assert_ne!(
        size_s, 0,
        "Cannot reinterpret a Vec of non-zero sized types as a Vec of zero sized types."
    );
    // We must be able to split the given vec into appropriately sized chunks.
    assert_eq!(
        (vec.len() * size_t) % size_s,
        0,
        "Vec cannot be safely reinterpreted due to a misaligned size"
    );
    let nu_len = (vec.len() * size_t) / size_s;
    assert_eq!(
        (vec.capacity() * size_t) % size_s,
        0,
        "Vec cannot be safely reinterpreted due to a misaligned capacity"
    );
    let nu_capacity = (vec.capacity() * size_t) / size_s;
    let vec_ptr = vec.as_mut_ptr();
    let nu_vec = unsafe { Vec::from_raw_parts(vec_ptr as *mut TargetTy, nu_len, nu_capacity) };

    mem::forget(vec);
    nu_vec
}

pub fn reinterpret_slice<SourceTy, TargetTy>(slice: &[SourceTy]) -> &[TargetTy] {
    let size_t = size_of::<SourceTy>();
    let size_s = size_of::<TargetTy>();

    assert_ne!(
        size_s, 0,
        "Cannot reinterpret a slice of non-zero sized types as a slice of zero sized types."
    );
    // We must be able to split the given slice into appropriately sized chunks.
    assert_eq!(
        (slice.len() * size_t) % size_s,
        0,
        "Slice cannot be safely reinterpreted due to a misaligned size"
    );
    let nu_len = (slice.len() * size_t) / size_s;
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const TargetTy, nu_len) }
}
