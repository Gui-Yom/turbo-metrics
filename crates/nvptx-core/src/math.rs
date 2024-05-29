// libdevice bindings
extern "C" {
    #[link_name = "__nv_fmaf"]
    pub fn fmaf(x: f32, y: f32, z: f32) -> f32;
    #[link_name = "__nv_cbrtf"]
    pub fn cbrtf(x: f32) -> f32;
    #[link_name = "__nv_powf"]
    pub fn powf(x: f32, y: f32) -> f32;
    #[link_name = "__nv_fabsf"]
    pub fn fabsf(x: f32) -> f32;
    #[link_name = "__nv_roundf"]
    pub fn roundf(x: f32) -> f32;
}

pub trait StdMathExt {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn abs(self) -> Self;
    fn round(self) -> Self;
}

impl StdMathExt for f32 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { fmaf(self, a, b) }
    }

    #[inline]
    fn cbrt(self) -> Self {
        unsafe { cbrtf(self) }
    }

    #[inline]
    fn powf(self, p: Self) -> Self {
        unsafe { powf(self, p) }
    }

    #[inline]
    fn abs(self) -> Self {
        unsafe { fabsf(self) }
    }

    #[inline]
    fn round(self) -> Self {
        unsafe { roundf(self) }
    }
}
