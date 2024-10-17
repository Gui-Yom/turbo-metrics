// libdevice bindings
extern "C" {
    #[link_name = "__nv_fmaf"]
    pub fn fmaf(x: f32, y: f32, z: f32) -> f32;
    #[link_name = "__nv_fma"]
    pub fn fma(x: f64, y: f64, z: f64) -> f64;
    #[link_name = "__nv_cbrtf"]
    pub fn cbrtf(x: f32) -> f32;
    #[link_name = "__nv_cbrt"]
    pub fn cbrt(x: f64) -> f64;
    #[link_name = "__nv_powf"]
    pub fn powf(x: f32, y: f32) -> f32;
    #[link_name = "__nv_pow"]
    pub fn pow(x: f64, y: f64) -> f64;
    #[link_name = "__nv_fast_powf"]
    pub fn fast_powf(x: f32, y: f32) -> f32;
    #[link_name = "__nv_fabsf"]
    pub fn fabsf(x: f32) -> f32;
    #[link_name = "__nv_fabs"]
    pub fn fabs(x: f64) -> f64;
    #[link_name = "__nv_roundf"]
    pub fn roundf(x: f32) -> f32;
    #[link_name = "__nv_float2uint_rn"]
    pub fn float2uint_rn(x: f32) -> u32;
    #[link_name = "__nv_rsqrt"]
    pub fn rsqrt(x: f64) -> f64;
    #[link_name = "__nv_rsqrtf"]
    pub fn rsqrtf(x: f32) -> f32;
}

pub trait StdMathExt {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn powf_fast(self, p: Self) -> Self;
    fn abs(self) -> Self;
    fn round(self) -> Self;
    fn round_to_u32(self) -> u32;
    fn round_to_u16(self) -> u16;
    fn round_to_u8(self) -> u8;
    fn rsqrt(self) -> Self;
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
    fn powf_fast(self, p: Self) -> Self {
        unsafe { fast_powf(self, p) }
    }

    #[inline]
    fn abs(self) -> Self {
        unsafe { fabsf(self) }
    }

    #[inline]
    fn round(self) -> Self {
        unsafe { roundf(self) }
    }
    #[inline]
    fn round_to_u32(self) -> u32 {
        unsafe { float2uint_rn(self) }
    }

    #[inline]
    fn round_to_u16(self) -> u16 {
        self.round_to_u32() as u16
    }

    #[inline]
    fn round_to_u8(self) -> u8 {
        self.round_to_u32() as u8
    }

    #[inline]
    fn rsqrt(self) -> Self {
        unsafe { rsqrtf(self) }
    }
}

impl StdMathExt for f64 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { fma(self, a, b) }
    }

    #[inline]
    fn cbrt(self) -> Self {
        unsafe { cbrt(self) }
    }

    #[inline]
    fn powf(self, p: Self) -> Self {
        unsafe { pow(self, p) }
    }

    #[inline]
    fn powf_fast(self, p: Self) -> Self {
        todo!()
    }

    #[inline]
    fn abs(self) -> Self {
        unsafe { fabs(self) }
    }

    #[inline]
    fn round(self) -> Self {
        todo!()
    }

    #[inline]
    fn round_to_u32(self) -> u32 {
        todo!()
    }

    #[inline]
    fn round_to_u16(self) -> u16 {
        todo!()
    }

    #[inline]
    fn round_to_u8(self) -> u8 {
        todo!()
    }

    #[inline]
    fn rsqrt(self) -> Self {
        unsafe { rsqrt(self) }
    }
}

pub trait RoundFromf32 {
    fn round(v: f32) -> Self;
}

impl RoundFromf32 for u32 {
    #[inline]
    fn round(v: f32) -> Self {
        v.round_to_u32()
    }
}

impl RoundFromf32 for u16 {
    #[inline]
    fn round(v: f32) -> Self {
        v.round_to_u16()
    }
}

impl RoundFromf32 for u8 {
    #[inline]
    fn round(v: f32) -> Self {
        v.round_to_u8()
    }
}
