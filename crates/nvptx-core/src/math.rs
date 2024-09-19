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
    #[link_name = "__nv_float2uint_rn"]
    pub fn float2uint_rn(x: f32) -> u32;
    #[link_name = "__nv_float2half_rn"]
    pub fn float2half_rn(x: f32) -> u16;
}

pub trait StdMathExt {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn abs(self) -> Self;
    fn round(self) -> Self;
    fn round_to_u32(self) -> u32;
    fn round_to_u16(self) -> u16;
    fn round_to_u8(self) -> u8;
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
    #[inline]
    fn round_to_u32(self) -> u32 {
        unsafe { float2uint_rn(self) }
    }

    #[inline]
    fn round_to_u16(self) -> u16 {
        unsafe { float2half_rn(self) }
    }

    #[inline]
    fn round_to_u8(self) -> u8 {
        self.round_to_u16() as u8
    }
}

pub trait RoundFromf32 {
    fn round(v: f32) -> Self;
}

impl RoundFromf32 for u32 {
    fn round(v: f32) -> Self {
        v.round_to_u32()
    }
}

impl RoundFromf32 for u16 {
    fn round(v: f32) -> Self {
        v.round_to_u16()
    }
}

impl RoundFromf32 for u8 {
    fn round(v: f32) -> Self {
        v.round_to_u8()
    }
}
