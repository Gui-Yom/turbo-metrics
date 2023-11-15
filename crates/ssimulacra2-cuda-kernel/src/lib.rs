#![no_std]
#![feature(abi_ptx)]
#![feature(stdsimd)]
// #![feature(core_intrisics)]

extern crate nvptx_panic_handler;

// libdevice bindings
extern "C" {
    #[link_name = "__nv_fmaf"]
    fn fma(x: f32, y: f32, z: f32) -> f32;
    #[link_name = "__nv_cbrtf"]
    fn cbrt(x: f32) -> f32;
    #[link_name = "__nv_powf"]
    fn powf(x: f32, y: f32) -> f32;
}

unsafe fn coords_1d() -> usize {
    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let bdx = core::arch::nvptx::_block_dim_x() as usize;
    (bx * bdx + tx)
}

unsafe fn coords_2d() -> (usize, usize) {
    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let ty = core::arch::nvptx::_thread_idx_y() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let by = core::arch::nvptx::_block_idx_y() as usize;
    let bdx = core::arch::nvptx::_block_dim_x() as usize;
    let bdy = core::arch::nvptx::_block_dim_y() as usize;
    (by * bdy + ty, bx * bdx + tx)
}

// Adjusted for continuity of first derivative.
const SRGB_ALPHA: f32 = 1.055_010_7;
const SRGB_BETA: f32 = 0.003_041_282_5;

/// Transfer function srgb to linear
unsafe fn srgb_eotf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < 12.92 * SRGB_BETA {
        x / 12.92
    } else {
        powf((x + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA, 2.4)
    }
}

#[no_mangle]
unsafe extern "ptx-kernel" fn plane_srgb_to_linear(len: usize, src: *const f32, dst: *mut f32) {
    let i = coords_1d();
    if i < len {
        *dst.add(i) = srgb_eotf(*src.add(i));
    }
}

const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;

const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;

const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;

const K_B0: f32 = 0.003_793_073_4_f32;
const K_B0_ROOT: f32 = 0.1559542025327239180319220163705;
const K_B1: f32 = K_B0;
const K_B2: f32 = K_B0;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];
const OPSIN_ABSORBANCE_BIAS_ROOT: [f32; 3] = [K_B0_ROOT, K_B0_ROOT, K_B0_ROOT];
const INVERSE_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    11.031_567_f32,
    -9.866_944_f32,
    -0.164_622_99_f32,
    -3.254_147_3_f32,
    4.418_770_3_f32,
    -0.164_622_99_f32,
    -3.658_851_4_f32,
    2.712_923_f32,
    1.945_928_2_f32,
];
const NEG_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [-K_B0, -K_B1, -K_B2];

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
/// Get all components in more or less 0..1 range
pub unsafe fn px_linear_rgb_to_positive_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (mut rg, mut gr, mut b) = opsin_absorbance(r, g, b);
    rg = cbrt(rg.max(0.0)) - OPSIN_ABSORBANCE_BIAS_ROOT[0];
    gr = cbrt(gr.max(0.0)) - OPSIN_ABSORBANCE_BIAS_ROOT[1];
    b = cbrt(b.max(0.0)) - OPSIN_ABSORBANCE_BIAS_ROOT[2];
    let x = 0.5f32 * (rg - gr);
    let y = 0.5f32 * (rg + gr);
    // Get all components in more or less 0..1 range
    (fma(x, 14.0, 0.42), y + 0.01, b - y + 0.55)
}

#[inline]
unsafe fn opsin_absorbance(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        fma(
            OPSIN_ABSORBANCE_MATRIX[0],
            r,
            fma(
                OPSIN_ABSORBANCE_MATRIX[1],
                g,
                fma(OPSIN_ABSORBANCE_MATRIX[2], b, OPSIN_ABSORBANCE_BIAS[0]),
            ),
        ),
        fma(
            OPSIN_ABSORBANCE_MATRIX[3],
            r,
            fma(
                OPSIN_ABSORBANCE_MATRIX[4],
                g,
                fma(OPSIN_ABSORBANCE_MATRIX[5], b, OPSIN_ABSORBANCE_BIAS[1]),
            ),
        ),
        fma(
            OPSIN_ABSORBANCE_MATRIX[6],
            r,
            fma(
                OPSIN_ABSORBANCE_MATRIX[7],
                g,
                fma(OPSIN_ABSORBANCE_MATRIX[8], b, OPSIN_ABSORBANCE_BIAS[2]),
            ),
        ),
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_xyb(
    len: usize,
    rp: *const f32,
    gp: *const f32,
    bp: *const f32,
    xp: *mut f32,
    yp: *mut f32,
    bp2: *mut f32,
) {
    let i = core::arch::nvptx::_thread_idx_x() as usize;
    if i < len {
        let (x, y, b) = px_linear_rgb_to_positive_xyb(*rp.add(i), *gp.add(i), *bp.add(i));
        *xp.add(i) = x;
        *yp.add(i) = y;
        *bp2.add(i) = b;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn downscale_by_2(
    iw: usize,
    ih: usize,
    ow: usize,
    oh: usize,
    pin: *const f32,
    pout: *mut f32,
) {
    const SCALE: usize = 2;
    const NORMALIZE: f32 = 1f32 / (SCALE * SCALE) as f32;

    let (ox, oy) = coords_2d();

    if ox < ow && oy < oh {
        let mut sum = 0f32;
        for iy in 0..SCALE {
            for ix in 0..SCALE {
                let x = (ox * SCALE + ix).min(iw - 1);
                let y = (oy * SCALE + iy).min(ih - 1);

                sum += *pin.add(y * iw + x);
            }
        }
        *pout.add(oy * ow + ox) = sum * NORMALIZE;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mul_planes(
    w: usize,
    h: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
) {
    let (x, y) = coords_2d();
    if x < w && y < h {
        *c.add(y * w + x) = *a.add(y * w + x) * *b.add(y * w + x);
    }
}
