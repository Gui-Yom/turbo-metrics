#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
// #![feature(core_intrisics)]

extern crate nvptx_panic_handler;

use core::mem;

// libdevice bindings
extern "C" {
    #[link_name = "__nv_fmaf"]
    fn fma(x: f32, y: f32, z: f32) -> f32;
    #[link_name = "__nv_cbrtf"]
    fn cbrt(x: f32) -> f32;
    #[link_name = "__nv_powf"]
    fn powf(x: f32, y: f32) -> f32;
    #[link_name = "__nv_fabsf"]
    fn abs(x: f32) -> f32;
}

unsafe fn coords_1d() -> usize {
    let tx = core::arch::nvptx::_thread_idx_x() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let bdx = core::arch::nvptx::_block_dim_x() as usize;
    bx * bdx + tx
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
pub unsafe extern "ptx-kernel" fn plane_linear_to_xyb(
    len: usize,
    rp: *const f32,
    gp: *const f32,
    bp: *const f32,
    xp: *mut f32,
    yp: *mut f32,
    bp2: *mut f32,
) {
    let i = coords_1d();
    if i < len {
        let (x, y, b) = px_linear_rgb_to_positive_xyb(*rp.add(i), *gp.add(i), *bp.add(i));
        *xp.add(i) = x;
        *yp.add(i) = y;
        *bp2.add(i) = b;
    }
}

/// Linear RGB to XYB for a packed image in place.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_xyb_packed(
    in_: *const f32,
    out: *mut f32,
    width: usize,
    height: usize,
    line_step: usize,
) {
    let (col, row) = coords_2d();
    if col < width && row < height {
        let in_ = in_.byte_add(row * line_step).add(col * 3);
        let r = *in_;
        let g = *in_.add(1);
        let b = *in_.add(2);
        let (x, y, b) = px_linear_rgb_to_positive_xyb(r, g, b);
        let out = out.byte_add(row * line_step).add(col * 3);
        *out = x;
        *out.add(1) = y;
        *out.add(2) = b;
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

#[no_mangle]
pub unsafe extern "ptx-kernel" fn ssim_map(
    w: usize,
    h: usize,
    stride: usize,
    mu1: *const f32,
    mu2: *const f32,
    sigma11: *const f32,
    sigma22: *const f32,
    sigma12: *const f32,
    out: *mut f32,
) {
    const C2: f32 = 0.0009f32;

    let (x, y) = coords_2d();
    if x < w && y < h {
        for i in 0..3 {
            let offset = y * stride + (x * 3 + i) * mem::size_of::<f32>();
            let mu1 = *mu1.byte_add(offset);
            let mu11 = mu1 * mu1;
            let mu2 = *mu2.byte_add(offset);
            let mu22 = mu2 * mu2;
            let mu12 = mu1 * mu2;
            let mu_diff = mu1 - mu2;
            let num_m = fma(mu_diff, -mu_diff, 1.0f32);

            let sigma12 = *sigma12.byte_add(offset);
            let num_s = fma(2f32, sigma12 - mu12, C2);
            let sigma11 = *sigma11.byte_add(offset);
            let sigma22 = *sigma22.byte_add(offset);
            let denom_s = (sigma11 - mu11) + (sigma22 - mu22) + C2;
            // Use 1 - SSIM' so it becomes an error score instead of a quality
            // index. This makes it make sense to compute an L_4 norm.
            *out.byte_add(offset) = (1.0 - (num_m * num_s) / denom_s).max(0.0);
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn edge_diff_map(
    w: usize,
    h: usize,
    stride: usize,
    source: *const f32,
    mu1: *const f32,
    distorted: *const f32,
    mu2: *const f32,
    artifact: *mut f32,
    detail_lost: *mut f32,
) {
    let (x, y) = coords_2d();
    if x < w && y < h {
        for i in 0..3 {
            let offset = y * stride + (x * 3 + i) * mem::size_of::<f32>();
            let source = *source.byte_add(offset);
            let mu1 = *mu1.byte_add(offset);
            let denom = 1.0 / (1.0 + abs(source - mu1));
            let distorted = *distorted.byte_add(offset);
            let mu2 = *mu2.byte_add(offset);
            let numer = 1.0 + abs(distorted - mu2);

            let d1 = fma(numer, denom, -1.0);

            // d1 > 0: distorted has an edge where original is smooth
            //         (indicating ringing, color banding, blockiness, etc)
            *artifact.byte_add(offset) = d1.max(0.0);

            // d1 < 0: original has an edge where distorted is smooth
            //         (indicating smoothing, blurring, smearing, etc)
            *detail_lost.byte_add(offset) = (-d1).max(0.0);
        }
    }
}
