#![no_std]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::mem;

use nvptx_core::{abs, cbrt, coords_1d, coords_2d, fma, lane, powf, shfl_down_sync, syncthreads};

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

const FROM_SRGB8_TABLE: [f32; 256] = [
    0.0,
    0.000303527,
    0.000607054,
    0.00091058103,
    0.001214108,
    0.001517635,
    0.0018211621,
    0.002124689,
    0.002428216,
    0.002731743,
    0.00303527,
    0.0033465356,
    0.003676507,
    0.004024717,
    0.004391442,
    0.0047769533,
    0.005181517,
    0.0056053917,
    0.0060488326,
    0.006512091,
    0.00699541,
    0.0074990317,
    0.008023192,
    0.008568125,
    0.009134057,
    0.009721218,
    0.010329823,
    0.010960094,
    0.011612245,
    0.012286487,
    0.012983031,
    0.013702081,
    0.014443844,
    0.015208514,
    0.015996292,
    0.016807375,
    0.017641952,
    0.018500218,
    0.019382361,
    0.020288562,
    0.02121901,
    0.022173883,
    0.023153365,
    0.02415763,
    0.025186857,
    0.026241222,
    0.027320892,
    0.028426038,
    0.029556843,
    0.03071345,
    0.03189604,
    0.033104774,
    0.03433981,
    0.035601325,
    0.036889452,
    0.038204376,
    0.039546248,
    0.04091521,
    0.042311423,
    0.043735042,
    0.045186214,
    0.046665095,
    0.048171833,
    0.049706575,
    0.051269468,
    0.052860655,
    0.05448028,
    0.056128494,
    0.057805434,
    0.05951124,
    0.06124607,
    0.06301003,
    0.06480328,
    0.06662595,
    0.06847818,
    0.07036011,
    0.07227186,
    0.07421358,
    0.07618539,
    0.07818743,
    0.08021983,
    0.082282715,
    0.084376216,
    0.086500466,
    0.088655606,
    0.09084173,
    0.09305898,
    0.095307484,
    0.09758736,
    0.09989874,
    0.10224175,
    0.10461649,
    0.10702311,
    0.10946172,
    0.111932434,
    0.11443538,
    0.116970696,
    0.11953845,
    0.12213881,
    0.12477186,
    0.12743773,
    0.13013652,
    0.13286836,
    0.13563336,
    0.13843165,
    0.14126332,
    0.1441285,
    0.1470273,
    0.14995982,
    0.15292618,
    0.1559265,
    0.15896086,
    0.16202943,
    0.16513224,
    0.16826946,
    0.17144115,
    0.17464745,
    0.17788847,
    0.1811643,
    0.18447503,
    0.1878208,
    0.19120172,
    0.19461787,
    0.19806935,
    0.2015563,
    0.20507877,
    0.2086369,
    0.21223079,
    0.21586053,
    0.21952623,
    0.22322798,
    0.22696589,
    0.23074007,
    0.23455065,
    0.23839766,
    0.2422812,
    0.2462014,
    0.25015837,
    0.25415218,
    0.2581829,
    0.26225072,
    0.26635566,
    0.27049786,
    0.27467737,
    0.27889434,
    0.2831488,
    0.2874409,
    0.2917707,
    0.29613832,
    0.30054384,
    0.30498737,
    0.30946895,
    0.31398875,
    0.31854683,
    0.32314324,
    0.32777813,
    0.33245158,
    0.33716366,
    0.34191445,
    0.3467041,
    0.3515327,
    0.35640025,
    0.36130688,
    0.3662527,
    0.37123778,
    0.37626222,
    0.3813261,
    0.38642952,
    0.39157256,
    0.3967553,
    0.40197787,
    0.4072403,
    0.4125427,
    0.41788515,
    0.42326775,
    0.42869055,
    0.4341537,
    0.43965724,
    0.44520125,
    0.45078585,
    0.45641106,
    0.46207705,
    0.46778384,
    0.47353154,
    0.47932023,
    0.48514998,
    0.4910209,
    0.49693304,
    0.5028866,
    0.50888145,
    0.5149178,
    0.5209957,
    0.52711535,
    0.5332766,
    0.5394797,
    0.5457247,
    0.5520116,
    0.5583406,
    0.5647117,
    0.57112503,
    0.57758063,
    0.5840786,
    0.590619,
    0.597202,
    0.60382754,
    0.61049575,
    0.61720675,
    0.62396055,
    0.63075733,
    0.637597,
    0.6444799,
    0.6514058,
    0.65837497,
    0.66538745,
    0.67244333,
    0.6795426,
    0.68668544,
    0.69387203,
    0.70110214,
    0.70837605,
    0.7156938,
    0.72305536,
    0.730461,
    0.7379107,
    0.7454045,
    0.75294244,
    0.76052475,
    0.7681514,
    0.77582246,
    0.78353804,
    0.79129815,
    0.79910296,
    0.8069525,
    0.8148468,
    0.822786,
    0.8307701,
    0.83879924,
    0.84687346,
    0.8549928,
    0.8631574,
    0.87136734,
    0.8796226,
    0.8879232,
    0.89626956,
    0.90466136,
    0.913099,
    0.92158204,
    0.93011117,
    0.9386859,
    0.9473069,
    0.9559735,
    0.9646866,
    0.9734455,
    0.98225087,
    0.9911022,
    1.0,
];

/// Works at the sample level for coalesced read & writes.
///
/// Does not do bound checks as it assumes it can read & write past bounds.
/// When working on full images, the image must be padded.
/// In other cases, this kernel will touch more samples on the edge.
#[no_mangle]
unsafe extern "ptx-kernel" fn srgb_to_linear(
    src: *mut u8,
    src_line_step: usize,
    dst: *mut f32,
    dst_line_step: usize,
) {
    let (x, y) = coords_2d();
    // if x < w && y < h {
    *dst.byte_add(y * dst_line_step).add(x) = FROM_SRGB8_TABLE[*src.byte_add(y * src_line_step).add(x) as usize];
    // }
}

/// Downscale by taking the average of 2x2 patches
#[no_mangle]
pub unsafe extern "ptx-kernel" fn downscale_by_2(
    src_w: usize,
    src_h: usize,
    src: *const f32,
    src_pitch: usize,
    dst_w: usize,
    dst_h: usize,
    dst: *mut f32,
    dst_pitch: usize,
) {
    const SCALE: usize = 2;
    const NORMALIZE: f32 = 1f32 / (SCALE * SCALE) as f32;
    const C: usize = 3;

    let (ox, oy) = coords_2d();

    if ox < dst_w && oy < dst_h {
        for c in 0..C {
            let mut sum = 0.0;
            for iy in 0..SCALE {
                for ix in 0..SCALE {
                    let x = (ox * SCALE + ix).min(src_w - 1);
                    let y = (oy * SCALE + iy).min(src_h - 1);

                    sum += *src.byte_add(y * src_pitch).add(x * C + c);
                }
            }
            *dst.byte_add(oy * dst_pitch).add(ox * C + c) = sum * NORMALIZE;
        }
    }
}

/// Downscale by taking the average of 2x2 patches
#[no_mangle]
pub unsafe extern "ptx-kernel" fn downscale_plane_by_2(
    src_w: usize,
    src_h: usize,
    src: *const f32,
    src_pitch: usize,
    dst_w: usize,
    dst_h: usize,
    dst: *mut f32,
    dst_pitch: usize,
) {
    const SCALE: usize = 2;
    const NORMALIZE: f32 = 1f32 / (SCALE * SCALE) as f32;

    let (x, y) = coords_2d();

    if x < src_w && y < src_h {
        let mut v = *src.byte_add(y * src_pitch).add(x);

        v += shfl_down_sync(0xffffffff, v, 16, 32);
        v += shfl_down_sync(0xffffffff, v, 1, 32);

        let id = lane();
        // Top left thread in the patch will write its value
        if id <= 16 && id & 1 == 0 {
            *dst.byte_add(y / 2 * dst_pitch).add(x / 2) = v * NORMALIZE;
        }
        syncthreads();
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
unsafe fn px_linear_rgb_to_positive_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
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
    width: usize,
    height: usize,
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
) {
    let (col, row) = coords_2d();
    // if col < width && row < height {
    let in_ = src.byte_add(row * src_pitch).add(col * 3);
    let r = *in_;
    let g = *in_.add(1);
    let b = *in_.add(2);
    let (x, y, b) = px_linear_rgb_to_positive_xyb(r, g, b);
    let out = dst.byte_add(row * dst_pitch).add(col * 3);
    *out = x;
    *out.add(1) = y;
    *out.add(2) = b;
    // }
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
