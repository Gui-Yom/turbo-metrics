use nvptx_core::coords_2d;
use nvptx_core::math::{cbrt, fma};

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
    K_M00, K_M01, K_M02,
    K_M10, K_M11, K_M12,
    K_M20, K_M21, K_M22,
];
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];
const OPSIN_ABSORBANCE_BIAS_ROOT: [f32; 3] = [K_B0_ROOT, K_B0_ROOT, K_B0_ROOT];
const INVERSE_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    11.031_567_f32, -9.866_944_f32, -0.164_622_99_f32,
    -3.254_147_3_f32, 4.418_770_3_f32, -0.164_622_99_f32,
    -3.658_851_4_f32, 2.712_923_f32, 1.945_928_2_f32,
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
    if col < width && row < height {
        let in_ = src.byte_add(row * src_pitch).add(col * 3);
        let r = *in_;
        let g = *in_.add(1);
        let b = *in_.add(2);
        let (x, y, b) = px_linear_rgb_to_positive_xyb(r, g, b);
        let out = dst.byte_add(row * dst_pitch).add(col * 3);
        *out = x;
        *out.add(1) = y;
        *out.add(2) = b;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn linear_to_xyb_planar(
    width: usize,
    height: usize,
    src_r: *const f32,
    src_g: *const f32,
    src_b: *const f32,
    src_pitch: usize,
    dst_x: *mut f32,
    dst_y: *mut f32,
    dst_b: *mut f32,
    dst_pitch: usize,
) {
    let (col, row) = coords_2d();
    let byte_offset = row * src_pitch;
    if col < width && row < height {
        let r = *src_r.byte_add(byte_offset).add(col);
        let g = *src_g.byte_add(byte_offset).add(col);
        let b = *src_b.byte_add(byte_offset).add(col);
        let (x, y, b) = px_linear_rgb_to_positive_xyb(r, g, b);
        let byte_offset = row * dst_pitch;
        *dst_x.byte_add(byte_offset).add(col) = x;
        *dst_y.byte_add(byte_offset).add(col) = y;
        *dst_b.byte_add(byte_offset).add(col) = b;
    }
}
