use core::hint::unreachable_unchecked;

use nvptx_core::prelude::*;

/// Filter values as computed by the build script
mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

/// Single vertical pass on a single plane
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_plane_pass(
    width: usize,
    height: usize,
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
) {
    let x = coords_1d();

    if x < width {
        let big_n = consts::RADIUS as isize;
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        for y in (-big_n + 1)..height as isize {
            let left = y - big_n - 1;
            let right = y + big_n - 1;
            let left_val = if left >= 0 {
                *src.byte_add(left as usize * src_pitch).add(x)
            } else {
                0f32
            };
            let right_val = if right < height as isize {
                *src.byte_add(right as usize * src_pitch).add(x)
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out_1 = sum * consts::MUL_IN_1;
            let mut out_3 = sum * consts::MUL_IN_3;
            let mut out_5 = sum * consts::MUL_IN_5;

            out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
            out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
            out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
            out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
            out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if y >= 0 {
                *dst.byte_add(y as usize * dst_pitch).add(x) = out_1 + out_3 + out_5;
            }
        }
    }
}

/// Single vertical pass on a single plane
#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_plane_pass_fused(
    width: usize,
    height: usize,
    src0: *const f32,
    src1: *const f32,
    src2: *const f32,
    src3: *const f32,
    src4: *const f32,
    src_pitch: usize,
    dst0: *mut f32,
    dst1: *mut f32,
    dst2: *mut f32,
    dst3: *mut f32,
    dst4: *mut f32,
    dst_pitch: usize,
) {
    // The y coord designs the plane on which this invocation is operating
    // With clever grid and block layout, we can compute 5 blur passes in one bigger grid launch,
    // effectively maximizing occupancy
    let (x, index) = coords_2d();

    let (src, dst) = match index {
        0 => (src0, dst0),
        1 => (src1, dst1),
        2 => (src2, dst2),
        3 => (src3, dst3),
        4 => (src4, dst4),
        _ => unreachable_unchecked(),
    };

    if x < width {
        let big_n = consts::RADIUS as isize;
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        for y in (-big_n + 1)..height as isize {
            let left = y - big_n - 1;
            let right = y + big_n - 1;
            let left_val = if left >= 0 {
                *src.byte_add(left as usize * src_pitch).add(x)
            } else {
                0f32
            };
            let right_val = if right < height as isize {
                *src.byte_add(right as usize * src_pitch).add(x)
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out_1 = sum * consts::MUL_IN_1;
            let mut out_3 = sum * consts::MUL_IN_3;
            let mut out_5 = sum * consts::MUL_IN_5;

            out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
            out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
            out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
            out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
            out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if y >= 0 {
                *dst.byte_add(y as usize * dst_pitch).add(x) = out_1 + out_3 + out_5;
            }
        }
    }
}
