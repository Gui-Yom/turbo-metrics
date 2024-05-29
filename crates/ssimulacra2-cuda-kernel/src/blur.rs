use nvptx_core::prelude::*;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur_plane(
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

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;
            let left_val = if left >= 0 {
                // SAFETY: `left` can never be bigger than `width`
                *src.byte_add(left as usize * src_pitch).add(x)
            } else {
                0f32
            };
            let right_val = if right < height as isize {
                // SAFETY: this branch ensures that `right` is not bigger than `width`
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

            if n >= 0 {
                // SAFETY: We know that this chunk of output is of size `width`,
                // which `n` cannot be larger than.
                *dst.byte_add(n as usize * dst_pitch).add(x) = out_1 + out_3 + out_5;
            }

            n += 1;
        }
    }
}

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}
