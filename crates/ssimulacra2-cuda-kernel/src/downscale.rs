use nvptx_core::prelude::*;

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
        // syncthreads();
    }
}
