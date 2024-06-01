#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;
use core::mem;

use nvptx_core::prelude::*;

mod blur;
mod downscale;
mod srgb;
mod xyb;

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
            let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);

            let sigma12 = *sigma12.byte_add(offset);
            let num_s = 2f32.mul_add(sigma12 - mu12, C2);
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
            let denom = 1.0 / (1.0 + (source - mu1).abs());
            let distorted = *distorted.byte_add(offset);
            let mu2 = *mu2.byte_add(offset);
            let numer = 1.0 + (distorted - mu2).abs();

            let d1 = numer.mul_add(denom, -1.0);

            // d1 > 0: distorted has an edge where original is smooth
            //         (indicating ringing, color banding, blockiness, etc)
            *artifact.byte_add(offset) = d1.max(0.0);

            // d1 < 0: original has an edge where distorted is smooth
            //         (indicating smoothing, blurring, smearing, etc)
            *detail_lost.byte_add(offset) = (-d1).max(0.0);
        }
    }
}
