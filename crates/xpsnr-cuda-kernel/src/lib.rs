#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;
use core::ops::Add;
use nvptx_std::prelude::*;
/*
For each block :
    sum of squared error
    spatial activity: high pass filter
    temporal activity: 1st and 2nd order diff
 */

const XPSNR_GAMMA: u32 = 2;

// unsafe fn sse8(
//     ref_block: *const u8,
//     ref_pitch: usize,
//     dis_block: *const u8,
//     dis_pitch: usize,
//     block_w: usize,
//     block_h: usize,
// ) -> u32 {
//     let mut sse = 0;
//     for y in 0..block_h {
//         for x in 0..block_w {
//             let ref_value = ref_block.byte_add(y * ref_pitch).add(x).read();
//             let dis_value = dis_block.byte_add(y * dis_pitch).add(x).read();
//             let error = ref_value as i32 - dis_value as i32;
//             sse += (error * error) as u32;
//         }
//     }
//     sse
// }

#[no_mangle]
pub unsafe extern "ptx-kernel" fn xpsnr_support_8(
    ref_: *const u8,
    ref_pitch: usize,
    prev: *mut u8,
    prev_pitch: usize,
    dis: *const u8,
    dis_pitch: usize,
    highpass: *const i16,
    highpass_pitch: usize,
    sse: *mut u32,
    sact: *mut u32,
    tact: *mut u32,
    width: usize,
    height: usize,
) {
    let tx = nvptx::_thread_idx_x() as usize;
    let ty = nvptx::_thread_idx_y() as usize;
    let bx = nvptx::_block_idx_x() as usize;
    let by = nvptx::_block_idx_y() as usize;
    let bdx = nvptx::_block_dim_x() as usize;
    let bdy = nvptx::_block_dim_y() as usize;
    let (x, y) = (bx * bdx + tx, by * bdy + ty);

    let block_idx = by * nvptx::_grid_dim_x() as usize + bx;
    // printf!(c"%d\n", u32; block_idx as u32);

    if x >= width || y >= height {
        return;
    }

    // Temporal activity
    let ref_value = ref_.byte_add(y * ref_pitch).add(x).read();
    let prev = prev.byte_add(y * prev_pitch).add(x);
    let prev_value = prev.read();
    *prev = ref_value;
    let t_act = ref_value.abs_diff(prev_value) as u32;
    let tact_value = warp_sum_u32(t_act);

    // SSE
    let dis_value = dis.byte_add(y * dis_pitch).add(x).read();
    let error = ref_value as i32 - dis_value as i32;
    let se = (error * error) as u32;
    let se = warp_sum_u32(se);

    // Spatial activity
    let highpass = highpass.byte_add(y * highpass_pitch).add(x).read();
    let sact_value = warp_sum_u32(highpass.abs() as u32);

    if lane() == 0 {
        atomic_add_global_u32(sact.add(block_idx), sact_value);
        atomic_add_global_u32(tact.add(block_idx), tact_value);
        atomic_add_global_u32(sse.add(block_idx), se);
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn xpsnr_postprocess(
    sse: *const u32,
    sact: *const u32,
    tact: *const u32,
    len: usize,
    wsse: *mut f32,
) {
    let x = coords_1d();

    if x >= len {
        return;
    }

    let num_samples = (16 * 16) as f32;
    let mut msact = 1.0 + sact.add(x).read() as f32 / num_samples;
    msact += 2.0 * tact.add(x).read() as f32 / num_samples;
    msact = msact.max((1 << (8 - 2)) as f32);
    msact *= msact;
    let weight = msact.rsqrt();
    let wsse_value = sse.add(x).read() as f32 * weight;
    let wsse_value = warp_sum_f32(wsse_value);

    if lane() == 0 {
        atomic_add_global_f32(wsse, wsse_value);
    }
}

// unsafe fn highpass_filter(
//     src: *const u8,
//     src_pitch: usize,
//     dst: *const u8,
//     dst_pitch: usize,
//     width: usize,
//     height: usize,
//     x: usize,
//     y: usize,
// ) {
//     if x >= width || y > height {}
//     if x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1 {
//         return;
//     }
//     let center = src.byte_add(y * src_pitch).add(x);
//     let value = 12 * center.read()
//         - 2 * (center.offset(-1).read()
//             + center.add(1).read()
//             + center.byte_offset(-(src_pitch as isize)).read()
//             + center.byte_add(src_pitch).read())
//         - (center.offset(-1).read()
//             + center.add(1).read()
//             + center.byte_offset(-(src_pitch as isize)).read()
//             + center.byte_add(src_pitch).read());
// }

unsafe fn dispatch(
    width: usize,
    height: usize,
    offsetx: usize,
    offsety: usize,
    block_w: usize,
    block_h: usize,
) {
    let bval = if width * height > 2048 * 1152 { 2 } else { 1 };
    let xact = if offsetx > 0 { 0 } else { bval };
    let yact = if offsety > 0 { 0 } else { bval };
    let wact = if offsetx + block_w < width {
        block_w
    } else {
        block_w - bval
    };
    let hact = if offsety + block_h < width {
        block_h
    } else {
        block_h - bval
    };
    if wact <= xact || hact <= yact {
        return;
    }
    if bval > 1 {
        // With downsampling
    } else {
        for y in yact..hact {
            for x in xact..wact {
                // Neighborhood sum
            }
        }
    }
}
