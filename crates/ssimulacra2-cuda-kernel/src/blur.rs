use core::hint::unreachable_unchecked;

use nvptx_std::prelude::*;

/// Filter values as computed by the build script
mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

// This must be kept in sync with the launch dimensions
const BLOCK_WIDTH: usize = 3 * 32;

// We can't allocate shared memory from Rust. This value is defined in shared.bc
extern "C" {
    /// The ring buffer.
    ///
    /// Logically, for each thread :
    ///   left=y-N-1        y            right=y+N-1
    ///   |                 |            |
    /// [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    ///   <------------------------------>
    ///            RING_SIZE=2N+1
    ///
    /// Here we allocate for 33 elements to solve the bank conflict problem.
    static mut RING: [[f32; 33]; BLOCK_WIDTH];
}

/// Single vertical pass on a single plane
///
/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016]. Derived from the ssimulacra2 impl.
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
    const N: isize = consts::RADIUS as isize;
    const RING_SIZE: usize = consts::RADIUS * 2 + 1;

    let tx = core::arch::nvptx::_thread_idx_x() as usize;

    // This suffices to remove bound checks
    if tx >= BLOCK_WIDTH {
        return;
    }

    // Initialize shared mem to zero
    for i in 0..RING_SIZE {
        RING[tx][i] = 0.0;
    }

    let ty = core::arch::nvptx::_thread_idx_y() as usize;
    let bx = core::arch::nvptx::_block_idx_x() as usize;
    let by = core::arch::nvptx::_block_idx_y() as usize;
    let bdx = core::arch::nvptx::_block_dim_x() as usize;
    let bdy = core::arch::nvptx::_block_dim_y() as usize;
    // The y coord designs the plane on which this invocation is operating
    // With clever grid and block layout, we can compute 5 blur passes in one bigger grid launch,
    // effectively maximizing occupancy
    let (x, index) = (bx * bdx + tx, by * bdy + ty);

    if x >= width {
        return;
    }

    let (src, dst) = match index {
        0 => (src0, dst0),
        1 => (src1, dst1),
        2 => (src2, dst2),
        3 => (src3, dst3),
        4 => (src4, dst4),
        _ => unreachable_unchecked(),
    };

    let mut prev_1 = 0f32;
    let mut prev_3 = 0f32;
    let mut prev_5 = 0f32;
    let mut prev2_1 = 0f32;
    let mut prev2_3 = 0f32;
    let mut prev2_5 = 0f32;

    for y in (-N + 1)..height as isize {
        let right = (y + N) as usize - 1;
        let right_val = if right < height {
            *src.byte_add(right * src_pitch).add(x)
        } else {
            0f32
        };

        let left = y - N - 1;
        let left_val = if left >= 0 {
            // *src.byte_add(left as usize * src_pitch).add(x)
            RING[tx].as_ptr().add(left as usize % RING_SIZE).read()
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

        *RING[tx].as_mut_ptr().add(right % RING_SIZE) = right_val;
    }
}
