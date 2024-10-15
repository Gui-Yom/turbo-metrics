use core::arch::asm;
use core::arch::nvptx;
use core::ops::Add;

pub const WARP_SIZE: usize = 32;

/// Local id of a thread in a warp
#[inline]
pub fn lane() -> u32 {
    let out;
    unsafe {
        asm!(
        "mov.u32 {out}, %laneid;",
        out = out(reg32) out
        )
    }
    out
}

#[inline]
pub fn syncthreads() {
    unsafe {
        nvptx::_syncthreads();
    }
}

#[inline]
pub fn coords_1d() -> usize {
    unsafe {
        let tx = nvptx::_thread_idx_x() as usize;
        let bx = nvptx::_block_idx_x() as usize;
        let bdx = nvptx::_block_dim_x() as usize;
        bx * bdx + tx
    }
}

#[inline]
pub fn coords_2d() -> (usize, usize) {
    unsafe {
        let tx = nvptx::_thread_idx_x() as usize;
        let ty = nvptx::_thread_idx_y() as usize;
        let bx = nvptx::_block_idx_x() as usize;
        let by = nvptx::_block_idx_y() as usize;
        let bdx = nvptx::_block_dim_x() as usize;
        let bdy = nvptx::_block_dim_y() as usize;
        (bx * bdx + tx, by * bdy + ty)
    }
}

#[inline]
pub fn coords_3d() -> (usize, usize, usize) {
    unsafe {
        let tx = nvptx::_thread_idx_x() as usize;
        let ty = nvptx::_thread_idx_y() as usize;
        let tz = nvptx::_thread_idx_z() as usize;
        let bx = nvptx::_block_idx_x() as usize;
        let by = nvptx::_block_idx_y() as usize;
        let bz = nvptx::_block_idx_z() as usize;
        let bdx = nvptx::_block_dim_x() as usize;
        let bdy = nvptx::_block_dim_y() as usize;
        let bdz = nvptx::_block_dim_z() as usize;
        (bx * bdx + tx, by * bdy + ty, bz * bdz + tz)
    }
}

#[inline(always)]
pub unsafe fn shfl_down_sync_f32(mask: u32, value: f32, offset: u32, width: u32) -> f32 {
    let out;
    asm!(
    "shfl.sync.down.b32 {out}, {v}, {offset}, {width}, {mask};",
    out = out(reg32) out,
    v = in(reg32) value,
    offset = in(reg32) offset,
    width = in(reg32) width,
    mask = in(reg32) mask
    );
    out
}

#[inline(always)]
pub unsafe fn shfl_down_sync_u32(mask: u32, value: u32, offset: u32, width: u32) -> u32 {
    let out;
    asm!(
    "shfl.sync.down.b32 {out}, {v}, {offset}, {width}, {mask};",
    out = out(reg32) out,
    v = in(reg32) value,
    offset = in(reg32) offset,
    width = in(reg32) width,
    mask = in(reg32) mask
    );
    out
}

#[inline]
pub unsafe fn warp_sum_u32(value: u32) -> u32 {
    warp_reduce_u32(value, Add::add)
}

#[inline]
pub unsafe fn warp_reduce_u32(mut value: u32, mut red: impl FnMut(u32, u32) -> u32) -> u32 {
    #[inline(always)]
    unsafe fn shfl_bfly_u32(mask: u32, value: u32, offset: u32, width: u32) -> u32 {
        let out;
        asm!(
            "shfl.sync.bfly.b32 {out}, {v}, {offset}, {width}, {mask};",
            out = out(reg32) out,
            v = in(reg32) value,
            offset = in(reg32) offset,
            width = in(reg32) width,
            mask = in(reg32) mask
        );
        out
    }

    value = red(value, shfl_bfly_u32(0xffffffff, value, 16, 32));
    value = red(value, shfl_bfly_u32(0xffffffff, value, 8, 32));
    value = red(value, shfl_bfly_u32(0xffffffff, value, 4, 32));
    value = red(value, shfl_bfly_u32(0xffffffff, value, 2, 32));
    value = red(value, shfl_bfly_u32(0xffffffff, value, 1, 32));
    value
}

/// Reads the 32-bit unsigned old located at the address in global or shared memory, computes (old + val), and stores
/// the result back to memory at the same address. These three operations are performed in one atomic transaction.
/// The function returns old.
#[inline(always)]
pub unsafe fn atomic_add_global_u32(ptr: *mut u32, value: u32) -> u32 {
    let out;
    asm!(
    "atom.global.add.u32 {out}, [{p}], {v};",
    out = out(reg32) out,
    p = in(reg64) ptr,
    v = in(reg32) value
    );
    out
}

/// Reads the 64-bit unsigned old located at the address in global or shared memory, computes (old + val), and stores
/// the result back to memory at the same address. These three operations are performed in one atomic transaction.
/// The function returns old.
#[inline(always)]
pub unsafe fn atomic_add_global_u64(ptr: *mut u64, value: u64) -> u64 {
    let out;
    asm!(
    "cvta.to.global.u64 {tmp}, {p};",
    "atom.global.add.u64 {out}, [{tmp}], {v};",
    tmp = out(reg64) _,
    out = out(reg64) out,
    p = in(reg64) ptr,
    v = in(reg64) value
    );
    out
}
