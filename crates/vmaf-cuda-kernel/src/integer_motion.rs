use nvptx_core::prelude::*;

trait Sample {
    type Type: Into<u32>;
}

struct Bit<const N: usize>;
impl Sample for Bit<8> {
    type Type = u8;
}
impl Sample for Bit<10> {
    type Type = u16;
}
impl Sample for Bit<12> {
    type Type = u16;
}

fn mirror(idx: isize, limit: usize) -> usize {
    let idx = idx.unsigned_abs();
    if idx < limit {
        idx
    } else {
        limit - (idx - limit + 1)
    }
}

#[inline]
unsafe fn motion_generic<const N: usize>(
    width: usize,
    height: usize,
    src: *const <Bit<N> as Sample>::Type,
    src_pitch: usize,
    blurred: *mut u16,
    blurred_pitch: usize,
    prev_blurred: *mut u16,
    prev_blurred_pitch: usize,
    sad: *mut u64,
) where
    Bit<N>: Sample,
{
    const FILTER: [u16; 5] = [3571, 16004, 26386, 16004, 3571];
    const RADIUS: usize = FILTER.len() / 2;
    const shift_var_x: u32 = 16;
    const add_before_shift_x: u32 = 32768;

    let (x, y) = coords_2d();
    if x >= width || y >= height {
        return;
    }

    let mut blurred_sample = 0;
    for xf in 0..FILTER.len() {
        let mut blurred_y = 0;
        for yf in 0..FILTER.len() {
            let sample = src
                .byte_add(mirror(y as isize - RADIUS as isize + yf as isize, height) * src_pitch)
                .add(mirror(x as isize - RADIUS as isize + xf as isize, width))
                .read();
            blurred_y += FILTER[yf] as u32 * sample.into();
        }
        // blurred_y = FILTER dot SAMPLES
        // blurred_sample = (FILTER dot ((blurred_y + A) / B) + C) / D
        // blurred_sample = (FILTER dot (FILTER dot SAMPLES) + FILTER dot A) / (B * D) + C / D
        blurred_sample += FILTER[xf] as u32 * ((blurred_y + 2u32.pow(N as u32 - 1)) >> N as u32);
    }
    blurred_sample = (blurred_sample + add_before_shift_x) >> shift_var_x;
    blurred
        .byte_add(y * blurred_pitch)
        .add(x)
        .write(blurred_sample as u16);
    let mut abs_dist =
        blurred_sample.abs_diff(prev_blurred.byte_add(y * prev_blurred_pitch).add(x).read() as u32);

    abs_dist += shfl_down_sync_u32(0xffffffff, abs_dist, 16, 32);
    abs_dist += shfl_down_sync_u32(0xffffffff, abs_dist, 8, 32);
    abs_dist += shfl_down_sync_u32(0xffffffff, abs_dist, 4, 32);
    abs_dist += shfl_down_sync_u32(0xffffffff, abs_dist, 2, 32);
    abs_dist += shfl_down_sync_u32(0xffffffff, abs_dist, 1, 32);

    let lane = lane();
    if lane == 0 {
        atomic_add_global_u64(sad, abs_dist as u64);
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn motion_8(
    width: usize,
    height: usize,
    src: *const u8,
    src_pitch: usize,
    blurred: *mut u16,
    blurred_pitch: usize,
    prev_blurred: *mut u16,
    prev_blurred_pitch: usize,
    sad: *mut u64,
) {
    motion_generic::<8>(
        width,
        height,
        src,
        src_pitch,
        blurred,
        blurred_pitch,
        prev_blurred,
        prev_blurred_pitch,
        sad,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn motion_10(
    width: usize,
    height: usize,
    src: *const u16,
    src_pitch: usize,
    blurred: *mut u16,
    blurred_pitch: usize,
    prev_blurred: *mut u16,
    prev_blurred_pitch: usize,
    sad: *mut u64,
) {
    motion_generic::<10>(
        width,
        height,
        src,
        src_pitch,
        blurred,
        blurred_pitch,
        prev_blurred,
        prev_blurred_pitch,
        sad,
    )
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn motion_12(
    width: usize,
    height: usize,
    src: *const u16,
    src_pitch: usize,
    blurred: *mut u16,
    blurred_pitch: usize,
    prev_blurred: *mut u16,
    prev_blurred_pitch: usize,
    sad: *mut u64,
) {
    motion_generic::<12>(
        width,
        height,
        src,
        src_pitch,
        blurred,
        blurred_pitch,
        prev_blurred,
        prev_blurred_pitch,
        sad,
    )
}
