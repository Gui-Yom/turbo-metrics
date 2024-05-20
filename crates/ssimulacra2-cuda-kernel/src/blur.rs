use nvptx_core::coords_2d;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn blur(
    width: usize,
    height: usize,
    src: *const f32,
    src_pitch: usize,
    dst: *mut f32,
    dst_pitch: usize,
) {
    const SCALE: usize = 2;
    const NORMALIZE: f32 = 1f32 / (SCALE * SCALE) as f32;
    const C: usize = 3;

    let (x, y) = coords_2d();

    if x < width && y < height {
        for c in 0..C {
            // *dst.byte_add(oy * dst_pitch).add(ox * C + c) = sum * NORMALIZE;
        }
    }
}
