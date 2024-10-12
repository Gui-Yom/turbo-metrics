use cudarse_driver::sys::CuResult;
use cudarse_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cudarse_npp::debug_assert_same_size;
use cudarse_npp::image::{Img, ImgMut, C, P};

pub struct Kernel {
    _module: CuModule,
    biplanaryuv420_to_linearrgb_8_l_bt709: CuFunction,
    biplanaryuv420_to_linearrgb_16_l_bt709: CuFunction,
    biplanaryuv420_to_linearrgb_8_l_bt601_525: CuFunction,
    biplanaryuv420_to_linearrgb_16_l_bt601_525: CuFunction,
    biplanaryuv420_to_linearrgb_8_l_bt601_625: CuFunction,
    biplanaryuv420_to_linearrgb_16_l_bt601_625: CuFunction,
    biplanaryuv420_to_linearrgb_debug: CuFunction,
    f32_to_8bit: CuFunction,
    srgb_to_linear_u8_lookup: CuFunction,
    srgb_to_linear_u8: CuFunction,
    srgb_to_linear_u16: CuFunction,
    srgb_to_linear_f32: CuFunction,
}

impl Kernel {
    pub fn load() -> CuResult<Self> {
        //let path = "target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";
        // let module = CuModule::load_ptx(include_str!(concat!(
        //     env!("CARGO_CDYLIB_DIR_CUDA_COLORSPACE_KERNEL"),
        //     "/cuda_colorspace_kernel.ptx"
        // )))?;
        let module = CuModule::load_ptx(include_str!(env!(
            "CARGO_CDYLIB_FILE_CUDA_COLORSPACE_KERNEL"
        )))?;
        Ok(Self {
            biplanaryuv420_to_linearrgb_8_l_bt709: module
                .function_by_name("biplanaryuv420_to_linearrgb_8_l_bt709")?,
            biplanaryuv420_to_linearrgb_16_l_bt709: module
                .function_by_name("biplanaryuv420_to_linearrgb_16_l_bt709")?,
            biplanaryuv420_to_linearrgb_8_l_bt601_525: module
                .function_by_name("biplanaryuv420_to_linearrgb_8_l_bt601_525")?,
            biplanaryuv420_to_linearrgb_16_l_bt601_525: module
                .function_by_name("biplanaryuv420_to_linearrgb_16_l_bt601_525")?,
            biplanaryuv420_to_linearrgb_8_l_bt601_625: module
                .function_by_name("biplanaryuv420_to_linearrgb_8_l_bt601_625")?,
            biplanaryuv420_to_linearrgb_16_l_bt601_625: module
                .function_by_name("biplanaryuv420_to_linearrgb_16_l_bt601_625")?,
            biplanaryuv420_to_linearrgb_debug: module
                .function_by_name("biplanaryuv420_to_linearrgb_debug")?,
            f32_to_8bit: module.function_by_name("f32_to_8bit")?,
            srgb_to_linear_u8_lookup: module.function_by_name("srgb_to_linear_u8_lookup")?,
            srgb_to_linear_u8: module.function_by_name("srgb_to_linear_u8")?,
            srgb_to_linear_u16: module.function_by_name("srgb_to_linear_u16")?,
            srgb_to_linear_f32: module.function_by_name("srgb_to_linear_f32")?,
            _module: module,
        })
    }

    pub fn biplanaryuv420_to_linearrgb_8_l_bt709(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_8_l_bt709.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_16_l_bt709(
        &self,
        src: impl Img<u16, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_16_l_bt709.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_8_l_bt601_525(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_8_l_bt601_525.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_16_l_bt601_525(
        &self,
        src: impl Img<u16, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_16_l_bt601_525.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_8_l_bt601_625(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_8_l_bt601_625.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_16_l_bt601_625(
        &self,
        src: impl Img<u16, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_16_l_bt601_625.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn biplanaryuv420_to_linearrgb_debug(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<u8, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let y = iter.next().unwrap();
            let uv = iter.next().unwrap();
            let w = dst.width() / 2;
            let h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_debug.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    // src.device_ptr(),
                    y,
                    uv,
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn f32_to_8bit(
        &self,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<u8, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let w = dst.width() * 3;
            let h = dst.height();
            self.f32_to_8bit.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    src.device_ptr(),
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn srgb_to_linear_u8_lookup(
        &self,
        src: impl Img<u8, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let w = dst.width() * 3;
            let h = dst.height();
            self.srgb_to_linear_u8_lookup.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    src.device_ptr(),
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn srgb_to_linear_u8(
        &self,
        src: impl Img<u8, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let w = dst.width() * 3;
            let h = dst.height();
            self.srgb_to_linear_u8.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    src.device_ptr(),
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn srgb_to_linear_u16(
        &self,
        src: impl Img<u16, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let w = dst.width() * 3;
            let h = dst.height();
            self.srgb_to_linear_u16.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    src.device_ptr(),
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }

    pub fn srgb_to_linear_f32(
        &self,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let w = dst.width() * 3;
            let h = dst.height();
            self.srgb_to_linear_f32.launch(
                &launch_config_2d(w, h),
                stream,
                kernel_params!(
                    src.device_ptr(),
                    src.pitch() as usize,
                    dst.device_ptr_mut(),
                    dst.pitch() as usize,
                    w as usize,
                    h as usize,
                ),
            )
        }
    }
}

fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
    // const MAX_THREADS_PER_BLOCK: u32 = 256;
    const THREADS_WIDTH: u32 = 16;
    const THREADS_HEIGHT: u32 = 8;
    let num_blocks_w = (width + THREADS_WIDTH - 1) / THREADS_WIDTH;
    let num_blocks_h = (height + THREADS_HEIGHT - 1) / THREADS_HEIGHT;
    LaunchConfig {
        grid_dim: (num_blocks_w, num_blocks_h, 1),
        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use crate::Kernel;
    use cudarse_driver::sys::CuResult;
    use cudarse_driver::{init_cuda_and_primary_ctx, CuStream};
    use cudarse_npp::image::idei::{Set, SetMany};
    use cudarse_npp::image::isu::Malloc;
    use cudarse_npp::image::{Image, Img, C};

    #[test]
    fn test_8bit() -> CuResult<()> {
        init_cuda_and_primary_ctx()?;
        let kernel = Kernel::load()?;
        let main = CuStream::new()?;
        cudarse_npp::set_stream(main.inner() as _).unwrap();
        let ctx = cudarse_npp::get_stream_ctx().unwrap();
        let mut y_plane = Image::<u8, C<1>>::malloc(8, 8).unwrap();
        let mut uv_plane = Image::<u8, C<2>>::malloc(4, 4).unwrap();
        y_plane.set(235, ctx).unwrap();
        uv_plane.set([128, 128], ctx).unwrap();
        assert_eq!(y_plane.pitch(), uv_plane.pitch());

        let mut dst = y_plane.malloc_same_size().unwrap();

        let src = Image::from_raw(
            y_plane.width(),
            y_plane.height(),
            y_plane.pitch(),
            [y_plane.into_inner(), uv_plane.into_inner()],
        );
        dbg!(&src);
        dbg!(&dst);

        kernel.biplanaryuv420_to_linearrgb_8_l_bt709(&src, &mut dst, &main)?;

        let data = dst.copy_to_cpu(cudarse_npp::get_stream()).unwrap();
        main.sync()?;
        dbg!(&data[0..3]);

        let mut dst2 = dst.malloc_same_size().unwrap();
        kernel.f32_to_8bit(&dst, &mut dst2, &main)?;

        let data = dst2.copy_to_cpu(cudarse_npp::get_stream()).unwrap();
        main.sync()?;
        dbg!(&data[0..3]);

        Ok(())
    }
}
