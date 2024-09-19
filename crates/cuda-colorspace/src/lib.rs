use cudarse_driver::sys::CuResult;
use cudarse_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cudarse_npp::debug_assert_same_size;
use cudarse_npp::image::{Img, ImgMut, C, P};

pub struct Kernel {
    module: CuModule,
    biplanaryuv420_to_linearrgb_8_L_BT709: CuFunction,
    biplanaryuv420_to_linearrgb_8_F_BT709: CuFunction,
    biplanaryuv420_to_linearrgb_10_L_BT709: CuFunction,
    biplanaryuv420_to_linearrgb_debug: CuFunction,
    rgb_f32_to_8bit: CuFunction,
}

impl Kernel {
    pub fn load() -> CuResult<Self> {
        //let path = "target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";
        let module = CuModule::load_ptx(include_str!(concat!(
            env!("OUT_DIR"),
            "/cuda_colorspace_kernel.ptx"
        )))?;
        Ok(Self {
            biplanaryuv420_to_linearrgb_8_L_BT709: module
                .function_by_name("biplanaryuv420_to_linearrgb_8_L_BT709")?,
            biplanaryuv420_to_linearrgb_8_F_BT709: module
                .function_by_name("biplanaryuv420_to_linearrgb_8_F_BT709")?,
            biplanaryuv420_to_linearrgb_10_L_BT709: module
                .function_by_name("biplanaryuv420_to_linearrgb_10_L_BT709")?,
            biplanaryuv420_to_linearrgb_debug: module
                .function_by_name("biplanaryuv420_to_linearrgb_debug")?,
            rgb_f32_to_8bit: module.function_by_name("rgb_f32_to_8bit")?,
            module,
        })
    }

    pub fn biplanaryuv420_to_linearrgb_8_L_BT709(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let mut y = iter.next().unwrap();
            let mut uv = iter.next().unwrap();
            let mut w = dst.width() / 2;
            let mut h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_8_L_BT709.launch(
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

    pub fn biplanaryuv420_to_linearrgb_8_F_BT709(
        &self,
        src: impl Img<u8, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let mut y = iter.next().unwrap();
            let mut uv = iter.next().unwrap();
            let mut w = dst.width() / 2;
            let mut h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_8_F_BT709.launch(
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

    pub fn biplanaryuv420_to_linearrgb_10_L_BT709(
        &self,
        src: impl Img<u16, P<2>>,
        mut dst: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut iter = src.alloc_ptrs();
            let mut y = iter.next().unwrap();
            let mut uv = iter.next().unwrap();
            let mut w = dst.width() / 2;
            let mut h = dst.height() / 2;
            self.biplanaryuv420_to_linearrgb_10_L_BT709.launch(
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
            let mut y = iter.next().unwrap();
            let mut uv = iter.next().unwrap();
            let mut w = dst.width() / 2;
            let mut h = dst.height() / 2;
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

    pub fn rgb_f32_to_8bit(
        &self,
        src: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<u8, C<3>>,
        stream: &CuStream,
    ) -> CuResult<()> {
        debug_assert_same_size!(src, dst);
        unsafe {
            let mut w = dst.width() * 3;
            let mut h = dst.height();
            self.rgb_f32_to_8bit.launch(
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
    const MAX_THREADS_PER_BLOCK: u32 = 256;
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
    fn test_to_rgb() -> CuResult<()> {
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

        kernel.biplanaryuv420_to_linearrgb_8_L_BT709(&src, &mut dst, &main)?;
        main.sync()?;

        let data = dst.copy_to_cpu(cudarse_npp::get_stream()).unwrap();
        main.sync()?;
        dbg!(&data[0..3]);

        Ok(())
    }
}
