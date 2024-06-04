use cuda_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cuda_npp::safe::{Img, ImgMut};
use cuda_npp::{assert_same_size, C, P};

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Kernel {
    module: CuModule,
    srgb_to_linear: CuFunction,
    downscale_by_2: CuFunction,
    downscale_plane_by_2: CuFunction,
    linear_to_xyb: CuFunction,
    linear_to_xyb_planar: CuFunction,
    blur_plane_pass_fused: CuFunction,
    compute_error_maps: CuFunction,
}

impl Kernel {
    pub fn load() -> Self {
        let path = "target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";
        let module = CuModule::load_from_file(path).unwrap();

        Self {
            srgb_to_linear: module.function_by_name("srgb_to_linear").unwrap(),
            downscale_by_2: module.function_by_name("downscale_by_2").unwrap(),
            downscale_plane_by_2: module.function_by_name("downscale_plane_by_2").unwrap(),
            linear_to_xyb: module.function_by_name("linear_to_xyb_packed").unwrap(),
            linear_to_xyb_planar: module.function_by_name("linear_to_xyb_planar").unwrap(),
            blur_plane_pass_fused: module.function_by_name("blur_plane_pass_fused").unwrap(),
            compute_error_maps: module.function_by_name("compute_error_maps").unwrap(),
            module,
        }
    }

    pub fn srgb_to_linear(&self, src: impl Img<u8, C<3>>, mut dst: impl ImgMut<f32, C<3>>) {
        assert_same_size!(src, dst);
        unsafe {
            self.srgb_to_linear
                .launch(
                    &launch_config_2d(src.width() * 3, src.height()),
                    CuStream::DEFAULT,
                    kernel_params!(
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch srgb_to_linear kernel");
        }
    }

    pub fn downscale_by_2(&self, src: impl Img<f32, C<3>>, mut dst: impl ImgMut<f32, C<3>>) {
        unsafe {
            self.downscale_by_2
                .launch(
                    &launch_config_2d(dst.width(), dst.height()),
                    CuStream::DEFAULT,
                    kernel_params!(
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch downscale_by_2 kernel");
        }
    }

    pub fn downscale_plane_by_2(&self, src: impl Img<f32, C<1>>, mut dst: impl ImgMut<f32, C<1>>) {
        unsafe {
            const THREADS_WIDTH: u32 = 16;
            const THREADS_HEIGHT: u32 = 16;
            let num_blocks_w = (src.width() + THREADS_WIDTH - 1) / THREADS_WIDTH;
            let num_blocks_h = (src.height() + THREADS_HEIGHT - 1) / THREADS_HEIGHT;

            self.downscale_plane_by_2
                .launch(
                    &LaunchConfig {
                        grid_dim: (num_blocks_w, num_blocks_h, 1),
                        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
                        shared_mem_bytes: 0,
                    },
                    CuStream::DEFAULT,
                    kernel_params!(
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch downscale_plane_by_2 kernel");
        }
    }

    pub fn downscale_plane_by_2_planar(
        &self,
        src: impl Img<f32, P<3>>,
        mut dst: impl ImgMut<f32, P<3>>,
    ) {
        unsafe {
            const THREADS_WIDTH: u32 = 16;
            const THREADS_HEIGHT: u32 = 16;
            let num_blocks_w = (src.width() + THREADS_WIDTH - 1) / THREADS_WIDTH;
            let num_blocks_h = (src.height() + THREADS_HEIGHT - 1) / THREADS_HEIGHT;

            let width = dst.width();
            let height = dst.height();
            let pitch = dst.pitch();

            for (r, w) in src.alloc_ptrs().zip(dst.alloc_ptrs_mut()) {
                self.downscale_plane_by_2
                    .launch(
                        &LaunchConfig {
                            grid_dim: (num_blocks_w, num_blocks_h, 1),
                            block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
                            shared_mem_bytes: 0,
                        },
                        CuStream::DEFAULT,
                        kernel_params!(
                            src.width() as usize,
                            src.height() as usize,
                            r,
                            src.pitch() as usize,
                            width as usize,
                            height as usize,
                            w,
                            pitch as usize,
                        ),
                    )
                    .expect("Could not launch downscale_plane_by_2 kernel");
            }
        }
    }

    pub fn linear_to_xyb(&self, src: impl Img<f32, C<3>>, mut dst: impl ImgMut<f32, C<3>>) {
        assert_same_size!(src, dst);
        unsafe {
            self.linear_to_xyb
                .launch(
                    &launch_config_2d(src.width(), src.height()),
                    CuStream::DEFAULT,
                    kernel_params!(
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr(),
                        src.pitch() as usize,
                        dst.device_ptr_mut(),
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
    }

    pub fn linear_to_xyb_planar(&self, src: impl Img<f32, P<3>>, mut dst: impl ImgMut<f32, P<3>>) {
        assert_same_size!(src, dst);
        let [src_r, src_g, src_b] = src.storage();
        let [dst_x, dst_y, dst_b] = src.storage();
        unsafe {
            self.linear_to_xyb_planar
                .launch(
                    &launch_config_2d(src.width(), src.height()),
                    CuStream::DEFAULT,
                    kernel_params!(
                        src.width() as usize,
                        src.height() as usize,
                        src_r,
                        src_g,
                        src_b,
                        src.pitch() as usize,
                        dst_x,
                        dst_y,
                        dst_b,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
    }

    /// Blur 5 packed images in one kernel launch.
    /// We can treat the images as a single plane because each thread only processes a single column.
    pub fn blur_pass_fused(
        &self,
        src0: impl Img<f32, C<3>>,
        mut dst0: impl ImgMut<f32, C<3>>,
        src1: impl Img<f32, C<3>>,
        mut dst1: impl ImgMut<f32, C<3>>,
        src2: impl Img<f32, C<3>>,
        mut dst2: impl ImgMut<f32, C<3>>,
        src3: impl Img<f32, C<3>>,
        mut dst3: impl ImgMut<f32, C<3>>,
        src4: impl Img<f32, C<3>>,
        mut dst4: impl ImgMut<f32, C<3>>,
    ) {
        assert_same_size!(src0, dst0);
        assert_same_size!(src1, dst1);
        assert_same_size!(src2, dst2);
        assert_same_size!(src3, dst3);
        assert_same_size!(src4, dst4);

        // A warp 32 wide is nice because cache line is 128 bytes (32 * 4).
        // The y coordinate selects the image pair to operate on.

        // 3 warps per block seems to be the sweet spot, more profiling is needed.
        // Remember to keep this value in sync with the BLOCK_WIDTH constant in the kernel.
        const THREADS_WIDTH: u32 = 3 * 32;
        const THREADS_HEIGHT: u32 = 1;
        let num_blocks_w = (src0.width() * 3 + THREADS_WIDTH - 1) / THREADS_WIDTH;
        let num_blocks_h = 5;

        unsafe {
            self.blur_plane_pass_fused
                .launch(
                    &LaunchConfig {
                        grid_dim: (num_blocks_w, num_blocks_h, 1),
                        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
                        shared_mem_bytes: 0,
                    },
                    CuStream::DEFAULT,
                    kernel_params!(
                        src0.width() as usize * 3,
                        src0.height() as usize,
                        src0.device_ptr(),
                        src1.device_ptr(),
                        src2.device_ptr(),
                        src3.device_ptr(),
                        src4.device_ptr(),
                        src0.pitch() as usize,
                        dst0.device_ptr_mut(),
                        dst1.device_ptr_mut(),
                        dst2.device_ptr_mut(),
                        dst3.device_ptr_mut(),
                        dst4.device_ptr_mut(),
                        dst0.pitch() as usize,
                    ),
                )
                .expect("Could not launch blur_plane_pass_fused kernel");
        }
    }

    pub fn compute_error_maps(
        &self,
        source: impl Img<f32, C<3>>,
        distorted: impl Img<f32, C<3>>,
        mu1: impl Img<f32, C<3>>,
        mu2: impl Img<f32, C<3>>,
        sigma11: impl Img<f32, C<3>>,
        sigma22: impl Img<f32, C<3>>,
        sigma12: impl Img<f32, C<3>>,
        mut ssim: impl ImgMut<f32, C<3>>,
        mut artifact: impl ImgMut<f32, C<3>>,
        mut detail_loss: impl ImgMut<f32, C<3>>,
    ) {
        assert_same_size!(source, distorted);
        assert_same_size!(distorted, mu1);
        assert_same_size!(mu1, mu2);
        assert_same_size!(mu2, sigma11);
        assert_same_size!(sigma11, sigma22);
        assert_same_size!(sigma22, sigma12);
        assert_same_size!(sigma12, ssim);
        assert_same_size!(ssim, artifact);
        assert_same_size!(artifact, detail_loss);
        unsafe {
            self.compute_error_maps
                .launch(
                    &launch_config_2d(mu1.width() * 3, mu1.height()),
                    CuStream::DEFAULT,
                    kernel_params!(
                        mu1.width() as usize * 3,
                        mu1.height() as usize,
                        mu1.pitch() as usize,
                        source.device_ptr(),
                        distorted.device_ptr(),
                        mu1.device_ptr(),
                        mu2.device_ptr(),
                        sigma11.device_ptr(),
                        sigma22.device_ptr(),
                        sigma12.device_ptr(),
                        ssim.device_ptr_mut(),
                        artifact.device_ptr_mut(),
                        detail_loss.device_ptr_mut(),
                    ),
                )
                .expect("Could not launch compute_error_maps kernel");
        }
    }
}

fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
    const MAX_THREADS_PER_BLOCK: u32 = 256;
    const THREADS_WIDTH: u32 = 32;
    const THREADS_HEIGHT: u32 = 8;
    let num_blocks_w = (width + THREADS_WIDTH - 1) / THREADS_WIDTH;
    let num_blocks_h = (height + THREADS_HEIGHT - 1) / THREADS_HEIGHT;
    LaunchConfig {
        grid_dim: (num_blocks_w, num_blocks_h, 1),
        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
        shared_mem_bytes: 0,
    }
}
