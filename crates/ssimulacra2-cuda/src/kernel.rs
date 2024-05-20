use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use cuda_npp::{assert_same_size, C, P};
use cuda_npp::safe::{Img, ImgMut};

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Kernel {
    dev: Arc<CudaDevice>,
    srgb_to_linear: CudaFunction,
    downscale_by_2: CudaFunction,
    downscale_plane_by_2: CudaFunction,
    linear_to_xyb: CudaFunction,
    linear_to_xyb_planar: CudaFunction,
    ssim_map: CudaFunction,
    edge_diff_map: CudaFunction,
}

impl Kernel {
    pub fn load(dev: &Arc<CudaDevice>) -> Self {
        let path = "target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";

        dev.load_ptx(
            Ptx::from_file(path),
            PTX_MODULE_NAME,
            &[
                "srgb_to_linear",
                "downscale_by_2",
                "downscale_plane_by_2",
                "linear_to_xyb_packed",
                "linear_to_xyb_planar",
                "ssim_map",
                "edge_diff_map",
            ],
        )
            .unwrap();

        Self {
            dev: dev.clone(),
            srgb_to_linear: dev
                .get_func(PTX_MODULE_NAME, "srgb_to_linear")
                .unwrap(),
            downscale_by_2: dev
                .get_func(PTX_MODULE_NAME, "downscale_by_2")
                .unwrap(),
            downscale_plane_by_2: dev
                .get_func(PTX_MODULE_NAME, "downscale_plane_by_2")
                .unwrap(),
            linear_to_xyb: dev
                .get_func(PTX_MODULE_NAME, "linear_to_xyb_packed")
                .unwrap(),
            linear_to_xyb_planar: dev
                .get_func(PTX_MODULE_NAME, "linear_to_xyb_planar")
                .unwrap(),
            ssim_map: dev.get_func(PTX_MODULE_NAME, "ssim_map").unwrap(),
            edge_diff_map: dev.get_func(PTX_MODULE_NAME, "edge_diff_map").unwrap(),
        }
    }

    pub fn srgb_to_linear(&self, src: impl Img<u8, C<3>>, mut dst: impl ImgMut<f32, C<3>>) {
        assert_same_size!(src, dst);
        unsafe {
            self.srgb_to_linear
                .clone()
                .launch(
                    launch_config_2d(src.width() * 3, src.height()),
                    (
                        src.device_ptr() as usize,
                        src.pitch() as usize,
                        dst.device_ptr_mut() as usize,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch srgb_to_linear_packed_coalesced kernel");
        }
    }

    pub fn downscale_by_2(&self, src: impl Img<f32, C<3>>, mut dst: impl ImgMut<f32, C<3>>) {
        unsafe {
            self.downscale_by_2
                .clone()
                .launch(
                    launch_config_2d(dst.width(), dst.height()),
                    (
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr() as usize,
                        src.pitch() as usize,
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.device_ptr_mut() as usize,
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
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (num_blocks_w, num_blocks_h, 1),
                        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr() as usize,
                        src.pitch() as usize,
                        dst.width() as usize,
                        dst.height() as usize,
                        dst.device_ptr_mut() as usize,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch downscale_plane_by_2 kernel");
        }
    }

    pub fn downscale_plane_by_2_planar(&self, src: impl Img<f32, P<3>>, mut dst: impl ImgMut<f32, P<3>>) {
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
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (num_blocks_w, num_blocks_h, 1),
                            block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            src.width() as usize,
                            src.height() as usize,
                            r as usize,
                            src.pitch() as usize,
                            width as usize,
                            height as usize,
                            w as usize,
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
                .clone()
                .launch(
                    launch_config_2d(src.width(), src.height()),
                    (
                        src.width() as usize,
                        src.height() as usize,
                        src.device_ptr() as usize,
                        src.pitch() as usize,
                        dst.device_ptr_mut() as usize,
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
                .clone()
                .launch(
                    launch_config_2d(src.width(), src.height()),
                    (
                        src.width() as usize,
                        src.height() as usize,
                        src_r as usize,
                        src_g as usize,
                        src_b as usize,
                        src.pitch() as usize,
                        dst_x as usize,
                        dst_y as usize,
                        dst_b as usize,
                        dst.pitch() as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
    }

    pub fn ssim_map(
        &self,
        mu1: impl Img<f32, C<3>>,
        mu2: impl Img<f32, C<3>>,
        sigma11: impl Img<f32, C<3>>,
        sigma22: impl Img<f32, C<3>>,
        sigma12: impl Img<f32, C<3>>,
        mut dst: impl ImgMut<f32, C<3>>,
    ) {
        assert_same_size!(mu1, mu2);
        assert_same_size!(mu2, sigma11);
        assert_same_size!(sigma11, sigma22);
        assert_same_size!(sigma22, sigma12);
        assert_same_size!(sigma12, dst);
        unsafe {
            self.ssim_map
                .clone()
                .launch(
                    launch_config_2d(mu1.width(), mu1.height()),
                    (
                        mu1.width() as usize,
                        mu1.height() as usize,
                        mu1.pitch() as usize,
                        mu1.device_ptr() as usize,
                        mu2.device_ptr() as usize,
                        sigma11.device_ptr() as usize,
                        sigma22.device_ptr() as usize,
                        sigma12.device_ptr() as usize,
                        dst.device_ptr_mut() as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
    }

    pub fn edge_diff_map(
        &self,
        src: impl Img<f32, C<3>>,
        mu1: impl Img<f32, C<3>>,
        distorted: impl Img<f32, C<3>>,
        mu2: impl Img<f32, C<3>>,
        mut artifact: impl ImgMut<f32, C<3>>,
        mut detail_loss: impl ImgMut<f32, C<3>>,
    ) {
        assert_same_size!(src, mu1);
        assert_same_size!(mu1, distorted);
        assert_same_size!(distorted, mu2);
        assert_same_size!(mu2, artifact);
        assert_same_size!(artifact, detail_loss);
        unsafe {
            self.edge_diff_map
                .clone()
                .launch(
                    launch_config_2d(src.width(), src.height()),
                    (
                        src.width() as usize,
                        src.height() as usize,
                        src.pitch() as usize,
                        src.device_ptr() as usize,
                        mu1.device_ptr() as usize,
                        distorted.device_ptr() as usize,
                        mu2.device_ptr() as usize,
                        artifact.device_ptr_mut() as usize,
                        detail_loss.device_ptr_mut() as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
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
