use std::sync::Arc;

use cuda_npp::safe::Image;
use cuda_npp::C;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::Img;

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Kernel {
    dev: Arc<CudaDevice>,
    packed_srgb_to_linear: CudaFunction,
    linear_to_xyb: CudaFunction,
    ssim_map: CudaFunction,
    edge_diff_map: CudaFunction,
}

impl Kernel {
    pub fn load(dev: &Arc<CudaDevice>) -> Self {
        let path = if cfg!(debug_assertions) {
            "../../target/nvptx64-nvidia-cuda/dev-nvptx/ssimulacra2_cuda_kernel.ptx"
        } else {
            "../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx"
        };

        let path = "../../target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";

        dev.load_ptx(
            Ptx::from_file(path),
            PTX_MODULE_NAME,
            &[
                "packed_srgb_to_linear",
                "linear_to_xyb_packed",
                "ssim_map",
                "edge_diff_map",
            ],
        )
        .unwrap();

        Self {
            dev: dev.clone(),
            packed_srgb_to_linear: dev
                .get_func(PTX_MODULE_NAME, "packed_srgb_to_linear")
                .unwrap(),
            linear_to_xyb: dev
                .get_func(PTX_MODULE_NAME, "linear_to_xyb_packed")
                .unwrap(),
            ssim_map: dev.get_func(PTX_MODULE_NAME, "ssim_map").unwrap(),
            edge_diff_map: dev.get_func(PTX_MODULE_NAME, "edge_diff_map").unwrap(),
        }
    }

    pub fn packed_srgb_to_linear(&self, src: &Image<u8, C<3>>, dst: &mut Image<f32, C<3>>) {
        unsafe {
            self.packed_srgb_to_linear
                .clone()
                .launch(
                    launch_config_2d(src.width, src.height),
                    (
                        src.width as usize,
                        src.height as usize,
                        src.data as usize,
                        src.line_step as usize,
                        dst.data as usize,
                        dst.line_step as usize,
                    ),
                )
                .expect("Could not launch packed_srgb_to_linear kernel");
        }
    }

    pub fn linear_to_xyb(&self, src: &Img) -> Img {
        let mut xyb = src.malloc_same().unwrap();
        unsafe {
            self.linear_to_xyb
                .clone()
                .launch(
                    launch_config_2d(src.width, src.height),
                    (
                        src.data as usize,
                        xyb.data as usize,
                        src.width as usize,
                        src.height as usize,
                        src.line_step as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
        xyb
    }

    pub fn ssim_map(
        &self,
        mu1: &Img,
        mu2: &Img,
        sigma11: &Img,
        sigma22: &Img,
        sigma12: &Img,
    ) -> Img {
        let mut out = mu1.malloc_same().unwrap();
        unsafe {
            self.ssim_map
                .clone()
                .launch(
                    launch_config_2d(mu1.width, mu1.height),
                    (
                        mu1.width as usize,
                        mu1.height as usize,
                        mu1.line_step as usize,
                        mu1.data as usize,
                        mu2.data as usize,
                        sigma11.data as usize,
                        sigma22.data as usize,
                        sigma12.data as usize,
                        out.data as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
        out
    }

    pub fn edge_diff_map(&self, src: &Img, mu1: &Img, distorted: &Img, mu2: &Img) -> (Img, Img) {
        let mut artifact = src.malloc_same().unwrap();
        let mut detail_loss = src.malloc_same().unwrap();
        unsafe {
            self.edge_diff_map
                .clone()
                .launch(
                    launch_config_2d(src.width, src.height),
                    (
                        src.width as usize,
                        src.height as usize,
                        src.line_step as usize,
                        src.data as usize,
                        mu1.data as usize,
                        distorted.data as usize,
                        mu2.data as usize,
                        artifact.data as usize,
                        detail_loss.data as usize,
                    ),
                )
                .expect("Could not launch linear_to_xyb kernel");
        }
        (artifact, detail_loss)
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
