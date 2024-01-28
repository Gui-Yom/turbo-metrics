use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::Img;

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Kernel {
    dev: Arc<CudaDevice>,
    linear_to_xyb: CudaFunction,
    ssim_map: CudaFunction,
    edge_diff_map: CudaFunction,
}

impl Kernel {
    pub fn load(dev: &Arc<CudaDevice>) -> Self {
        dev.load_ptx(
            Ptx::from_file("../ssimulacra2-cuda-kernel/ssimulacra2.ptx"),
            PTX_MODULE_NAME,
            &["linear_to_xyb_packed", "ssim_map", "edge_diff_map"],
        )
        .unwrap();

        Self {
            dev: dev.clone(),
            linear_to_xyb: dev
                .get_func(PTX_MODULE_NAME, "linear_to_xyb_packed")
                .unwrap(),
            ssim_map: dev.get_func(PTX_MODULE_NAME, "ssim_map").unwrap(),
            edge_diff_map: dev.get_func(PTX_MODULE_NAME, "edge_diff_map").unwrap(),
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
    const MAX_THREADS_PER_BLOCK: u32 = 32;
    let num_blocks_w = (width + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    let num_blocks_h = (height + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    LaunchConfig {
        grid_dim: (num_blocks_w, num_blocks_h, 1),
        block_dim: (MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    }
}
