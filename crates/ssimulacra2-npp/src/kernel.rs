use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaView, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use cuda_npp::safe::Image;
use cuda_npp::C;

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Kernel {
    dev: Arc<CudaDevice>,
    linear_to_xyb: CudaFunction,
}

impl Kernel {
    pub fn load(dev: &Arc<CudaDevice>) -> Self {
        dev.load_ptx(
            Ptx::from_file("../ssimulacra2-cuda-kernel/ssimulacra2.ptx"),
            PTX_MODULE_NAME,
            &["linear_to_xyb_packed"],
        )
        .unwrap();

        Self {
            dev: dev.clone(),
            linear_to_xyb: dev
                .get_func(PTX_MODULE_NAME, "linear_to_xyb_packed")
                .unwrap(),
        }
    }

    pub fn linear_to_xyb(&self, src: &Image<f32, C<3>>) -> Image<f32, C<3>> {
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
