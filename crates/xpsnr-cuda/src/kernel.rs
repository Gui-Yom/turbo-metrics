use cudarse_driver::{kernel_params, CuFunction, CuModule, CuStream, LaunchConfig};
use cudarse_npp::image::{Img, ImgMut, C};
use cudarse_npp::{debug_assert_same_size, ScratchBuffer};

pub struct Kernel {
    _module: CuModule,
    xpsnr_support_8: CuFunction,
    xpsnr_postprocess: CuFunction,
}

impl Kernel {
    pub fn load() -> Self {
        //let path = "target/nvptx64-nvidia-cuda/release-nvptx/ssimulacra2_cuda_kernel.ptx";
        let module = CuModule::load_ptx(include_str!(concat!(
            env!("OUT_DIR"),
            "/xpsnr_cuda_kernel.ptx"
        )))
        .unwrap();
        Self {
            xpsnr_support_8: module.function_by_name("xpsnr_support_8").unwrap(),
            xpsnr_postprocess: module.function_by_name("xpsnr_postprocess").unwrap(),
            _module: module,
        }
    }

    pub fn xpsnr_support_8(
        &self,
        stream: &CuStream,
        ref_: impl Img<u8, C<1>>,
        mut prev: impl ImgMut<u8, C<1>>,
        dis: impl Img<u8, C<1>>,
        highpass: impl Img<i16, C<1>>,
        sse: &mut ScratchBuffer,
        sact: &mut ScratchBuffer,
        tact: &mut ScratchBuffer,
    ) {
        debug_assert_same_size!(ref_, prev);
        debug_assert_same_size!(ref_, dis);
        debug_assert_same_size!(ref_, highpass);
        assert_eq!(sse.len, sact.len);
        assert_eq!(sse.len, tact.len);
        unsafe {
            self.xpsnr_support_8
                .launch(
                    &launch_config_2d(ref_.width(), ref_.height()),
                    stream,
                    kernel_params!(
                        ref_.device_ptr(),
                        ref_.pitch() as usize,
                        prev.device_ptr_mut(),
                        prev.pitch() as usize,
                        dis.device_ptr(),
                        dis.pitch() as usize,
                        highpass.device_ptr(),
                        highpass.pitch() as usize,
                        sse.ptr,
                        sact.ptr,
                        tact.ptr,
                        ref_.width() as usize,
                        ref_.height() as usize,
                    ),
                )
                .expect("Could not launch srgb_to_linear kernel");
        }
    }

    pub fn xpsnr_postprocess(
        &self,
        stream: &CuStream,
        sse: &ScratchBuffer,
        sact: &ScratchBuffer,
        tact: &ScratchBuffer,
        wsse: &mut ScratchBuffer,
    ) {
        assert_eq!(sse.len, sact.len);
        assert_eq!(sse.len, tact.len);
        assert_eq!(wsse.len, 4);
        unsafe {
            self.xpsnr_postprocess
                .launch(
                    &launch_config_1d(sse.len as u32),
                    stream,
                    kernel_params!(sse.ptr, sact.ptr, tact.ptr, sse.len, wsse.ptr),
                )
                .expect("Could not launch srgb_to_linear kernel");
        }
    }
}

fn launch_config_1d(width: u32) -> LaunchConfig {
    // const MAX_THREADS_PER_BLOCK: u32 = 256;
    const THREADS_WIDTH: u32 = 128;
    let num_blocks_w = (width + THREADS_WIDTH - 1) / THREADS_WIDTH;
    LaunchConfig {
        grid_dim: (num_blocks_w, 1, 1),
        block_dim: (THREADS_WIDTH, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_config_2d(width: u32, height: u32) -> LaunchConfig {
    // const MAX_THREADS_PER_BLOCK: u32 = 256;
    const THREADS_WIDTH: u32 = 16;
    const THREADS_HEIGHT: u32 = 16;
    let num_blocks_w = (width + THREADS_WIDTH - 1) / THREADS_WIDTH;
    let num_blocks_h = (height + THREADS_HEIGHT - 1) / THREADS_HEIGHT;
    LaunchConfig {
        grid_dim: (num_blocks_w, num_blocks_h, 1),
        block_dim: (THREADS_WIDTH, THREADS_HEIGHT, 1),
        shared_mem_bytes: 0,
    }
}
