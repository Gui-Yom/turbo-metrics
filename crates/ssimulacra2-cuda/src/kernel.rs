use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

const PTX_MODULE_NAME: &str = "ssimulacra2";

pub struct Img {
    pub width: u32,
    pub height: u32,
    pub planes: [CudaSlice<f32>; 3],
}

impl Img {
    pub fn new(width: u32, height: u32, planes: [CudaSlice<f32>; 3]) -> Self {
        Self {
            width,
            height,
            planes,
        }
    }

    pub fn new_empty(dev: &Arc<CudaDevice>, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            planes: alloc_planes(dev, (width * height) as usize).unwrap(),
        }
    }

    pub fn new_with_planes(&self, planes: [CudaSlice<f32>; 3]) -> Self {
        Self { planes, ..*self }
    }

    pub fn map_planes(&self, map: impl FnMut(&CudaSlice<f32>) -> CudaSlice<f32>) -> Self {
        self.new_with_planes(array_init::map_array_init(&self.planes, map))
    }

    pub fn new_with_shape(dev: &Arc<CudaDevice>, img: &Self) -> Self {
        img.new_with_planes(alloc_planes(dev, img.plane_len()).unwrap())
    }

    pub fn plane_len(&self) -> usize {
        self.planes[0].len()
    }
}

pub struct Kernel {
    dev: Arc<CudaDevice>,
    plane_srgb_to_linear: CudaFunction,
    linear_to_xyb: CudaFunction,
    downscale_by_2: CudaFunction,
    mul_planes: CudaFunction,
}

impl Kernel {
    pub fn load(dev: &Arc<CudaDevice>) -> Self {
        dev.load_ptx(
            Ptx::from_file("ssimulacra2-cuda-kernel/ssimulacra2.ptx"),
            PTX_MODULE_NAME,
            &[
                "plane_srgb_to_linear",
                "linear_to_xyb",
                "downscale_by_2",
                "mul_planes",
            ],
        )
        .unwrap();

        Self {
            dev: dev.clone(),
            plane_srgb_to_linear: dev
                .get_func(PTX_MODULE_NAME, "plane_srgb_to_linear")
                .unwrap(),
            linear_to_xyb: dev.get_func(PTX_MODULE_NAME, "linear_to_xyb").unwrap(),
            downscale_by_2: dev.get_func(PTX_MODULE_NAME, "downscale_by_2").unwrap(),
            mul_planes: dev.get_func(PTX_MODULE_NAME, "mul_planes").unwrap(),
        }
    }

    pub fn srgb_to_linear(&self, src: &Img) -> Img {
        unsafe {
            src.map_planes(|pin| {
                let mut pout = self.dev.alloc(pin.len()).unwrap();
                self.plane_srgb_to_linear
                    .clone()
                    .launch(
                        LaunchConfig::for_num_elems(pin.len() as u32),
                        (pin.len(), pin, &mut pout),
                    )
                    .unwrap();
                pout
            })
        }
    }

    pub fn linear_to_xyb(&self, src: &Img) -> Img {
        let mut xyb = Img::new_with_shape(&self.dev, src);
        unsafe {
            let [x, y, b] = xyb.planes.as_mut_slice() else {
                unreachable!()
            };
            self.linear_to_xyb
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(src.plane_len() as u32),
                    (
                        src.plane_len(),
                        &src.planes[0],
                        &src.planes[1],
                        &src.planes[2],
                        x,
                        y,
                        b,
                    ),
                )
                .unwrap();
        }
        xyb
    }

    pub fn downscale_by_2(&self, src: &Img) -> Img {
        let new_width = (src.width + 1) / 2;
        let new_height = (src.height + 1) / 2;
        let mut dst = Img::new_empty(&self.dev, new_width, new_height);
        unsafe {
            for (pin, pout) in src.planes.iter().zip(&mut dst.planes) {
                self.downscale_by_2
                    .clone()
                    .launch(
                        launch_config_2d(new_width, new_height),
                        (
                            src.width as usize,
                            src.height as usize,
                            new_width as usize,
                            new_height as usize,
                            pin,
                            pout,
                        ),
                    )
                    .unwrap();
            }
        }
        dst
    }

    pub fn mul_planes(&self, a: &Img, b: &Img) -> Img {
        let mut dst = Img::new_with_shape(&self.dev, &a);
        unsafe {
            for ((pa, pb), pc) in a.planes.iter().zip(&b.planes).zip(&mut dst.planes) {
                self.mul_planes
                    .clone()
                    .launch(
                        launch_config_2d(dst.width, dst.height),
                        (dst.width as usize, dst.height as usize, pa, pb, pc),
                    )
                    .unwrap()
            }
        }
        dst
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

fn alloc_planes<const N: usize>(
    dev: &Arc<CudaDevice>,
    len: usize,
) -> Result<[CudaSlice<f32>; N], cudarc::driver::result::DriverError> {
    unsafe { array_init::try_array_init(|i| dev.alloc_zeros::<f32>(len)) }
}
