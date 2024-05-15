use std::mem;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use zune_image::codecs::png::zune_core::options::DecoderOptions;

use cuda_npp::C;
use cuda_npp::safe::{Image, Img, ImgMut};
use cuda_npp::safe::ial::Mul;
use cuda_npp::safe::isu::Malloc;
use cuda_npp::sys::NppiRect;

use crate::kernel::Kernel;

mod kernel;

fn main() {
    let dev = CudaDevice::new(0).unwrap();

    let ref_img = zune_image::image::Image::open_with_options("crates/ssimulacra2-cuda/source.png", DecoderOptions::new_fast()).unwrap();
    let dis_img = zune_image::image::Image::open_with_options("crates/ssimulacra2-cuda/distorted.png", DecoderOptions::new_fast()).unwrap();

    // Upload to gpu
    let (width, height) = ref_img.dimensions();
    let ref_bytes = &ref_img.flatten_to_u8()[0];

    let (dwidth, dheight) = dis_img.dimensions();
    assert_eq!((width, height), (dwidth, dheight));
    let dis_bytes = &dis_img.flatten_to_u8()[0];

    let mut ssimulacra2 = Ssimulacra2::new(&dev, width as u32, height as u32);
    dbg!(ssimulacra2.compute(ref_bytes, dis_bytes));
}

struct Ssimulacra2 {
    dev: Arc<CudaDevice>,
    kernel: Kernel,
    ref_input: Image<u8, C<3>>,
    dis_input: Image<u8, C<3>>,
    ref_linear: Image<f32, C<3>>,
    dis_linear: Image<f32, C<3>>,
    ref_linear_resized: Image<f32, C<3>>,
    dis_linear_resized: Image<f32, C<3>>,
    ref_xyb: Image<f32, C<3>>,
    dis_xyb: Image<f32, C<3>>,
}

impl Ssimulacra2 {
    pub fn new(dev: &Arc<CudaDevice>, width: u32, height: u32) -> Self {
        let ref_input = Image::<u8, C<3>>::malloc(width, height).unwrap();
        let dis_input = Image::<u8, C<3>>::malloc(width, height).unwrap();

        let ref_linear = ref_input.malloc_same_size().unwrap();
        let dis_linear = dis_input.malloc_same_size().unwrap();

        let ref_linear_resized = ref_linear.malloc_same_size().unwrap();
        let dis_linear_resized = dis_linear.malloc_same_size().unwrap();

        let ref_xyb = ref_linear.malloc_same_size().unwrap();
        let dis_xyb = dis_linear.malloc_same_size().unwrap();

        Self {
            dev: Arc::clone(dev),
            kernel: Kernel::load(dev),
            ref_input,
            dis_input,
            ref_linear,
            dis_linear,
            ref_linear_resized,
            dis_linear_resized,
            ref_xyb,
            dis_xyb,
        }
    }

    pub fn compute(&mut self, ref_bytes: &[u8], dis_bytes: &[u8]) -> f64 {
        self.ref_input.copy_from_cpu(ref_bytes).unwrap();
        self.dis_input.copy_from_cpu(dis_bytes).unwrap();

        // Convert to linear
        self.kernel.srgb_to_linear(&self.ref_input, &mut self.ref_linear);
        self.kernel.srgb_to_linear(&self.dis_input, &mut self.dis_linear);

        // linear -> xyb -> ...
        //    |-> /2 -> xyb -> ...
        //         |-> /2 -> xyb -> ...

        let mut width = self.ref_input.width();
        let mut height = self.ref_input.height();

        for scale in 0..6 {
            if scale > 0 {
                {
                    let rect = NppiRect {
                        x: 0,
                        y: 0,
                        width: width as _,
                        height: height as _,
                    };

                    let ref_linear = self.ref_linear.view_mut(rect);
                    let dis_linear = self.dis_linear.view_mut(rect);

                    width = (width + 1) / 2;
                    height = (height + 1) / 2;

                    dbg!(width, height);

                    let rect = NppiRect {
                        x: 0,
                        y: 0,
                        width: width as _,
                        height: height as _,
                    };

                    self.kernel.downscale_by_2(&ref_linear, self.ref_linear_resized.view_mut(rect));
                    self.kernel.downscale_by_2(&dis_linear, self.dis_linear_resized.view_mut(rect));
                }

                mem::swap(&mut self.ref_linear, &mut self.ref_linear_resized);
                mem::swap(&mut self.dis_linear, &mut self.dis_linear_resized);

                let rect = NppiRect {
                    x: 0,
                    y: 0,
                    width: width as _,
                    height: height as _,
                };

                // dbg!(ref_linear_resized.device_ptr());
                self.kernel.linear_to_xyb(self.ref_linear_resized.view(rect), self.ref_xyb.view_mut(rect));
                self.kernel.linear_to_xyb(self.dis_linear_resized.view(rect), self.dis_xyb.view_mut(rect));
            } else {
                // dbg!(ref_linear_resized.device_ptr());
                self.kernel.linear_to_xyb(&self.ref_linear, &mut self.ref_xyb);
                self.kernel.linear_to_xyb(&self.dis_linear, &mut self.dis_xyb);
            }

            // source_xyb.mul();

            // let ref_sq = kernel.mul_planes(&ref_xyb, &ref_xyb);
            // let dis_sq = kernel.mul_planes(&dis_xyb, &dis_xyb);
            // let ref_dis = kernel.mul_planes(&ref_xyb, &dis_xyb);
            // dev.synchronize().unwrap();
            //
            // cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);
            // cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);
            // cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);
            //
            // let mut ref_sigma = Img::new_with_shape(&dev, &ref_sq);
            // for (pin, pout) in ref_sq.planes.iter().zip(&mut ref_sigma.planes) {
            //     cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_sq.width, ref_sq.height);
            // }
            //
            // let mut dis_sigma = Img::new_with_shape(&dev, &dis_sq);
            // for (pin, pout) in dis_sq.planes.iter().zip(&mut dis_sigma.planes) {
            //     cuda_npp::npp_gauss_5x5(&dev, pin, pout, dis_sq.width, dis_sq.height);
            // }
            //
            // let mut ref_dis_sigma = Img::new_with_shape(&dev, &ref_dis);
            // for (pin, pout) in ref_dis.planes.iter().zip(&mut ref_dis_sigma.planes) {
            //     cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_dis.width, ref_dis.height);
            // }
            //
            // let mut ref_mu = Img::new_with_shape(&dev, &ref_xyb);
            // for (pin, pout) in ref_xyb.planes.iter().zip(&mut ref_mu.planes) {
            //     cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_xyb.width, ref_xyb.height);
            // }
            //
            // let mut dis_mu = Img::new_with_shape(&dev, &dis_xyb);
            // for (pin, pout) in dis_xyb.planes.iter().zip(&mut dis_mu.planes) {
            //     cuda_npp::npp_gauss_5x5(&dev, pin, pout, dis_xyb.width, dis_xyb.height);
            // }
        }

        self.dev.synchronize().unwrap();

        0.0
    }
}
