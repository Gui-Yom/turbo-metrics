use std::fs;

use cudarc::driver::{CudaDevice, LaunchAsync};
use image::{ImageFormat, Rgb32FImage};

use crate::kernel::{Img, Kernel};

mod kernel;

fn to_planes(img: Rgb32FImage) -> [Vec<f32>; 3] {
    let size = (img.width() * img.height()) as usize;
    let mut rgb = [vec![0.0; size], vec![0.0; size], vec![0.0; size]];
    for (i, chunk) in img.chunks_exact(3).enumerate() {
        if let &[r, g, b] = chunk {
            rgb[0][i] = r;
            rgb[1][i] = g;
            rgb[2][i] = b;
        }
    }
    rgb
}

fn main() {
    let dev = CudaDevice::new(0).unwrap();
    let kernel = Kernel::load(&dev);

    println!("Loading and converting images ...");
    let (mut ref_linear, mut dis_linear) = {
        let ref_data = fs::read("crates/ssimulacra2-cuda/source.png").unwrap();
        let ref_img = image::load_from_memory_with_format(&ref_data, ImageFormat::Png)
            .unwrap()
            .to_rgb32f();
        let width = ref_img.width();
        let height = ref_img.height();

        assert!(width > 8);
        assert!(height > 8);

        let [ref_r, ref_g, ref_b] = to_planes(ref_img);

        let dis_data = fs::read("crates/ssimulacra2-cuda/distorted.png").unwrap();
        let dis_img = image::load_from_memory_with_format(&dis_data, ImageFormat::Png)
            .unwrap()
            .to_rgb32f();

        assert_eq!(width, dis_img.width());
        assert_eq!(height, dis_img.height());

        let [dis_r, dis_g, dis_b] = to_planes(dis_img);

        // Copy reference image planes
        let rr = dev.htod_sync_copy(&ref_r).unwrap();
        let rg = dev.htod_sync_copy(&ref_g).unwrap();
        let rb = dev.htod_sync_copy(&ref_b).unwrap();

        let dr = dev.htod_sync_copy(&dis_r).unwrap();
        let dg = dev.htod_sync_copy(&dis_g).unwrap();
        let db = dev.htod_sync_copy(&dis_b).unwrap();

        // Convert images to linear rgb
        let ref_linear = kernel.srgb_to_linear(&Img::new(width, height, [rr, rg, rb]));
        let dis_linear = kernel.srgb_to_linear(&Img::new(width, height, [dr, dg, db]));
        (ref_linear, dis_linear)
    };

    println!("Start processing ...");

    for scale in 0..6 {
        if scale > 0 {
            ref_linear = kernel.downscale_by_2(&ref_linear);
            dis_linear = kernel.downscale_by_2(&dis_linear);
        }

        let ref_xyb = kernel.linear_to_xyb(&ref_linear);
        let dis_xyb = kernel.linear_to_xyb(&dis_linear);

        let ref_sq = kernel.mul_planes(&ref_xyb, &ref_xyb);
        let dis_sq = kernel.mul_planes(&dis_xyb, &dis_xyb);
        let ref_dis = kernel.mul_planes(&ref_xyb, &dis_xyb);
        dev.synchronize().unwrap();

        cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);
        cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);
        cuda_npp::npp_new_img(ref_sq.width, ref_sq.height);

        let mut ref_sigma = Img::new_with_shape(&dev, &ref_sq);
        for (pin, pout) in ref_sq.planes.iter().zip(&mut ref_sigma.planes) {
            cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_sq.width, ref_sq.height);
        }

        let mut dis_sigma = Img::new_with_shape(&dev, &dis_sq);
        for (pin, pout) in dis_sq.planes.iter().zip(&mut dis_sigma.planes) {
            cuda_npp::npp_gauss_5x5(&dev, pin, pout, dis_sq.width, dis_sq.height);
        }

        let mut ref_dis_sigma = Img::new_with_shape(&dev, &ref_dis);
        for (pin, pout) in ref_dis.planes.iter().zip(&mut ref_dis_sigma.planes) {
            cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_dis.width, ref_dis.height);
        }

        let mut ref_mu = Img::new_with_shape(&dev, &ref_xyb);
        for (pin, pout) in ref_xyb.planes.iter().zip(&mut ref_mu.planes) {
            cuda_npp::npp_gauss_5x5(&dev, pin, pout, ref_xyb.width, ref_xyb.height);
        }

        let mut dis_mu = Img::new_with_shape(&dev, &dis_xyb);
        for (pin, pout) in dis_xyb.planes.iter().zip(&mut dis_mu.planes) {
            cuda_npp::npp_gauss_5x5(&dev, pin, pout, dis_xyb.width, dis_xyb.height);
        }
    }

    dev.synchronize().unwrap();
}
