use cuda_npp::safe::icc::GammaFwdIP;
use cuda_npp::safe::idei::Scale;
use cuda_npp::safe::ig::Resize;
use cuda_npp::safe::Image;
use cuda_npp::safe::Result;
use cuda_npp::sys::{NppiInterpolationMode, NppiMaskSize};
use cuda_npp::{get_stream_ctx, C};

use crate::kernel::Kernel;

mod kernel;

pub fn ssimulacra2(mut source: Image<u8, C<3>>, mut distorted: Image<u8, C<3>>) -> Result<f32> {
    let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
    let kernel = Kernel::load(&dev);
    let ctx = get_stream_ctx()?;
    println!("Starting work ...");
    source.gamma_fwd_ip(ctx)?;
    let mut source = source.scale_float_new(0.0..1.0, ctx)?;
    distorted.gamma_fwd_ip(ctx)?;
    let mut distorted = distorted.scale_float_new(0.0..1.0, ctx)?;
    dev.synchronize().unwrap();

    for scale in 0..6 {
        // Scaling in linear colorspace
        if scale > 0 {
            let new_width = (source.width + 1) / 2;
            let new_height = (source.height + 1) / 2;
            source = source.resize_new(
                new_width,
                new_height,
                NppiInterpolationMode::NPPI_INTER_LINEAR,
                ctx,
            )?;
            distorted = distorted.resize_new(
                new_width,
                new_height,
                NppiInterpolationMode::NPPI_INTER_LINEAR,
                ctx,
            )?;
            dev.synchronize().unwrap();
        }

        // Operations in XYB perceptual colorspace
        let source = kernel.linear_to_xyb(&source);
        let distorted = kernel.linear_to_xyb(&distorted);
        dev.synchronize().unwrap();

        let ref_sq = source.mul_new(&source, ctx)?;
        let sigma1_sq = ref_sq.filter_gauss_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let dis_sq = distorted.mul_new(&distorted, ctx)?;
        let sigma2_sq = dis_sq.filter_gauss_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let ref_dis = source.mul_new(&distorted, ctx)?;
        let sigma12 = ref_dis.filter_gauss_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;

        let mu1 = source.filter_gauss_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let mu2 = distorted.filter_gauss_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        dev.synchronize().unwrap();
    }
    dev.synchronize().unwrap();

    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use cuda_npp::safe::isu::Malloc;

    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        let source = zune_image::image::Image::open("../ssimulacra2-cuda/source.png").unwrap();
        assert_eq!(source.channels_ref(true).len(), 3);
        let source_bytes = &source.flatten_to_u8()[0];
        let mut source_img =
            Image::<u8, C<3>>::malloc(source.dimensions().0 as u32, source.dimensions().1 as u32)?;
        source_img.copy_from_cpu(&source_bytes)?;

        let dis = zune_image::image::Image::open("../ssimulacra2-cuda/distorted.png").unwrap();
        assert_eq!(dis.channels_ref(true).len(), 3);
        let dis_bytes = &dis.flatten_to_u8()[0];
        let mut dis_img =
            Image::<u8, C<3>>::malloc(dis.dimensions().0 as u32, dis.dimensions().1 as u32)?;
        dis_img.copy_from_cpu(&dis_bytes)?;

        dbg!(ssimulacra2(source_img, dis_img));
        Ok(())
    }
}
