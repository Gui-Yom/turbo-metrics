use std::sync::Arc;

use cudarc::driver::CudaDevice;
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use cuda_npp::safe::ial::SqrIP;
use cuda_npp::safe::icc::{GammaFwdIP, GammaInvIP};
use cuda_npp::safe::idei::Scale;
use cuda_npp::safe::ig::Resize;
use cuda_npp::safe::ist::Sum;
use cuda_npp::safe::Image;
use cuda_npp::safe::Result;
use cuda_npp::sys::{NppiInterpolationMode, NppiMaskSize};
use cuda_npp::{get_stream_ctx, C};

use crate::kernel::Kernel;

mod cpu;
mod kernel;

pub(crate) type Img = Image<f32, C<3>>;

pub fn ssimulacra2(source_: Image<u8, C<3>>, distorted_: Image<u8, C<3>>) -> Result<f64> {
    let dev = CudaDevice::new(0).unwrap();
    let kernel = Kernel::load(&dev);
    let ctx = get_stream_ctx()?;
    println!("Starting work ...");

    let mut source = source_.malloc_same_size()?;
    kernel.packed_srgb_to_linear(&source_, &mut source);
    let mut distorted = distorted_.malloc_same_size()?;
    kernel.packed_srgb_to_linear(&distorted_, &mut distorted);

    // source.gamma_inv_ip(ctx)?;
    // let mut source = source.scale_float_new(0.0..1.0, ctx)?;
    // distorted.gamma_inv_ip(ctx)?;
    // let mut distorted = distorted.scale_float_new(0.0..1.0, ctx)?;
    dev.synchronize().unwrap();

    let mut scores = [0.0; 108];
    for scale in 0..6 {
        // Scaling in linear colorspace
        if scale > 0 {
            let new_width = (source.width + 1) / 2;
            let new_height = (source.height + 1) / 2;
            source = source.resize_new(
                new_width,
                new_height,
                NppiInterpolationMode::NPPI_INTER_NN,
                ctx,
            )?;
            distorted = distorted.resize_new(
                new_width,
                new_height,
                NppiInterpolationMode::NPPI_INTER_NN,
                ctx,
            )?;
            dev.synchronize().unwrap();
        }

        // Operations in XYB perceptual colorspace
        let source = kernel.linear_to_xyb(&source);
        let distorted = kernel.linear_to_xyb(&distorted);
        dev.synchronize().unwrap();

        save_img(&dev, &source, &format!("xyb_src_{scale}"));
        save_img(&dev, &distorted, &format!("xyb_dis_{scale}"));

        let ref_sq = source.mul_new(&source, ctx)?;
        let sigma1_sq = ref_sq.filter_gauss_border_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let dis_sq = distorted.mul_new(&distorted, ctx)?;
        let sigma2_sq = dis_sq.filter_gauss_border_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let ref_dis = source.mul_new(&distorted, ctx)?;
        let sigma12 = ref_dis.filter_gauss_border_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;

        let mu1 = source.filter_gauss_border_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        let mu2 = distorted.filter_gauss_border_new(NppiMaskSize::NPP_MASK_SIZE_9_X_9, ctx)?;
        dev.synchronize().unwrap();

        let mut ssim = kernel.ssim_map(&mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        // save_img(&dev, &ssim, &format!("{scale}_ssim"));
        let mut sum_scratch = ssim.alloc_scratch();
        let sum_ssim = ssim.sum(&mut sum_scratch, ctx)?;
        ssim.sqr_ip(ctx)?;
        ssim.sqr_ip(ctx)?;
        let sum_ssim_4 = ssim.sum(&mut sum_scratch, ctx)?;

        let (mut artifact, mut detail_loss) = kernel.edge_diff_map(&source, &mu1, &distorted, &mu2);
        // save_img(&dev, &ssim, &format!("{scale}_artifact"));
        // save_img(&dev, &ssim, &format!("{scale}_detail_loss"));
        let sum_artifact = artifact.sum(&mut sum_scratch, ctx)?;
        artifact.sqr_ip(ctx)?;
        artifact.sqr_ip(ctx)?;
        let sum_artifact_4 = artifact.sum(&mut sum_scratch, ctx)?;
        let sum_detail_loss = detail_loss.sum(&mut sum_scratch, ctx)?;
        detail_loss.sqr_ip(ctx)?;
        detail_loss.sqr_ip(ctx)?;
        let sum_detail_loss_4 = detail_loss.sum(&mut sum_scratch, ctx)?;
        dev.synchronize().unwrap();

        let opp = 1.0 / (source.width * source.height) as f64;

        for c in 0..3 {
            let offset = c * 6 * 6 + scale * 6;
            scores[offset] = (sum_ssim[c] * opp).abs();
            scores[offset + 1] = (sum_artifact[c] * opp).abs();
            scores[offset + 2] = (sum_detail_loss[c] * opp).abs();
            scores[offset + 3] = (sum_ssim_4[c] * opp).sqrt().sqrt();
            scores[offset + 4] = (sum_artifact_4[c] * opp).sqrt().sqrt();
            scores[offset + 5] = (sum_detail_loss_4[c] * opp).sqrt().sqrt();
        }
    }
    dev.synchronize().unwrap();

    Ok(post_process_scores(&scores))
}

fn post_process_scores(scores: &[f64; 108]) -> f64 {
    const WEIGHT: [f64; 108] = [
        0.0,
        0.000_737_660_670_740_658_6,
        0.0,
        0.0,
        0.000_779_348_168_286_730_9,
        0.0,
        0.0,
        0.000_437_115_573_010_737_9,
        0.0,
        1.104_172_642_665_734_6,
        0.000_662_848_341_292_71,
        0.000_152_316_327_837_187_52,
        0.0,
        0.001_640_643_745_659_975_4,
        0.0,
        1.842_245_552_053_929_8,
        11.441_172_603_757_666,
        0.0,
        0.000_798_910_943_601_516_3,
        0.000_176_816_438_078_653,
        0.0,
        1.878_759_497_954_638_7,
        10.949_069_906_051_42,
        0.0,
        0.000_728_934_699_150_807_2,
        0.967_793_708_062_683_3,
        0.0,
        0.000_140_034_242_854_358_84,
        0.998_176_697_785_496_7,
        0.000_319_497_559_344_350_53,
        0.000_455_099_211_379_206_3,
        0.0,
        0.0,
        0.001_364_876_616_324_339_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.466_890_328_078_848,
        0.0,
        17.445_833_984_131_262,
        0.000_623_560_163_404_146_6,
        0.0,
        0.0,
        6.683_678_146_179_332,
        0.000_377_244_079_796_112_96,
        1.027_889_937_768_264,
        225.205_153_008_492_74,
        0.0,
        0.0,
        19.213_238_186_143_016,
        0.001_140_152_458_661_836_1,
        0.001_237_755_635_509_985,
        176.393_175_984_506_94,
        0.0,
        0.0,
        24.433_009_998_704_76,
        0.285_208_026_121_177_57,
        0.000_448_543_692_383_340_8,
        0.0,
        0.0,
        0.0,
        34.779_063_444_837_72,
        44.835_625_328_877_896,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_868_055_657_329_169_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_531_319_187_435_874_7,
        0.0,
        0.000_165_338_141_613_791_12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_417_917_180_325_133_6,
        0.001_729_082_823_472_283_3,
        0.0,
        0.002_082_700_584_663_643_7,
        0.0,
        0.0,
        8.826_982_764_996_862,
        23.192_433_439_989_26,
        0.0,
        95.108_049_881_108_6,
        0.986_397_803_440_068_2,
        0.983_438_279_246_535_3,
        0.001_228_640_504_827_849_3,
        171.266_725_589_730_7,
        0.980_785_887_243_537_9,
        0.0,
        0.0,
        0.0,
        0.000_513_006_458_899_067_9,
        0.0,
        0.000_108_540_578_584_115_37,
    ];

    let mut score = WEIGHT
        .into_iter()
        .zip(scores)
        .map(|(w, &s)| w * s)
        .sum::<f64>();

    score *= 0.956_238_261_683_484_4_f64;
    score = (6.248_496_625_763_138e-5 * score * score).mul_add(
        score,
        2.326_765_642_916_932f64.mul_add(score, -0.020_884_521_182_843_837 * score * score),
    );

    if score > 0.0f64 {
        score = score
            .powf(0.627_633_646_783_138_7)
            .mul_add(-10.0f64, 100.0f64);
    } else {
        score = 100.0f64;
    }

    score
}

fn save_image(dev: &Arc<CudaDevice>, img: &Image<u8, C<3>>, name: &str) {
    dev.synchronize().unwrap();
    let bytes = img.copy_to_cpu().unwrap();
    let mut img = zune_image::image::Image::from_u8(
        &bytes,
        img.width as usize,
        img.height as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("./gpu_{name}.png")).unwrap()
}

fn save_img(dev: &Arc<CudaDevice>, img: &Img, name: &str) {
    dev.synchronize().unwrap();
    let bytes = img.copy_to_cpu().unwrap();
    let mut img = zune_image::image::Image::from_f32(
        &bytes,
        img.width as usize,
        img.height as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("./gpu_{name}.png")).unwrap()
}

#[cfg(test)]
mod tests {
    use cuda_npp::safe::isu::Malloc;

    use crate::cpu::CpuImg;

    use super::*;

    #[test]
    fn gamma() -> Result<()> {
        let source = zune_image::image::Image::open("../ssimulacra2-cuda/source.png").unwrap();
        assert_eq!(source.channels_ref(true).len(), 3);
        let source_bytes = &source.flatten_to_u8()[0];
        let mut source_img =
            Image::<u8, C<3>>::malloc(source.dimensions().0 as u32, source.dimensions().1 as u32)?;
        source_img.copy_from_cpu(&source_bytes)?;

        let dev = CudaDevice::new(0).unwrap();
        let kernel = Kernel::load(&dev);
        let ctx = get_stream_ctx()?;

        {
            let mut src = source_img.malloc_same()?;
            dev.dtod_copy(&source_img, &mut src).unwrap();
            src.gamma_fwd_ip(ctx)?;
            save_image(&dev, &src, "source_npp_fwd");
        }

        {
            let mut src = source_img.malloc_same()?;
            dev.dtod_copy(&source_img, &mut src).unwrap();
            src.gamma_inv_ip(ctx)?;
            save_image(&dev, &src, "source_npp_inv");
        }

        {
            let mut src = source_img.malloc_same()?;
            dev.dtod_copy(&source_img, &mut src).unwrap();
            let mut dst = src.malloc_same_size()?;
            kernel.packed_srgb_to_linear(&src, &mut dst);
            save_image(&dev, &src, "source_custom_cuda");
        }

        Ok(())
    }

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

        let result = dbg!(ssimulacra2(source_img, dis_img))?;

        let expected = 17.398_505_f64;
        assert!(
            (result - expected).abs() < 0.25f64,
            "Result {result:.6} not equal to expected {expected:.6}",
        );
        Ok(())
    }

    #[test]
    fn cpu() -> Result<()> {
        let source = zune_image::image::Image::open("../ssimulacra2-cuda/source.png").unwrap();
        assert_eq!(source.channels_ref(true).len(), 3);
        let source_bytes = &source.flatten_to_u8()[0];
        let source_img =
            CpuImg::from_srgb(source_bytes, source.dimensions().0, source.dimensions().1);

        let dis = zune_image::image::Image::open("../ssimulacra2-cuda/distorted.png").unwrap();
        assert_eq!(dis.channels_ref(true).len(), 3);
        let dis_bytes = &dis.flatten_to_u8()[0];
        let dis_img = CpuImg::from_srgb(dis_bytes, dis.dimensions().0, dis.dimensions().1);

        let result = dbg!(cpu::compute_frame_ssimulacra2(&source_img, &dis_img));

        let expected = 17.398_505_f64;
        assert!(
            (result - expected).abs() < 0.25f64,
            "Result {result:.6} not equal to expected {expected:.6}",
        );
        Ok(())
    }
}
