use std::mem;
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
use zune_image::codecs::png::zune_core::options::DecoderOptions;

use cuda_driver::{profiler_start, profiler_stop, CuStream};
use cuda_npp::safe::ial::{Mul, SqrIP};
use cuda_npp::safe::idei::{ConvertChannel, Transpose};
use cuda_npp::safe::if_::FilterGaussBorder;
use cuda_npp::safe::ist::Sum;
use cuda_npp::safe::isu::Malloc;
use cuda_npp::safe::{Image, Img, ImgMut, ScratchBuffer};
use cuda_npp::sys::NppStreamContext;
use cuda_npp::{get_stream_ctx, C};

use crate::kernel::Kernel;

mod cpu;
mod kernel;

fn main() {
    let dev = CudaDevice::new(0).unwrap();

    let ref_img = zune_image::image::Image::open_with_options(
        "crates/ssimulacra2-cuda/source.png",
        DecoderOptions::new_fast(),
    )
    .unwrap();
    let dis_img = zune_image::image::Image::open_with_options(
        "crates/ssimulacra2-cuda/distorted.png",
        DecoderOptions::new_fast(),
    )
    .unwrap();

    // Upload to gpu
    let (width, height) = ref_img.dimensions();
    let ref_bytes = &ref_img.flatten_to_u8()[0];

    let (dwidth, dheight) = dis_img.dimensions();
    assert_eq!((width, height), (dwidth, dheight));
    let dis_bytes = &dis_img.flatten_to_u8()[0];

    let mut ssimulacra2 = Ssimulacra2::new(&dev, width as u32, height as u32);
    dbg!(ssimulacra2.mem_usage());
    dbg!(ssimulacra2.compute(ref_bytes, dis_bytes));

    // let ref_img = CpuImg::from_srgb(ref_bytes, width, height);
    // let dis_img = CpuImg::from_srgb(dis_bytes, width, height);
    // dbg!(cpu::compute_frame_ssimulacra2(&ref_img, &dis_img));
}

/// An instance is valid for a specific width and height.
///
/// This implementation never allocates during processing and requires a minimum
/// of `270 * width * height` bytes.
/// e.g. ~400 MiB of device memory for processing 1440x1080 frames.
/// Actual memory usage is higher because of padding and other state.
///
/// Processing a single image pair results in 192 kernels launches !
struct Ssimulacra2 {
    dev: Arc<CudaDevice>,
    kernel: Kernel,
    npp: NppStreamContext,
    ref_input: Image<u8, C<3>>,
    dis_input: Image<u8, C<3>>,
    ref_linear: Image<f32, C<3>>,
    dis_linear: Image<f32, C<3>>,
    ref_xyb: Image<f32, C<3>>,
    dis_xyb: Image<f32, C<3>>,

    sum_scratch: ScratchBuffer,

    tmp0: Image<f32, C<3>>,
    tmp1: Image<f32, C<3>>,
    tmp2: Image<f32, C<3>>,
    tmp3: Image<f32, C<3>>,
    tmp4: Image<f32, C<3>>,
    tmp5: Image<f32, C<3>>,
    tmp6: Image<f32, C<3>>,
    tmp7: Image<f32, C<3>>,

    tmpt0: Image<f32, C<3>>,
    tmpt1: Image<f32, C<3>>,
    tmpt2: Image<f32, C<3>>,
    tmpt3: Image<f32, C<3>>,
    tmpt4: Image<f32, C<3>>,
    tmpt5: Image<f32, C<3>>,
    tmpt6: Image<f32, C<3>>,
    tmpt7: Image<f32, C<3>>,
    tmpt8: Image<f32, C<3>>,
    tmpt9: Image<f32, C<3>>,
}

impl Ssimulacra2 {
    pub fn new(dev: &Arc<CudaDevice>, width: u32, height: u32) -> Self {
        let ref_input = Image::<u8, C<3>>::malloc(width, height).unwrap();
        let dis_input = Image::<u8, C<3>>::malloc(width, height).unwrap();

        let ref_linear = ref_input.malloc_same_size().unwrap();
        let dis_linear = dis_input.malloc_same_size().unwrap();

        let ref_xyb = ref_linear.malloc_same_size().unwrap();
        let dis_xyb = dis_linear.malloc_same_size().unwrap();

        let tmp0 = ref_input.malloc_same_size().unwrap();
        let tmp1 = ref_input.malloc_same_size().unwrap();
        let tmp2 = ref_input.malloc_same_size().unwrap();

        let tmp3 = ref_input.malloc_same_size().unwrap();
        let tmp4 = ref_input.malloc_same_size().unwrap();
        let tmp5 = ref_input.malloc_same_size().unwrap();
        let tmp6 = ref_input.malloc_same_size().unwrap();
        let tmp7 = ref_input.malloc_same_size().unwrap();

        let tmpt0 = Image::malloc(ref_input.height(), ref_input.width()).unwrap();
        let tmpt1 = tmpt0.malloc_same_size().unwrap();
        let tmpt2 = tmpt0.malloc_same_size().unwrap();
        let tmpt3 = tmpt0.malloc_same_size().unwrap();
        let tmpt4 = tmpt0.malloc_same_size().unwrap();

        let tmpt5 = tmpt0.malloc_same_size().unwrap();
        let tmpt6 = tmpt0.malloc_same_size().unwrap();
        let tmpt7 = tmpt0.malloc_same_size().unwrap();
        let tmpt8 = tmpt0.malloc_same_size().unwrap();
        let tmpt9 = tmpt0.malloc_same_size().unwrap();

        let sum_scratch = tmpt0.sum_alloc_scratch();

        Self {
            dev: Arc::clone(dev),
            kernel: Kernel::load(),
            npp: get_stream_ctx().unwrap(),
            ref_input,
            dis_input,
            ref_linear,
            dis_linear,
            ref_xyb,
            dis_xyb,
            sum_scratch,
            tmp0,
            tmp1,
            tmp2,
            tmp3,
            tmp4,
            tmp5,
            tmp6,
            tmp7,
            tmpt0,
            tmpt1,
            tmpt2,
            tmpt3,
            tmpt4,
            tmpt5,
            tmpt6,
            tmpt7,
            tmpt8,
            tmpt9,
        }
    }

    /// Estimate the approximate memory usage
    pub fn mem_usage(&self) -> usize {
        self.ref_input.device_mem_usage()
            + self.dis_input.device_mem_usage()
            + self.ref_linear.device_mem_usage()
            + self.dis_linear.device_mem_usage()
            + self.ref_xyb.device_mem_usage()
            + self.dis_xyb.device_mem_usage()
            + self.tmp0.device_mem_usage()
            + self.tmp1.device_mem_usage()
            + self.tmp2.device_mem_usage()
            + self.tmp3.device_mem_usage()
            + self.tmp4.device_mem_usage()
            + self.tmp5.device_mem_usage()
            + self.tmp6.device_mem_usage()
            + self.tmp7.device_mem_usage()
            + self.tmpt0.device_mem_usage()
            + self.tmpt1.device_mem_usage()
            + self.tmpt2.device_mem_usage()
            + self.tmpt3.device_mem_usage()
            + self.tmpt4.device_mem_usage()
            + self.tmpt5.device_mem_usage()
            + self.tmpt6.device_mem_usage()
            + self.tmpt7.device_mem_usage()
            + self.tmpt8.device_mem_usage()
            + self.tmpt9.device_mem_usage()
    }

    pub fn compute(&mut self, ref_bytes: &[u8], dis_bytes: &[u8]) -> f64 {
        profiler_stop().unwrap();

        const SCALES: usize = 6;

        self.ref_input.copy_from_cpu(ref_bytes).unwrap();
        self.dis_input.copy_from_cpu(dis_bytes).unwrap();

        // TODO we should work with planar images, as it would allow us to coalesce read and writes
        //  coalescing can already be achieved for kernels which doesn't require access to neighbouring pixels or samples

        // Convert to linear
        self.kernel
            .srgb_to_linear(&self.ref_input, &mut self.ref_linear);
        self.kernel
            .srgb_to_linear(&self.dis_input, &mut self.dis_linear);

        // save_img(&self.dev, &self.ref_linear, &format!("ref_linear"));

        // linear -> xyb -> ...
        //    |-> /2 -> xyb -> ...
        //         |-> /2 -> xyb -> ...

        let mut scores = [0.0; 108];

        let mut size = self.ref_input.rect();
        let mut sizet = self.tmpt0.rect();

        // TODO launch independent computations on different streams for parallel execution and maximum gpu utilization.
        //  Iterations only depend on the previous downscale operation to complete.
        for scale in 0..SCALES {
            if scale > 0 {
                {
                    let ref_linear = self.ref_linear.view_mut(size);
                    let dis_linear = self.dis_linear.view_mut(size);

                    // Divide size by 2
                    size.width = (size.width + 1) / 2;
                    size.height = (size.height + 1) / 2;

                    sizet.width = (sizet.width + 1) / 2;
                    sizet.height = (sizet.height + 1) / 2;

                    dbg!(size);

                    // TODO this can be done with warp level primitives by having warps sized 16x2
                    //  block size 16x16 would be perfect
                    //  warps would contain 8 2x2 patches which can be summed using shfl_down_sync and friends
                    self.kernel
                        .downscale_by_2(&ref_linear, self.tmp0.view_mut(size));
                    self.kernel
                        .downscale_by_2(&dis_linear, self.tmp1.view_mut(size));

                    // let mut planar = self.ref_linear.malloc_same_size().unwrap();
                    // self.ref_linear.convert_channel(&mut planar, get_stream_ctx().unwrap()).unwrap();
                    // let mut dst = Image::malloc(size.width as u32, size.height as u32).unwrap();
                    // self.kernel.downscale_plane_by_2_planar(&planar, &mut dst);
                    // self.dev.synchronize().unwrap();

                    // save_img(&self.dev, self.ref_linear_resized.view(size), &format!("ref_linear_resized_{scale}"));
                }

                mem::swap(&mut self.ref_linear, &mut self.tmp0);
                mem::swap(&mut self.dis_linear, &mut self.tmp1);

                // dbg!(ref_linear_resized.device_ptr());
            }

            // let mut planar = self.ref_linear.view(size).malloc_same_size().unwrap();
            // self.ref_linear.view(size).convert_channel(&mut planar, get_stream_ctx().unwrap()).unwrap();
            // let mut dst = Image::malloc(size.width as u32, size.height as u32).unwrap();
            // self.kernel.linear_to_xyb_planar(&planar, &mut dst);
            // self.dev.synchronize().unwrap();

            self.kernel
                .linear_to_xyb(self.ref_linear.view(size), self.ref_xyb.view_mut(size));
            self.kernel
                .linear_to_xyb(self.dis_linear.view(size), self.dis_xyb.view_mut(size));
            // save_img(&self.dev, self.ref_xyb.view(size), &format!("ref_xyb_{scale}"));

            // tmp0: sigma11
            self.ref_xyb
                .view(size)
                .mul(self.ref_xyb.view(size), self.tmp0.view_mut(size), self.npp)
                .unwrap();
            // tmp1: sigma22
            self.dis_xyb
                .view(size)
                .mul(self.dis_xyb.view(size), self.tmp1.view_mut(size), self.npp)
                .unwrap();
            // tmp2: sigma12
            self.ref_xyb
                .view(size)
                .mul(self.dis_xyb.view(size), self.tmp2.view_mut(size), self.npp)
                .unwrap();

            // TODO make blur work in place
            // We currently can't compute our blur pass in place,
            // which means we need 10 full buffers allocated :(
            #[rustfmt::skip]
            self.kernel.blur_pass_fused(
                self.tmp0.view(size), self.tmp3.view_mut(size),
                self.tmp1.view(size), self.tmp4.view_mut(size),
                self.tmp2.view(size), self.tmp5.view_mut(size),
                self.ref_xyb.view(size), self.tmp6.view_mut(size),
                self.dis_xyb.view(size), self.tmp7.view_mut(size),
            );

            self.tmp3
                .view(size)
                .transpose(self.tmpt0.view_mut(sizet), self.npp)
                .unwrap();
            self.tmp4
                .view(size)
                .transpose(self.tmpt1.view_mut(sizet), self.npp)
                .unwrap();
            self.tmp5
                .view(size)
                .transpose(self.tmpt2.view_mut(sizet), self.npp)
                .unwrap();
            self.tmp6
                .view(size)
                .transpose(self.tmpt3.view_mut(sizet), self.npp)
                .unwrap();
            self.tmp7
                .view(size)
                .transpose(self.tmpt4.view_mut(sizet), self.npp)
                .unwrap();

            // tmpt5: sigma11
            // tmpt6: sigma22
            // tmpt7: sigma12
            // tmpt8: mu1
            // tmpt9: mu2
            #[rustfmt::skip]
            self.kernel.blur_pass_fused(
                self.tmpt0.view(sizet), self.tmpt5.view_mut(sizet),
                self.tmpt1.view(sizet), self.tmpt6.view_mut(sizet),
                self.tmpt2.view(sizet), self.tmpt7.view_mut(sizet),
                self.tmpt3.view(sizet), self.tmpt8.view_mut(sizet),
                self.tmpt4.view(sizet), self.tmpt9.view_mut(sizet),
            );

            // tmpt0: source
            // tmpt1: distorted
            self.ref_xyb
                .view(size)
                .transpose(self.tmpt0.view_mut(sizet), self.npp)
                .unwrap();
            self.dis_xyb
                .view(size)
                .transpose(self.tmpt1.view_mut(sizet), self.npp)
                .unwrap();

            profiler_start().unwrap();

            // tmpt2: ssim
            // tmpt3: artifact
            // tmpt4: detail_loss
            self.kernel.compute_error_maps(
                self.tmpt0.view(sizet),
                self.tmpt1.view(sizet),
                self.tmpt8.view(sizet),
                self.tmpt9.view(sizet),
                self.tmpt5.view(sizet),
                self.tmpt6.view(sizet),
                self.tmpt7.view(sizet),
                self.tmpt2.view_mut(sizet),
                self.tmpt3.view_mut(sizet),
                self.tmpt4.view_mut(sizet),
            );
            // save_img(&self.dev, self.tmp0.view(size), &format!("ssim_{scale}"));

            let sum_ssim = self
                .tmpt2
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();
            // self.dev.synchronize().unwrap();
            // dbg!(sum_ssim);
            self.tmpt2.view_mut(sizet).sqr_ip(self.npp).unwrap();
            self.tmpt2.view_mut(sizet).sqr_ip(self.npp).unwrap();
            let sum_ssim_4 = self
                .tmpt2
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();

            // save_img(&dev, &ssim, &format!("{scale}_artifact"));
            // save_img(&dev, &ssim, &format!("{scale}_detail_loss"));
            // TODO fuse those computions
            let sum_artifact = self
                .tmpt3
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();
            self.tmpt3.view_mut(sizet).sqr_ip(self.npp).unwrap();
            self.tmpt3.view_mut(sizet).sqr_ip(self.npp).unwrap();
            let sum_artifact_4 = self
                .tmpt3
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();
            let sum_detail_loss = self
                .tmpt4
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();
            self.tmpt4.view_mut(sizet).sqr_ip(self.npp).unwrap();
            self.tmpt4.view_mut(sizet).sqr_ip(self.npp).unwrap();
            let sum_detail_loss_4 = self
                .tmpt4
                .view(sizet)
                .sum(&mut self.sum_scratch, self.npp)
                .unwrap();

            profiler_stop().unwrap();
            self.dev.synchronize().unwrap();

            let opp = 1.0 / (size.width * size.height) as f64;

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

        self.dev.synchronize().unwrap();

        dbg!(scores);
        post_process_scores(&scores)
    }
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

fn save_img(dev: &Arc<CudaDevice>, img: impl Img<f32, C<3>>, name: &str) {
    // dev.synchronize().unwrap();
    let bytes = img.copy_to_cpu().unwrap();
    dev.synchronize().unwrap();
    let mut img = zune_image::image::Image::from_f32(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("./{name}.png")).unwrap()
}
