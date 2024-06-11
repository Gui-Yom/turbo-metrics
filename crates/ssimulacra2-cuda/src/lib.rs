use indices::indices_ordered;
use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use cuda_driver::CuStream;
use cuda_npp::image::ial::{Mul, Sqr, SqrIP};
use cuda_npp::image::idei::Transpose;
use cuda_npp::image::ist::Sum;
use cuda_npp::image::isu::Malloc;
use cuda_npp::image::{Image, Img, ImgMut, C};
use cuda_npp::sys::{cudaStream_t, NppStreamContext, NppiRect, Result};
use cuda_npp::{get_stream_ctx, ScratchBuffer};

use crate::kernel::Kernel;

pub mod cpu;
mod kernel;

/// Number of scales to compute
const SCALES: usize = 6;

/// An instance is valid for a specific width and height.
///
/// This implementation never allocates during processing and requires a minimum
/// of `270 * width * height` bytes.
/// e.g. ~800 MiB of device memory for processing 1440x1080 frames.
/// Actual memory usage is higher because of padding and other state.
///
/// Processing a single image pair results in 192 kernels launches !
pub struct Ssimulacra2 {
    kernel: Kernel,
    npp: NppStreamContext,

    sizes: [NppiRect; SCALES],
    sizes_t: [NppiRect; SCALES],

    ref_input: Image<u8, C<3>>,
    dis_input: Image<u8, C<3>>,

    ref_linear: [Image<f32, C<3>>; SCALES],
    dis_linear: [Image<f32, C<3>>; SCALES],

    img: [[Image<f32, C<3>>; 10]; SCALES],
    imgt: [[Image<f32, C<3>>; 10]; SCALES],

    sum_scratch: [[ScratchBuffer; 6]; SCALES],

    main_ref: CuStream,
    main_dis: CuStream,
    streams: [[CuStream; 6]; SCALES],
}

impl Ssimulacra2 {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        // Do not use the default stream
        let main_ref = CuStream::new().unwrap();
        let main_dis = CuStream::new().unwrap();
        let streams =
            array_init::try_array_init(|_| array_init::try_array_init(|_| CuStream::new()))
                .unwrap();
        let mut npp = get_stream_ctx()?;
        npp.hStream = main_ref.inner() as _;
        // Set in the global context too
        cuda_npp::set_stream(npp.hStream)?;

        // Input images
        let ref_input = Image::<u8, C<3>>::malloc(width, height)?;
        let dis_input = Image::<u8, C<3>>::malloc(width, height)?;

        let mut sizes = [ref_input.rect(); SCALES];
        for scale in 1..SCALES {
            sizes[scale].width = (sizes[scale - 1].width + 1) / 2;
            sizes[scale].height = (sizes[scale - 1].height + 1) / 2;
        }
        let sizes_t = array_init::array_init(|i| sizes[i].transpose());

        // Buffers needed for computations
        let img = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| {
                Image::malloc(sizes[i].width as u32, sizes[i].height as u32)
            })
        })?;
        // Transposed buffers
        let imgt = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| {
                Image::malloc(sizes_t[i].width as u32, sizes_t[i].height as u32)
            })
        })?;

        let ref_linear = array_init::try_array_init(|i| img[i][0].malloc_same_size())?;
        let dis_linear = array_init::try_array_init(|i| img[i][0].malloc_same_size())?;

        let sum_scratch = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| imgt[i][0].sum_alloc_scratch(npp))
        })?;

        // Wait for allocations to complete
        main_ref.sync().unwrap();

        Ok(Self {
            kernel: Kernel::load(),
            npp,
            sizes,
            sizes_t,
            ref_input,
            dis_input,
            ref_linear,
            dis_linear,
            img,
            imgt,
            sum_scratch,
            main_ref,
            main_dis,
            streams,
        })
    }

    /// Estimate the minimum memory usage
    pub fn mem_usage(&self) -> usize {
        self.ref_input.device_mem_usage()
            + self.dis_input.device_mem_usage()
            + self
                .ref_linear
                .iter()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .dis_linear
                .iter()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .img
                .iter()
                .flatten()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .imgt
                .iter()
                .flatten()
                .map(|i| i.device_mem_usage())
                .sum::<usize>()
            + self
                .sum_scratch
                .iter()
                .flatten()
                .map(|b| b.len())
                .sum::<usize>()
    }

    pub fn compute(&mut self, ref_bytes: &[u8], dis_bytes: &[u8]) -> Result<f64> {
        // profiler_stop().unwrap();

        self.ref_input
            .copy_from_cpu(ref_bytes, self.main_ref.inner() as _)?;
        self.dis_input
            .copy_from_cpu(dis_bytes, self.main_dis.inner() as _)?;

        // TODO we should work with planar images, as it would allow us to coalesce read and writes
        //  coalescing can already be achieved for kernels which doesn't require access to neighbouring pixels or samples

        // Convert to linear
        self.kernel
            .srgb_to_linear(&self.main_ref, &self.ref_input, &mut self.ref_linear[0]);
        self.kernel
            .srgb_to_linear(&self.main_dis, &self.dis_input, &mut self.dis_linear[0]);

        // save_img(&self.dev, &self.ref_linear, &format!("ref_linear"));

        // linear -> xyb -> ...
        //    |-> /2 -> xyb -> ...
        //         |-> /2 -> xyb -> ...

        let mut scores = [0.0; 108];

        for scale in 0..SCALES {
            if scale > 0 {
                // TODO this can be done with warp level primitives by having warps sized 16x2
                //  block size 16x16 would be perfect
                //  warps would contain 8 2x2 patches which can be summed using shfl_down_sync and friends
                //  This would require a planar format ...

                let (prev, curr) = indices_ordered!(&mut self.ref_linear, scale - 1, scale);
                self.kernel.downscale_by_2(&self.main_ref, prev, curr);
                let (prev, curr) = indices_ordered!(&mut self.dis_linear, scale - 1, scale);
                self.kernel.downscale_by_2(&self.main_dis, prev, curr);
            }

            let [sum_ssim, sum_artifact, sum_detail_loss, sum_ssim_4, sum_artifact_4, sum_detail_loss_4] =
                self.process_scale(scale)?;

            for i in 1..6 {
                self.streams[scale][i]
                    .wait_for_stream(&self.streams[scale][0])
                    .unwrap();
            }

            // profiler_stop().unwrap();
            self.streams[scale][0].sync().unwrap();

            let opp = self.sizes_t[scale].norm();

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

        CuStream::DEFAULT.sync().unwrap();

        dbg!(scores);
        Ok(post_process_scores(&scores))
    }

    fn process_scale(&mut self, scale: usize) -> Result<[Box<[f64; 3]>; 6]> {
        let streams: [&CuStream; 6] = self.streams[scale].each_ref().try_into().unwrap();

        streams[0].wait_for_stream(&self.main_ref).unwrap();
        streams[1].wait_for_stream(&self.main_dis).unwrap();

        self.kernel
            .linear_to_xyb(streams[0], &self.ref_linear[scale], &mut self.img[scale][8]);
        self.kernel
            .linear_to_xyb(streams[1], &self.dis_linear[scale], &mut self.img[scale][9]);
        // save_img(&self.dev, self.ref_xyb.view(size), &format!("ref_xyb_{scale}"));

        streams[0].join(streams[1]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();

        {
            let (sigma11, sigma22, sigma12, ref_xyb, dis_xyb) =
                indices_ordered!(&mut self.img[scale], 0, 1, 2, 8, 9);
            ref_xyb.mul(
                &ref_xyb,
                sigma11,
                self.npp.with_stream(streams[0].inner() as _),
            )?;
            dis_xyb.mul(
                &dis_xyb,
                sigma22,
                self.npp.with_stream(streams[1].inner() as _),
            )?;
            ref_xyb.mul(
                &dis_xyb,
                sigma12,
                self.npp.with_stream(streams[2].inner() as _),
            )?;
        }

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.img[scale].each_mut();
            // TODO make blur work in place
            // We currently can't compute our blur pass in place,
            // which means we need 10 full buffers allocated :(
            #[rustfmt::skip]
            self.kernel.blur_pass_fused(streams[0],
                i0, i3,
                i1, i4,
                i2, i5,
                i8, i6,
                i9, i7,
            );
        }

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();
        streams[4].wait_for_stream(streams[0]).unwrap();

        self.img[scale][3].transpose(
            &mut self.imgt[scale][0],
            self.npp.with_stream(streams[0].inner() as _),
        )?;
        self.img[scale][4].transpose(
            &mut self.imgt[scale][1],
            self.npp.with_stream(streams[1].inner() as _),
        )?;
        self.img[scale][5].transpose(
            &mut self.imgt[scale][2],
            self.npp.with_stream(streams[2].inner() as _),
        )?;
        self.img[scale][6].transpose(
            &mut self.imgt[scale][3],
            self.npp.with_stream(streams[3].inner() as _),
        )?;
        self.img[scale][7].transpose(
            &mut self.imgt[scale][4],
            self.npp.with_stream(streams[4].inner() as _),
        )?;

        streams[0].wait_for_stream(streams[1]).unwrap();
        streams[0].wait_for_stream(streams[2]).unwrap();
        streams[0].wait_for_stream(streams[3]).unwrap();
        streams[0].wait_for_stream(streams[4]).unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.imgt[scale].each_mut();

            // tmpt5: sigma11
            // tmpt6: sigma22
            // tmpt7: sigma12
            // tmpt8: mu1
            // tmpt9: mu2
            #[rustfmt::skip]
            self.kernel.blur_pass_fused(streams[0],
                i0, i5,
                i1, i6,
                i2, i7,
                i3, i8,
                i4, i9,
            );
        }

        streams[1].wait_for_stream(streams[0]).unwrap();

        // tmpt0: source
        // tmpt1: distorted
        self.img[scale][8].transpose(
            &mut self.imgt[scale][0],
            self.npp.with_stream(streams[0].inner() as _),
        )?;
        self.img[scale][9].transpose(
            &mut self.imgt[scale][1],
            self.npp.with_stream(streams[1].inner() as _),
        )?;

        streams[0].wait_for_stream(streams[1]).unwrap();

        // profiler_start().unwrap();

        {
            let [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9] = self.imgt[scale].each_mut();

            // tmpt2: ssim
            // tmpt3: artifact
            // tmpt4: detail_loss
            self.kernel
                .compute_error_maps(streams[0], i0, i1, i8, i9, i5, i6, i7, i2, i3, i4);
        }
        // save_img(&self.dev, self.tmp0.view(size), &format!("ssim_{scale}"));

        streams[1].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[3].wait_for_stream(streams[0]).unwrap();
        streams[4].wait_for_stream(streams[0]).unwrap();
        streams[5].wait_for_stream(streams[0]).unwrap();

        let (sum_ssim, sum_ssim_4) = self.reduce(scale, 2, 5, 0)?;
        let (sum_artifact, sum_artifact_4) = self.reduce(scale, 3, 6, 2)?;
        let (sum_detail_loss, sum_detail_loss_4) = self.reduce(scale, 4, 7, 4)?;

        Ok([
            sum_ssim,
            sum_artifact,
            sum_detail_loss,
            sum_ssim_4,
            sum_artifact_4,
            sum_detail_loss_4,
        ])
    }

    fn reduce(
        &mut self,
        scale: usize,
        data: usize,
        tmp: usize,
        offset: usize,
    ) -> Result<(Box<[f64; 3]>, Box<[f64; 3]>)> {
        let [scratch0, scratch1] = &mut self.sum_scratch[scale][offset..offset + 2] else {
            unreachable!()
        };
        let ctx1 = self
            .npp
            .with_stream(self.streams[scale][offset + 1].inner() as _);
        let sum_ssim = {
            let (ssim, tmp) = indices_ordered!(&mut self.imgt[scale], data, tmp);
            let sum_ssim = ssim.sum(
                scratch0,
                self.npp
                    .with_stream(self.streams[scale][offset].inner() as _),
            )?;
            ssim.sqr(tmp, ctx1)?;
            sum_ssim
        };
        self.imgt[scale][tmp].sqr_ip(ctx1)?;
        let sum_ssim_4 = self.imgt[scale][tmp].sum(scratch1, ctx1)?;
        Ok((sum_ssim, sum_ssim_4))
    }
}

fn post_process_scores(scores: &[f64; 108]) -> f64 {
    // TODO jeez that's a lot of zeros
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

fn save_img(img: impl Img<f32, C<3>>, name: &str, stream: cudaStream_t) {
    // dev.synchronize().unwrap();
    let bytes = img.copy_to_cpu(stream).unwrap();
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
