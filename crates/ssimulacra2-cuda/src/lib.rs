use indices::{indices, indices_ordered};

use cudarse_driver::{CuGraphExec, CuStream};
use cudarse_npp::image::ial::{Mul, Sqr, SqrIP};
use cudarse_npp::image::idei::Transpose;
use cudarse_npp::image::ist::Sum;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgMut, C};
use cudarse_npp::sys::{NppStreamContext, NppiRect, Result};
use cudarse_npp::{assert_same_size, get_stream_ctx, ScratchBuffer};

use crate::kernel::Kernel;

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
/// Processing a single image pair results in 305 kernels launches !
pub struct Ssimulacra2 {
    kernel: Kernel,
    npp: NppStreamContext,
    exec: Option<CuGraphExec>,

    sizes: [NppiRect; SCALES],
    sizes_t: [NppiRect; SCALES],

    ref_linear: [Image<f32, C<3>>; SCALES - 1],
    dis_linear: [Image<f32, C<3>>; SCALES - 1],

    img: [[Image<f32, C<3>>; 10]; SCALES],
    imgt: [[Image<f32, C<3>>; 10]; SCALES],

    sum_scratch: [[ScratchBuffer; 6]; SCALES],

    streams: [[CuStream; 6]; SCALES],
    scores: Box<[f64; 3 * 6 * SCALES]>,
}

impl Ssimulacra2 {
    pub fn new(
        src_ref_linear: impl Img<f32, C<3>>,
        src_dis_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<Self> {
        assert_same_size!(src_ref_linear, src_dis_linear);
        let rect = src_ref_linear.rect();

        let streams =
            array_init::try_array_init(|_| array_init::try_array_init(|_| CuStream::new()))
                .unwrap();
        let mut npp = get_stream_ctx()?;
        npp.hStream = stream.inner() as _;

        let mut sizes = [rect; SCALES];
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

        let ref_linear = array_init::try_array_init(|i| img[i + 1][0].malloc_same_size())?;
        let dis_linear = array_init::try_array_init(|i| img[i + 1][0].malloc_same_size())?;

        let sum_scratch = array_init::try_array_init(|i| {
            array_init::try_array_init(|_| imgt[i][0].sum_alloc_scratch(npp))
        })?;

        let mut s = Self {
            kernel: Kernel::load(),
            npp,
            exec: None,
            sizes,
            sizes_t,
            ref_linear,
            dis_linear,
            img,
            imgt,
            sum_scratch,
            streams,
            scores: Box::new([0.0; 3 * 6 * SCALES]),
        };

        s.exec = Some(s.record(src_ref_linear, src_dis_linear, stream)?);

        Ok(s)
    }

    /// Estimate the minimum memory usage
    pub fn mem_usage(&self) -> usize {
        self.ref_linear
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

    fn record(
        &mut self,
        src_ref_linear: impl Img<f32, C<3>>,
        src_dis_linear: impl Img<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<CuGraphExec> {
        // TODO we should work with planar images, as it would allow us to coalesce read and writes
        //  coalescing can already be achieved for kernels which doesn't require access to neighbouring pixels or samples

        let alt_stream = CuStream::new().unwrap();
        stream.begin_capture().unwrap();

        // Bring main_dis into the graph capture scope
        alt_stream.wait_for_stream(stream).unwrap();

        // save_img(&self.dev, &self.ref_linear, &format!("ref_linear"));

        // linear -> xyb -> ...
        //    |-> /2 -> xyb -> ...
        //         |-> /2 -> xyb -> ...

        for scale in 0..SCALES {
            if scale == 1 {
                self.kernel.downscale_by_2(
                    stream,
                    &src_ref_linear,
                    &mut self.ref_linear[scale - 1],
                );
                self.kernel.downscale_by_2(
                    &alt_stream,
                    &src_dis_linear,
                    &mut self.dis_linear[scale - 1],
                );
            } else if scale > 1 {
                // TODO this can be done with warp level primitives by having warps sized 16x2
                //  block size 16x16 would be perfect
                //  warps would contain 8 2x2 patches which can be summed using shfl_down_sync and friends
                //  This would require a planar format ...

                let (prev, curr) = indices!(&mut self.ref_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(stream, prev, curr);
                let (prev, curr) = indices!(&mut self.dis_linear, scale - 2, scale - 1);
                self.kernel.downscale_by_2(&alt_stream, prev, curr);
            }

            self.streams[scale][0].wait_for_stream(stream).unwrap();
            self.streams[scale][1].wait_for_stream(&alt_stream).unwrap();

            if scale == 0 {
                self.kernel.linear_to_xyb(
                    &self.streams[scale][0],
                    &src_ref_linear,
                    &mut self.img[scale][8],
                );
                self.kernel.linear_to_xyb(
                    &self.streams[scale][1],
                    &src_dis_linear,
                    &mut self.img[scale][9],
                );
            } else {
                self.kernel.linear_to_xyb(
                    &self.streams[scale][0],
                    &self.ref_linear[scale - 1],
                    &mut self.img[scale][8],
                );
                self.kernel.linear_to_xyb(
                    &self.streams[scale][1],
                    &self.dis_linear[scale - 1],
                    &mut self.img[scale][9],
                );
            }

            self.process_scale(scale)?;

            // profiler_stop().unwrap();
        }

        stream.wait_for_stream(&alt_stream).unwrap();
        for scale in 0..SCALES {
            for i in 0..self.streams[scale].len() {
                stream.wait_for_stream(&self.streams[scale][i]).unwrap();
            }
        }

        let graph = stream.end_capture().unwrap();
        // graph.dot("ssimulacra2-cuda-graph.gviz").unwrap();
        let exec = graph.instantiate().unwrap();
        // self.main_ref.sync().unwrap();
        Ok(exec)
    }

    /// Compute ssimulacra2 metric using image bytes in CPU memory. Useful for processing a single pair of images.
    pub fn compute_from_cpu_srgb_sync(
        &mut self,
        ref_bytes: &[u8],
        dis_bytes: &[u8],
        mut tmp_ref: impl ImgMut<u8, C<3>>,
        mut tmp_dis: impl ImgMut<u8, C<3>>,
        src_ref_linear: impl ImgMut<f32, C<3>>,
        src_dis_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        // profiler_stop().unwrap();

        tmp_ref.copy_from_cpu(ref_bytes, stream.inner() as _)?;
        tmp_dis.copy_from_cpu(dis_bytes, stream.inner() as _)?;

        self.compute_srgb_sync(tmp_ref, tmp_dis, src_ref_linear, src_dis_linear, stream)
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the [ref_input] and [dis_input] fields.
    /// This will block until CUDA is done to post process scores as that last part is done on the CPU.
    pub fn compute_srgb_sync(
        &mut self,
        src_ref: impl Img<u8, C<3>>,
        src_dis: impl Img<u8, C<3>>,
        src_ref_linear: impl ImgMut<f32, C<3>>,
        src_dis_linear: impl ImgMut<f32, C<3>>,
        stream: &CuStream,
    ) -> Result<f64> {
        // Convert to linear
        self.kernel.srgb_to_linear(stream, src_ref, src_ref_linear);
        self.kernel.srgb_to_linear(stream, src_dis, src_dis_linear);

        self.compute_sync(stream)
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the [ref_input] and [dis_input] fields.
    /// This will block until CUDA is done to post process scores as that last part is done on the CPU.
    pub fn compute_sync(&mut self, stream: &CuStream) -> Result<f64> {
        self.compute(stream)?;

        // Wait for CUDA to transfer scores back to the CPU before post-processing.
        stream.sync().unwrap();

        Ok(self.get_score())
    }

    /// Compute ssimulacra2 metric using images already in CUDA memory.
    /// Reference and distorted images must be copied to the linear input images.
    /// This one does not block. To retrieve the score, you must sync with the `main_ref` stream and call [Self::get_score] afterward.
    pub fn compute(&mut self, stream: &CuStream) -> Result<()> {
        self.exec.as_ref().unwrap().launch(stream).unwrap();
        Ok(())
    }

    /// Post process and retrieve the score for the last computation.
    pub fn get_score(&mut self) -> f64 {
        self.post_process_scores()
    }

    fn process_scale(&mut self, scale: usize) -> Result<()> {
        let streams: [&CuStream; 6] = self.streams[scale].each_ref().try_into().unwrap();

        streams[2].wait_for_stream(streams[0]).unwrap();
        streams[2].wait_for_stream(streams[1]).unwrap();

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

        self.reduce(scale, 2, 5, 0)?;
        self.reduce(scale, 3, 6, 2)?;
        self.reduce(scale, 4, 7, 4)?;

        Ok(())
    }

    fn reduce(&mut self, scale: usize, data: usize, tmp: usize, offset: usize) -> Result<()> {
        let [scratch0, scratch1] = &mut self.sum_scratch[scale][offset..offset + 2] else {
            unreachable!()
        };
        let ctx1 = self
            .npp
            .with_stream(self.streams[scale][offset + 1].inner() as _);
        {
            let (ssim, tmp) = indices!(&mut self.imgt[scale], data, tmp);
            ssim.sum_into(
                scratch0,
                (&mut self.scores
                    [scale * 6 * 3 + offset / 2 * 3..scale * 6 * 3 + offset / 2 * 3 + 3])
                    .try_into()
                    .unwrap(),
                self.npp
                    .with_stream(self.streams[scale][offset].inner() as _),
            )?;
            ssim.sqr(tmp, ctx1)?;
        }
        self.imgt[scale][tmp].sqr_ip(ctx1)?;
        self.imgt[scale][tmp].sum_into(
            scratch1,
            (&mut self.scores
                [scale * 6 * 3 + offset / 2 * 3 + 9..scale * 6 * 3 + offset / 2 * 3 + 12])
                .try_into()
                .unwrap(),
            ctx1,
        )?;
        Ok(())
    }

    fn post_process_scores(&mut self) -> f64 {
        // TODO jeez that's a lot of zeros
        //  Computing with a finer granularity (e.g. separated planes)
        //  may allow us to reduce total computations since there are a lot of zeros
        #[rustfmt::skip]
        const WEIGHT: [f64; 108] = [
            // X
            // Scale 0
            0.0,
            0.000_737_660_670_740_658_6,
            0.0,
            0.0,
            0.000_779_348_168_286_730_9,
            0.0,
            // Scale 1
            0.0,
            0.000_437_115_573_010_737_9,
            0.0,
            1.104_172_642_665_734_6,
            0.000_662_848_341_292_71,
            0.000_152_316_327_837_187_52,
            // Scale 2
            0.0,
            0.001_640_643_745_659_975_4,
            0.0,
            1.842_245_552_053_929_8,
            11.441_172_603_757_666,
            0.0,
            // Scale 3
            0.000_798_910_943_601_516_3,
            0.000_176_816_438_078_653,
            0.0,
            1.878_759_497_954_638_7,
            10.949_069_906_051_42,
            0.0,
            // Scale 4
            0.000_728_934_699_150_807_2,
            0.967_793_708_062_683_3,
            0.0,
            0.000_140_034_242_854_358_84,
            0.998_176_697_785_496_7,
            0.000_319_497_559_344_350_53,
            // Scale 5
            0.000_455_099_211_379_206_3,
            0.0,
            0.0,
            0.001_364_876_616_324_339_8,
            0.0,
            0.0,
            // Y
            // Scale 0
            0.0,
            0.0,
            0.0,
            7.466_890_328_078_848,
            0.0,
            17.445_833_984_131_262,
            // Scale 1
            0.000_623_560_163_404_146_6,
            0.0,
            0.0,
            6.683_678_146_179_332,
            0.000_377_244_079_796_112_96,
            1.027_889_937_768_264,
            // Scale 2
            225.205_153_008_492_74,
            0.0,
            0.0,
            19.213_238_186_143_016,
            0.001_140_152_458_661_836_1,
            0.001_237_755_635_509_985,
            // Scale 3
            176.393_175_984_506_94,
            0.0,
            0.0,
            24.433_009_998_704_76,
            0.285_208_026_121_177_57,
            0.000_448_543_692_383_340_8,
            // Scale 4
            0.0,
            0.0,
            0.0,
            34.779_063_444_837_72,
            44.835_625_328_877_896,
            0.0,
            // Scale 5
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            // B
            // Scale 0
            0.0,
            0.000_868_055_657_329_169_8,
            0.0,
            0.0,
            0.0,
            0.0,
            // Scale 1
            0.0,
            0.000_531_319_187_435_874_7,
            0.0,
            0.000_165_338_141_613_791_12,
            0.0,
            0.0,
            // Scale 2
            0.0,
            0.0,
            0.0,
            0.000_417_917_180_325_133_6,
            0.001_729_082_823_472_283_3,
            0.0,
            // Scale 3
            0.002_082_700_584_663_643_7,
            0.0,
            0.0,
            8.826_982_764_996_862,
            23.192_433_439_989_26,
            0.0,
            // Scale 4
            95.108_049_881_108_6,
            0.986_397_803_440_068_2,
            0.983_438_279_246_535_3,
            0.001_228_640_504_827_849_3,
            171.266_725_589_730_7,
            0.980_785_887_243_537_9,
            // Scale 5
            0.0,
            0.0,
            0.0,
            0.000_513_006_458_899_067_9,
            0.0,
            0.000_108_540_578_584_115_37,
        ];

        for scale in 0..SCALES {
            let opp = self.sizes_t[scale].norm();
            let offset = 3 * 6 * scale;
            for c in 0..3 {
                let offset = offset + c;
                let offsetw = c * 6 * 6 + 6 * scale;
                self.scores[offset] = (self.scores[offset] * opp).abs() * WEIGHT[offsetw];
                self.scores[offset + 3] =
                    (self.scores[offset + 3] * opp).abs() * WEIGHT[offsetw + 1];
                self.scores[offset + 2 * 3] =
                    (self.scores[offset + 2 * 3] * opp).abs() * WEIGHT[offsetw + 2];
                self.scores[offset + 3 * 3] =
                    (self.scores[offset + 3 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 3];
                self.scores[offset + 4 * 3] =
                    (self.scores[offset + 4 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 4];
                self.scores[offset + 5 * 3] =
                    (self.scores[offset + 5 * 3] * opp).sqrt().sqrt() * WEIGHT[offsetw + 5];
            }
        }

        let mut score: f64 = self.scores.iter().sum();

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
}

/*
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
}*/
