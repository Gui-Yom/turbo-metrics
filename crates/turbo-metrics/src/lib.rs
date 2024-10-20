use crate::color::{convert_frame_to_linearrgb, ColorRange};
use codec_bitstream::{Codec, ColorCharacteristics};
use cuda_colorspace::ColorspaceConversion;
pub use cudarse_driver;
use cudarse_driver::{CuDevice, CuStream};
pub use cudarse_npp as npp;
use cudarse_npp::image::ist::{PSNR, SSIM, WMSSSIM};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, ImgView, C};
use cudarse_npp::{get_stream_ctx, set_stream, ScratchBuffer};
pub use cudarse_video;
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::sys::cudaVideoCodec;
pub use quick_stats;
use quick_stats::full::Stats;
use ssimulacra2_cuda::Ssimulacra2;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::LazyLock;
use tracing::debug;

pub mod color;
pub mod img;
pub mod input_image;
pub mod input_video;

#[derive(Debug, Default)]
pub struct Metrics {
    /// Compute PSNR score
    pub psnr: bool,
    /// Compute SSIM score
    pub ssim: bool,
    /// Compute MSSSIM score
    pub msssim: bool,
    /// Compute ssimulacra2 score
    pub ssimulacra2: bool,
}

#[derive(Debug, Default)]
pub struct Options {
    /// Only compute metrics every few frames, effectively down-sampling the measurements.
    /// Still, this tool will decode all frames, hence increasing overhead. Check Mpx/s to see what I mean.
    ///
    /// E.g. 8 invocations with --every 8 will perform around 50% worse than a single pass computing every frame.
    pub every: u32,
    /// Index of the first frame to start computing at. Useful for overlaying separate computations with `every`.
    pub skip: u32,
    /// Index of the first frame to start computing at for the reference frame.
    pub skip_ref: u32,
    /// Index of the first frame to start computing at for the distorted frame.
    pub skip_dis: u32,
    /// Amount of frames to compute. Useful for computing subsets with `skip`, `skip-ref`, and `skip-dis`.
    pub frames: u32,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MetricResults {
    pub scores: Vec<f64>,
    pub stats: Stats,
}

impl From<Vec<f64>> for MetricResults {
    fn from(value: Vec<f64>) -> Self {
        Self {
            stats: Stats::compute(&value),
            scores: value,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MetricsResults {
    pub frame_count: usize,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub psnr: Option<MetricResults>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub ssim: Option<MetricResults>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub msssim: Option<MetricResults>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub ssimulacra2: Option<MetricResults>,
}

pub enum HwFrame<'dec> {
    NvDec(NvDecFrame<'dec>),
    Npp8(ImgView<'dec, u8, C<3>>),
    Npp16(ImgView<'dec, u16, C<3>>),
    Npp32(ImgView<'dec, f32, C<3>>),
}

#[derive(Debug)]
pub struct FormatIdentifier {
    container: Option<String>,
    codec: String,
    decoder: String,
}

impl Display for FormatIdentifier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(container) = &self.container {
            write!(f, "{container}/")?;
        }
        write!(f, "{}/{}", self.codec, self.decoder)
    }
}

pub trait FrameSource {
    fn format_id(&self) -> FormatIdentifier;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn color_characteristics(&self) -> (ColorCharacteristics, ColorRange);
    fn frame_count(&self) -> usize;
    fn skip_frames(&mut self, n: u32);
    fn next_frame(&mut self, stream: &CuStream) -> Result<Option<HwFrame>, Box<dyn Error>>;
}

impl FrameSource for Box<dyn FrameSource> {
    fn format_id(&self) -> FormatIdentifier {
        Box::as_ref(self).format_id()
    }

    fn width(&self) -> u32 {
        Box::as_ref(self).width()
    }

    fn height(&self) -> u32 {
        Box::as_ref(self).height()
    }

    fn color_characteristics(&self) -> (ColorCharacteristics, ColorRange) {
        Box::as_ref(self).color_characteristics()
    }

    fn frame_count(&self) -> usize {
        Box::as_ref(self).frame_count()
    }

    fn skip_frames(&mut self, n: u32) {
        Box::as_mut(self).skip_frames(n)
    }

    fn next_frame(&mut self, stream: &CuStream) -> Result<Option<HwFrame>, Box<dyn Error>> {
        Box::as_mut(self).next_frame(stream)
    }
}

pub struct TurboMetrics {
    colorspace: ColorspaceConversion,
    lrgb_ref: Image<f32, C<3>>,
    lrgb_dis: Image<f32, C<3>>,
    quantized: Option<(Image<u8, C<3>>, Image<u8, C<3>>)>,
    psnr: Option<ScratchBuffer>,
    ssim: Option<ScratchBuffer>,
    msssim: Option<ScratchBuffer>,
    ssimulacra2: Option<Ssimulacra2>,
    streams: [CuStream; 5],
}

impl TurboMetrics {
    pub fn new(width: u32, height: u32, metrics: &Metrics) -> Result<Self, Box<dyn Error>> {
        let streams = [(); 5].map(|()| CuStream::new().unwrap());
        set_stream(streams[0].inner() as _)?;
        let npp = get_stream_ctx()?;

        let lrgb_ref = cudarse_npp::image::Image::malloc(width, height)?;
        let lrgb_dis = lrgb_ref.malloc_same_size()?;

        let mut quantized = None;
        let mut psnr = None;
        let mut ssim = None;
        let mut msssim = None;

        if metrics.psnr || metrics.ssim || metrics.msssim {
            let quant_ref = lrgb_ref.malloc_same_size()?;

            psnr = metrics.psnr.then(|| {
                debug!("init psnr");
                quant_ref.psnr_alloc_scratch(npp).unwrap()
            });
            ssim = metrics.ssim.then(|| {
                debug!("init ssim");
                quant_ref.ssim_alloc_scratch(npp).unwrap()
            });
            msssim = metrics.msssim.then(|| {
                debug!("init msssim");
                quant_ref.wmsssim_alloc_scratch(npp).unwrap()
            });
            quantized = Some((quant_ref, lrgb_ref.malloc_same_size()?));
        }

        let ssimulacra2 = metrics.ssimulacra2.then(|| {
            debug!("init ssimulacra2");

            Ssimulacra2::new(&lrgb_ref, &lrgb_dis, &streams[0]).unwrap()
        });

        Ok(Self {
            colorspace: ColorspaceConversion::new(),
            streams,
            lrgb_ref,
            lrgb_dis,
            quantized,
            psnr,
            ssim,
            msssim,
            ssimulacra2,
        })
    }

    pub fn has_psnr(&self) -> bool {
        self.psnr.is_some()
    }

    pub fn has_ssim(&self) -> bool {
        self.ssim.is_some()
    }

    pub fn has_msssim(&self) -> bool {
        self.msssim.is_some()
    }

    pub fn has_ssimulacra2(&self) -> bool {
        self.ssimulacra2.is_some()
    }

    pub fn stream_ref(&self) -> &CuStream {
        &self.streams[0]
    }

    pub fn stream_dis(&self) -> &CuStream {
        &self.streams[1]
    }

    pub fn compute_one(
        &mut self,
        fref: HwFrame<'_>,
        (cc_ref, cr_ref): (ColorCharacteristics, ColorRange),
        fdis: HwFrame<'_>,
        (cc_dis, cr_dis): (ColorCharacteristics, ColorRange),
    ) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        let npp = get_stream_ctx().unwrap();
        convert_frame_to_linearrgb(
            fref,
            (cc_ref, cr_ref),
            &self.colorspace,
            &mut self.lrgb_ref,
            &self.streams[0],
        );

        convert_frame_to_linearrgb(
            fdis,
            (cc_dis, cr_dis),
            &self.colorspace,
            &mut self.lrgb_dis,
            &self.streams[1],
        );

        let mut psnr = 0.0;
        let mut ssim = 0.0;
        let mut msssim = 0.0;

        if let Some((ref_8bit, dis_8bit)) = &mut self.quantized {
            self.streams[2].wait_for_stream(&self.streams[0]).unwrap();
            self.streams[3].wait_for_stream(&self.streams[1]).unwrap();
            self.colorspace
                .f32_to_8bit(&self.lrgb_ref, &mut *ref_8bit, &self.streams[2])
                .unwrap();
            self.colorspace
                .f32_to_8bit(&self.lrgb_dis, &mut *dis_8bit, &self.streams[3])
                .unwrap();

            self.streams[2].wait_for_stream(&self.streams[3]).unwrap();

            if let Some(scratch_psnr) = &mut self.psnr {
                self.streams[4].wait_for_stream(&self.streams[2]).unwrap();
                ref_8bit
                    .psnr_into(
                        &mut *dis_8bit,
                        scratch_psnr,
                        &mut psnr,
                        npp.with_stream(self.streams[4].inner() as _),
                    )
                    .unwrap();
            }
            if let Some(scratch_ssim) = &mut self.ssim {
                self.streams[3].wait_for_stream(&self.streams[2]).unwrap();
                ref_8bit
                    .ssim_into(
                        &mut *dis_8bit,
                        scratch_ssim,
                        &mut ssim,
                        npp.with_stream(self.streams[3].inner() as _),
                    )
                    .unwrap();
            }
            if let Some(scratch_msssim) = &mut self.msssim {
                ref_8bit
                    .wmsssim_into(
                        dis_8bit,
                        scratch_msssim,
                        &mut msssim,
                        npp.with_stream(self.streams[2].inner() as _),
                    )
                    .unwrap();
            }
        }

        if let Some(ssimu) = &mut self.ssimulacra2 {
            self.streams[0].wait_for_stream(&self.streams[1]).unwrap();
            ssimu.compute(&self.streams[0]).unwrap();
        }

        self.streams[0].wait_for_stream(&self.streams[1]).unwrap();
        self.streams[0].wait_for_stream(&self.streams[2]).unwrap();
        self.streams[0].wait_for_stream(&self.streams[3]).unwrap();
        self.streams[0].wait_for_stream(&self.streams[4]).unwrap();

        self.streams[0].sync().unwrap();

        (
            self.psnr.as_ref().map(|_| psnr as f64),
            self.ssim.as_ref().map(|_| ssim as f64),
            self.msssim.as_ref().map(|_| msssim as f64),
            self.ssimulacra2.as_mut().map(|s| s.get_score()),
        )
    }

    pub fn compute_all(
        mut self,
        mut frames_ref: impl FrameSource,
        mut frames_dis: impl FrameSource,
        opts: &Options,
    ) -> Option<MetricsResults> {
        assert_eq!(
            (frames_ref.width(), frames_ref.height()),
            (frames_dis.width(), frames_dis.height()),
            "Reference and distorted are not the same size"
        );

        let (cc_ref, cr_ref) = frames_ref.color_characteristics();
        let (cc_dis, cr_dis) = frames_dis.color_characteristics();

        let mut scores_psnr = self.psnr.as_ref().map(|_| Vec::with_capacity(4096));
        let mut scores_ssim = self.ssim.as_ref().map(|_| Vec::with_capacity(4096));
        let mut scores_msssim = self.msssim.as_ref().map(|_| Vec::with_capacity(4096));
        let mut scores_ssimu = self.ssimulacra2.as_ref().map(|_| Vec::with_capacity(4096));

        let mut decode_count = 0;
        let mut compute_count = 0;

        frames_ref.skip_frames(opts.skip_ref + opts.skip);
        frames_dis.skip_frames(opts.skip_dis + opts.skip);

        while let Some((fref, fdis)) = frames_ref
            .next_frame(&self.streams[0])
            .unwrap()
            .zip(frames_dis.next_frame(&self.streams[1]).unwrap())
        {
            if opts.every > 1 && decode_count != 0 && decode_count % opts.every != 0 {
                decode_count += 1;
                continue;
            }

            if opts.frames > 0 && decode_count >= opts.frames {
                break;
            }

            decode_count += 1;

            let (psnr, ssim, msssim, ssimu) =
                self.compute_one(fref, (cc_ref, cr_ref), fdis, (cc_dis, cr_dis));

            if let Some((scores, value)) = scores_psnr.as_mut().zip(psnr) {
                scores.push(value);
            }
            if let Some((scores, value)) = scores_ssim.as_mut().zip(ssim) {
                scores.push(value);
            }
            if let Some((scores, value)) = scores_msssim.as_mut().zip(msssim) {
                scores.push(value);
            }
            if let Some((scores, value)) = scores_ssimu.as_mut().zip(ssimu) {
                scores.push(value);
            }

            compute_count += 1;
        }

        // Default drop impl for npp image buffers are using this global stream
        // The stream we set before is being destroyed before the drop
        set_stream(CuStream::DEFAULT.inner() as _).unwrap();

        Some(MetricsResults {
            frame_count: compute_count,
            psnr: scores_psnr.map(Into::into),
            ssim: scores_ssim.map(Into::into),
            msssim: scores_msssim.map(Into::into),
            ssimulacra2: scores_ssimu.map(Into::into),
        })
    }
}

/// Init the CUDA API and binds the primary context to the current thread.
/// This can be called many times safely.
pub fn init_cuda() {
    struct Cuda(CuDevice);
    static CUDA_INIT: LazyLock<Cuda> = LazyLock::new(|| {
        cudarse_driver::init_cuda().expect("Could not initialize the CUDA API");
        let dev = CuDevice::get(0).unwrap();
        debug!(
            "Using device {} with CUDA version {}",
            dev.name().unwrap(),
            cudarse_driver::cuda_driver_version().unwrap()
        );
        Cuda(dev)
    });
    CUDA_INIT
        .0
        .retain_primary_ctx()
        .unwrap()
        .set_current()
        .unwrap();
}

pub fn nvdec_to_codec(codec: cudaVideoCodec) -> Codec {
    use cudarse_video::sys::cudaVideoCodec::*;
    match codec {
        cudaVideoCodec_MPEG2 => Codec::MPEG2,
        cudaVideoCodec_H264 => Codec::H264,
        cudaVideoCodec_AV1 => Codec::AV1,
        _ => todo!(),
    }
}

pub fn codec_to_nvdec(codec: Codec) -> cudaVideoCodec {
    use cudarse_video::sys::cudaVideoCodec::*;
    match codec {
        Codec::MPEG2 => cudaVideoCodec_MPEG2,
        Codec::H264 => cudaVideoCodec_H264,
        Codec::AV1 => cudaVideoCodec_AV1,
    }
}

//region debug utilities

// Use like this
//save_img_f32(&lrgb_ref, &format!("lrgb_ref{}", decode_count), &streams[0]);

// pub fn save_img(img: impl Img<u8, C<3>>, name: &str, stream: &CuStream) {
//     use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
//     let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
//     stream.sync().unwrap();
//     let mut img = zune_image::image::Image::from_u8(
//         &bytes,
//         img.width() as usize,
//         img.height() as usize,
//         ColorSpace::RGB,
//     );
//     img.metadata_mut()
//         .set_color_trc(ColorCharacteristics::Linear);
//     img.save(format!("frames/{name}.png")).unwrap()
// }
//
// pub fn save_img_u16(img: impl Img<u16, C<3>>, name: &str, stream: &CuStream) {
//     use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
//     let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
//     stream.sync().unwrap();
//     let mut img = zune_image::image::Image::from_u16(
//         &bytes,
//         img.width() as usize,
//         img.height() as usize,
//         ColorSpace::RGB,
//     );
//     img.metadata_mut()
//         .set_color_trc(ColorCharacteristics::Linear);
//     img.save(format!("frames/{name}.png")).unwrap()
// }
//
// pub fn save_img_f32(img: impl Img<f32, C<3>>, name: &str, stream: &CuStream) {
//     use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
//     let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
//     stream.sync().unwrap();
//     let mut img = zune_image::image::Image::from_f32(
//         &bytes,
//         img.width() as usize,
//         img.height() as usize,
//         ColorSpace::RGB,
//     );
//     img.metadata_mut()
//         .set_color_trc(ColorCharacteristics::Linear);
//     img.save(format!("frames/{name}.png")).unwrap()
// }

//endregion
