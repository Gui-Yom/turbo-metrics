use crate::color::{convert_frame_to_linearrgb, ColorRange};
use codec_bitstream::{Codec, ColorCharacteristics};
use cuda_colorspace::ColorspaceConversion;
pub use cudarse_driver;
use cudarse_driver::{CuDevice, CuStream};
pub use cudarse_npp as npp;
use cudarse_npp::image::ist::{PSNR, SSIM, WMSSSIM};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Img, ImgView, C};
use cudarse_npp::{get_stream_ctx, set_stream};
pub use cudarse_video;
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::sys::cudaVideoCodec;
pub use quick_stats;
use quick_stats::full::Stats;
use ssimulacra2_cuda::Ssimulacra2;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tracing::{debug, info, trace};

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
    pub frame_count: u32,
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

pub fn compute_metrics(
    mut frames_ref: impl FrameSource,
    mut frames_dis: impl FrameSource,
    metrics: &Metrics,
    opts: &Options,
) -> Option<MetricsResults> {
    assert_eq!(
        (frames_ref.width(), frames_ref.height()),
        (frames_dis.width(), frames_dis.height()),
        "Reference and distorted are not the same size"
    );

    // Init the colorspace conversion module
    let colorspace = ColorspaceConversion::new();

    let (cc_ref, cr_ref) = frames_ref.color_characteristics();
    info!(
        target: "reference",
        codec=%frames_ref.format_id(),
        width=frames_ref.width(),
        height=frames_ref.height(),
        cp=?cc_ref.cp,
        mc=?cc_ref.mc,
        tc=?cc_ref.tc,
        cr=?cr_ref,
    );

    let (cc_dis, cr_dis) = frames_dis.color_characteristics();
    info!(
        target: "distorted",
        codec=%frames_dis.format_id(),
        width=frames_dis.width(),
        height=frames_dis.height(),
        cp=?cc_dis.cp,
        mc=?cc_dis.mc,
        tc=?cc_dis.tc,
        cr=?cr_dis,
    );

    let streams = [(); 5].map(|()| CuStream::new().unwrap());
    set_stream(streams[0].inner() as _).unwrap();
    let npp = get_stream_ctx().unwrap();

    let mut lrgb_ref =
        cudarse_npp::image::Image::malloc(frames_ref.width(), frames_ref.height()).unwrap();
    let mut lrgb_dis = lrgb_ref.malloc_same_size().unwrap();

    let (mut quant_ref, mut quant_dis) = if metrics.psnr || metrics.ssim || metrics.msssim {
        (
            Some(lrgb_ref.malloc_same_size().unwrap()),
            Some(lrgb_ref.malloc_same_size().unwrap()),
        )
    } else {
        (None, None)
    };

    let (mut scratch_psnr, mut scores_psnr) = if metrics.psnr {
        debug!("Initializing PSNR");
        (
            Some(quant_ref.as_ref().unwrap().psnr_alloc_scratch(npp).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    let (mut scratch_ssim, mut scores_ssim) = if metrics.ssim {
        debug!("Initializing SSIM");
        (
            Some(quant_ref.as_ref().unwrap().ssim_alloc_scratch(npp).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    let (mut scratch_msssim, mut scores_msssim) = if metrics.msssim {
        debug!("Initializing MSSSIM");
        (
            Some(
                quant_ref
                    .as_ref()
                    .unwrap()
                    .wmsssim_alloc_scratch(npp)
                    .unwrap(),
            ),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    let (mut ssimu, mut scores_ssimu) = if metrics.ssimulacra2 {
        debug!("Initializing SSIMULACRA2");
        (
            Some(Ssimulacra2::new(&lrgb_ref, &lrgb_dis, &streams[0]).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    streams[0].sync().unwrap();

    debug!("Initialized, now processing ...");
    let start = Instant::now();
    let mut decode_count = 0;
    let mut compute_count = 0;

    frames_ref.skip_frames(opts.skip_ref + opts.skip);
    frames_dis.skip_frames(opts.skip_dis + opts.skip);

    // println!();

    while let Some((fref, fdis)) = frames_ref
        .next_frame(&streams[0])
        .unwrap()
        .zip(frames_dis.next_frame(&streams[1]).unwrap())
    {
        if opts.every > 1 && decode_count != 0 && decode_count % opts.every != 0 {
            decode_count += 1;
            continue;
        }

        if opts.frame_count > 0 && decode_count - opts.skip >= opts.frame_count {
            break;
        }

        decode_count += 1;
        trace!(frame = decode_count, "Computing metrics for frame");

        convert_frame_to_linearrgb(
            fref,
            (cc_ref, cr_ref),
            &colorspace,
            &mut lrgb_ref,
            &streams[0],
        );

        convert_frame_to_linearrgb(
            fdis,
            (cc_dis, cr_dis),
            &colorspace,
            &mut lrgb_dis,
            &streams[1],
        );

        let mut psnr = 0.0;
        let mut ssim = 0.0;
        let mut msssim = 0.0;

        if let (Some(ref_8bit), Some(dis_8bit)) = (&mut quant_ref, &mut quant_dis) {
            streams[2].wait_for_stream(&streams[0]).unwrap();
            streams[3].wait_for_stream(&streams[1]).unwrap();
            colorspace
                .f32_to_8bit(&lrgb_ref, &mut *ref_8bit, &streams[2])
                .unwrap();
            colorspace
                .f32_to_8bit(&lrgb_dis, &mut *dis_8bit, &streams[3])
                .unwrap();

            streams[2].wait_for_stream(&streams[3]).unwrap();

            if let Some(scratch_psnr) = &mut scratch_psnr {
                streams[4].wait_for_stream(&streams[2]).unwrap();
                ref_8bit
                    .psnr_into(
                        &mut *dis_8bit,
                        scratch_psnr,
                        &mut psnr,
                        npp.with_stream(streams[4].inner() as _),
                    )
                    .unwrap();
            }
            if let Some(scratch_ssim) = &mut scratch_ssim {
                streams[3].wait_for_stream(&streams[2]).unwrap();
                ref_8bit
                    .ssim_into(
                        &mut *dis_8bit,
                        scratch_ssim,
                        &mut ssim,
                        npp.with_stream(streams[3].inner() as _),
                    )
                    .unwrap();
            }
            if let Some(scratch_msssim) = &mut scratch_msssim {
                ref_8bit
                    .wmsssim_into(
                        dis_8bit,
                        scratch_msssim,
                        &mut msssim,
                        npp.with_stream(streams[2].inner() as _),
                    )
                    .unwrap();
            }
        }

        if let Some(ssimu) = &mut ssimu {
            streams[0].wait_for_stream(&streams[1]).unwrap();
            ssimu.compute(&streams[0]).unwrap();
        }

        streams[0].wait_for_stream(&streams[1]).unwrap();
        streams[0].wait_for_stream(&streams[2]).unwrap();
        streams[0].wait_for_stream(&streams[3]).unwrap();
        streams[0].wait_for_stream(&streams[4]).unwrap();

        streams[0].sync().unwrap();

        if let Some(scores_psnr) = &mut scores_psnr {
            trace!("PSNR {} {}", compute_count, psnr);
            scores_psnr.push(psnr as f64);
        }
        if let Some(scores_ssim) = &mut scores_ssim {
            trace!("SSIM {} {}", compute_count, ssim);
            scores_ssim.push(ssim as f64);
        }
        if let Some(scores_msssim) = &mut scores_msssim {
            trace!("MSSSIM {} {}", compute_count, msssim);
            scores_msssim.push(msssim as f64);
        }
        if let Some(scores_ssimu) = &mut scores_ssimu {
            let ssimu = ssimu.as_mut().unwrap().get_score();
            trace!("SSIMULACRA2 {} {}", compute_count, ssimu);
            scores_ssimu.push(ssimu);
        }

        compute_count += 1;
    }

    let duration = start.elapsed();
    let fps = compute_count as u128 * 1000 / duration.as_millis();
    let perf_score = frames_ref.width() as f64 * frames_ref.height() as f64 * compute_count as f64
        / duration.as_millis() as f64
        / 1000.0;
    info!(
        "Decoded: {}, processed: {} frame pairs in {} ({} fps) (Mpx/s: {:.3})",
        decode_count,
        compute_count,
        format_duration(duration),
        fps,
        perf_score
    );

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

fn format_duration(duration: Duration) -> impl Display {
    struct DurationFmt(Duration);
    impl Display for DurationFmt {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let mut secs = self.0.as_secs();
            let minutes = secs / 60;
            secs = secs % 60;
            let millis = self.0.subsec_millis();
            if minutes > 0 {
                write!(f, "{} m", minutes)?;
                if secs > 0 {
                    write!(f, " ")?;
                }
            }
            if secs > 0 {
                write!(f, "{} s", secs)?;
                if millis > 0 {
                    write!(f, " ")?;
                }
            }
            if millis > 0 {
                write!(f, "{} ms", millis)?;
            }
            Ok(())
        }
    }
    DurationFmt(duration)
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
