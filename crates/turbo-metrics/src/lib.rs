use crate::color::{
    color_characteristics_from_format, convert_frame_to_linearrgb, cpu_to_linear, video_color_print,
};
use crate::input::{decode_image_frames, DemuxerParser, ImageProbe};
use cuda_colorspace::ColorspaceConversion;
pub use cudarse_driver;
use cudarse_driver::{CuDevice, CuStream};
pub use cudarse_npp as npp;
use cudarse_npp::image::ist::{PSNR, SSIM, WMSSSIM};
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Img, C};
use cudarse_npp::{get_stream_ctx, set_stream};
pub use cudarse_video;
use cudarse_video::dec_simple::NvDecoderSimple;
use ssimulacra2_cuda::Ssimulacra2;
pub use stats;
use stats::full::Stats;
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;
use std::time::Instant;

pub mod color;
pub mod img;
pub mod input;

#[derive(Debug, Default)]
pub struct MetricsToCompute {
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
pub struct VideoOptions {
    /// Only compute metrics every few frames, effectively down-sampling the measurements.
    /// Still, this tool will decode all frames, hence increasing overhead. Check Mpx/s to see what I mean.
    ///
    /// E.g. 8 invocations with --every 8 will perform around 50% worse than a single pass computing every frame.
    pub every: u32,
    /// Index of the first frame to start computing at. Useful for overlaying separate computations with `every`.
    pub skip: u32,
}

pub fn init_cuda() {
    struct Cuda(CuDevice);
    static CUDA_INIT: LazyLock<Cuda> = LazyLock::new(|| {
        cudarse_driver::init_cuda().expect("Could not initialize the CUDA API");
        let dev = CuDevice::get(0).unwrap();
        println!(
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

pub fn process_img_pair(
    in_ref: impl Read,
    in_dis: impl Read,
    probe_ref: ImageProbe,
    probe_dis: ImageProbe,
    metrics: &MetricsToCompute,
) {
    let cpu_img_ref = decode_image_frames(in_ref, probe_ref);
    let cpu_img_dis = decode_image_frames(in_dis, probe_dis);

    eprintln!("Reference: {:?}, {}", probe_ref, cpu_img_ref[0]);
    eprintln!("Distorted: {:?}, {}", probe_dis, cpu_img_dis[0]);

    let width = cpu_img_ref[0].width;
    let height = cpu_img_ref[0].height;

    if cpu_img_ref.len() != cpu_img_dis.len() {
        eprintln!("Images have a different number of frames. Aborting.");
        return;
    }

    if cpu_img_ref.len() > 1 {
        eprintln!(
            "Note that animated images are unsupported, only the first frame will be evaluated"
        );
    }

    if width != cpu_img_dis[0].width || height != cpu_img_dis[0].height {
        eprintln!("Images are not the same size. Aborting.");
        return;
    }

    let stream = CuStream::new().unwrap();
    let stream2 = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();
    let ctx = get_stream_ctx().unwrap();

    let conversion = ColorspaceConversion::new();

    let mut linear_ref = npp::image::Image::malloc(width, height).unwrap();
    let mut linear_dis = linear_ref.malloc_same_size().unwrap();

    let mut quant_ref = linear_ref.malloc_same_size().unwrap();
    let mut quant_dis = linear_ref.malloc_same_size().unwrap();

    let mut psnr = metrics
        .psnr
        .then(|| quant_ref.psnr_alloc_scratch(ctx).unwrap());
    let mut ssim = metrics
        .ssim
        .then(|| quant_ref.ssim_alloc_scratch(ctx).unwrap());
    let mut msssim = metrics
        .msssim
        .then(|| quant_ref.wmsssim_alloc_scratch(ctx).unwrap());
    let mut ssimu2 = metrics
        .ssimulacra2
        .then(|| Ssimulacra2::new(&linear_ref, &linear_dis, &stream).unwrap());

    cpu_to_linear(&cpu_img_ref[0], &mut linear_ref, &conversion, &stream);
    cpu_to_linear(&cpu_img_dis[0], &mut linear_dis, &conversion, &stream2);

    if psnr.is_some() || ssim.is_some() || msssim.is_some() {
        conversion
            .f32_to_8bit(&linear_ref, &mut quant_ref, &stream)
            .unwrap();
        conversion
            .f32_to_8bit(&linear_dis, &mut quant_dis, &stream2)
            .unwrap();
    }

    stream.join(&stream2).unwrap();

    if let Some(psnr) = &mut psnr {
        let score = quant_ref.psnr(&quant_dis, psnr, ctx).unwrap();
        stream.sync().unwrap();
        println!("PSNR: {score:.3}");
    }

    if let Some(ssim) = &mut ssim {
        let score = quant_ref.ssim(&quant_dis, ssim, ctx).unwrap();
        stream.sync().unwrap();
        println!("SSIM: {score:.3}");
    }

    if let Some(msssim) = &mut msssim {
        let score = quant_ref.wmsssim(&quant_dis, msssim, ctx).unwrap();
        stream.sync().unwrap();
        println!("MSSSIM: {score:.3}");
    }

    if let Some(ssimu2) = &mut ssimu2 {
        let score = ssimu2.compute_sync(&stream).unwrap();
        println!("SSIMULACRA2: {score:.3}");
    }
}

pub fn process_video_pair(
    reference: &Path,
    distorted: &Path,
    metrics: &MetricsToCompute,
    opts: &VideoOptions,
) {
    // Init the colorspace conversion module
    let colorspace = ColorspaceConversion::new();

    let cb_ref = NvDecoderSimple::new(3, None);
    let cb_dis = NvDecoderSimple::new(3, None);

    let mut demuxer_ref = DemuxerParser::new(reference, &cb_ref);
    // let mut demuxer_dis = DemuxerParser::new("data/dummy_encode2.mkv", &cb_dis);
    let mut demuxer_dis = DemuxerParser::new(distorted, &cb_dis);

    while cb_ref.format().is_none() {
        demuxer_ref.demux();
    }
    let format = cb_ref.format().unwrap();
    let colors_ref = color_characteristics_from_format(&format);
    println!("ref: {}", video_color_print(&format));

    let size = format.size();

    while cb_dis.format().is_none() {
        demuxer_dis.demux();
    }
    let format_dis = cb_dis.format().unwrap();
    let colors_dis = color_characteristics_from_format(&format_dis);
    println!("dis: {}", video_color_print(&format_dis));

    assert_eq!(size, format_dis.size());

    let streams = [(); 5].map(|()| CuStream::new().unwrap());
    set_stream(streams[0].inner() as _).unwrap();
    let npp = get_stream_ctx().unwrap();

    let mut lrgb_ref =
        cudarse_npp::image::Image::malloc(size.width as _, size.height as _).unwrap();
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
        println!("Initializing PSNR");
        (
            Some(quant_ref.as_ref().unwrap().psnr_alloc_scratch(npp).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    let (mut scratch_ssim, mut scores_ssim) = if metrics.ssim {
        println!("Initializing SSIM");
        (
            Some(quant_ref.as_ref().unwrap().ssim_alloc_scratch(npp).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    let (mut scratch_msssim, mut scores_msssim) = if metrics.msssim {
        println!("Initializing MSSSIM");
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
        println!("Initializing SSIMULACRA2");
        (
            Some(Ssimulacra2::new(&lrgb_ref, &lrgb_dis, &streams[0]).unwrap()),
            Some(Vec::with_capacity(4096)),
        )
    } else {
        (None, None)
    };

    streams[0].sync().unwrap();

    println!("Initialized, now processing ...");
    let start = Instant::now();
    let mut decode_count = 0;
    let mut compute_count = 0;

    'main: loop {
        while !cb_ref.has_frames() {
            demuxer_ref.demux();
        }
        while !cb_dis.has_frames() {
            demuxer_dis.demux();
        }
        for (fref, fdis) in cb_ref.frames_sync(&cb_dis) {
            if let (Some(fref), Some(fdis)) = (fref, fdis) {
                if decode_count < opts.skip
                    || (opts.every > 1 && decode_count != 0 && decode_count % opts.every != 0)
                {
                    decode_count += 1;
                    continue;
                }
                decode_count += 1;

                convert_frame_to_linearrgb(
                    cb_ref.map_npp(&fref, &streams[0]).unwrap(),
                    colors_ref,
                    &colorspace,
                    &mut lrgb_ref,
                    &streams[0],
                );

                // if decode_count == 130
                //     || decode_count == 200
                //     || decode_count == 250
                //     || decode_count == 300
                // {
                //     save_img_f32(&lrgb_ref, &format!("lrgb_ref{}", decode_count), &streams[0]);
                //     // break 'main;
                // }

                convert_frame_to_linearrgb(
                    cb_dis.map_npp(&fdis, &streams[1]).unwrap(),
                    colors_dis,
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
                    scores_psnr.push(psnr as f64);
                }
                if let Some(scores_ssim) = &mut scores_ssim {
                    scores_ssim.push(ssim as f64);
                }
                if let Some(scores_msssim) = &mut scores_msssim {
                    scores_msssim.push(msssim as f64);
                }
                if let Some(scores_ssimu) = &mut scores_ssimu {
                    scores_ssimu.push(ssimu.as_mut().unwrap().get_score());
                }

                compute_count += 1;
            } else {
                break 'main;
            }
        }
    }

    let total = start.elapsed().as_millis();
    let fps = compute_count as u128 * 1000 / total;
    let perf_score =
        format.size().width as f64 * format.size().height as f64 * compute_count as f64
            / total as f64
            / 1000.0;
    println!("Done !");
    println!(
        "Decoded: {}, processed: {} frame pairs in {total} ms ({} fps) (Mpx/s: {:.3})",
        decode_count, compute_count, fps, perf_score
    );
    println!("Stats :");
    if let Some(scores) = &scores_psnr {
        println!("  psnr: {:#?}", Stats::compute(scores));
    }
    if let Some(scores) = &scores_ssim {
        println!("  ssim: {:#?}", Stats::compute(scores));
    }
    if let Some(scores) = &scores_msssim {
        println!("  msssim: {:#?}", Stats::compute(scores));
    }
    if let Some(scores) = &scores_ssimu {
        println!("  ssimulacra2: {:#?}", Stats::compute(scores));
    }
}

pub fn save_img(img: impl Img<u8, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_u8(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}

pub fn save_img_u16(img: impl Img<u16, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_u16(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}

pub fn save_img_f32(img: impl Img<f32, C<3>>, name: &str, stream: &CuStream) {
    use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};
    let bytes = img.copy_to_cpu(stream.inner() as _).unwrap();
    stream.sync().unwrap();
    let mut img = zune_image::image::Image::from_f32(
        &bytes,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::RGB,
    );
    img.metadata_mut()
        .set_color_trc(ColorCharacteristics::Linear);
    img.save(format!("frames/{name}.png")).unwrap()
}
