use crate::output::Output;
use clap::{Parser, ValueEnum};
use indicatif::ProgressStyle;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{stdin, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing::{debug, error, info, info_span, trace};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use turbo_metrics::cudarse_driver::CuStream;
use turbo_metrics::input_image::{ImageFrameSource, ImageProbe, PROBE_LEN};
use turbo_metrics::input_video::{VideoFrameSource, VideoProbe};
use turbo_metrics::npp::set_stream;
use turbo_metrics::{init_cuda, FrameSource, MetricsResults, Options, TurboMetrics};

mod output;

/// Turbo metrics compares two images or videos using quality metrics.
///
/// Video decoding and metric computations happen on your Nvidia GPU using CUDA.
///
/// Use the `RUST_LOG` environment variable to configure logging. The trace level can be very verbose, with many events per frame.
#[derive(Parser, Debug)]
#[command(version, author)]
struct CliArgs {
    /// Reference media. Use `-` to read from stdin.
    reference: PathBuf,
    /// Distorted media. Use `-` to read from stdin.
    distorted: PathBuf,

    /// Select the metrics to compute, the video will only be decoded once.
    #[arg(short, long)]
    metrics: Vec<Metrics>,

    /// Only compute metrics every few frames, effectively down-sampling the measurements.
    /// Frames in between will still be decoded.
    #[arg(long, default_value = "0")]
    every: u32,
    /// Index of the first frame to start computing at. Useful for overlaying separate computations with `every`.
    /// No efficient seeking is implemented yet. All frames will be decoded.
    #[arg(long, default_value = "0")]
    skip: u32,
    /// Index of the first frame to start computing at the reference frame. Additive with `skip`.
    #[arg(long, default_value = "0")]
    skip_ref: u32,
    /// Index of the first frame to start computing at the distorted frame. Additive with `skip`.
    #[arg(long, default_value = "0")]
    skip_dis: u32,
    /// Amount of frames to compute. Useful for computing subsets with `skip`, `skip-ref`, and `skip-dis`.
    #[arg(long, default_value = "0")]
    frames: u32,

    /// Choose the CLI stdout format. Omit the option for the default.
    /// Status messages will be printed to stderr in all cases.
    #[arg(long, value_enum)]
    output: Option<Output>,
}

#[derive(Debug, Copy, Clone, PartialEq, ValueEnum)]
pub enum Metrics {
    /// PSNR computed with NPP in linear RGB
    PSNR,
    /// SSIM computed with NPP in linear RGB
    SSIM,
    /// MSSSIM computed with NPP in linear RGB
    MSSSIM,
    /// SSIMULACRA2 computed with CUDA
    SSIMULACRA2,
}

impl CliArgs {
    fn metrics(&self) -> turbo_metrics::Metrics {
        turbo_metrics::Metrics {
            psnr: self.metrics.contains(&Metrics::PSNR),
            ssim: self.metrics.contains(&Metrics::SSIM),
            msssim: self.metrics.contains(&Metrics::MSSSIM),
            ssimulacra2: self.metrics.contains(&Metrics::SSIMULACRA2),
        }
    }

    fn video_options(&self) -> Options {
        Options {
            every: self.every,
            skip: self.skip,
            skip_ref: self.skip_ref,
            skip_dis: self.skip_dis,
            frames: self.frames,
        }
    }

    fn output(&self) -> Output {
        self.output.unwrap_or_default()
    }
}

fn main() -> ExitCode {
    let args = CliArgs::parse();

    let mut env_filter = EnvFilter::builder();
    env_filter = if cfg!(debug_assertions) {
        env_filter.with_default_directive(LevelFilter::DEBUG.into())
    } else {
        env_filter.with_default_directive(LevelFilter::INFO.into())
    };
    let env_filter = env_filter.from_env_lossy();

    let indicatif_layer = IndicatifLayer::new();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .compact()
                .without_time()
                .with_writer(indicatif_layer.get_stderr_writer())
                .with_filter(env_filter),
        )
        .with(indicatif_layer)
        .init();

    let ref_is_stdin = args.reference.to_str() == Some("-");
    let dis_is_stdin = args.distorted.to_str() == Some("-");

    if ref_is_stdin && dis_is_stdin {
        error!("Can't read both reference and distorted from stdin");
        return ExitCode::FAILURE;
    }

    // Frame sources might need cuda
    init_cuda();

    let source_ref = match create_source(&args.reference, ref_is_stdin) {
        Ok(source) => source,
        Err(e) => {
            error!("Could not read reference : {e}");
            return ExitCode::FAILURE;
        }
    };
    let source_dis = match create_source(&args.distorted, dis_is_stdin) {
        Ok(source) => source,
        Err(e) => {
            error!("Could not read distorted : {e}");
            return ExitCode::FAILURE;
        }
    };

    if (source_ref.width(), source_ref.height()) != (source_dis.width(), source_dis.height()) {
        error!("Reference and distorted are not the same size");
    }

    let turbo = match TurboMetrics::new(source_ref.width(), source_ref.height(), &args.metrics()) {
        Ok(turbo) => turbo,
        Err(e) => {
            error!("Could not initialize engine : {e}");
            return ExitCode::FAILURE;
        }
    };

    compute(
        turbo,
        source_ref,
        source_dis,
        &args.video_options(),
        args.output(),
    );
    ExitCode::SUCCESS
}

fn create_source(path: &Path, is_stdin: bool) -> Result<Box<dyn FrameSource>, Box<dyn Error>> {
    let mut stream: BufReader<Box<dyn Read>> = BufReader::with_capacity(
        PROBE_LEN,
        if is_stdin {
            // Use stdin
            Box::new(stdin().lock())
        } else {
            Box::new(File::open(path).unwrap())
        },
    );
    if let Some(probe_ref) = ImageProbe::probe_image(&mut stream)? {
        if probe_ref.can_decode() {
            Ok(Box::new(ImageFrameSource::new(&mut stream, probe_ref)?))
        } else {
            error!(
                "'{}' detected as {:?} but no decoder is available (missing crate feature or unimplemented).",
                path.display(),
                probe_ref
            );
            Err("no decoder".into())
        }
    } else {
        let probe = if is_stdin {
            VideoProbe::probe_stream(stream)
        } else {
            VideoProbe::probe_file(BufReader::new(File::open(path).unwrap()))
        };

        match probe {
            Ok(Ok(probe)) => Ok(Box::new(VideoFrameSource::new(probe.make_demuxer()))),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

fn compute(
    mut turbo: TurboMetrics,
    mut frames_ref: impl FrameSource,
    mut frames_dis: impl FrameSource,
    opts: &Options,
    output: Output,
) {
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
        frame_count=frames_ref.frame_count()
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
        frame_count=frames_dis.frame_count()
    );

    let mut scores_psnr = turbo.has_psnr().then(|| Vec::with_capacity(4096));
    let mut scores_ssim = turbo.has_ssim().then(|| Vec::with_capacity(4096));
    let mut scores_msssim = turbo.has_msssim().then(|| Vec::with_capacity(4096));
    let mut scores_ssimu = turbo.has_ssimulacra2().then(|| Vec::with_capacity(4096));

    let start = Instant::now();
    let mut decode_count = 0;
    let mut compute_count = 0;

    // This can be 0 if the source does not report the total frame count
    let useful_decode_amount = frames_ref
        .frame_count()
        .saturating_sub(opts.skip_ref as _)
        .max(frames_dis.frame_count().saturating_sub(opts.skip_dis as _))
        .saturating_sub(opts.skip as _)
        .min(opts.frames as _);

    let span = info_span!("pb");
    if useful_decode_amount > 0 {
        span.pb_set_style(
            &ProgressStyle::with_template("{wide_bar} {pos}/{len} {msg} (eta: {eta})").unwrap(),
        );
    } else {
        span.pb_set_style(
            &ProgressStyle::with_template("{spinner} {pos}/? {msg} (elapsed: {elapsed})").unwrap(),
        )
    }
    if useful_decode_amount > 0 {
        span.pb_set_length(useful_decode_amount as u64);
    }
    span.pb_set_message("Seeking");

    debug!("Initialized, now processing ...");

    let pb = span.enter();

    frames_ref.skip_frames(opts.skip_ref + opts.skip);
    frames_dis.skip_frames(opts.skip_dis + opts.skip);

    span.pb_set_message("Computing");

    while let Some((fref, fdis)) = frames_ref
        .next_frame(turbo.stream_ref())
        .unwrap()
        .zip(frames_dis.next_frame(turbo.stream_dis()).unwrap())
    {
        if opts.every > 1 && decode_count != 0 && decode_count % opts.every != 0 {
            decode_count += 1;
            continue;
        }

        if opts.frames > 0 && decode_count >= opts.frames {
            break;
        }

        decode_count += 1;
        trace!(frame = decode_count, "Computing metrics for frame");

        let (psnr, ssim, msssim, ssimu) =
            turbo.compute_one(fref, (cc_ref, cr_ref), fdis, (cc_dis, cr_dis));

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
        span.pb_inc(1);
    }

    drop(pb);
    drop(span);

    let duration = start.elapsed();
    let fps = compute_count as u128 * 1000 / duration.as_millis();
    let perf_score = frames_ref.width() as f64 * frames_ref.height() as f64 * compute_count as f64
        / duration.as_millis() as f64
        / 1000.0;
    info!(
        "Processed: {} (decoded: ~{}) frame pairs in {} ({} fps) (Mpx/s: {:.3})",
        compute_count,
        decode_count + opts.skip,
        format_duration(duration),
        fps,
        perf_score
    );

    // Default drop impl for npp image buffers are using this global stream
    // The stream we set before is being destroyed before the drop
    set_stream(CuStream::DEFAULT.inner() as _).unwrap();

    output.display_results(&MetricsResults {
        frame_count: compute_count,
        psnr: scores_psnr.map(Into::into),
        ssim: scores_ssim.map(Into::into),
        msssim: scores_msssim.map(Into::into),
        ssimulacra2: scores_ssimu.map(Into::into),
    });
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
