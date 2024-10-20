use crate::output::Output;
use clap::{Parser, ValueEnum};
use std::error::Error;
use std::fs::File;
use std::io::{stdin, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing::error;
use tracing::level_filters::LevelFilter;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use turbo_metrics::input_image::{ImageFrameSource, ImageProbe};
use turbo_metrics::input_video::{DynDemuxer, VideoFrameSource, VideoProbe};
use turbo_metrics::{compute_metrics, init_cuda, FrameSource, Options};

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

    let in_ref: BufReader<Box<dyn Read>> = BufReader::new(if ref_is_stdin {
        // Use stdin
        Box::new(stdin().lock())
    } else {
        Box::new(File::open(&args.reference).unwrap())
    });
    let in_dis: BufReader<Box<dyn Read>> = BufReader::new(if dis_is_stdin {
        // Use stdin
        Box::new(stdin().lock())
    } else {
        Box::new(File::open(&args.distorted).unwrap())
    });

    // Frame sources might need cuda
    init_cuda();

    let source_ref = match create_source(&args.reference, ref_is_stdin, in_ref) {
        Ok(source) => source,
        Err(e) => {
            error!("Could not read reference : {e}");
            return ExitCode::FAILURE;
        }
    };
    let source_dis = match create_source(&args.distorted, dis_is_stdin, in_dis) {
        Ok(source) => source,
        Err(e) => {
            error!("Could not read distorted : {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = compute_metrics(
        source_ref,
        source_dis,
        &args.metrics(),
        &args.video_options(),
    );

    if let Some(result) = &result {
        args.output().display_results(result);
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn create_source(
    path: &Path,
    is_stdin: bool,
    mut stream: BufReader<impl Read + 'static>,
) -> Result<Box<dyn FrameSource>, Box<dyn Error>> {
    if let Some(probe_ref) = ImageProbe::probe_image(&mut stream) {
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
            Ok(Ok(probe)) => Ok(Box::new(VideoFrameSource::<DynDemuxer>::new(
                probe.make_demuxer(),
            ))),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}
