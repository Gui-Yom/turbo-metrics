use clap::Parser;
use std::fs::File;
use std::io::{stdin, BufReader, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use turbo_metrics::input_image::probe_image;
use turbo_metrics::input_video::{Demuxer, VideoProbe};
use turbo_metrics::{
    init_cuda, process_img_pair, process_video_pair, MetricsToCompute, VideoOptions,
};

/// Turbo metrics compare the video tracks of two mkv files.
///
/// You can select many metrics to be computed together, which will reduce overhead.
#[derive(Parser, Debug)]
#[command(version, author)]
struct CliArgs {
    /// Reference media, either a video muxed in mkv or a single image.
    reference: PathBuf,
    /// Distorted media, either a video muxed in mkv or a single image. Use '-' to read from stdin.
    distorted: PathBuf,

    /// Compute PSNR score (computed using NPP in linear RGB)
    #[arg(long)]
    psnr: bool,
    /// Compute SSIM score (computed using NPP in linear RGB)
    #[arg(long)]
    ssim: bool,
    /// Compute MSSSIM score (computed using NPP in linear RGB)
    #[arg(long)]
    msssim: bool,
    /// Compute ssimulacra2 score
    #[arg(long)]
    ssimulacra2: bool,

    /// Only compute metrics every few frames, effectively down-sampling the measurements.
    /// Still, this tool will decode all frames, hence increasing overhead. Check Mpx/s to see what I mean.
    ///
    /// E.g. 8 invocations with --every 8 will perform around 50% worse than a single pass computing every frame.
    #[arg(long, default_value = "0")]
    every: u32,
    /// Index of the first frame to start computing at. Useful for overlaying separate computations with `every`.
    #[arg(long, default_value = "0")]
    skip: u32,
}

impl CliArgs {
    fn metrics(&self) -> MetricsToCompute {
        MetricsToCompute {
            psnr: self.psnr,
            ssim: self.ssim,
            msssim: self.msssim,
            ssimulacra2: self.ssimulacra2,
        }
    }

    fn video_options(&self) -> VideoOptions {
        VideoOptions {
            every: self.every,
            skip: self.skip,
        }
    }
}

fn main() -> ExitCode {
    let args = CliArgs::parse();

    let dis_is_stdin = args.distorted.to_str() == Some("-");

    let mut in_ref = BufReader::new(File::open(&args.reference).unwrap());
    let mut in_dis: BufReader<Box<dyn Read>> = BufReader::new(if dis_is_stdin {
        // Use stdin
        Box::new(stdin().lock())
    } else {
        Box::new(File::open(&args.distorted).unwrap())
    });

    // Try with an image first
    if let Some(probe_ref) = probe_image(&mut in_ref) {
        if probe_ref.can_decode() {
            if let Some(probe_dis) = probe_image(&mut in_dis) {
                if probe_dis.can_decode() {
                    if args.every != 0 || args.skip != 0 {
                        eprintln!("WARN: --every and --skip are useless with a pair of images");
                    }

                    init_cuda();
                    // Yay, we can decode both as an image
                    process_img_pair(
                        &mut in_ref,
                        &mut in_dis,
                        probe_ref,
                        probe_dis,
                        &args.metrics(),
                    );
                    ExitCode::SUCCESS
                } else {
                    eprintln!(
                        "Distorted '{}' detected as {:?} but no decoder is available (missing crate feature / unimplemented).",
                        args.distorted.display(),
                        probe_dis
                    );
                    eprintln!("Aborting.");
                    ExitCode::FAILURE
                }
            } else {
                eprintln!("Reference is an image, so we expect an image as distorted.");
                eprintln!("Aborting.");
                ExitCode::FAILURE
            }
        } else {
            eprintln!(
                "Reference '{}' detected as {:?} but no decoder is available (missing crate feature or unimplemented).",
                args.reference.display(),
                probe_ref
            );
            eprintln!("Aborting.");
            ExitCode::FAILURE
        }
    } else {
        match VideoProbe::probe_file(in_ref) {
            Ok(Ok(probe_ref)) => {
                let probe_dis = if dis_is_stdin {
                    VideoProbe::probe_stream(in_dis)
                } else {
                    VideoProbe::probe_file(BufReader::new(File::open(&args.distorted).unwrap()))
                };
                match probe_dis {
                    Ok(Ok(probe_dis)) => {
                        init_cuda();
                        process_video_pair(
                            <VideoProbe as Into<Box<dyn Demuxer>>>::into(probe_ref),
                            <VideoProbe as Into<Box<dyn Demuxer>>>::into(probe_dis),
                            &args.metrics(),
                            &args.video_options(),
                        );
                        ExitCode::SUCCESS
                    }
                    Ok(Err(e)) => {
                        if dis_is_stdin {
                            eprintln!("Unsupported distorted from stream : {e:?}");
                        } else {
                            eprintln!(
                                "Unsupported distorted file '{}' : {e:?}",
                                args.distorted.display()
                            );
                        }
                        ExitCode::FAILURE
                    }
                    Err(e) => {
                        if dis_is_stdin {
                            eprintln!("Could not read distorted from stdin : {e}");
                        } else {
                            eprintln!(
                                "Could not read distorted file '{}' : {e}",
                                args.distorted.display()
                            );
                        }
                        ExitCode::FAILURE
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!(
                    "Unsupported reference file '{}' : {e:?}",
                    args.reference.display(),
                );
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!(
                    "Can't read reference file '{}' : {e}",
                    args.reference.display(),
                );
                ExitCode::FAILURE
            }
        }
    }
}
