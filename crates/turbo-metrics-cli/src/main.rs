use clap::{Args, Parser};
use std::fmt::Display;
use std::fs::File;
use std::io::{stdin, BufReader, Read};
use std::path::PathBuf;
use turbo_metrics::input::peekable::PeekExt;
use turbo_metrics::input::{probe_image, PROBE_LEN};
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

    /// Compute PSNR score
    #[arg(long)]
    psnr: bool,
    /// Compute SSIM score
    #[arg(long)]
    ssim: bool,
    /// Compute MSSSIM score
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

fn main() {
    let args = CliArgs::parse();

    let mut in_ref =
        BufReader::new(File::open(&args.reference).unwrap()).peekable_with_capacity(PROBE_LEN);
    let in_dis: Box<dyn Read> = if args.distorted.to_str() == Some("-") {
        // Use stdin
        Box::new(BufReader::new(stdin().lock()))
    } else {
        Box::new(BufReader::new(File::open(&args.distorted).unwrap()))
    };
    let mut in_dis = in_dis.peekable_with_capacity(PROBE_LEN);

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
                    return;
                } else {
                    eprintln!(
                        "Distorted '{}' detected as {:?} but no decoder is available (missing crate feature / unimplemented).",
                        args.distorted.display(),
                        probe_dis
                    );
                    eprintln!("Aborting.");
                    return;
                }
            } else {
                eprintln!("Reference is an image, so we expect an image as distorted.");
                eprintln!("Aborting.");
                return;
            }
        } else {
            eprintln!(
                "Reference '{}' detected as {:?} but no decoder is available (missing crate feature or unimplemented).",
                args.reference.display(),
                probe_ref
            );
            eprintln!("Aborting.");
            return;
        }
    } else {
        // not an image
        init_cuda();
        process_video_pair(
            &args.reference,
            &args.distorted,
            &args.metrics(),
            &args.video_options(),
        );
    }
}
