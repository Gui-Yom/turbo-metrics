use clap::ValueEnum;
use std::io::stdout;
use std::{io, iter};
use turbo_metrics::{ResultsImage, ResultsVideo};

#[derive(Debug, Copy, Clone, Default, ValueEnum)]
pub(crate) enum Output {
    /// Default classic output for human reading. This won't print the score for each individual frames.
    #[default]
    Default,
    /// Json object output. Contains both per-frame scores and aggregated stats.
    Json,
    /// CSV output. Only contains per-frame scores.
    CSV,
}

impl Output {
    pub(crate) fn display_image_result(&self, results: &ResultsImage) {
        match self {
            Output::Default => {
                if let Some(score) = results.psnr {
                    println!("PSNR: {score:.3}");
                }
                if let Some(score) = results.ssim {
                    println!("SSIM: {score:.3}");
                }
                if let Some(score) = results.msssim {
                    println!("MSSSIM: {score:.3}");
                }
                if let Some(score) = results.ssimulacra2 {
                    println!("SSIMULACRA2: {score:.3}");
                }
            }
            Output::Json => {
                println!("{}", serde_json::to_string_pretty(results).unwrap());
            }
            Output::CSV => {
                let mut csv = csv::Writer::from_writer(stdout().lock());
                // CSV header
                csv.write_record(
                    results
                        .psnr
                        .map(|_| "psnr")
                        .into_iter()
                        .chain(results.ssim.map(|_| "ssim"))
                        .chain(results.msssim.map(|_| "msssim"))
                        .chain(results.ssimulacra2.map(|_| "ssimulacra2")),
                )
                .unwrap();
                csv.write_record(
                    results
                        .psnr
                        .into_iter()
                        .chain(results.ssim)
                        .chain(results.msssim)
                        .chain(results.ssimulacra2)
                        .map(|s| s.to_string()),
                )
                .unwrap();
            }
        }
    }

    pub(crate) fn display_video_result(&self, results: &ResultsVideo) {
        match self {
            Output::Default => {
                if let Some(results) = &results.psnr {
                    println!("PSNR: {:#?}", results.stats);
                }
                if let Some(results) = &results.ssim {
                    println!("SSIM: {:#?}", results.stats);
                }
                if let Some(results) = &results.msssim {
                    println!("MSSSIM: {:#?}", results.stats);
                }
                if let Some(results) = &results.ssimulacra2 {
                    println!("SSIMULACRA2: {:#?}", results.stats);
                }
            }
            Output::Json => {
                println!("{}", serde_json::to_string_pretty(results).unwrap());
            }
            Output::CSV => {
                let mut csv = csv::Writer::from_writer(stdout().lock());
                // CSV header
                csv.write_record(
                    results
                        .psnr
                        .as_ref()
                        .map(|_| "psnr")
                        .into_iter()
                        .chain(results.ssim.as_ref().map(|_| "ssim"))
                        .chain(results.msssim.as_ref().map(|_| "msssim"))
                        .chain(results.ssimulacra2.as_ref().map(|_| "ssimulacra2")),
                )
                .unwrap();
                let mut fmt_buffer = String::with_capacity(16);
                let mut print_score = |csv: &mut csv::Writer<io::StdoutLock>, x: f64| {
                    use std::fmt::Write;
                    write!(&mut fmt_buffer, "{}", x).unwrap();
                    csv.write_field(&fmt_buffer).unwrap();
                    fmt_buffer.clear();
                };
                for i in 0..results.frame_count {
                    if let Some(r) = &results.psnr {
                        print_score(&mut csv, r.scores[i]);
                    }
                    if let Some(r) = &results.ssim {
                        print_score(&mut csv, r.scores[i]);
                    }
                    if let Some(r) = &results.msssim {
                        print_score(&mut csv, r.scores[i]);
                    }
                    if let Some(r) = &results.ssimulacra2 {
                        print_score(&mut csv, r.scores[i]);
                    }
                    csv.write_record(iter::empty::<&str>()).unwrap()
                }
            }
        }
    }
}
