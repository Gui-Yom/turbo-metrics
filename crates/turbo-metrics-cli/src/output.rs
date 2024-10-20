use clap::ValueEnum;
use std::io::stdout;
use std::{io, iter};
use turbo_metrics::{FrameScores, Metrics, MetricsResults, MetricsStats};

#[derive(Debug, Copy, Clone, Default, ValueEnum)]
pub(crate) enum Output {
    /// Default classic output for human reading. This won't print the score for each individual frames.
    #[default]
    Default,
    /// Json object output. Contains both per-frame scores and aggregated stats.
    Json,
    /// Output a json object per frame. The last object contains the stats.
    JsonLines,
    /// CSV output. Only contains per-frame scores.
    CSV,
}

impl Output {
    pub(crate) fn prepare(&self, metrics: &Metrics) {
        match self {
            Output::Default => {}
            Output::Json => {}
            Output::JsonLines => {}
            Output::CSV => {
                let mut csv = csv::Writer::from_writer(stdout().lock());
                // CSV header
                csv.write_record(
                    metrics
                        .psnr
                        .then(|| "psnr")
                        .into_iter()
                        .chain(metrics.ssim.then(|| "ssim"))
                        .chain(metrics.msssim.then(|| "msssim"))
                        .chain(metrics.ssimulacra2.then(|| "ssimulacra2")),
                )
                .unwrap();
            }
        }
    }

    pub(crate) fn output_single_score(&self, results: &FrameScores) {
        match self {
            Output::Default => {
                // nothing
            }
            Output::Json => {
                // nothing
            }
            Output::JsonLines => {
                println!("{}", serde_json::to_string(results).unwrap());
            }
            Output::CSV => {
                let mut csv = csv::Writer::from_writer(stdout().lock());
                let mut fmt_buffer = String::with_capacity(16);
                let mut print_score = |csv: &mut csv::Writer<io::StdoutLock>, x: f64| {
                    use std::fmt::Write;
                    write!(&mut fmt_buffer, "{}", x).unwrap();
                    csv.write_field(&fmt_buffer).unwrap();
                    fmt_buffer.clear();
                };
                if let Some(r) = results.psnr {
                    print_score(&mut csv, r);
                }
                if let Some(r) = results.ssim {
                    print_score(&mut csv, r);
                }
                if let Some(r) = results.msssim {
                    print_score(&mut csv, r);
                }
                if let Some(r) = results.ssimulacra2 {
                    print_score(&mut csv, r);
                }
                csv.write_record(iter::empty::<&str>()).unwrap()
            }
        }
    }

    pub(crate) fn output_results(&self, results: MetricsResults) {
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
                println!("{}", serde_json::to_string_pretty(&results).unwrap());
            }
            Output::JsonLines => {
                println!(
                    "{}",
                    serde_json::to_string(&MetricsStats::from(results)).unwrap()
                );
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
