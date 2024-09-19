use crate::metrics::{CudaMetric, CudaMetrics};
use std::collections::HashMap;

pub mod metrics;
pub use cudarse_driver;
use cudarse_driver::CuStream;
pub use cudarse_npp;
use cudarse_npp::image::{ImgView, C};
use cudarse_npp::sys::Result;
pub use cudarse_video;
pub use stats;
use stats::full::Stats;

pub struct TurboMetrics {
    metrics: CudaMetrics,
    scores: Vec<Vec<f64>>,
}

impl TurboMetrics {
    pub fn new(metrics: Vec<Box<dyn CudaMetric>>) -> Self {
        Self {
            scores: (0..metrics.len()).map(|_| Vec::new()).collect(),
            metrics: CudaMetrics::new(metrics),
        }
    }

    pub fn compute(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        self.metrics.compute(
            reference,
            distorted,
            stream,
            self.scores.iter_mut().map(|v| {
                v.push(0.0);
                let len = v.len();
                &mut v[len - 1]
            }),
        )
    }

    pub fn stats(&self) -> HashMap<&'static str, Stats> {
        self.scores
            .iter()
            .zip(self.metrics.metrics.iter())
            .map(|(v, m)| (m.name(), Stats::compute(v)))
            .collect()
    }
}
