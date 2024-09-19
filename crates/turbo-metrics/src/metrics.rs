use cudarse_driver::CuStream;
use cudarse_npp::image::ist::{QualityIndex, SSIM};
use cudarse_npp::image::{Img, ImgView, C};
use cudarse_npp::sys::{NppiSize, Result};
use cudarse_npp::{get_stream_ctx, ScratchBuffer};

pub trait CudaMetric {
    fn name(&self) -> &'static str;

    /// Schedule the computation of the metric, should not block.
    fn schedule_compute(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<()>;

    /// Get the last computed score, can block.
    fn get_score(&mut self) -> f64;
}

/// Application of the following formula : https://docs.nvidia.com/cuda/npp/image_statistics_functions.html#group__image__quality__index_1image_quality_index
pub struct NppQualityIndex {
    scratch: ScratchBuffer,
    // FIXME it is unsound to hold the result here,
    //  bad things can happen if the struct is moved while npp has a reference to it.
    result: f32,
}
impl NppQualityIndex {
    pub fn new(size: NppiSize, stream: &CuStream) -> Result<Self> {
        Ok(Self {
            scratch: <ImgView<u8, C<3>> as QualityIndex<u8, C<3>>>::alloc_scratch(
                size,
                get_stream_ctx()?.with_stream(stream.inner() as _),
            )?,
            result: f32::NAN,
        })
    }
}
impl CudaMetric for NppQualityIndex {
    fn name(&self) -> &'static str {
        "QualityIndex (Cuda NPP)"
    }

    fn schedule_compute(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        reference
            .qualityindex_into(
                distorted,
                &mut self.scratch,
                &mut self.result,
                get_stream_ctx()?.with_stream(stream.inner() as _),
            )
            .unwrap();
        Ok(())
    }

    fn get_score(&mut self) -> f64 {
        self.result as f64
    }
}
pub struct NppPSNR;
pub struct NppSSIM {
    scratch: ScratchBuffer,
    result: f32,
}
impl NppSSIM {
    pub fn new(size: NppiSize, stream: &CuStream) -> Result<Self> {
        Ok(Self {
            scratch: <ImgView<u8, C<3>> as SSIM<u8, C<3>>>::alloc_scratch(
                size,
                get_stream_ctx()?.with_stream(stream.inner() as _),
            )?,
            result: f32::NAN,
        })
    }
}
impl CudaMetric for NppSSIM {
    fn name(&self) -> &'static str {
        "SSIM (Cuda NPP)"
    }

    fn schedule_compute(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        reference.ssim_into(
            distorted,
            &mut self.scratch,
            &mut self.result,
            get_stream_ctx()?.with_stream(stream.inner() as _),
        )
    }

    fn get_score(&mut self) -> f64 {
        self.result as f64
    }
}
pub struct NppMSSSIM;
pub struct Ssimulacra2 {
    inner: ssimulacra2_cuda::Ssimulacra2,
}
impl Ssimulacra2 {
    pub fn new(size: NppiSize, stream: &CuStream) -> Result<Self> {
        Ok(Self {
            inner: ssimulacra2_cuda::Ssimulacra2::new(size.width as u32, size.height as u32)?,
        })
    }
}
impl CudaMetric for Ssimulacra2 {
    fn name(&self) -> &'static str {
        "Ssimulacra2 (Cuda)"
    }

    fn schedule_compute(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
    ) -> Result<()> {
        self.inner.compute()
    }

    fn get_score(&mut self) -> f64 {
        self.inner.get_score()
    }
}

pub struct CudaMetrics {
    pub(crate) metrics: Vec<Box<dyn CudaMetric>>,
    streams: Vec<CuStream>,
}

impl CudaMetrics {
    pub fn new(metrics: Vec<Box<dyn CudaMetric>>) -> Self {
        Self {
            streams: (0..metrics.len())
                .map(|_| CuStream::new().unwrap())
                .collect(),
            metrics,
        }
    }

    /// Return the number of metrics to be computed
    pub fn len(&self) -> usize {
        self.len()
    }

    pub fn compute<'a>(
        &mut self,
        reference: &ImgView<u8, C<3>>,
        distorted: &ImgView<u8, C<3>>,
        stream: &CuStream,
        scores: impl IntoIterator<Item = &'a mut f64>,
    ) -> Result<()> {
        for (m, s) in self.metrics.iter_mut().zip(self.streams.iter()) {
            s.wait_for_stream(&stream).unwrap();
            m.schedule_compute(reference, distorted, s)?;
            stream.wait_for_stream(s).unwrap()
        }
        stream.sync().unwrap();
        for (m, score) in self.metrics.iter_mut().zip(scores) {
            *score = m.get_score();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics::{NppQualityIndex, NppSSIM};
    use crate::CudaMetrics;
    use cudarse_driver::CuStream;
    use cudarse_npp::get_stream_ctx;
    use cudarse_npp::image::idei::SetMany;
    use cudarse_npp::image::isu::Malloc;
    use cudarse_npp::image::Image;
    use cudarse_npp::sys::{NppiRect, NppiSize};

    #[test]
    fn cuda_metrics() {
        cudarse_driver::init_cuda_and_primary_ctx().unwrap();
        let main = CuStream::new().unwrap();
        cudarse_npp::set_stream(main.inner() as _).unwrap();
        let mut scores = vec![vec![0.0]; 2];
        let mut turbo = CudaMetrics::new(vec![
            Box::new(
                NppQualityIndex::new(
                    NppiSize {
                        width: 256,
                        height: 256,
                    },
                    &main,
                )
                .unwrap(),
            ),
            Box::new(
                NppSSIM::new(
                    NppiSize {
                        width: 256,
                        height: 256,
                    },
                    &main,
                )
                .unwrap(),
            ),
        ]);
        let mut ref_ = Image::malloc(256, 256).unwrap();
        ref_.view_mut(NppiRect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        })
        .set([255, 255, 255], get_stream_ctx().unwrap())
        .unwrap();
        let mut dis = Image::malloc(256, 256).unwrap();
        dis.view_mut(NppiRect {
            x: 0,
            y: 0,
            width: 15,
            height: 15,
        })
        .set([255, 255, 255], get_stream_ctx().unwrap())
        .unwrap();
        turbo
            .compute(
                &ref_.full_view(),
                &dis.full_view(),
                &main,
                scores.iter_mut().map(|v| &mut v[0]),
            )
            .unwrap();
        dbg!(scores);
    }
}
