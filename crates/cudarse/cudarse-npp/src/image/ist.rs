use crate::sys::*;

use crate::{debug_assert_same_size, Result, ScratchBuffer};

use super::{Channels, Img, Sample, C};
use paste::paste;

pub trait Sum<S: Sample, C: Channels> {
    type SumResult: Default;

    /// You should not use the result until synchronized with the CUDA stream.
    fn sum(
        &self,
        scratch: &mut ScratchBuffer,
        ctx: NppStreamContext,
    ) -> Result<Box<Self::SumResult>> {
        let mut out = Box::<Self::SumResult>::default();
        self.sum_into(scratch, out.as_mut(), ctx)?;
        Ok(out)
    }
    fn sum_into(
        &self,
        scratch: &mut ScratchBuffer,
        out: &mut Self::SumResult,
        ctx: NppStreamContext,
    ) -> Result<()>;
    fn sum_scratch_size(&self) -> Result<usize>;
    fn sum_alloc_scratch(&self, ctx: NppStreamContext) -> Result<ScratchBuffer> {
        ScratchBuffer::alloc_len(self.sum_scratch_size()?, ctx.hStream)
    }
}

impl<T: Img<f32, C<3>>> Sum<f32, C<3>> for T {
    type SumResult = [f64; 3];

    fn sum_into(
        &self,
        scratch: &mut ScratchBuffer,
        out: &mut Self::SumResult,
        ctx: NppStreamContext,
    ) -> Result<()> {
        let out_dev = ScratchBuffer::alloc::<[f64; 3]>(ctx.hStream)?;
        unsafe {
            nppiSum_32f_C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                self.size(),
                scratch.ptr.cast(),
                out_dev.ptr.cast(),
                ctx,
            )
            .result()?;
        }

        out_dev.copy_to_cpu(out, ctx.hStream)?;
        // This alloc will be freed asynchronously
        out_dev.manual_drop(ctx.hStream)?;

        Ok(())
    }

    fn sum_scratch_size(&self) -> Result<usize> {
        let mut size = 0;
        unsafe { nppiSumGetBufferHostSize_32f_C3R(self.size(), &mut size).result_with(size) }
    }
}

macro_rules! declare_quality_trait {
    ($quality:ident) => {
        paste! {
            pub trait $quality<S: Sample, C: Channels>: Img<S, C> {
                fn [<$quality:lower>](
                    &self,
                    other: impl Img<S, C>,
                    scratch: &mut ScratchBuffer,
                    ctx: NppStreamContext,
                ) -> Result<Box<f32>> {
                    let mut out = Box::new(f32::NAN);
                    self.[<$quality:lower _into>](other, scratch, &mut out, ctx)?;
                    Ok(out)
                }

                fn [<$quality:lower _into>](
                    &self,
                    other: impl Img<S, C>,
                    scratch: &mut ScratchBuffer,
                    out: &mut f32,
                    ctx: NppStreamContext,
                ) -> Result<()>;

                fn [<$quality:lower _scratch_size>](&self, ctx: NppStreamContext) -> Result<usize> {
                    Self::scratch_size(self.size(), ctx)
                }
                fn scratch_size(size: NppiSize, ctx: NppStreamContext) -> Result<usize>;
                fn [<$quality:lower _alloc_scratch>](&self, ctx: NppStreamContext) -> Result<ScratchBuffer> {
                    Self::alloc_scratch(self.size(), ctx)
                }
                fn alloc_scratch(size: NppiSize, ctx: NppStreamContext) -> Result<ScratchBuffer> {
                    ScratchBuffer::alloc_len(Self::scratch_size(size, ctx)?, ctx.hStream)
                }
            }
        }
    }
}

macro_rules! impl_quality_metric {
    ($quality:ident, $sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        paste! {
            impl<T: Img<$sample_ty, $channel_ty>> $quality<$sample_ty, $channel_ty> for T {
                fn [<$quality:lower _into>](
                    &self,
                    other: impl Img<$sample_ty, $channel_ty>,
                    scratch: &mut ScratchBuffer,
                    out: &mut f32,
                    ctx: NppStreamContext,
                ) -> Result<()> {
                    debug_assert_same_size!(self, other);
                    let out_dev = ScratchBuffer::alloc::<f32>(ctx.hStream)?;
                    unsafe {
                        [<nppi $quality $sample_id _ $channel_id R_Ctx>](
                            self.device_ptr(),
                            self.pitch(),
                            other.device_ptr(),
                            other.pitch(),
                            self.size(),
                            out_dev.ptr.cast(),
                            scratch.ptr.cast(),
                            ctx,
                        )
                        .result()?;
                    }

                    out_dev.copy_to_cpu(out, ctx.hStream)?;
                    // This alloc will be freed asynchronously
                    out_dev.manual_drop(ctx.hStream)?;

                    Ok(())
                }


                fn scratch_size(size: NppiSize, ctx: NppStreamContext) -> Result<usize> {
                    let mut len = 0;
                    unsafe {
                        [<nppi $quality GetBufferHostSize $sample_id _ $channel_id R_Ctx>](size, &mut len, ctx).result_with(len)
                    }
                }
            }
        }
    };
}

// I love macros

declare_quality_trait!(QualityIndex);
impl_quality_metric!(QualityIndex, u8, C<1>, _8u32f, C1);
impl_quality_metric!(QualityIndex, u8, C<3>, _8u32f, C3);
impl_quality_metric!(QualityIndex, u8, C<4>, _8u32f, AC4);
impl_quality_metric!(QualityIndex, u16, C<1>, _16u32f, C1);
impl_quality_metric!(QualityIndex, u16, C<3>, _16u32f, C3);
impl_quality_metric!(QualityIndex, u16, C<4>, _16u32f, AC4);
impl_quality_metric!(QualityIndex, f32, C<1>, _32f, C1);
impl_quality_metric!(QualityIndex, f32, C<3>, _32f, C3);
impl_quality_metric!(QualityIndex, f32, C<4>, _32f, AC4);

declare_quality_trait!(MSE);
impl_quality_metric!(MSE, u8, C<1>, _8u, C1);
impl_quality_metric!(MSE, u8, C<3>, _8u, C3);

declare_quality_trait!(PSNR);
impl_quality_metric!(PSNR, u8, C<1>, _8u, C1);
impl_quality_metric!(PSNR, u8, C<3>, _8u, C3);

declare_quality_trait!(SSIM);
impl_quality_metric!(SSIM, u8, C<1>, _8u, C1);
impl_quality_metric!(SSIM, u8, C<3>, _8u, C3);

declare_quality_trait!(WMSSSIM);
impl_quality_metric!(WMSSSIM, u8, C<1>, _8u, C1);
impl_quality_metric!(WMSSSIM, u8, C<3>, _8u, C3);

#[cfg(test)]
mod tests {
    use cudarse_driver::{init_cuda_and_primary_ctx, sync_ctx};

    use crate::get_stream_ctx;
    use crate::image::idei::SetMany;
    use crate::image::ist::Sum;
    use crate::image::isu::Malloc;
    use crate::image::{Image, C};

    #[test]
    fn sum() -> crate::Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let mut scratch = img.sum_alloc_scratch(ctx)?;
        let sum = img.sum(&mut scratch, ctx)?;
        sync_ctx().unwrap();
        dbg!(sum);

        Ok(())
    }

    #[test]
    fn sum_value() -> crate::Result<()> {
        init_cuda_and_primary_ctx().unwrap();
        let mut img = Image::<f32, C<3>>::malloc(128, 128)?;
        let ctx = get_stream_ctx()?;
        img.set([1.0, 0.0, 0.0], ctx)?;
        let mut scratch = img.sum_alloc_scratch(ctx)?;
        let sum = img.sum(&mut scratch, ctx)?;
        sync_ctx().unwrap();
        dbg!(&sum);
        assert_eq!(sum.as_ref(), &[128.0 * 128.0, 0.0, 0.0]);

        Ok(())
    }
}
