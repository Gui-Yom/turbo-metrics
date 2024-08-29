use std::mem;

use crate::sys::*;

use crate::{Result, ScratchBuffer};

use super::{Channels, Img, Sample, C};

pub trait Sum<S: Sample, C: Channels> {
    type SumResult;

    /// You should not use the result until synchronized with the CUDA stream.
    fn sum(
        &self,
        scratch: &mut ScratchBuffer,
        ctx: NppStreamContext,
    ) -> Result<Box<Self::SumResult>>;
    fn sum_into(
        &self,
        scratch: &mut ScratchBuffer,
        out: &mut Self::SumResult,
        ctx: NppStreamContext,
    ) -> Result<()>;
    fn sum_scratch_size(&self) -> Result<usize>;
    fn sum_alloc_scratch(&self, ctx: NppStreamContext) -> Result<ScratchBuffer>;
}

impl<T: Img<f32, C<3>>> Sum<f32, C<3>> for T {
    type SumResult = [f64; 3];

    fn sum(
        &self,
        scratch: &mut ScratchBuffer,
        ctx: NppStreamContext,
    ) -> Result<Box<Self::SumResult>> {
        let mut out = Box::new([0.0; 3]);
        self.sum_into(scratch, &mut out, ctx)?;
        Ok(out)
    }

    fn sum_into(
        &self,
        scratch: &mut ScratchBuffer,
        out: &mut Self::SumResult,
        ctx: NppStreamContext,
    ) -> Result<()> {
        let mut out_dev = ScratchBuffer::alloc(3 * mem::size_of::<f64>(), ctx.hStream)?;
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

        unsafe {
            cudaMemcpyAsync(
                out.as_mut_ptr().cast(),
                out_dev.ptr.cast_const(),
                out_dev.len,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ctx.hStream,
            )
                .result()?;
        }
        // This alloc will be freed asynchronously
        out_dev.manual_drop(ctx.hStream)?;

        Ok(())
    }

    fn sum_scratch_size(&self) -> Result<usize> {
        let mut size = 0;
        unsafe { nppiSumGetBufferHostSize_32f_C3R(self.size(), &mut size).result_with(size) }
    }

    fn sum_alloc_scratch(&self, ctx: NppStreamContext) -> Result<ScratchBuffer> {
        ScratchBuffer::alloc(self.sum_scratch_size()?, ctx.hStream)
    }
}

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
