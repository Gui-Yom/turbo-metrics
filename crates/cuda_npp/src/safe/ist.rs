use std::mem;
use std::ptr::null_mut;

use cuda_npp_sys::*;

use crate::{C, Channels, Sample};

use super::{Img, Result, ScratchBuffer};

pub trait Sum<S: Sample, C: Channels> {
    type SumResult;

    fn sum(&self, scratch: &mut ScratchBuffer, ctx: NppStreamContext) -> Result<Self::SumResult>;
    fn scratch_size(&self) -> usize;
    fn alloc_scratch(&self) -> ScratchBuffer;
}

impl<T: Img<f32, C<3>>> Sum<f32, C<3>> for T {
    type SumResult = [f64; 3];

    fn sum(&self, scratch: &mut ScratchBuffer, ctx: NppStreamContext) -> Result<Self::SumResult> {
        let mut out_dev = ScratchBuffer::alloc(3 * mem::size_of::<f64>());
        unsafe {
            nppiSum_32f_C3R_Ctx(
                self.device_ptr(),
                self.pitch(),
                self.size(),
                scratch.ptr as _,
                out_dev.ptr as _,
                ctx,
            )
        }
            .result()?;
        let mut out = [0.0; 3];
        unsafe {
            assert_eq!(
                cudaMemcpyAsync(
                    out.as_mut_ptr() as _,
                    out_dev.ptr,
                    out_dev.size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    null_mut(),
                ),
                cudaError_t::cudaSuccess
            );
        }

        Ok(out)
    }

    fn scratch_size(&self) -> usize {
        let mut size = 0;
        unsafe {
            nppiSumGetBufferHostSize_32f_C3R(self.size(), &mut size);
        }
        size as usize
    }

    fn alloc_scratch(&self) -> ScratchBuffer {
        ScratchBuffer::alloc(self.scratch_size())
    }
}

#[cfg(test)]
mod tests {
    use crate::{C, get_stream_ctx};
    use crate::safe::idei::SetMany;
    use crate::safe::Image;
    use crate::safe::ist::Sum;
    use crate::safe::isu::Malloc;

    #[test]
    fn sum() -> crate::safe::Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        let ctx = get_stream_ctx()?;
        let mut scratch = img.alloc_scratch();
        let sum = img.sum(&mut scratch, ctx)?;
        drop(scratch);
        dev.synchronize().unwrap();
        dbg!(sum);

        Ok(())
    }

    #[test]
    fn sum_value() -> crate::safe::Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let mut img = Image::<f32, C<3>>::malloc(128, 128)?;
        let ctx = get_stream_ctx()?;
        img.set([1.0, 0.0, 0.0], ctx)?;
        let mut scratch = img.alloc_scratch();
        let sum = img.sum(&mut scratch, ctx)?;
        drop(scratch);
        dev.synchronize().unwrap();
        dbg!(sum);
        assert_eq!(sum, [128.0 * 128.0, 0.0, 0.0]);

        Ok(())
    }
}
