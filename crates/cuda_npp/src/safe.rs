use std::marker::PhantomData;

use cuda_npp_sys::{
    nppiFree, NppStatus, NppStreamContext, NppiInterpolationMode, NppiRect, NppiSize,
};

use crate::{
    safeish, ChannelLayout, ChannelLayoutPacked, ChannelLayoutResizePacked, Sample, SampleResize,
};

#[derive(Debug)]
pub enum E {
    NppStatus(NppStatus),
}

impl From<NppStatus> for E {
    fn from(value: NppStatus) -> Self {
        E::NppStatus(value)
    }
}

pub type Result<T> = std::result::Result<T, E>;

struct Image<S: Sample, C: ChannelLayout> {
    width: u32,
    height: u32,
    /// Line step in bytes
    line_step: i32,
    data: *const S,
    marker_: PhantomData<S>,
    marker__: PhantomData<C>,
}

impl<S: Sample, C: ChannelLayoutPacked> Image<S, C> {
    /// `malloc` a new image on device
    pub fn new_device(width: u32, height: u32) -> Result<Self> {
        let (ptr, line_step) = safeish::malloc::<S, C>(width as i32, height as i32)?;
        Ok(Self {
            width,
            height,
            line_step,
            data: ptr,
            marker_: PhantomData,
            marker__: PhantomData,
        })
    }
}

impl<S: SampleResize, C: ChannelLayoutResizePacked> Image<S, C> {
    fn resize(&self, new_width: u32, new_height: u32, ctx: NppStreamContext) -> Result<Self> {
        let mut dst = Image::<S, C>::new_device(new_width, new_height)?;
        safeish::resize::<S, C>(
            self.data,
            self.line_step,
            NppiSize {
                width: self.width as i32,
                height: self.height as i32,
            },
            NppiRect {
                x: 0,
                y: 0,
                width: self.width as i32,
                height: self.height as i32,
            },
            dst.data as _,
            dst.line_step,
            NppiSize {
                width: dst.width as i32,
                height: dst.height as i32,
            },
            NppiRect {
                x: 0,
                y: 0,
                width: dst.width as i32,
                height: dst.height as i32,
            },
            NppiInterpolationMode::NPPI_INTER_LANCZOS,
            ctx,
        )?;
        Ok(dst)
    }
}

impl<S: Sample, C: ChannelLayout> Drop for Image<S, C> {
    fn drop(&mut self) {
        println!("Dropping image");
        unsafe {
            nppiFree(self.data as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use cuda_npp_sys::nppGetStreamContext;

    use crate::safe::Image;
    use crate::safe::Result;
    use crate::C;

    #[test]
    fn new_image() -> Result<()> {
        let img = Image::<f32, C<3>>::new_device(1024, 1024)?;
        Ok(())
    }

    #[test]
    fn resize() -> Result<()> {
        let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
        let img = Image::<f32, C<3>>::new_device(1024, 1024)?;
        let mut ctx = Default::default();
        unsafe {
            nppGetStreamContext(&mut ctx);
        }
        img.resize(2048, 2048, ctx)?;
        Ok(())
    }
}
