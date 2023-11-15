use std::marker::PhantomData;

use crate::{safeish, ChannelLayout, ChannelLayoutPacked, Sample};
use cuda_npp_sys::NppStatus;

enum E {
    NppStatus(NppStatus),
}

impl From<NppStatus> for E {
    fn from(value: NppStatus) -> Self {
        E::NppStatus(value)
    }
}

pub type Result<T> = std::result::Result<T, E>;

struct Image<S: Sample, C: ChannelLayout, Stor> {
    width: u32,
    height: u32,
    /// Line step in bytes
    line_step: i32,
    buffer: Stor,
    marker_: PhantomData<S>,
    marker__: PhantomData<C>,
}

impl<S: Sample, C: ChannelLayoutPacked> Image<S, C, *const S> {
    pub fn new_device(width: u32, height: u32) -> Result<Self> {
        let (ptr, line_step) = safeish::malloc::<S, C>(width as i32, height as i32)?;
        Ok(Self {
            width,
            height,
            line_step,
            buffer: ptr,
            marker_: PhantomData,
            marker__: PhantomData,
        })
    }

    fn resize() {}
}
