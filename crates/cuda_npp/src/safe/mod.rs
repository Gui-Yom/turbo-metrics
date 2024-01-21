use std::marker::PhantomData;

use cuda_npp_sys::{NppStatus, NppiSize};

use crate::{Channel, ChannelPacked, ChannelResize, ChannelSet, Sample, SampleResize};

#[cfg(feature = "icc")]
pub mod icc;
#[cfg(feature = "idei")]
pub mod idei;
#[cfg(feature = "ig")]
pub mod ig;
#[cfg(feature = "isu")]
pub mod isu;

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

pub struct Image<S: Sample, C: Channel> {
    width: u32,
    height: u32,
    /// Line step in bytes
    line_step: i32,
    data: *const S,
    marker_: PhantomData<S>,
    marker__: PhantomData<C>,
}

impl<S: Sample, C: Channel> Image<S, C> {
    pub fn size(&self) -> NppiSize {
        NppiSize {
            width: self.width as i32,
            height: self.height as i32,
        }
    }
}
