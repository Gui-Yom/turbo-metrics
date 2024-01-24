use std::marker::PhantomData;
use std::mem;

use cuda_npp_sys::{NppStatus, NppiSize};

use crate::{Channel, Sample};

#[cfg(feature = "ial")]
pub mod ial;
#[cfg(feature = "icc")]
pub mod icc;
#[cfg(feature = "idei")]
pub mod idei;
#[cfg(feature = "if")]
pub mod if_;
#[cfg(feature = "ig")]
pub mod ig;
#[cfg(feature = "isu")]
pub mod isu;

#[derive(Debug)]
pub enum E {
    NppError(NppStatus),
}

impl From<NppStatus> for E {
    fn from(value: NppStatus) -> Self {
        E::NppError(value)
    }
}

pub type Result<T> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Image<S: Sample, C: Channel> {
    pub width: u32,
    pub height: u32,
    /// Line step in bytes
    pub line_step: i32,
    pub data: *const S,
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

    pub fn has_padding(&self) -> bool {
        dbg!(self.width as usize * C::NUM_SAMPLES * mem::size_of::<S>())
            != dbg!(self.line_step as usize)
    }

    #[cfg(feature = "cudarc")]
    pub fn copy_from_cpu(&mut self, data: &[S]) -> Result<()> {
        if self.has_padding() {
            for c in 0..self.height as usize {
                unsafe {
                    cudarc::driver::result::memcpy_htod_sync(
                        self.data.byte_add(c * self.line_step as usize) as _,
                        &data[c * self.width as usize * C::NUM_SAMPLES
                            ..(c + 1) * self.width as usize * C::NUM_SAMPLES],
                    )
                }
                .unwrap();
            }
            Ok(())
        } else {
            let res = unsafe { cudarc::driver::result::memcpy_htod_sync(self.data as _, data) };
            res.unwrap();
            Ok(())
        }
    }
}

#[cfg(feature = "cudarc")]
impl<S: Sample + Default + Copy, C: Channel> Image<S, C> {
    pub fn copy_to_cpu(&self) -> Result<Vec<S>> {
        let mut dst =
            vec![S::default(); self.width as usize * self.height as usize * C::NUM_SAMPLES];
        if self.has_padding() {
            for c in 0..self.height as usize {
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_sync(
                        &mut dst[c * self.width as usize * C::NUM_SAMPLES
                            ..(c + 1) * self.width as usize * C::NUM_SAMPLES],
                        self.data.byte_add(c * self.line_step as usize) as _,
                    )
                }
                .unwrap();
            }
            Ok(dst)
        } else {
            let res = unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut dst, self.data as _) };
            res.unwrap();
            Ok(dst)
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::C;
//     use crate::safe::Image;
//     use crate::safe::isu::Malloc;
//
//     #[test]
//     fn copy_and_back() {
//         let source_bytes = &source.flatten_to_u8()[0];
//         let mut source_img =
//             Image::<u8, C<3>>::malloc(source.dimensions().0 as u32, source.dimensions().1 as u32)
//                 .unwrap();
//         source_img.copy_from_cpu(&source_bytes).unwrap();
//
//         let source_back = source_img.copy_to_cpu().unwrap();
//         assert_eq!(source_bytes, &source_back);
//     }
// }
