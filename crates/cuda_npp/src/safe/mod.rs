use core::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ptr::null_mut;

use cuda_npp_sys::{
    cudaError, cudaFreeAsync, cudaMallocAsync, cudaMemcpy2DAsync, cudaMemcpyKind, NppStatus,
    NppiRect, NppiSize,
};

use crate::{Channels, Sample, __priv};

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
#[cfg(feature = "ist")]
pub mod ist;
#[cfg(feature = "isu")]
pub mod isu;

#[derive(Debug)]
pub enum E {
    NppError(NppStatus),
    CudaError(cudaError),
}

impl From<NppStatus> for E {
    fn from(value: NppStatus) -> Self {
        E::NppError(value)
    }
}

impl From<cudaError> for E {
    fn from(value: cudaError) -> Self {
        E::CudaError(value)
    }
}

pub type Result<T> = std::result::Result<T, E>;

/// Owned image
#[derive(Debug)]
pub struct Image<S: Sample, C: Channels> {
    width: u32,
    height: u32,
    /// in bytes
    pitch: i32,
    data: C::Storage<S>,
    marker: PhantomData<S>,
    marker_: PhantomData<C>,
}

impl<S: Sample, C: Channels> Image<S, C> {
    fn into_inner(self) -> C::Storage<S> {
        let inner = self.data.clone();
        mem::forget(self);
        return inner;
    }
}

// TODO impl view offset

/// [ImgView] is to [Image] what [str] is to [String]
#[derive(Debug, Clone)]
pub struct ImgView<'a, S: Sample, C: Channels> {
    parent: &'a Image<S, C>,
    width: u32,
    height: u32,
}

#[derive(Debug)]
pub struct ImgViewMut<'a, S: Sample, C: Channels> {
    parent: &'a mut Image<S, C>,
    width: u32,
    height: u32,
}

impl<S: Sample, C: Channels> __priv::Sealed for Image<S, C> {}

impl<'a, S: Sample, C: Channels> __priv::Sealed for ImgView<'a, S, C> {}

impl<'a, S: Sample, C: Channels> __priv::Sealed for ImgViewMut<'a, S, C> {}

// #[cfg(feature = "cudarc")]
// impl<S: Sample, C: Channel> cudarc::driver::DeviceSlice<S> for Image<S, C> {
//     fn len(&self) -> usize {
//         (self.line_step as u32 * self.height) as usize
//     }
// }
//
// #[cfg(feature = "cudarc")]
// impl<S: Sample, C: Channel> cudarc::driver::DeviceSlice<S> for ImgMut<S, C> {
//     fn len(&self) -> usize {
//         (self.line_step as u32 * self.height) as usize
//     }
// }
//
// #[cfg(feature = "cudarc")]
// impl<S: Sample, C: Channel> cudarc::driver::DevicePtr<S> for Img<S, C> {
//     fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
//         &self.parent.data
//     }
// }
//
// #[cfg(feature = "cudarc")]
// impl<S: Sample, C: Channel> cudarc::driver::DevicePtr<S> for ImgMut<S, C> {
//     fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
//         &self.parent.data
//     }
// }
//
// #[cfg(feature = "cudarc")]
// impl<S: Sample, C: Channel> cudarc::driver::DevicePtrMut<S> for ImgMut<S, C> {
//     fn device_ptr_mut(&mut self) -> &mut cudarc::driver::sys::CUdeviceptr {
//         &mut self.parent.data
//     }
// }

#[macro_export]
macro_rules! assert_same_size {
    ($img1:expr, $img2:expr) => {
        debug_assert_eq!(
            ($img1.width(), $img1.height()),
            ($img2.width(), $img2.height())
        )
    };
}

impl<'a, S: Sample, C: Channels> From<&'a Image<S, C>> for ImgView<'a, S, C> {
    fn from(value: &'a Image<S, C>) -> Self {
        value.full_view()
    }
}

impl<'a, S: Sample, C: Channels> From<&'a mut Image<S, C>> for ImgViewMut<'a, S, C> {
    fn from(value: &'a mut Image<S, C>) -> Self {
        value.full_view_mut()
    }
}

impl<S: Sample, C: Channels> Image<S, C> {
    pub fn full_view(&self) -> ImgView<S, C> {
        ImgView {
            parent: self,
            width: self.width,
            height: self.height,
        }
    }

    pub fn view(&self, rect: NppiRect) -> ImgView<S, C> {
        assert!(rect.width as u32 <= self.width() && rect.height as u32 <= self.height());
        ImgView {
            parent: self,
            width: rect.width as u32,
            height: rect.height as u32,
        }
    }

    pub fn full_view_mut(&mut self) -> ImgViewMut<S, C> {
        let (width, height) = (self.width, self.height);
        ImgViewMut {
            parent: self,
            width,
            height,
        }
    }

    pub fn view_mut(&mut self, rect: NppiRect) -> ImgViewMut<S, C> {
        assert!(rect.width as u32 <= self.width() && rect.height as u32 <= self.height());
        ImgViewMut {
            parent: self,
            width: rect.width as u32,
            height: rect.height as u32,
        }
    }
}

pub trait Img<S: Sample, C: Channels>: __priv::Sealed {
    const SAMPLE_SIZE: usize = mem::size_of::<S>();
    /// Pixel size in bytes
    const PIXEL_SIZE: usize = C::NUM_SAMPLES * Self::SAMPLE_SIZE;

    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// Pitch in bytes
    fn pitch(&self) -> i32;

    fn storage(&self) -> C::Storage<S>;

    /// The pointer to give when passing the image to NPP for reads
    fn device_ptr(&self) -> C::Ref<S>;

    /// Iterator over the pointers of the underlying allocations on device
    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S>;

    fn size(&self) -> NppiSize {
        NppiSize {
            width: self.width() as i32,
            height: self.height() as i32,
        }
    }

    fn rect(&self) -> NppiRect {
        NppiRect {
            x: 0,
            y: 0,
            width: self.width() as i32,
            height: self.height() as i32,
        }
    }

    fn has_padding(&self) -> bool {
        self.width() as usize * Self::PIXEL_SIZE != self.pitch() as usize
    }

    fn is_same_size<S2: Sample, C2: Channels>(&self, other: impl Img<S2, C2>) -> bool {
        self.width() == other.width() && self.height() == other.height()
    }

    #[cfg(feature = "isu")]
    fn malloc_same_size<S2: Sample, C2: Channels>(&self) -> Result<Image<S2, C2>>
    where
        Image<S2, C2>: isu::Malloc,
    {
        isu::Malloc::malloc(self.width(), self.height())
    }

    fn copy_to_cpu(&self) -> Result<Vec<S>> {
        let mut dst =
            vec![S::default(); self.width() as usize * self.height() as usize * C::NUM_SAMPLES];

        let pixel_size = if C::IS_PLANAR {
            Self::SAMPLE_SIZE
        } else {
            Self::PIXEL_SIZE
        };

        for (i, ptr) in self.alloc_ptrs().enumerate() {
            let res = unsafe {
                cudaMemcpy2DAsync(
                    dst[self.width() as usize * self.height() as usize * i..]
                        .as_mut_ptr()
                        .cast(),
                    self.width() as usize * pixel_size,
                    ptr.cast(),
                    self.pitch() as usize,
                    self.width() as usize * pixel_size,
                    self.height() as usize,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    null_mut(),
                )
            };
            match res {
                cudaError::cudaSuccess => {}
                e => {
                    return Err(e.into());
                }
            }
        }

        Ok(dst)
    }
}

pub trait ImgMut<S: Sample, C: Channels>: Img<S, C> {
    /// The pointer to give when passing the image to NPP for writes
    fn device_ptr_mut(&mut self) -> C::RefMut<S>;

    /// Iterator over the pointers of the underlying allocations on device
    fn alloc_ptrs_mut(&mut self) -> impl ExactSizeIterator<Item = *mut S>;

    fn copy_from_cpu(&mut self, data: &[S]) -> Result<()> {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let pitch = self.pitch() as usize;
        assert_eq!(data.len(), width * height * C::NUM_SAMPLES);

        let pixel_size = if C::IS_PLANAR {
            Self::SAMPLE_SIZE
        } else {
            Self::PIXEL_SIZE
        };

        for (i, ptr) in self.alloc_ptrs_mut().enumerate() {
            let res = unsafe {
                cudaMemcpy2DAsync(
                    ptr.cast(),
                    pitch,
                    data[width * height * i..].as_ptr().cast(),
                    width * pixel_size,
                    width * pixel_size,
                    height,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    null_mut(),
                )
            };
            match res {
                cudaError::cudaSuccess => {}
                e => {
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }
}

impl<S: Sample, C: Channels, T: Img<S, C>> Img<S, C> for &T {
    fn width(&self) -> u32 {
        (*self).width()
    }

    fn height(&self) -> u32 {
        (*self).height()
    }

    fn pitch(&self) -> i32 {
        (*self).pitch()
    }

    fn storage(&self) -> C::Storage<S> {
        (*self).storage()
    }

    fn device_ptr(&self) -> <C>::Ref<S> {
        (*self).device_ptr()
    }

    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S> {
        (*self).alloc_ptrs()
    }
}

impl<S: Sample, C: Channels, T: Img<S, C>> Img<S, C> for &mut T {
    fn width(&self) -> u32 {
        Img::width(*self)
    }

    fn height(&self) -> u32 {
        Img::height(*self)
    }

    fn pitch(&self) -> i32 {
        Img::pitch(*self)
    }

    fn storage(&self) -> C::Storage<S> {
        Img::storage(*self)
    }

    fn device_ptr(&self) -> C::Ref<S> {
        Img::device_ptr(*self)
    }

    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S> {
        Img::alloc_ptrs(*self)
    }
}

impl<S: Sample, C: Channels> Img<S, C> for Image<S, C> {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn pitch(&self) -> i32 {
        self.pitch
    }

    fn storage(&self) -> C::Storage<S> {
        self.data.clone()
    }

    fn device_ptr(&self) -> C::Ref<S> {
        C::make_ref(&self.data)
    }

    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S> {
        C::iter_ptrs(&self.data)
    }
}

impl<'a, S: Sample, C: Channels> Img<S, C> for ImgView<'a, S, C> {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn pitch(&self) -> i32 {
        self.parent.pitch()
    }

    fn storage(&self) -> C::Storage<S> {
        self.parent.storage()
    }

    fn device_ptr(&self) -> C::Ref<S> {
        self.parent.device_ptr()
    }

    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S> {
        self.parent.alloc_ptrs()
    }
}

impl<'a, S: Sample, C: Channels> Img<S, C> for ImgViewMut<'a, S, C> {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn pitch(&self) -> i32 {
        self.parent.pitch()
    }

    fn storage(&self) -> C::Storage<S> {
        self.parent.storage()
    }

    fn device_ptr(&self) -> C::Ref<S> {
        self.parent.device_ptr()
    }

    fn alloc_ptrs(&self) -> impl ExactSizeIterator<Item = *const S> {
        self.parent.alloc_ptrs()
    }
}

impl<S: Sample, C: Channels> ImgMut<S, C> for Image<S, C> {
    fn device_ptr_mut(&mut self) -> C::RefMut<S> {
        C::make_ref_mut(&mut self.data)
    }

    fn alloc_ptrs_mut(&mut self) -> impl ExactSizeIterator<Item = *mut S> {
        C::iter_ptrs_mut(&mut self.data)
    }
}

impl<S: Sample, C: Channels> ImgMut<S, C> for &mut Image<S, C> {
    fn device_ptr_mut(&mut self) -> C::RefMut<S> {
        C::make_ref_mut(&mut self.data)
    }

    fn alloc_ptrs_mut(&mut self) -> impl ExactSizeIterator<Item = *mut S> {
        C::iter_ptrs_mut(&mut self.data)
    }
}

impl<'a, S: Sample, C: Channels> ImgMut<S, C> for ImgViewMut<'a, S, C> {
    fn device_ptr_mut(&mut self) -> C::RefMut<S> {
        self.parent.device_ptr_mut()
    }

    fn alloc_ptrs_mut(&mut self) -> impl ExactSizeIterator<Item = *mut S> {
        self.parent.alloc_ptrs_mut()
    }
}

impl<'a, S: Sample, C: Channels> ImgMut<S, C> for &mut ImgViewMut<'a, S, C> {
    fn device_ptr_mut(&mut self) -> C::RefMut<S> {
        self.parent.device_ptr_mut()
    }

    fn alloc_ptrs_mut(&mut self) -> impl ExactSizeIterator<Item = *mut S> {
        self.parent.alloc_ptrs_mut()
    }
}

/// An opaque scratch buffer needed by some npp routines
#[derive(Debug, Clone)]
pub struct ScratchBuffer {
    pub ptr: *mut c_void,
    pub size: usize,
}

impl ScratchBuffer {
    /// Uses stream ordered cuda malloc and free
    pub fn alloc(size: usize) -> Self {
        let mut ptr = null_mut();
        unsafe {
            cudaMallocAsync(&mut ptr, size, null_mut());
        }
        Self { ptr, size }
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFreeAsync(self.ptr, null_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::safe::isu::Malloc;
    use crate::safe::Image;
    use crate::C;

    #[test]
    fn into_inner_does_not_drop() {
        let inner = Image::<f32, C<1>>::malloc(16, 16).unwrap().into_inner();
        dbg!(inner);
    }
}
