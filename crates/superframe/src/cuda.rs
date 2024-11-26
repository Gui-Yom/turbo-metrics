use crate::{
    assert_same_size, row_align, AsPlane, AsPlaneMut, CastStorage, HostAccessible,
    HostAccessibleMut, OwnedSampleStorage, Sample, SampleStorage, SetPlane, StaticSample,
    TransferPlane,
};
use cudarse_driver::{CuBox, CuPin, CuResult, CuStream};
use std::borrow::{Borrow, BorrowMut};
use std::error::Error;
use std::ptr::{null, null_mut};

pub use cudarse_driver;

pub struct Cuda<S: Sample>(CuBox<[S]>);
impl<S: Sample> Cuda<S> {
    pub fn from_ptr_len(ptr: cudarse_driver::sys::CUdeviceptr, len: usize) -> Self {
        Self(CuBox::from_ptr_len(ptr, len))
    }

    /// Base pointer to this storage in Cuda memory
    pub fn device_ptr(&self) -> cudarse_driver::sys::CUdeviceptr {
        self.0.ptr()
    }

    /// Pointer computed from 2D coordinates.
    pub fn compute_ptr(
        &self,
        x: usize,
        y: usize,
        pitch_bytes: usize,
    ) -> cudarse_driver::sys::CUdeviceptr {
        (self.device_ptr() as usize + y * pitch_bytes + x * S::SIZE)
            as cudarse_driver::sys::CUdeviceptr
    }
}

impl<S: Sample> SampleStorage for Cuda<S> {
    type SampleType = S;
}
impl<S: Sample, Target: Sample> CastStorage<Target> for Cuda<S> {
    type Out = Cuda<Target>;

    fn cast(self) -> Self::Out {
        Cuda(self.0.cast())
    }
}
impl<S: Sample> OwnedSampleStorage for Cuda<S> {
    type Ext<'a> = &'a CuStream;

    fn alloc_ext(
        width: usize,
        height: usize,
        ext: Self::Ext<'_>,
    ) -> Result<(usize, Self), Box<dyn Error>> {
        // This should be a good enough alignment for most cases with CUDA (32 warps accessing a float each)
        const ROW_ALIGN: usize = 128;
        let pitch = row_align(width * S::SIZE, ROW_ALIGN);
        Ok((pitch, Self(CuBox::<[S]>::new_uninit(pitch * height, ext)?)))
    }

    fn drop_ext(&mut self, ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        unsafe { self.0.drop_inner(ext)? };
        Ok(())
    }
}

// Cuda to Host
impl<'b, S, Dst, DstPlane> TransferPlane<S, Cuda<S>, Dst> for DstPlane
where
    S: StaticSample,
    Dst: HostAccessibleMut<SampleType = S>,
    DstPlane: AsPlaneMut<Storage = Dst>,
{
    type Ext<'a> = &'a CuStream;

    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Cuda<S>>,
        ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>> {
        assert_same_size!(self, src);
        let src_rect = src.absolute_rect();
        let dst_rect = self.absolute_rect();
        unsafe {
            cudarse_driver::sys::cuMemcpy2DAsync_v2(
                &cudarse_driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: src_rect.x * S::SIZE,
                    srcY: src_rect.y,
                    srcMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: null(),
                    srcDevice: src.storage().device_ptr(),
                    srcArray: null_mut(),
                    srcPitch: src.pitch_bytes(),
                    dstXInBytes: dst_rect.x * S::SIZE,
                    dstY: dst_rect.y,
                    dstMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                    dstHost: self.storage_mut().mut_ptr().cast(),
                    dstDevice: 0,
                    dstArray: null_mut(),
                    dstPitch: self.pitch_bytes(),
                    WidthInBytes: self.width() * S::SIZE,
                    Height: self.height(),
                },
                ext.raw(),
            )
            .result()?;
        }
        Ok(())
    }

    fn copy_from(&mut self, src: impl AsPlane<Storage = Cuda<S>>) -> Result<(), Box<dyn Error>> {
        self.copy_from_ext(src, CuStream::DEFAULT_)
    }
}

/// Host to Cuda
impl<'b, S, Src, Dst> TransferPlane<S, Src, Cuda<S>> for Dst
where
    S: StaticSample,
    Src: HostAccessible<SampleType = S>,
    Dst: AsPlaneMut<Storage = Cuda<S>>,
{
    type Ext<'a> = &'a CuStream;

    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Src>,
        ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>> {
        assert_same_size!(self, src);
        let src_rect = src.absolute_rect();
        let dst_rect = self.absolute_rect();
        unsafe {
            cudarse_driver::sys::cuMemcpy2DAsync_v2(
                &cudarse_driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: src_rect.x * S::SIZE,
                    srcY: src_rect.y,
                    srcMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: src.storage().ptr().cast(),
                    srcDevice: 0,
                    srcArray: null_mut(),
                    srcPitch: src.pitch_bytes(),
                    dstXInBytes: dst_rect.x * S::SIZE,
                    dstY: dst_rect.y,
                    dstMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: null_mut(),
                    dstDevice: self.storage_mut().device_ptr(),
                    dstArray: null_mut(),
                    dstPitch: self.pitch_bytes(),
                    WidthInBytes: self.width() * S::SIZE,
                    Height: self.height(),
                },
                ext.raw(),
            )
            .result()
            .unwrap();
        }
        Ok(())
    }

    fn copy_from(&mut self, src: impl AsPlane<Storage = Src>) -> Result<(), Box<dyn Error>> {
        self.copy_from_ext(src, CuStream::DEFAULT_)
    }
}

// Cuda to Cuda
impl<'b, 'c, S, Dst> TransferPlane<S, Cuda<S>, Cuda<S>> for Dst
where
    S: StaticSample,
    Dst: AsPlaneMut<Storage = Cuda<S>>,
{
    type Ext<'a> = &'a CuStream;

    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Cuda<S>>,
        ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>> {
        assert_same_size!(self, src);
        let src_rect = src.absolute_rect();
        let dst_rect = self.absolute_rect();
        unsafe {
            cudarse_driver::sys::cuMemcpy2DAsync_v2(
                &cudarse_driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: src_rect.x * S::SIZE,
                    srcY: src_rect.y,
                    srcMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: null(),
                    srcDevice: src.storage().device_ptr(),
                    srcArray: null_mut(),
                    srcPitch: src.pitch_bytes(),
                    dstXInBytes: dst_rect.x * S::SIZE,
                    dstY: dst_rect.y,
                    dstMemoryType: cudarse_driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: null_mut(),
                    dstDevice: self.storage_mut().device_ptr(),
                    dstArray: null_mut(),
                    dstPitch: self.pitch_bytes(),
                    WidthInBytes: self.width() * S::SIZE,
                    Height: self.height(),
                },
                ext.raw(),
            )
            .result()?;
        }
        Ok(())
    }

    fn copy_from(&mut self, src: impl AsPlane<Storage = Cuda<S>>) -> Result<(), Box<dyn Error>> {
        self.copy_from_ext(src, CuStream::DEFAULT_)
    }
}

impl<'a, P: AsPlaneMut<Storage = Cuda<u8>>> SetPlane<Cuda<u8>> for P {
    type Ext<'aux> = &'aux CuStream;

    fn set_ext(&mut self, value: u8, ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        let rect = self.absolute_rect();
        let pitch = self.pitch_bytes();
        unsafe {
            cudarse_driver::sys::cuMemsetD2D8Async(
                self.storage_mut().compute_ptr(rect.x, rect.y, pitch),
                pitch,
                value,
                self.width(),
                self.height(),
                ext.raw(),
            )
            .result()?;
            Ok(())
        }
    }

    fn set(&mut self, value: u8) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, CuStream::DEFAULT_)
    }
}

impl<'a, P: AsPlaneMut<Storage = Cuda<u16>>> SetPlane<Cuda<u16>> for P {
    type Ext<'aux> = &'aux CuStream;

    fn set_ext(&mut self, value: u16, ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        let rect = self.absolute_rect();
        let pitch = self.pitch_bytes();
        unsafe {
            cudarse_driver::sys::cuMemsetD2D16Async(
                self.storage_mut().compute_ptr(rect.x, rect.y, pitch),
                pitch,
                value,
                self.width(),
                self.height(),
                ext.raw(),
            )
            .result()?;
            Ok(())
        }
    }

    fn set(&mut self, value: u16) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, CuStream::DEFAULT_)
    }
}

impl<'a, P: AsPlaneMut<Storage = Cuda<f32>>> SetPlane<Cuda<f32>> for P {
    type Ext<'aux> = &'aux CuStream;

    fn set_ext(&mut self, value: f32, ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        let rect = self.absolute_rect();
        let pitch = self.pitch_bytes();
        unsafe {
            cudarse_driver::sys::cuMemsetD2D32Async(
                self.storage_mut().compute_ptr(rect.x, rect.y, pitch),
                pitch,
                value.to_bits(),
                self.width(),
                self.height(),
                ext.raw(),
            )
            .result()?;
            Ok(())
        }
    }

    fn set(&mut self, value: f32) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, CuStream::DEFAULT_)
    }
}

impl<'a, P: AsPlaneMut<Storage = Cuda<[u8; 2]>>> SetPlane<Cuda<[u8; 2]>> for P {
    type Ext<'aux> = &'aux CuStream;

    fn set_ext(&mut self, value: [u8; 2], ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        let rect = self.absolute_rect();
        let pitch = self.pitch_bytes();
        unsafe {
            cudarse_driver::sys::cuMemsetD2D16Async(
                self.storage_mut().compute_ptr(rect.x, rect.y, pitch),
                pitch,
                u16::from_ne_bytes(value),
                self.width(),
                self.height(),
                ext.raw(),
            )
            .result()?;
            Ok(())
        }
    }

    fn set(&mut self, value: [u8; 2]) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, CuStream::DEFAULT_)
    }
}

impl<'a, P: AsPlaneMut<Storage = Cuda<[u16; 2]>>> SetPlane<Cuda<[u16; 2]>> for P {
    type Ext<'aux> = &'aux CuStream;

    fn set_ext(&mut self, value: [u16; 2], ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        let rect = self.absolute_rect();
        let pitch = self.pitch_bytes();
        unsafe {
            cudarse_driver::sys::cuMemsetD2D32Async(
                self.storage_mut().compute_ptr(rect.x, rect.y, pitch),
                pitch,
                ((value[1] as u32) << 16) | value[0] as u32,
                self.width(),
                self.height(),
                ext.raw(),
            )
            .result()?;
            Ok(())
        }
    }

    fn set(&mut self, value: [u16; 2]) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, CuStream::DEFAULT_)
    }
}

pub struct CuPinned<S>(CuPin<Box<[S]>>);
impl<S> CuPinned<S> {
    pub fn new(boxed: Box<[S]>) -> CuResult<Self> {
        Ok(Self(CuPin::new(boxed)?))
    }
}

impl<S: Sample> SampleStorage for CuPinned<S> {
    type SampleType = S;
}
// impl<S: Sample, Target: Sample> CastStorage<Target> for CuPinned<S> {
//     type Out = CuPinned<Target>;
//
//     fn cast(self) -> Self::Out {
//
//     }
// }
impl<S: Sample> OwnedSampleStorage for CuPinned<S> {
    type Ext<'a> = ();

    fn alloc_ext(
        width: usize,
        height: usize,
        ext: Self::Ext<'_>,
    ) -> Result<(usize, Self), Box<dyn Error>> {
        let (pitch, alloc) = Box::alloc_ext(width, height, ext)?;
        Ok((pitch, Self(CuPin::new(alloc)?)))
    }

    fn drop_ext(&mut self, _ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
impl<S: Sample> Borrow<[S]> for CuPinned<S> {
    fn borrow(&self) -> &[S] {
        &self.0
    }
}
impl<S: Sample> BorrowMut<[S]> for CuPinned<S> {
    fn borrow_mut(&mut self) -> &mut [S] {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda::Cuda;
    use crate::ops::TransferPlane;
    use crate::plane::Plane;
    use crate::{AsPlaneMut, Rect, SetPlane};
    use cudarse_driver::CuDevice;
    use std::error::Error;
    use std::sync::LazyLock;

    pub fn init_cuda() -> Result<(), Box<dyn Error>> {
        static CUDA_INIT: LazyLock<CuDevice> = LazyLock::new(|| {
            cudarse_driver::init_cuda().expect("Could not initialize the CUDA API");
            let dev = CuDevice::get(0).unwrap();
            dev
        });
        CUDA_INIT.retain_primary_ctx()?.set_current()?;
        Ok(())
    }

    #[test]
    fn transfer() -> Result<(), Box<dyn Error>> {
        init_cuda()?;
        let mut cpu_img = Plane::<Box<[[u8; 3]]>>::new(128, 128)?;
        cpu_img[(0, 0)] = [1, 1, 1];
        let mut gpu_img = Plane::<Cuda<[u8; 3]>>::new(128, 128)?;
        gpu_img.copy_from(&cpu_img)?;
        Ok(())
    }

    #[test]
    fn set() -> Result<(), Box<dyn Error>> {
        init_cuda()?;
        let mut plane = Plane::<Cuda<u8>>::new(4, 4)?;
        plane.set(0)?;
        plane
            .view_mut(&Rect {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            })
            .unwrap()
            .set(255)?;
        Ok(())
    }

    #[test]
    fn pinning() -> Result<(), Box<dyn Error>> {
        init_cuda()?;
        let plane = Plane::<Box<[u8]>>::new(4, 4)?.cupin()?;
        let mut gpu = plane.new_like::<Cuda<u8>>()?;
        gpu.copy_from(&plane)?;
        Ok(())
    }
}
