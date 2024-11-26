use std::any::TypeId;
use std::borrow::{Borrow, BorrowMut};
use std::error::Error;
use std::ptr::slice_from_raw_parts_mut;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod image;
pub mod ops;
pub mod plane;
pub mod rect;

pub use crate::image::*;
pub use crate::ops::*;
pub use crate::plane::*;
pub use crate::rect::*;

pub trait Sample: Copy {
    /// Tag that must be stored out of band to identify this sample type.
    type Id: Clone;
    const SIZE: usize = size_of::<Self>();
}
impl Sample for u8 {
    type Id = ();
}
impl Sample for u16 {
    type Id = ();
}
impl Sample for f32 {
    type Id = ();
}

impl<S: StaticSample, const N: usize> Sample for [S; N] {
    type Id = S::Id;
}

pub trait StaticSample: Sample<Id = ()> {}
impl<S: Sample<Id = ()>> StaticSample for S {}

/// Erasure for the original sample type. A tag must be used to store the information of the original type.
#[derive(Default, Copy, Clone)]
#[repr(transparent)]
pub struct DynSample(u8);
impl DynSample {
    pub fn create_tag<S: StaticSample + 'static>() -> <Self as Sample>::Id {
        TypeId::of::<S>()
    }
}
impl Sample for DynSample {
    type Id = TypeId;
}

/// Compute optimal pitch in bytes given a width in bytes
pub(crate) const fn row_align(row_bytes: usize, align: usize) -> usize {
    // After writing this, I realize this is just modulo with a 2^n operand.
    (row_bytes & !(align - 1)) + align
}

pub trait SampleStorage {
    type SampleType: Sample;
}

pub trait CastStorage<Target: Sample>: SampleStorage {
    type Out: SampleStorage<SampleType = Target>;
    /// Reinterpret the container to hold `Target` samples instead.
    fn cast(self) -> Self::Out;
}

pub trait OwnedSampleStorage: SampleStorage + Sized {
    type Ext<'a>;

    /// Allocate enough memory to hold a plane of size `width` * `height`.
    /// Memory may not be cleared and can be set with whatever rubbish was here before.
    fn alloc_ext(
        width: usize,
        height: usize,
        ext: Self::Ext<'_>,
    ) -> Result<(usize, Self), Box<dyn Error>>;
    /// Allocate enough memory to hold a plane of size `width` * `height`.
    /// Memory may not be cleared and can be set with whatever rubbish was here before.
    fn alloc<'a>(width: usize, height: usize) -> Result<(usize, Self), Box<dyn Error>>
    where
        Self::Ext<'a>: Default,
    {
        Self::alloc_ext(width, height, Self::Ext::default())
    }
    /// Run custom logic before drop. Caller must ensure the storage is properly dropped after calling this.
    fn drop_ext(&mut self, ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>>;
    /// Run custom logic before drop. Caller must ensure the storage is properly dropped after calling this.
    fn drop<'a>(&mut self) -> Result<(), Box<dyn Error>>
    where
        Self::Ext<'a>: Default,
    {
        self.drop_ext(Self::Ext::default())
    }
}

pub trait HostAccessible: SampleStorage {
    fn ptr(&self) -> *const Self::SampleType;
    fn slice(&self) -> &[Self::SampleType];
    fn lines<'a>(
        &'a self,
        rect: &Rect,
        pitch: usize,
    ) -> impl Iterator<Item = &'a [Self::SampleType]>
    where
        Self::SampleType: 'a;
}

impl<S, C: SampleStorage<SampleType = S> + Borrow<[S]>> HostAccessible for C {
    fn ptr(&self) -> *const Self::SampleType {
        self.borrow().as_ptr()
    }

    fn slice(&self) -> &[Self::SampleType] {
        self.borrow()
    }

    fn lines<'a>(
        &'a self,
        rect: &Rect,
        pitch: usize,
    ) -> impl Iterator<Item = &'a [Self::SampleType]>
    where
        Self::SampleType: 'a,
    {
        let Rect { x, y, w, h } = rect;
        // rect.rows().map(move |y| {
        //     let base = y * pitch;
        //     &stor[base + start..base + end]
        // })
        let start = *x;
        let end = x + w;
        assert_eq!(self.borrow().len() % pitch, 0);
        self.borrow()
            .chunks_exact(pitch)
            .skip(*y)
            .take(*h)
            .map(move |c| &c[start..end])
    }

    // pub fn lines_unchecked<'a, S: StaticSample>(
    //     stor: &'a <Self as Device>::Storage<S>,
    //     rect: &Rect,
    //     pitch: usize,
    // ) -> impl Iterator<Item = &'a [S]> {
    //     let Rect { x, y, w, h } = rect;
    //     // rect.rows().map(move |y| {
    //     //     let base = y * pitch;
    //     //     &stor[base + start..base + end]
    //     // })
    //     let start = *x;
    //     let end = x + w;
    //     stor.chunks_exact(pitch)
    //         .skip(*y)
    //         .take(*h)
    //         .map(move |c| &c[start..end])
    // }
}

pub trait HostAccessibleMut: SampleStorage {
    fn mut_ptr(&mut self) -> *mut Self::SampleType;
    fn slice_mut(&mut self) -> &mut [Self::SampleType];
    fn lines_mut<'a>(
        &'a mut self,
        rect: &Rect,
        pitch: usize,
    ) -> impl Iterator<Item = &'a mut [Self::SampleType]>
    where
        Self::SampleType: 'a;
}

impl<S, C: SampleStorage<SampleType = S> + BorrowMut<[S]>> HostAccessibleMut for C {
    fn mut_ptr(&mut self) -> *mut Self::SampleType {
        self.borrow_mut().as_mut_ptr()
    }

    fn slice_mut(&mut self) -> &mut [Self::SampleType] {
        self.borrow_mut()
    }

    fn lines_mut<'a>(
        &'a mut self,
        rect: &Rect,
        pitch: usize,
    ) -> impl Iterator<Item = &'a mut [Self::SampleType]>
    where
        Self::SampleType: 'a,
    {
        let Rect { x, y, w, h } = rect;
        let start = *x;
        let end = x + w;
        assert_eq!(self.borrow_mut().len() % pitch, 0);
        self.borrow_mut()
            .chunks_exact_mut(pitch)
            .skip(*y)
            .take(*h)
            .map(move |c| &mut c[start..end])
    }
}

impl<S: Sample> SampleStorage for Box<[S]> {
    type SampleType = S;
}
impl<S: Sample, Target: Sample> CastStorage<Target> for Box<[S]> {
    type Out = Box<[Target]>;
    fn cast(self) -> Self::Out {
        let raw = Box::leak(self);
        let ptr = raw.as_ptr();
        let len = raw.len();
        // FIXME this does not sound safe
        unsafe {
            Box::from_raw(slice_from_raw_parts_mut(
                ptr.cast::<Target>().cast_mut(),
                len * S::SIZE / Target::SIZE,
            ))
        }
    }
}

impl<S: Sample> OwnedSampleStorage for Box<[S]> {
    type Ext<'a> = ();

    fn alloc_ext(
        width: usize,
        height: usize,
        _ext: Self::Ext<'_>,
    ) -> Result<(usize, Self), Box<dyn Error>> {
        // This should be large enough alignment for a CPU
        const ROW_ALIGN: usize = 64;
        let pitch = row_align(width * S::SIZE, ROW_ALIGN);
        let b = Box::new_uninit_slice(pitch / S::SIZE * height);
        Ok((pitch, unsafe { b.assume_init() }))
    }

    fn drop_ext(&mut self, _ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

impl<'a, S: Sample> SampleStorage for &'a [S] {
    type SampleType = S;
}
impl<'a, S: Sample, Target: Sample + 'a> CastStorage<Target> for &'a [S] {
    type Out = &'a [Target];

    fn cast(self) -> Self::Out {
        todo!()
    }
}

impl<S: Sample, const N: usize> SampleStorage for [S; N] {
    type SampleType = S;
}
impl<S: Sample, const N: usize> SampleStorage for &[S; N] {
    type SampleType = S;
}

#[macro_export]
macro_rules! assert_same_size {
    ($a:expr, $b:expr) => {
        assert_eq!(
            ($a.width(), $a.height()),
            ($b.width(), $b.height()),
            "Not the same size"
        )
    };
}

#[cfg(test)]
mod tests {
    use crate::{HostAccessible, HostAccessibleMut, OwnedSampleStorage, Rect};

    #[test]
    fn generic() {
        let (pitch, alloc) = Box::alloc(8, 8).unwrap();
        fn a(a: impl HostAccessible<SampleType = u8>) {}
        a(alloc);
    }

    #[test]
    fn iter_lines() {
        let (pitch, alloc): (_, Box<[u8]>) = Box::alloc(8, 8).unwrap();
        let lines: Vec<_> = alloc
            .lines(
                &Rect {
                    x: 2,
                    y: 2,
                    w: 4,
                    h: 4,
                },
                pitch,
            )
            .collect();
        assert_eq!(lines.len(), 4);
        for line in lines {
            assert_eq!(line.len(), 4);
        }
    }

    #[test]
    fn iter_lines_mut() {
        let (pitch, mut alloc): (_, Box<[u8]>) = Box::alloc(8, 8).unwrap();
        let lines: Vec<_> = alloc
            .lines_mut(
                &Rect {
                    x: 2,
                    y: 2,
                    w: 4,
                    h: 4,
                },
                pitch,
            )
            .collect();
        assert_eq!(lines.len(), 4);
        for line in lines {
            assert_eq!(line.len(), 4);
        }
    }
}
