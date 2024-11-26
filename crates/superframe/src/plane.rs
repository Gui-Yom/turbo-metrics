use crate::rect::Rect;
use crate::{
    CastStorage, DynSample, HostAccessible, OwnedSampleStorage, Sample, SampleStorage, StaticSample,
};
use std::any::TypeId;
use std::error::Error;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Represent a read only plane. Implemented by [Plane], [PlaneRef] and [PlaneMut] and their references.
pub trait AsPlane {
    type Storage: SampleStorage;

    fn width(&self) -> usize;

    fn height(&self) -> usize;

    /// ¨Pitch in bytes
    fn pitch_bytes(&self) -> usize;

    /// Pitch in count of the sample type
    fn pitch(&self) -> usize {
        self.pitch_bytes() / <Self::Storage as SampleStorage>::SampleType::SIZE
    }

    /// Reference to the underlying storage in full
    fn storage(&self) -> &Self::Storage;

    /// View rectangle relative to the original plane
    fn absolute_rect(&self) -> Rect;

    /// View rectangle relative to this view (x=0,y=0)
    fn rect(&self) -> Rect;

    /// returns: true if this view represent the full underlying plane and the whole storage can be used.
    /// Note: this can be used for optimizations where we don't care about padding.
    fn is_full_view(&self) -> bool {
        self.rect() == self.absolute_rect()
    }

    /// Create a read only view of this plane. This is akin to slicing but in 2D.
    ///
    /// # Arguments
    ///
    /// * `rect`: view rectangle given relative to this view.
    ///
    /// returns: The view or [None] if `rect` does not fit this view.
    fn view_checked(&self, rect: &Rect) -> Option<PlaneRef<Self>> {
        if self.rect().contains(&rect) {
            Some(self.view(rect))
        } else {
            None
        }
    }

    fn view(&self, rect: &Rect) -> PlaneRef<Self> {
        assert!(self.rect().contains(&rect));
        PlaneRef {
            rect: rect.with_base(&self.absolute_rect()),
            parent: self,
        }
    }
}

/// Represent a mutable plane. Implemented by [Plane] and [PlaneMut] and their mutable references.
pub trait AsPlaneMut: AsPlane {
    fn storage_mut(&mut self) -> &mut Self::Storage;

    /// Create a mutable view of this plane. This is akin to slicing but in 2D.
    ///
    /// # Arguments
    ///
    /// * `rect`: view rectangle given relative to this view.
    ///
    /// returns: The view or [None] if `rect` does not fit this view.
    fn view_mut(&mut self, rect: &Rect) -> Option<PlaneMut<Self>> {
        if self.rect().contains(&rect) {
            Some(PlaneMut {
                rect: rect.with_base(&self.absolute_rect()),
                parent: self,
            })
        } else {
            None
        }
    }
}

impl<S: Sample, Stor: SampleStorage<SampleType = S>> AsPlane for Plane<Stor> {
    type Storage = Stor;

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn pitch_bytes(&self) -> usize {
        self.pitch
    }

    fn storage(&self) -> &Self::Storage {
        &self.data
    }

    fn absolute_rect(&self) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: self.width,
            h: self.height,
        }
    }

    fn rect(&self) -> Rect {
        self.absolute_rect()
    }
}

impl<Stor: SampleStorage> AsPlaneMut for Plane<Stor> {
    fn storage_mut(&mut self) -> &mut Self::Storage {
        &mut self.data
    }
}

impl<'a, P: AsPlane> AsPlane for PlaneRef<'a, P> {
    type Storage = P::Storage;

    fn width(&self) -> usize {
        self.rect.w
    }

    fn height(&self) -> usize {
        self.rect.h
    }

    fn pitch_bytes(&self) -> usize {
        self.parent.pitch_bytes()
    }

    fn storage(&self) -> &Self::Storage {
        self.parent.storage()
    }

    fn absolute_rect(&self) -> Rect {
        self.rect.clone()
    }

    fn rect(&self) -> Rect {
        self.rect.relative_to_self()
    }
}

impl<'a, P: AsPlane> AsPlane for PlaneMut<'a, P> {
    type Storage = P::Storage;

    fn width(&self) -> usize {
        self.rect.w
    }

    fn height(&self) -> usize {
        self.rect.h
    }

    fn pitch_bytes(&self) -> usize {
        self.parent.pitch_bytes()
    }

    fn storage(&self) -> &Self::Storage {
        self.parent.storage()
    }

    fn absolute_rect(&self) -> Rect {
        self.rect.clone()
    }

    fn rect(&self) -> Rect {
        self.rect.relative_to_self()
    }
}

impl<'a, P: AsPlaneMut> AsPlaneMut for PlaneMut<'a, P> {
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.parent.storage_mut()
    }
}

impl<P: AsPlane<Storage: SampleStorage<SampleType: Sample>>> AsPlane for &P {
    type Storage = P::Storage;

    fn width(&self) -> usize {
        AsPlane::width(*self)
    }

    fn height(&self) -> usize {
        AsPlane::height(*self)
    }

    fn pitch_bytes(&self) -> usize {
        AsPlane::pitch_bytes(*self)
    }

    fn storage(&self) -> &Self::Storage {
        AsPlane::storage(*self)
    }

    fn absolute_rect(&self) -> Rect {
        AsPlane::absolute_rect(*self)
    }

    fn rect(&self) -> Rect {
        AsPlane::rect(*self)
    }
}

impl<'a, P: AsPlane<Storage: SampleStorage<SampleType: Sample>>> AsPlane for &mut P {
    type Storage = P::Storage;

    fn width(&self) -> usize {
        AsPlane::width(*self)
    }

    fn height(&self) -> usize {
        AsPlane::height(*self)
    }

    fn pitch_bytes(&self) -> usize {
        AsPlane::pitch_bytes(*self)
    }

    fn storage(&self) -> &Self::Storage {
        AsPlane::storage(*self)
    }

    fn absolute_rect(&self) -> Rect {
        AsPlane::absolute_rect(*self)
    }

    fn rect(&self) -> Rect {
        AsPlane::rect(*self)
    }
}

impl<'a, P: AsPlaneMut<Storage: SampleStorage<SampleType: Sample>>> AsPlaneMut for &mut P {
    fn storage_mut(&mut self) -> &mut Self::Storage {
        AsPlaneMut::storage_mut(*self)
    }
}

/// Contiguous 2D array of samples. Rows can be padded for alignment. Addressing in the plane storage is usually done with `y * pitch + x`.
pub struct Plane<Stor: SampleStorage = Box<[u8]>> {
    width: usize,
    height: usize,
    /// Pitch (aka stride) is the length of a line in bytes including padding.
    pitch: usize,
    /// Tag for identifying the sample type when using [DynSample]. This is zero sized when the sample type is known at compile time.
    sample_type: <<Stor as SampleStorage>::SampleType as Sample>::Id,
    /// The underlying storage.
    data: Stor,
}

impl<S: StaticSample, Stor: OwnedSampleStorage<SampleType = S>> Plane<Stor> {
    /// Allocate a new plane. Data is set to whatever was here before. This is like [Plane::new_in] but with a default initialized device context.
    pub fn new<'a>(width: usize, height: usize) -> Result<Self, Box<dyn Error>>
    where
        Stor::Ext<'a>: Default,
    {
        Self::new_ext(width, height, Stor::Ext::default())
    }

    /// Allocate a new plane. Data is set to whatever was here before.
    pub fn new_ext(
        width: usize,
        height: usize,
        ext: Stor::Ext<'_>,
    ) -> Result<Self, Box<dyn Error>> {
        let (pitch, data) = Stor::alloc_ext(width, height, ext)?;
        Ok(Self {
            width,
            height,
            pitch,
            sample_type: (),
            data,
        })
    }

    /// Manually drop the plane, specifying drop parameters.
    pub fn drop<'a>(mut self) -> Result<(), Box<dyn Error>>
    where
        Stor::Ext<'a>: Default,
    {
        self.data.drop()
    }

    /// Manually drop the plane, specifying drop parameters.
    pub fn drop_ext(mut self, ext: Stor::Ext<'_>) -> Result<(), Box<dyn Error>> {
        self.data.drop_ext(ext)
    }
}

impl<Stor: SampleStorage> Plane<Stor> {
    pub fn new_like_ext<DstStor: OwnedSampleStorage<SampleType: StaticSample>>(
        &self,
        ext: DstStor::Ext<'_>,
    ) -> Result<Plane<DstStor>, Box<dyn Error>> {
        Plane::new_ext(self.width, self.height, ext)
    }

    pub fn new_like<'a, DstStor: OwnedSampleStorage<SampleType: StaticSample>>(
        &self,
    ) -> Result<Plane<DstStor>, Box<dyn Error>>
    where
        DstStor::Ext<'a>: Default,
    {
        self.new_like_ext(DstStor::Ext::default())
    }
}

impl<Stor: SampleStorage + Clone> Clone for Plane<Stor> {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            sample_type: self.sample_type.clone(),
            data: self.data.clone(),
        }
    }
}

impl<Stor: HostAccessible<SampleType: StaticSample>> Plane<Stor> {
    // pitch in count of S
    pub fn from_host(width: usize, height: usize, pitch: usize, data: Stor) -> Self {
        assert!(width <= pitch);
        assert_eq!(data.slice().len() % pitch, 0);
        assert!(data.slice().len() >= width * height);
        Self {
            width,
            height,
            pitch,
            sample_type: (),
            data,
        }
    }
}

#[cfg(feature = "cuda")]
impl<S: StaticSample> Plane<crate::cuda::Cuda<S>> {
    // pitch in count of S
    pub unsafe fn from_parts(
        width: usize,
        height: usize,
        pitch: usize,
        data: crate::cuda::Cuda<S>,
    ) -> Self {
        assert!(width <= pitch);
        Self {
            width,
            height,
            pitch,
            sample_type: (),
            data,
        }
    }
}

#[cfg(feature = "cuda")]
impl<S: StaticSample> Plane<Box<[S]>> {
    pub fn cupin(self) -> cudarse_driver::CuResult<Plane<crate::cuda::CuPinned<S>>> {
        Ok(Plane::from_host(
            self.width,
            self.height,
            self.pitch,
            crate::cuda::CuPinned::new(self.data)?,
        ))
    }
}

impl<S: StaticSample + 'static, Stor: SampleStorage<SampleType = S> + CastStorage<DynSample>>
    Plane<Stor>
{
    pub fn to_dyn(self) -> Plane<<Stor as CastStorage<DynSample>>::Out> {
        Plane {
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            sample_type: DynSample::create_tag::<S>(),
            data: self.data.cast(),
        }
    }
}

impl<Stor: SampleStorage<SampleType = DynSample>> Plane<Stor> {
    pub fn sample_type(&self) -> &<DynSample as Sample>::Id {
        &self.sample_type
    }
}

impl<Stor: SampleStorage<SampleType = DynSample>> Plane<Stor> {
    pub fn to_concrete<Target: StaticSample + 'static>(
        self,
    ) -> Result<Plane<<Stor as CastStorage<Target>>::Out>, Self>
    where
        Stor: CastStorage<Target>,
    {
        if self.sample_type == TypeId::of::<Target>() {
            Ok(unsafe { self.to_concrete_unchecked() })
        } else {
            Err(self)
        }
    }

    pub unsafe fn to_concrete_unchecked<Target: StaticSample + 'static>(
        self,
    ) -> Plane<<Stor as CastStorage<Target>>::Out>
    where
        Stor: CastStorage<Target>,
    {
        debug_assert_eq!(self.sample_type, TypeId::of::<Target>());
        Plane {
            width: self.width,
            height: self.height,
            pitch: self.pitch,
            sample_type: (),
            data: self.data.cast(),
        }
    }
}

#[macro_export]
macro_rules! dispatch_sample {
    ($val:expr; $($t:ty => $tt:tt,)* @other => $other:tt) => {
        {let base = $val;
        match base.sample_type() {
            $(id if *id == ::std::any::TypeId::of::<$t>() => {
                let self_ = unsafe { base.to_concrete::<$t>() };
                $tt
            })*
            other => $other,
        }}
    }
}

/// 2D read only slice into another plane view.
pub struct PlaneRef<'a, P: AsPlane + ?Sized> {
    /// Absolute positioning into the original buffer (prioritize quick evaluation)
    rect: Rect,
    parent: &'a P,
    // FIXME I have no idea what I'm doing with variance here
    // _marker: PhantomData<(&'a S, &'a mut D)>,
}

/// 2D mutable slice into another plane view.
pub struct PlaneMut<'a, P: AsPlane + ?Sized> {
    /// Absolute positioning into the original buffer (prioritize quick evaluation)
    rect: Rect,
    parent: &'a mut P,
    // _marker: PhantomData<(&'a S, &'a mut D)>,
}

impl<S: StaticSample, Stor: SampleStorage + Deref<Target: Index<usize, Output = S>>>
    Index<(usize, usize)> for Plane<Stor>
{
    type Output = S;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.data.index(y * self.pitch / S::SIZE + x)
    }
}

impl<S: StaticSample, Stor: SampleStorage + DerefMut<Target: IndexMut<usize, Output = S>>>
    IndexMut<(usize, usize)> for Plane<Stor>
{
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.data.index_mut(y * self.pitch / S::SIZE + x)
    }
}

// impl<S: Sample, D: SampleStorage> Img<S> for Plane<S, D> {
//     fn width(&self, _plane: usize) -> usize {
//         self.width
//     }
//
//     fn height(&self, _plane: usize) -> usize {
//         self.height
//     }
//
//     fn pitch(&self, _plane: usize) -> usize {
//         self.pitch
//     }
// }

#[cfg(test)]
mod tests {
    use crate::plane::Plane;
    use crate::{AsPlane, Rect};
    use std::error::Error;

    #[test]
    fn view() -> Result<(), Box<dyn Error>> {
        let plane = Plane::<Box<[u8]>>::new(16, 16)?;
        let quarter = plane
            .view_checked(&Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            })
            .unwrap();
        dbg!(&quarter.rect);
        let smaller = quarter
            .view_checked(&Rect {
                x: 4,
                y: 4,
                w: 4,
                h: 4,
            })
            .unwrap();
        dbg!(&smaller.rect);
        Ok(())
    }

    #[test]
    fn dispatch() -> Result<(), Box<dyn Error>> {
        let plane = Plane::<Box<[f32]>>::new(16, 16)?;
        println!(
            "{}",
            dispatch_sample!(plane.to_dyn();
                u8 => { "u8" },
                u16 => { "u16" },
                f32 => { "f32" },
                @other => { todo!() }
            )
        );
        Ok(())
    }

    #[test]
    fn borrowed() {
        let data: &[u8] = &[0u8, 0, 0, 0];
        let plane = Plane::from_host(2, 2, 2, data);
    }
}
