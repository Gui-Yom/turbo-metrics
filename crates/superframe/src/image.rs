use crate::plane::Plane;
use crate::{OwnedSampleStorage, Sample, SampleStorage, StaticSample};
use std::borrow::Borrow;
use std::error::Error;
use std::ops::{Index, IndexMut};

/// Describes how plane information is stored in an image.
/// This makes [Image] work no matter if the number of planes is known at compile time or not.
pub trait PlaneStorage {
    /// This is just storage for the plane information, not the actual plane data (that would be the role of [SampleStorage]).
    type Storage<P>: IntoIterator;

    fn len<P>(stor: &Self::Storage<P>) -> usize;
    fn index<P>(stor: &Self::Storage<P>, index: usize) -> &P;
    fn index_mut<P>(stor: &mut Self::Storage<P>, index: usize) -> &mut P;
    fn iter<'a, P: 'a>(stor: &'a Self::Storage<P>) -> impl Iterator<Item = &'a P>;
    fn iter_mut<'a, P: 'a>(stor: &'a mut Self::Storage<P>) -> impl Iterator<Item = &'a mut P>;
}

/// Storage in an array
#[derive(Default, Copy, Clone, Debug)]
pub struct Array<const N: usize>;

/// Storage in a [Box]ed slice
#[derive(Default, Copy, Clone, Debug)]
pub struct Boxed;

/// Storage in a [Vec]
#[derive(Default, Copy, Clone, Debug)]
pub struct Vector;

impl<const N: usize> PlaneStorage for Array<N> {
    type Storage<P> = [P; N];

    fn len<P>(stor: &Self::Storage<P>) -> usize {
        stor.len()
    }

    fn index<P>(stor: &Self::Storage<P>, index: usize) -> &P {
        stor.index(index)
    }

    fn index_mut<P>(stor: &mut Self::Storage<P>, index: usize) -> &mut P {
        stor.index_mut(index)
    }

    fn iter<'a, P: 'a>(stor: &'a Self::Storage<P>) -> impl Iterator<Item = &'a P> {
        stor.iter()
    }

    fn iter_mut<'a, P: 'a>(stor: &'a mut Self::Storage<P>) -> impl Iterator<Item = &'a mut P> {
        stor.iter_mut()
    }
}

impl PlaneStorage for Boxed {
    type Storage<P> = Box<[P]>;

    fn len<P>(stor: &Self::Storage<P>) -> usize {
        stor.len()
    }

    fn index<P>(stor: &Self::Storage<P>, index: usize) -> &P {
        stor.index(index)
    }

    fn index_mut<P>(stor: &mut Self::Storage<P>, index: usize) -> &mut P {
        stor.index_mut(index)
    }

    fn iter<'a, P: 'a>(stor: &'a Self::Storage<P>) -> impl Iterator<Item = &'a P> {
        stor.iter()
    }

    fn iter_mut<'a, P: 'a>(stor: &'a mut Self::Storage<P>) -> impl Iterator<Item = &'a mut P> {
        stor.iter_mut()
    }
}

impl PlaneStorage for Vector {
    type Storage<P> = Vec<P>;

    fn len<P>(stor: &Self::Storage<P>) -> usize {
        stor.len()
    }

    fn index<P>(stor: &Self::Storage<P>, index: usize) -> &P {
        stor.index(index)
    }

    fn index_mut<P>(stor: &mut Self::Storage<P>, index: usize) -> &mut P {
        stor.index_mut(index)
    }

    fn iter<'a, P: 'a>(stor: &'a Self::Storage<P>) -> impl Iterator<Item = &'a P> {
        stor.iter()
    }

    fn iter_mut<'a, P: 'a>(stor: &'a mut Self::Storage<P>) -> impl Iterator<Item = &'a mut P> {
        stor.iter_mut()
    }
}

pub trait Img<S: Sample> {
    fn width(&self, plane: usize) -> usize;
    fn height(&self, plane: usize) -> usize;
    /// pitch in bytes
    fn pitch(&self, plane: usize) -> usize;
}

/// Owned image composed of many planes, with optional metadata.
pub struct Image<Stor: SampleStorage, P: PlaneStorage = Boxed, M = ()> {
    planes: P::Storage<Plane<Stor>>,
    metadata: M,
}

impl<const N: usize, Stor: OwnedSampleStorage<SampleType: StaticSample>, M>
    Image<Stor, Array<N>, M>
{
    pub fn new<'a>(width: usize, height: usize, metadata: M) -> Result<Self, Box<dyn Error>>
    where
        Stor::Ext<'a>: Default,
    {
        let planes = array_init::try_array_init(|_| Plane::new(width, height))?;

        Ok(Self { planes, metadata })
    }
    pub fn new_ext<'a>(
        width: usize,
        height: usize,
        ext: Stor::Ext<'a>,
        metadata: M,
    ) -> Result<Self, Box<dyn Error>>
    where
        Stor::Ext<'a>: Clone,
    {
        let planes = array_init::try_array_init(|_| Plane::new_ext(width, height, ext.clone()))?;

        Ok(Self { planes, metadata })
    }
}

impl<Stor: OwnedSampleStorage<SampleType: StaticSample>, M> Image<Stor, Boxed, M> {
    pub fn new<'a>(
        width: usize,
        height: usize,
        plane_count: usize,
        metadata: M,
    ) -> Result<Self, Box<dyn Error>>
    where
        Stor::Ext<'a>: Default,
    {
        let mut planes = Box::new_uninit_slice(plane_count);
        for p in &mut planes {
            p.write(Plane::new(width, height)?);
        }

        Ok(Self {
            planes: unsafe { planes.assume_init() },
            metadata,
        })
    }

    pub fn new_ext<'a>(
        width: usize,
        height: usize,
        plane_count: usize,
        ext: Stor::Ext<'a>,
        metadata: M,
    ) -> Result<Self, Box<dyn Error>>
    where
        Stor::Ext<'a>: Clone,
    {
        let mut planes = Box::new_uninit_slice(plane_count);
        for p in &mut planes {
            p.write(Plane::new_ext(width, height, ext.clone())?);
        }

        Ok(Self {
            planes: unsafe { planes.assume_init() },
            metadata,
        })
    }
}

impl<const N: usize, Stor: SampleStorage, M> Image<Stor, Array<N>, M> {
    pub fn to_boxed_planes(self) -> Image<Stor, Boxed, M> {
        Image {
            planes: Box::new(self.planes),
            metadata: self.metadata,
        }
    }

    pub fn to_vec_planes(self) -> Image<Stor, Vector, M> {
        Image {
            planes: Vec::from(self.planes),
            metadata: self.metadata,
        }
    }

    pub fn from_array(planes: [Plane<Stor>; N], metadata: M) -> Self {
        Self { planes, metadata }
    }
}

impl<Stor: SampleStorage, M> Image<Stor, Boxed, M> {
    pub fn to_static_planes<const N: usize>(self) -> Result<Image<Stor, Array<N>, M>, Self> {
        match TryInto::<Box<[Plane<Stor>; N]>>::try_into(self.planes) {
            Ok(a) => Ok(Image {
                planes: *a,
                metadata: self.metadata,
            }),
            Err(b) => Err(Self {
                planes: b,
                metadata: self.metadata,
            }),
        }
    }
}

impl<Stor: SampleStorage, M> Image<Stor, Vector, M> {
    pub fn to_static_planes<const N: usize>(self) -> Result<Image<Stor, Array<N>, M>, Self> {
        match TryInto::<[Plane<Stor>; N]>::try_into(self.planes) {
            Ok(a) => Ok(Image {
                planes: a,
                metadata: self.metadata,
            }),
            Err(v) => Err(Self {
                planes: v,
                metadata: self.metadata,
            }),
        }
    }
}

impl<P: PlaneStorage, Stor: SampleStorage, M> Image<Stor, P, M> {
    pub fn from_parts(planes: P::Storage<Plane<Stor>>, metadata: M) -> Self {
        Self { planes, metadata }
    }

    pub fn into_parts(self) -> (P::Storage<Plane<Stor>>, M) {
        (self.planes, self.metadata)
    }

    pub fn planes(&self) -> &P::Storage<Plane<Stor>> {
        &self.planes
    }

    pub fn planes_mut(&mut self) -> &mut P::Storage<Plane<Stor>> {
        &mut self.planes
    }

    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut M {
        &mut self.metadata
    }

    pub fn len(&self) -> usize {
        P::len(&self.planes)
    }

    pub fn iter(&'_ self) -> impl Iterator<Item = &'_ Plane<Stor>> {
        P::iter(&self.planes)
    }

    pub fn iter_mut(&'_ mut self) -> impl Iterator<Item = &'_ mut Plane<Stor>> {
        P::iter_mut(&mut self.planes)
    }
}

impl<P: PlaneStorage, Stor: SampleStorage, M> Into<(P::Storage<Plane<Stor>>, M)>
    for Image<Stor, P, M>
{
    fn into(self) -> (P::Storage<Plane<Stor>>, M) {
        (self.planes, self.metadata)
    }
}

impl<P: PlaneStorage, Stor: SampleStorage, M> Index<usize> for Image<Stor, P, M> {
    type Output = Plane<Stor>;

    fn index(&self, index: usize) -> &Self::Output {
        P::index(&self.planes, index)
    }
}

impl<P: PlaneStorage, Stor: SampleStorage, M> IndexMut<usize> for Image<Stor, P, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        P::index_mut(&mut self.planes, index)
    }
}

impl<P: PlaneStorage, Stor: SampleStorage, M> IntoIterator for Image<Stor, P, M> {
    type Item = <<P as PlaneStorage>::Storage<Plane<Stor>> as IntoIterator>::Item;
    type IntoIter = <P::Storage<Plane<Stor>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.planes.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch_sample;
    use crate::image::{Array, Image};
    use crate::plane::Plane;
    use std::error::Error;

    #[test]
    fn nv12() -> Result<(), Box<dyn Error>> {
        let luma = Plane::<Box<[u8]>>::new(16, 16)?;
        let uv = Plane::<Box<[u8; 2]>>::new(8, 8)?;
        let img: Image<_, Array<2>> = Image::from_parts([luma.to_dyn(), uv.to_dyn()], ());
        let [l, uv] = img.planes();
        for (i, plane) in img.into_iter().enumerate() {
            println!(
                "plane {i} is {}",
                dispatch_sample!(plane;
                    u8 => { "u8" },
                    [u8; 2] => { "[u8; 2]" },
                    @other => { "unknown" }
                )
            );
        }
        Ok(())
    }
}
