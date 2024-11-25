use crate::{
    assert_same_size, AsPlane, AsPlaneMut, HostAccessible, HostAccessibleMut, SampleStorage,
    StaticSample,
};
use std::error::Error;

/// Data transfer between planes. The copy works across devices.
pub trait TransferPlane<S: StaticSample, Src: SampleStorage, Dst: SampleStorage> {
    type Aux<'a>;
    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Src>,
        aux: Self::Aux<'_>,
    ) -> Result<(), Box<dyn Error>>;
    fn copy_from(&mut self, src: impl AsPlane<Storage = Src>) -> Result<(), Box<dyn Error>>;
}

// Host to Host
impl<
        S: StaticSample,
        Src: HostAccessible<SampleType = S>,
        Dst: HostAccessibleMut<SampleType = S>,
        DstPlane: AsPlaneMut<Storage = Dst>,
    > TransferPlane<S, Src, Dst> for DstPlane
{
    type Aux<'a> = ();

    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Src>,
        _aux: Self::Aux<'_>,
    ) -> Result<(), Box<dyn Error>> {
        assert_same_size!(self, src);
        if self.is_full_view() && src.is_full_view() {
            // Do the whole operation in one copy if we are working with the full planes
            self.storage_mut()
                .slice_mut()
                .copy_from_slice(src.storage().slice());
        } else {
            // Fallback to line copy for other cases
            let dst_rect = self.absolute_rect();
            let dst_pitch = self.pitch();
            for (dst_line, src_line) in self
                .storage_mut()
                .lines_mut(&dst_rect, dst_pitch)
                .zip(src.storage().lines(&src.absolute_rect(), src.pitch()))
            {
                dst_line.copy_from_slice(src_line)
            }
        }
        Ok(())
    }

    fn copy_from(&mut self, src: impl AsPlane<Storage = Src>) -> Result<(), Box<dyn Error>> {
        self.copy_from_ext(src, ())
    }
}

pub trait SetPlane<Stor: SampleStorage> {
    type Aux<'aux>;
    fn set_ext(
        &mut self,
        value: Stor::SampleType,
        aux: Self::Aux<'_>,
    ) -> Result<(), Box<dyn Error>>;
    fn set(&mut self, value: Stor::SampleType) -> Result<(), Box<dyn Error>>;
}

impl<S: StaticSample, Stor: HostAccessibleMut<SampleType = S>, P: AsPlaneMut<Storage = Stor>>
    SetPlane<Stor> for P
{
    type Aux<'aux> = ();

    fn set_ext(&mut self, value: S, _aux: Self::Aux<'_>) -> Result<(), Box<dyn Error>> {
        if self.is_full_view() {
            self.storage_mut().slice_mut().fill(value);
        } else {
            let rect = self.absolute_rect();
            let pitch = self.pitch();
            for line in self.storage_mut().lines_mut(&rect, pitch) {
                line.fill(value);
            }
        }
        Ok(())
    }

    fn set(&mut self, value: S) -> Result<(), Box<dyn Error>> {
        self.set_ext(value, ())
    }
}

#[cfg(test)]
mod tests {
    use crate::{AsPlaneMut, Plane, Rect, SetPlane, TransferPlane};
    use std::error::Error;

    #[test]
    fn copy() -> Result<(), Box<dyn Error>> {
        let mut p1 = Plane::<Box<[[u8; 3]]>>::new(2, 2)?;
        p1[(0, 0)] = [1, 1, 1];
        let mut p2 = Plane::<Box<[[u8; 3]]>>::new(2, 2)?;
        p2.copy_from(&p1)?;
        assert_eq!(p2[(0, 0)], [1, 1, 1]);
        Ok(())
    }

    #[test]
    fn set() -> Result<(), Box<dyn Error>> {
        let mut plane = Plane::<Box<[u8]>>::new(4, 4)?;
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
}
