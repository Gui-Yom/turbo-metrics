use crate::{
    assert_same_size, AsPlane, AsPlaneMut, HostAccessible, HostAccessibleMut, SampleStorage,
    StaticSample,
};
use std::error::Error;

/// Data transfer between planes. The copy works across storages.
pub trait TransferPlane<S: StaticSample, Src: SampleStorage, Dst: SampleStorage> {
    /// Extra implementation defined data.
    type Ext<'a>;

    /// Copy data to this plane from a `src` plane. Plane sizes must match.
    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Src>,
        ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>>;

    /// Like [Self::copy_from_ext] but with a default initialized extra data.
    fn copy_from(&mut self, src: impl AsPlane<Storage = Src>) -> Result<(), Box<dyn Error>>;
}

// Host to Host
impl<S, Src, Dst, DstPlane> TransferPlane<S, Src, Dst> for DstPlane
where
    S: StaticSample,
    Src: HostAccessible<SampleType = S>,
    Dst: HostAccessibleMut<SampleType = S>,
    DstPlane: AsPlaneMut<Storage = Dst>,
{
    type Ext<'a> = ();

    fn copy_from_ext(
        &mut self,
        src: impl AsPlane<Storage = Src>,
        _ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>> {
        assert_same_size!(self, src);
        if self.is_full_view() && src.is_full_view() && self.pitch_bytes() == src.pitch_bytes() {
            // Do the whole operation in one copy if we are working with the same full planes
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

/// Set a plane to a fixed value.
pub trait SetPlane<Stor: SampleStorage> {
    /// Extra implementation defined data.
    type Ext<'aux>;

    /// Set the plane to a fixed value.
    fn set_ext(
        &mut self,
        value: Stor::SampleType,
        ext: Self::Ext<'_>,
    ) -> Result<(), Box<dyn Error>>;

    /// Like [Self::set_ext] but with a default initialized extra data.
    fn set(&mut self, value: Stor::SampleType) -> Result<(), Box<dyn Error>>;
}

impl<S, Stor, P> SetPlane<Stor> for P
where
    S: StaticSample,
    Stor: HostAccessibleMut<SampleType = S>,
    P: AsPlaneMut<Storage = Stor>,
{
    type Ext<'aux> = ();

    fn set_ext(&mut self, value: S, _ext: Self::Ext<'_>) -> Result<(), Box<dyn Error>> {
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
