use std::marker::PhantomData;

use cuda_npp_sys::*;

use crate::safe::{Image, E};
use crate::{Channel, ChannelPacked, Sample, C};

use super::Result;

pub trait Malloc {
    /// `malloc` a new image on device
    fn malloc(width: u32, height: u32) -> Result<Self>
    where
        Self: Sized;
}

macro_rules! malloc_impl {
    ($sample_ty:ty, $channel_ty:ty, $sample_id:ident, $channel_id:ident) => {
        impl Malloc for Image<$sample_ty, $channel_ty> {
            fn malloc(width: u32, height: u32) -> Result<Self>
            where
                Self: Sized,
            {
                let mut line_step = 0;
                let ptr = unsafe { paste::paste!([<nppi Malloc $sample_id _ $channel_id>])(width as i32, height as i32, &mut line_step) };
                if ptr.is_null() {
                    Err(E::from(NppStatus::NPP_MEMORY_ALLOCATION_ERR))
                } else {
                    Ok(Self {
                        width,
                        height,
                        line_step,
                        data: ptr,
                        marker_: PhantomData,
                        marker__: PhantomData,
                    })
                }
            }
        }
    };
}

malloc_impl!(u8, C<1>, _8u, C1);
malloc_impl!(u8, C<2>, _8u, C2);
malloc_impl!(u8, C<3>, _8u, C3);
malloc_impl!(u8, C<4>, _8u, C4);

malloc_impl!(u16, C<1>, _16u, C1);
malloc_impl!(u16, C<2>, _16u, C2);
malloc_impl!(u16, C<3>, _16u, C3);
malloc_impl!(u16, C<4>, _16u, C4);

malloc_impl!(i16, C<1>, _16s, C1);
malloc_impl!(i16, C<2>, _16s, C2);
malloc_impl!(i16, C<4>, _16s, C4);

malloc_impl!(f32, C<1>, _32f, C1);
malloc_impl!(f32, C<2>, _32f, C2);
malloc_impl!(f32, C<3>, _32f, C3);
malloc_impl!(f32, C<4>, _32f, C4);

impl<S: Sample, C: Channel> Drop for Image<S, C> {
    fn drop(&mut self) {
        println!("Dropping image");
        unsafe {
            nppiFree(self.data as _);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::safe::isu::Malloc;
    use crate::safe::Image;
    use crate::safe::Result;
    use crate::C;

    #[test]
    fn new_image() -> Result<()> {
        let img = Image::<f32, C<3>>::malloc(1024, 1024)?;
        Ok(())
    }
}
