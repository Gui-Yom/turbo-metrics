use std::fmt::Debug;

pub use cuda_npp_sys as sys;
use sys::{nppGetStreamContext, NppStatus, NppStreamContext};

pub mod safe;

mod __priv {
    /// So people don't implement child trait for themselves.
    /// Hacky way of doing closed polymorphism with traits.
    pub trait Sealed {}

    impl<T: Sealed> Sealed for &T {}

    impl<T: Sealed> Sealed for &mut T {}

    impl Sealed for u8 {}

    impl Sealed for u16 {}

    impl Sealed for i16 {}

    impl Sealed for i32 {}

    impl Sealed for f32 {}
}

/// Image sample type
pub trait Sample: __priv::Sealed + Default + Copy + 'static {}

impl Sample for u8 {}

impl Sample for u16 {}

impl Sample for i16 {}

impl Sample for i32 {}

impl Sample for f32 {}

/// Layout of the image, either packed channels or planar
pub trait Channels: __priv::Sealed + 'static {
    const NUM_SAMPLES: usize;
    const IS_PLANAR: bool;

    type Storage<S>: Debug + Clone;
    type Ref<S>: Copy;
    type RefMut<S>;

    fn make_ref<S>(s: &Self::Storage<S>) -> Self::Ref<S>;
    fn make_ref_mut<S>(s: &mut Self::Storage<S>) -> Self::RefMut<S>;

    fn iter_ptrs<S>(s: &Self::Storage<S>) -> impl ExactSizeIterator<Item = *const S>;
    fn iter_ptrs_mut<S>(s: &mut Self::Storage<S>) -> impl ExactSizeIterator<Item = *mut S>;
}

/// Packed channels
#[derive(Debug)]
pub struct C<const N: usize>;

/// Planar channels
#[derive(Debug)]
pub struct P<const N: usize>;

macro_rules! impl_channels_packed {
    ($n:literal) => {
        impl __priv::Sealed for C<$n> {}
        impl Channels for C<$n> {
            const NUM_SAMPLES: usize = $n;
            const IS_PLANAR: bool = false;
            type Storage<S> = *mut S;
            type Ref<S> = *const S;
            type RefMut<S> = *mut S;

            fn make_ref<S>(s: &Self::Storage<S>) -> Self::Ref<S> {
                *s
            }

            fn make_ref_mut<S>(s: &mut Self::Storage<S>) -> Self::RefMut<S> {
                *s
            }

            fn iter_ptrs<S>(s: &Self::Storage<S>) -> impl ExactSizeIterator<Item = *const S> {
                [(*s).cast_const()].into_iter()
            }

            fn iter_ptrs_mut<S>(s: &mut Self::Storage<S>) -> impl ExactSizeIterator<Item = *mut S> {
                [*s].into_iter()
            }
        }
    };
}

impl_channels_packed!(1);
impl_channels_packed!(2);
impl_channels_packed!(3);
impl_channels_packed!(4);

macro_rules! impl_channels_planar {
    ($n:literal) => {
        impl __priv::Sealed for P<$n> {}
        impl Channels for P<$n> {
            const NUM_SAMPLES: usize = $n;
            const IS_PLANAR: bool = true;
            type Storage<S> = [*mut S; Self::NUM_SAMPLES];
            type Ref<S> = *const *const S;
            type RefMut<S> = *const *mut S;

            fn make_ref<S>(s: &Self::Storage<S>) -> Self::Ref<S> {
                s.as_ptr() as _
            }

            fn make_ref_mut<S>(s: &mut Self::Storage<S>) -> Self::RefMut<S> {
                s.as_ptr()
            }

            fn iter_ptrs<S>(s: &Self::Storage<S>) -> impl ExactSizeIterator<Item = *const S> {
                s.clone().into_iter().map(|p| p.cast_const())
            }

            fn iter_ptrs_mut<S>(s: &mut Self::Storage<S>) -> impl ExactSizeIterator<Item = *mut S> {
                s.iter().copied()
            }
        }
    };
}

impl_channels_planar!(3);
impl_channels_planar!(4);

pub fn get_stream_ctx() -> Result<NppStreamContext, NppStatus> {
    let mut ctx = Default::default();
    unsafe { nppGetStreamContext(&mut ctx) }.result()?;
    Ok(ctx)
}
