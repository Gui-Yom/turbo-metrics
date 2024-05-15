pub use cuda_npp_sys as sys;
use cuda_npp_sys::{nppGetStreamContext, NppStatus, NppStreamContext};

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
}

/// Packed channels
#[derive(Debug)]
pub struct C<const N: usize>;

/// Planar channels
pub struct P<const N: usize>;

impl<const N: usize> __priv::Sealed for C<N> {}

impl<const N: usize> __priv::Sealed for P<N> {}

impl Channels for C<1> {
    const NUM_SAMPLES: usize = 1;
}

impl Channels for C<2> {
    const NUM_SAMPLES: usize = 2;
}

impl Channels for C<3> {
    const NUM_SAMPLES: usize = 3;
}

impl Channels for C<4> {
    const NUM_SAMPLES: usize = 4;
}

pub fn get_stream_ctx() -> Result<NppStreamContext, NppStatus> {
    let mut ctx = Default::default();
    unsafe { nppGetStreamContext(&mut ctx) }.result()?;
    Ok(ctx)
}
