use cudarc::driver::{DevicePtr, DevicePtrMut};

pub use cuda_npp_sys as sys;
use cuda_npp_sys::{nppGetStreamContext, NppStatus, NppStreamContext};

/// Emulated npp functions for some missing sample/channel count combinations.
//pub mod emulated;
/// Generic wrappers for npp functions (using macros and [std::any::TypeId])
//pub mod generic;
/// Generic wrappers for npp functions (using macros and const LUT)
//pub mod generic_lut;
/// Safer abstractions
pub mod safe;

/// Safeish wrapper over the result layer
//pub mod safeish;

pub type NppResult<T> = Result<T, NppStatus>;

mod __priv {
    /// So people don't implement child trait for themselves.
    /// Hacky way of doing closed polymorphism with traits.
    pub trait PrivateSupertrait {}

    impl PrivateSupertrait for u8 {}

    impl PrivateSupertrait for u16 {}

    impl PrivateSupertrait for i16 {}

    impl PrivateSupertrait for i32 {}

    impl PrivateSupertrait for f32 {}
}

/// Image sample type
pub trait Sample: __priv::PrivateSupertrait + 'static {
    const LUT_INDEX: usize;
}

const SAMPLE_LUT_SIZE: usize = 5;

/// Image sample type restricted to u8
pub trait SampleU8: Sample {}

impl SampleU8 for u8 {}

pub trait SampleResize: Sample {
    const LUT_INDEX: usize;
}

const SAMPLE_RESIZE_LUT_SIZE: usize = 4;

impl Sample for u8 {
    const LUT_INDEX: usize = 0;
}

impl Sample for u16 {
    const LUT_INDEX: usize = 1;
}

impl Sample for i16 {
    const LUT_INDEX: usize = 2;
}

impl Sample for i32 {
    const LUT_INDEX: usize = 3;
}

impl Sample for f32 {
    const LUT_INDEX: usize = 4;
}

impl SampleResize for u8 {
    const LUT_INDEX: usize = 0;
}

impl SampleResize for u16 {
    const LUT_INDEX: usize = 1;
}

impl SampleResize for i16 {
    const LUT_INDEX: usize = 2;
}

impl SampleResize for f32 {
    const LUT_INDEX: usize = 3;
}

/// Layout of the image, either packed channels or planar
pub trait Channel: __priv::PrivateSupertrait + 'static {
    const LUT_INDEX: usize;
}

const CHANNEL_LAYOUT_LUT_SIZE: usize = 6;

/// Marker trait to restrict channel layout to packed channels.
pub trait ChannelPacked: Channel {
    const LUT_INDEX: usize;
}

const CHANNEL_PACKED_LUT_SIZE: usize = 4;

pub trait ChannelResize: ChannelPacked {
    const LUT_INDEX: usize;
}

const CHANNEL_RESIZE_LUT_SIZE: usize = 3;

pub trait ChannelSet: ChannelPacked {
    const LUT_INDEX: usize;
}

const CHANNEL_SET_LUT_SIZE: usize = 3;

/// Packed channels
struct C<const N: usize>;

/// Planar channels
struct P<const N: usize>;

impl<const N: usize> __priv::PrivateSupertrait for C<N> {}

impl<const N: usize> __priv::PrivateSupertrait for P<N> {}

impl Channel for C<1> {
    const LUT_INDEX: usize = 0;
}

impl Channel for C<2> {
    const LUT_INDEX: usize = 1;
}

impl Channel for C<3> {
    const LUT_INDEX: usize = 2;
}

impl Channel for C<4> {
    const LUT_INDEX: usize = 3;
}

impl Channel for P<3> {
    const LUT_INDEX: usize = 4;
}

impl Channel for P<4> {
    const LUT_INDEX: usize = 5;
}

impl ChannelPacked for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelPacked for C<2> {
    const LUT_INDEX: usize = 1;
}

impl ChannelPacked for C<3> {
    const LUT_INDEX: usize = 2;
}

impl ChannelPacked for C<4> {
    const LUT_INDEX: usize = 3;
}

impl ChannelResize for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelResize for C<3> {
    const LUT_INDEX: usize = 1;
}

impl ChannelResize for C<4> {
    const LUT_INDEX: usize = 2;
}

impl ChannelSet for C<2> {
    const LUT_INDEX: usize = 0;
}

impl ChannelSet for C<3> {
    const LUT_INDEX: usize = 1;
}

impl ChannelSet for C<4> {
    const LUT_INDEX: usize = 2;
}

pub fn get_stream_ctx() -> NppResult<NppStreamContext> {
    let mut ctx = Default::default();
    unsafe {
        let status = nppGetStreamContext(&mut ctx);
        if status == NppStatus::NPP_NO_ERROR {
            Ok(ctx)
        } else {
            Err(status)
        }
    }
}
