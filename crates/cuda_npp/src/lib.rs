use cudarc::driver::{DevicePtr, DevicePtrMut};

pub use cuda_npp_sys as sys;
use cuda_npp_sys::NppStatus;

/// Emulated npp functions for some missing sample/channel count combinations.
pub mod emulated;
/// Generic wrappers for npp functions (using macros and [std::any::TypeId])
pub mod generic;
/// Generic wrappers for npp functions (using macros and const LUT)
pub mod generic_lut;
/// Safer abstractions
pub mod safe;
/// Safeish wrapper over the result layer
pub mod safeish;

pub type NppResult<T> = Result<T, NppStatus>;

// pub fn npp_gauss_5x5(
//     dev: &Arc<CudaDevice>,
//     src: &CudaSlice<f32>,
//     dst: &mut CudaSlice<f32>,
//     width: u32,
//     height: u32,
// ) {
//     unsafe {
//         println!(
//             "{:#p}",
//             src.device_ptr() as *const cudarc::driver::sys::CUdeviceptr as *const Npp32f
//         );
//         println!(
//             "{:#p}",
//             dst.device_ptr_mut() as *mut cudarc::driver::sys::CUdeviceptr as *mut Npp32f
//         );
//         let mut ctx = Default::default();
//         nppGetStreamContext(&mut ctx);
//         let res = nppiFilterGauss_32f_C1R_Ctx(
//             src.device_ptr() as *const cudarc::driver::sys::CUdeviceptr as *const Npp32f,
//             (width * mem::size_of::<f32>() as u32) as i32,
//             dst.device_ptr_mut() as *mut cudarc::driver::sys::CUdeviceptr as *mut Npp32f,
//             (width * mem::size_of::<f32>() as u32) as i32,
//             NppiSize {
//                 width: width as c_int,
//                 height: height as c_int,
//             },
//             NppiMaskSize::NPP_MASK_SIZE_5_X_5,
//             ctx,
//         );
//         dbg!(res);
//     }
// }

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
pub trait ChannelLayout: __priv::PrivateSupertrait + 'static {
    const LUT_INDEX: usize;
}

const CHANNEL_LAYOUT_LUT_SIZE: usize = 6;

/// Marker trait to restrict channel layout to packed channels.
pub trait ChannelLayoutPacked: ChannelLayout {
    const LUT_INDEX: usize;
}

const CHANNEL_LAYOUT_PACKED_LUT_SIZE: usize = 4;

pub trait ChannelLayoutResizePacked: ChannelLayoutPacked {
    const LUT_INDEX: usize;
}

const CHANNEL_LAYOUT_RESIZE_PACKED_LUT_SIZE: usize = 3;

/// Packed channels
struct C<const N: usize>;

/// Planar channels
struct P<const N: usize>;

impl<const N: usize> __priv::PrivateSupertrait for C<N> {}

impl<const N: usize> __priv::PrivateSupertrait for P<N> {}

impl ChannelLayout for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelLayout for C<2> {
    const LUT_INDEX: usize = 1;
}

impl ChannelLayout for C<3> {
    const LUT_INDEX: usize = 2;
}

impl ChannelLayout for C<4> {
    const LUT_INDEX: usize = 3;
}

impl ChannelLayout for P<3> {
    const LUT_INDEX: usize = 4;
}

impl ChannelLayout for P<4> {
    const LUT_INDEX: usize = 5;
}

impl ChannelLayoutPacked for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelLayoutPacked for C<2> {
    const LUT_INDEX: usize = 1;
}

impl ChannelLayoutPacked for C<3> {
    const LUT_INDEX: usize = 2;
}

impl ChannelLayoutPacked for C<4> {
    const LUT_INDEX: usize = 3;
}

impl ChannelLayoutResizePacked for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelLayoutResizePacked for C<3> {
    const LUT_INDEX: usize = 1;
}

impl ChannelLayoutResizePacked for C<4> {
    const LUT_INDEX: usize = 2;
}
