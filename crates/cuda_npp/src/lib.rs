use cudarc::driver::{DevicePtr, DevicePtrMut};

pub use cuda_npp_sys as sys;
use cuda_npp_sys::Npp32f;

/// Generic wrappers for npp functions
mod generic;
mod generic2;
/// Wraps the generic wrappers to provide a Result return type
mod result;
/// Safer abstractions
mod safe;
/// Safeish wrapper over the result layer
mod safeish;

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
}

/// Image sample type
pub trait Sample: __priv::PrivateSupertrait + 'static {
    const LUT_INDEX: usize;
}

/// Image sample type restricted to u8
pub trait SampleU8: Sample {}

impl SampleU8 for u8 {}

impl __priv::PrivateSupertrait for u8 {}

impl __priv::PrivateSupertrait for f32 {}

impl Sample for u8 {
    const LUT_INDEX: usize = 0;
}

impl Sample for f32 {
    const LUT_INDEX: usize = 1;
}

/// Layout of the image, either packed channels or planar
pub trait ChannelLayout: __priv::PrivateSupertrait + 'static {
    const LUT_INDEX: usize;
}

/// Marker trait to restrict channel layout to packed channels.
pub trait ChannelLayoutPacked: ChannelLayout {}

/// Packed channels
struct C<const N: usize>;

/// Planar channels
struct P<const N: usize>;

impl<const N: usize> __priv::PrivateSupertrait for C<N> {}

impl<const N: usize> __priv::PrivateSupertrait for P<N> {}

impl ChannelLayout for C<1> {
    const LUT_INDEX: usize = 0;
}

impl ChannelLayout for C<3> {
    const LUT_INDEX: usize = 1;
}

impl ChannelLayout for C<4> {
    const LUT_INDEX: usize = 2;
}

impl ChannelLayout for P<3> {
    const LUT_INDEX: usize = 3;
}

impl ChannelLayout for P<4> {
    const LUT_INDEX: usize = 4;
}

impl ChannelLayoutPacked for C<1> {}

impl ChannelLayoutPacked for C<3> {}

impl ChannelLayoutPacked for C<4> {}

// trait Storage<S: Sample> {}
//
// impl<S: Sample> Storage<S> for Vec<S> {}

struct Device;

struct Host;
