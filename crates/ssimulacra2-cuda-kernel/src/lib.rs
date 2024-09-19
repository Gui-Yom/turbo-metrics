#![no_std]
#![feature(stdarch_nvptx)]
#![feature(abi_ptx)]
#![feature(asm_experimental_arch)]
#![feature(array_ptr_get)]
#![feature(core_intrinsics)]

mod blur;
mod downscale;
mod error_maps;
mod srgb;
mod xyb;
