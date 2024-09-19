#![no_std]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

mod helpers;
/// Replacements for some math functions that are only present in std using libdevice.
pub mod math;
mod panic;

pub mod prelude {
    pub use crate::helpers::*;
    pub use crate::math::*;
    pub use crate::panic::*;
}
