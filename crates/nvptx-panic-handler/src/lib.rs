#![no_std]
#![feature(stdsimd)]
#![feature(core_intrinsics)]

use core::mem::transmute;
use core::panic::PanicInfo;
use core::ptr::null;

#[repr(C)]
struct PanicFmt<'a>(&'a str, u32, u32);

#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    unsafe {
        core::arch::nvptx::vprintf("CUDA code panicked :(".as_ptr(), null());
        if let Some(loc) = info.location() {
            core::arch::nvptx::vprintf(
                " (in %s at %d:%d)".as_ptr(),
                transmute(&PanicFmt(loc.file(), loc.line(), loc.column())),
            );
        }
        core::arch::nvptx::vprintf("\n".as_ptr(), null());
    }
    core::intrinsics::abort()
}
