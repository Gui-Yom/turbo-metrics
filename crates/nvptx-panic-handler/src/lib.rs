#![no_std]
#![feature(stdarch_nvptx)]

use core::ffi::CStr;
use core::mem::transmute;
use core::panic::PanicInfo;
use core::ptr::null;

#[repr(C)]
struct PanicFmt<'a>(&'a CStr, u32, u32);

#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    unsafe {
        core::arch::nvptx::vprintf("CUDA code panicked :(".as_ptr(), null());
        if let Some(loc) = info.location() {
            let mut buffer = [0; 64];
            let len = loc.file().len().max(63);
            buffer[..len].copy_from_slice(&loc.file().as_bytes()[..len]);
            let str = CStr::from_bytes_until_nul(&buffer).unwrap();
            core::arch::nvptx::vprintf(
                " (in %s at %d:%d)".as_ptr(),
                transmute(&PanicFmt(str, loc.line(), loc.column())),
            );
        }
        core::arch::nvptx::vprintf("\n".as_ptr(), null());
        core::arch::nvptx::trap();
    }
}
