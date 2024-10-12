use crate::print;
use core::arch::nvptx;
use core::ffi::CStr;
use core::panic::PanicInfo;

/// You're better off removing any panic places in your kernel code as this adds 3000 lines to the compiled ptx.
#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    unsafe {
        print(c"CUDA code panicked :(", &());
        if let Some(loc) = info.location() {
            let mut buffer = [0; 64];
            let len = loc.file().len().min(63);
            buffer[..len].copy_from_slice(&loc.file().as_bytes()[..len]);
            let str = CStr::from_bytes_until_nul(&buffer).unwrap();

            #[repr(C)]
            struct PanicFmt<'a>(&'a CStr, u32, u32);
            print(
                c" (in %s at %d:%d)",
                &PanicFmt(str, loc.line(), loc.column()),
            );
        }
        print(c"\n", &());
        nvptx::trap();
    }
}
