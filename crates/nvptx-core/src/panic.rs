use core::arch::nvptx;
use core::ffi::CStr;
use core::panic::PanicInfo;

#[macro_export]
macro_rules! print {
    ($fmt:literal, $($ty:ty),*; $($p:expr),*) => {
        {
            #[repr(C)]
            struct __Fmt($($ty),*);
            print($fmt, &__Fmt($($p),*));
        }
    };
}

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

/// You can use the standard C printf template arguments.
/// `T` must be a `#[repr(C)]` struct.
#[inline]
pub unsafe fn print<T>(fmt: &CStr, params: &T) {
    nvptx::vprintf(fmt.as_ptr().cast(), params as *const T as _);
}
