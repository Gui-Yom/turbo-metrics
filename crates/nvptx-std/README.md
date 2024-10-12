# nvptx-core

Core functionality for developing cuda kernels in rust. This crate only contains device code and should only be compiled
with the `nvptx64-nvidia-cuda` target.

## Math std

Some math functions are defined in libstd rather than core (`mul_add`, `cbrt`, ...). This crate defines the extension
trait `StdMathExt` that mimic those functions. The goal is to be able to import code without changing much.

## Linking libdevice

The llvm bitcode linker can't link bitcode files in rlib so we cannot link to libdevice in this library directly. You
need to add the following build script to your kernel crate.

```rust
use std::env;

fn main() {
    let cuda_path =
        env::var("CUDA_PATH").expect("CUDA_PATH must be set to the path of your CUDA installation");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-arg={cuda_path}/nvvm/libdevice/libdevice.10.bc");
}
```
