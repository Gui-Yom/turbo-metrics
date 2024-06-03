# ssimulacra2-cuda-kernel

ssimulacra2 routines implemented in a cuda kernel in Rust. This requires a
recent nightly (2024-04-24) to build with cargo.

Thanks to recent work by @kjetilkjeka
in https://github.com/rust-lang/rust/pull/117458, we can now link crates as llvm
bitcode before emitting ptx.

```shell
rustup +nightly component add llvm-bitcode-linker
# Also requires llvm-tools if you don't have a full llvm toolchain available
rustup +nightly component add llvm-tools
```

The full rustc command :

```shell
rustc +nightly --edition 2021 --crate-name ssimulacra2 --crate-type cdylib --target nvptx64-nvidia-cuda --extern nvptx_panic_handler=../nvptx-panic-handler/libnvptx_panic_handler.rlib src/lib.rs -Z unstable-options -Clinker-flavor=llbc -C opt-level=3 -C target-cpu="sm_60" -C link-arg="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\nvvm\libdevice\libdevice.10.bc"
```

This project has cargo config setup already so there is no need to invoke rustc
directly :

```shell
cargo build --package ssimulacra2-cuda-kernel --release --target nvptx64-nvidia-cuda
```

## Safety

The kernels are unsafe by definition and use unsafe everywhere. There is manual
calculation and checks happening everywhere, which means we're basically just
writing plain C++ code with a fancy syntax.

I recommend using the compute sanitizer tool from the CUDA SDK as it does not
even require recompilation or anything. Just look at its output and see if it
complains.

```shell
compute-sanitizer.bat target\debug\ssimulacra2-cuda.exe
```

## How it used to be

We could not link llvm bitcode directly within rustc, so we had to link it
manually, which means we could not integrate this with cargo.

```shell
rustc +nightly --edition 2021 --emit llvm-bc --crate-type rlib --crate-name ssimulacra2 --target nvptx64-nvidia-cuda --extern nvptx_panic_handler=../nvptx-panic-handler/libnvptx_panic_handler.rlib src/lib.rs -C opt-level=3

C:/apps/LLVM-18/bin/llvm-link ssimulacra2.bc "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\nvvm\libdevice\libdevice.10.bc" -o ssimulacra2.linked.bc
C:/apps/LLVM-18/bin/opt -p "default<O3>,internalize,globaldce" -internalize-public-api-list=plane_srgb_to_linear,linear_to_xyb_packed,downscale_by_2,mul_planes,ssim_map,edge_diff_map ssimulacra2.linked.bc -o ssimulacra2.opt.bc

C:/apps/LLVM-18/bin/llc -O3 -mcpu=sm_30 ssimulacra2.opt.bc -o ssimulacra2.ptx
```

## CUDA device code to llvm bitcode with clang

```shell
clang -S -emit-llvm --cuda-device-only --cuda-gpu-arch=sm_86 --cuda-path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5" shared.cu -o shared.ll
llvm-as shared.ll
```


