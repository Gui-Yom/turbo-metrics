# Video Hardware acceleration

My personal cache of Rust video encoding tools and libraries. It's all in a rough state, I'm experimenting with bindings
and safe abstractions.

## Libraries

### nv_video_codec_sdk

Bindings to NVDEC. Decode video with hardware acceleration on NVidia GPUs.

### cuda_npp

Bindings to CUDA NPP libraries.

### ssimulacra2-cuda

Reference implementation : https://github.com/cloudinary/ssimulacra2

Rust implementation : https://github.com/rust-av/ssimulacra2

An attempt at computing the ssimulacra2 metric with GPU acceleration leveraging NPP and custom written kernels in Rust.
Preliminary profiling shows that I'm really bad at writing GPU code that runs fast.

## Prerequisites

- A recent CUDA SDK (tested with 12.3)
- Everything should build with Rust stable, except for the cuda kernel which requires a recent (2024-04-24) nightly Rust
  toolchain

## Planned

- Bindings to oneVPL for hardware accelerated video decoding on Intel platforms.
- Vulkan video decoding library.
- ssimulacra2 on WebGPU or vulkan.
- Bindings to NVENC
