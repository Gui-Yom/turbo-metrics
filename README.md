# Video Hardware acceleration

My personal cache of Rust video encoding tools and libraries. It's all in a rough state, I'm experimenting with bindings
and safe abstractions.

## Libraries

### cuda-driver

Raw and safeish bindings to the CUDA Driver API. The safeish bindings are designed to not get in your way, at the
expense of dubious safety.

### cuda_npp

Raw and safe bindings to CUDA NPP libraries. Currently only for image functions, but it comes with a nice type system.

### nv_video_codec_sdk

Bindings to NVDEC. Decode video with hardware acceleration on NVidia GPUs.

### nvptx-core

Helper library to write CUDA kernels in Rust. Acts as some kind of libstd for the nvptx64-nvidia-cuda target. Provides a
math extension trait to replace std based on bindings to libdevice.

### ssimulacra2-cuda

An attempt at computing the ssimulacra2 metric with GPU acceleration leveraging NPP and custom written kernels written
in Rust.
Preliminary profiling shows that I'm terrible at writing GPU code that runs fast.

Reference implementation : https://github.com/cloudinary/ssimulacra2

Rust implementation : https://github.com/rust-av/ssimulacra2

## Prerequisites

- A recent CUDA SDK (tested with 12.5)
- Everything should build with Rust stable, except for the cuda kernel which requires a recent (2024-04-24) nightly Rust
  toolchain

## Planned

- Bindings to oneVPL for hardware accelerated video decoding on Intel platforms.
- Vulkan video decoding library.
- ssimulacra2 on WebGPU or vulkan.
- Bindings to NVENC
