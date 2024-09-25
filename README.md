# TurboMetrics

A collection of video related libraries and tools oriented at performance and hardware acceleration.
Including :

- [cudarse](crates/cudarse) : General purpose (no ML) CUDA bindings for the Driver API,
  NPP and the Video Codec SDK.
- A workflow and its tools to develop CUDA kernels in Rust just as another crate in the workspace.
- A working [ssimulacra2](https://github.com/cloudinary/ssimulacra2)
  implementation with CUDA.
- In progress bindings to libvmaf and in progress implementation of vmaf
- Utilities and foundational libraries for codec bitstream demuxing and statistics.

Nearly the whole stack here is custom-built from the CUDA headers and a matroska demuxer.

## Goal

The goal is to make things go fast by powering up the RTX 3060 in my laptop instead of burning my
CPU.

Best case is :

1. Demux a video file on the CPU
2. Decode the bitstream on hardware and keep the frame in CUDA memory
3. Do any costly processing on the frames (IQA, postprocessing ...) using the GPU
4. Get the results back to the CPU

In some instances, it would be impossible to decode the frame on the GPU, which means one has to
stream decoded frames from the CPU (PNG or unsupported codec), this would reduce performance but
still be faster than full CPU processing if the frames can stay in gpu memory long enough.

## Libraries

### cudarse

[Here](crates/cudarse)

### codec-bitstream

Transform codec bitstream for feeding into GPU decoders. Also provides parsing for metadata like
color information.

### nvptx-core

Nightly only helper library to write CUDA kernels in Rust. Acts as some kind of libstd for the
nvptx64-nvidia-cuda target. Provides a math extension trait to replace std based on bindings to
libdevice.

### nvptx-builder

Allows a crate to define a dependency on a nvptx crate and have it built with a single
`cargo build`.

### ssimulacra2-cuda

An attempt at computing the ssimulacra2 metric with GPU acceleration leveraging NPP and custom
written kernels written in Rust. Preliminary profiling shows that I'm terrible at writing GPU code
that runs fast.

Reference implementation : https://github.com/cloudinary/ssimulacra2

Rust implementation : https://github.com/rust-av/ssimulacra2

### vmaf

Bindings to libvmaf.

## Tools

### ssimulacra2-cuda-cli

CLI frontend for ssimulacra2-cuda that can compare videos blazingly fast by decoding on the GPU and
computing on the GPU.

At the time of writing, I'm maxing out my RTX 3060 laptop compute while using 45% of NVDEC total
capacity. With two h264 videos, I can process around 110 image pairs per second (which is more than
4x speedup over realtime and orders of magnitude over existing tools).

### turbo-metrics

CLi to process a pair of videos and compute various metrics and statistics.
Included metrics :

- PSNR
- SSIM
- MSSSIM
- SSIMULACRA2

Supported containers :

- MKV

Supported video codecs :

- AV1
- AVC/H.264
- MPEG-2 Part 2/H.262

Build with `cargo build --release -p turbo-metrics`. Start with `turbo-metrics --help`.

## Prerequisites

- Only tested on Windows 10 x64, but should work elsewhere
- A recent [CUDA SDK](https://developer.nvidia.com/cuda-toolkit) (tested with 12.5), this project
  uses the `CUDA_PATH` env var, please make sure it is available and correct.
- Everything should build with Rust stable, except for the cuda kernel which requires a recent (at
  least 2024-04-24) nightly Rust toolchain
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

```shell
rustup toolchain install stable
rustup toolchain install nightly
rustup +nightly target add nvptx64-nvidia-cuda
rustup +nightly component add llvm-bitcode-linker
rustup +nightly component add llvm-tools
```

## Planned

- Remove usage of CUDA NPP as it's impossible to link the libraries statically on windows. It's also
  responsible for most of the code bloat.
- New improved ssimulacra2 computation leveraging separated planes and fat kernels. (requires the
  previous point)
- Move cuda bindings and into their own repository. It's easier to develop in-tree for now.
- Scene detection for AV1 using CUDA (adapting rav1e scene detection on the gpu). Not even sure
  that's possible.
- Other forms of scene detection like histogram analysis.
- VMAF implementation
- ~~Intel hardware decoding~~, AMD hardware decoding, Vulkan video decoding
- ssimulacra2 using wgpu or vulkan (not locked to CUDA), OpenCL ?
- NVENC for encoding using the GPU
