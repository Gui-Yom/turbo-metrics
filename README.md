# TurboMetrics

A collection of video related libraries and tools oriented at performance and hardware acceleration.
Including :

- [cudarse](crates/cudarse) : General purpose (understand no ML) CUDA bindings for the Driver API,
  NPP and the Video Codec SDK.
- A working [ssimulacra2](https://github.com/cloudinary/ssimulacra2)
  implementation with CUDA.
- In progress bindings to libvmaf
- In progress implementation of vmaf
- Utilities and foundational libraries for codec bitstream demuxing and statistics.

## Goal

The goal is to make things go fast by powering up the RTX 3060 in my laptop instead of burning my
CPU.

Best case is :

1. Demux a video file on the CPU
2. Decode the bitstream on hardware and keep the frame in CUDA memory
3. Do any costly processing on the frames (IQA, postprocessing ...) using the GPU
4. Get the results back to the CPU

In some instances, it would be impossible to decode the frame on the GPU, which means one has to
stream decoded frames from the CPU (PNG or unsupported codec).

## Libraries

### cudarse

[Here](crates/cudarse)

### codec-bitstream

Transform codec bitstream for feeding into GPU decoders.

### nvptx-core

Nightly only helper library to write CUDA kernels in Rust. Acts as some kind of libstd for the
nvptx64-nvidia-cuda target. Provides a math extension trait to replace std based on bindings to
libdevice.

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

## Prerequisites

- Only tested on Windows 10 x64
- A recent [CUDA SDK](https://developer.nvidia.com/cuda-toolkit) (tested with 12.5)
- Everything should build with Rust stable, except for the cuda kernel which requires a recent (at
  least 2024-04-24) nightly Rust toolchain
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

## Planned

- Move cuda bindings and into their own repository. It's easier to develop in-tree for now.
- Scene detection for AV1 using CUDA (adapting rav1e scene detection on the gpu)
- Reimplement VMAF in Rust
- Intel hardware decoding
- Vulkan video decoding
- ssimulacra2 on WebGPU or vulkan (not locked to CUDA)
- NVENC for encoding using the GPU
