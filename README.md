# Video Hardware acceleration

My personal cache of Rust video encoding tools and libraries. It's all in a rough state, I'm experimenting with bindings
and safe abstractions.

## Goal

The goal is to make things go fast by powering up the RTX 3060 in my laptop instead of burning my CPU.

Best case is :

1. Demux a video file on the CPU
2. Decode the bitstream on hardware and keep the frame in CUDA memory
3. Do any costly processing on the frames (IQA, postprocessing ...) using the GPU
4. Get the results back to the CPU

In some instances, it would be impossible to decode the frame on the GPU, which means one has to stream decoded frames
from the CPU (PNG or unsupported codec).

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

### ssimulacra2-cuda-cli

CLI frontend for ssimulacra2-cuda that can compare full videos blazingly fast by decoding on the GPU.

At the time of writing, I'm maxing out my RTX 3060 laptop compute while using 45% of NVDEC total capacity.
With two h264 videos, I can process around 110 image pairs per second (which is more than 4x speedup over realtime).

## Prerequisites

- A recent [CUDA SDK](https://developer.nvidia.com/cuda-toolkit) (tested with 12.5)
- Everything should build with Rust stable, except for the cuda kernel which requires a recent (at least 2024-04-24)
  nightly Rust toolchain
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)

## Planned

- Move cuda bindings and cuda-npp into its own repository. It's easier to develop in-tree for now.
- Scene detection for AV1 using CUDA (adapting rav1e scene detection on the gpu)
- Reimplement VMAF in Rust
- Intel hardware decoding
- Vulkan video decoding
- ssimulacra2 on WebGPU or vulkan (not locked to CUDA)
- NVENC for encoding using the GPU
