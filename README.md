# TurboMetrics

A collection of video related libraries and tools oriented at performance and hardware acceleration.
Including :

- [cudarse](crates/cudarse) : General purpose (no ML) CUDA bindings for the Driver API,
  NPP and the Video Codec SDK.
- A workflow and its tools to develop CUDA kernels in Rust just as another crate in the workspace.
- A working [ssimulacra2](https://github.com/cloudinary/ssimulacra2)
  implementation with CUDA.
- Utilities and foundational libraries for codec bitstream demuxing and statistics.
- Kernels for colorspace conversion and linearization.

---

## Goal

This project started as me noticing my GPU usage at 0% while my CPU was overloaded while doing video
processing.

The strategy is to offload as much work as possible onto the GPU :

1. Demux a video file on the CPU
2. Decode the bitstream on hardware and keep the frame in CUDA memory
3. Do any costly processing on the frames (IQA, postprocessing ...) using the GPU
4. Get the results back to the CPU

In some instances, it would be impossible to decode the frame on the GPU, which means one has to
stream decoded frames from the CPU (e.g. image formats), this would reduce performance but
still be faster than full CPU processing if the frames can stay in gpu memory long enough.

## Subprojects

### turbo-metrics

CLI to process a pair of videos or images and compute various metrics and statistics.
Available [here](crates/turbo-metrics-cli/README.md).

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

### cuda-colorspace

Colorspace conversion CUDA kernels used in other crates.

### ssimulacra2-cuda

An attempt at computing the ssimulacra2 metric with GPU acceleration leveraging NPP and custom
written kernels written in Rust. Preliminary profiling shows that I'm terrible at writing GPU code
that runs fast.

Reference implementation : https://github.com/cloudinary/ssimulacra2

### vmaf

Bindings to libvmaf.

## Prerequisites

This repository is particularly difficult to set up for a Rust project due to the dependencies on
various vendor SDKs.
You need to be patient and be able to read error message from builds.

Also, it uses a novel approach enabled by recent rustc developments to colocate CUDA kernels written
in Rust within the same cargo workspace. This is very much bleeding edge and the way the crates are
linked together prevent publishing to crates.io. The only supported way to build any crate in this
repo is by cloning the git repo.

### Common

- 64-bit system.
- CUDA 12.x (tested with 12.5 and 12.6, it might work with previous versions, I don't know)
- CUDA NPP (normally packaged with CUDA by default, but it's optional component)
- Rust nightly (because of CUDA kernels and unstable Cargo features)
- Various rustup components for the nightly channel :
  ```shell
  rustup +nightly target add nvptx64-nvidia-cuda
  rustup +nightly component add llvm-bitcode-linker
  rustup +nightly component add llvm-tools
  ```
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) (need
  headers only on Linux, full sdk on Windows) with the `NV_VIDEO_CODEC_SDK` env var
- For the AMF backend : AMD AMF SDK headers
- (in progress) For the libvmaf bindings : libvmaf
- libclang available somewhere (that's for bindgen)

### Windows

- Tested on Windows 10, but should work elsewhere.
- `CUDA_PATH` env var pointing to your CUDA install
- `AMF_SDK_PATH` env var pointing to your AMF SDK install
- NPP dlls can't be built statically in the resulting binary and must be redistributed with whatever
  binary that depends on it.

### Linux

- Tested on Fedora 41 with proprietary Nvidia drivers (I do not think CUDA works with Nouveau ?)
- `CUDA_PATH` env var is optional, by default it will look in `/usr/local/cuda`.
- `AMF_SDK_PATH` env var is optional, by default it will look in `/usr/include/AMF` as AMF headers
  were present in my system packages.
- I need to link to `libstdc++` for NPP libraries, but it should be possible to use `libc++`
  instead.

## Support this project

There are various ways you can support development.

- File a detailed issue when you encounter a problem
- Support me through [ko-fi](https://ko-fi.com/ganthouard)
  or [GitHub Sponsors](https://github.com/sponsors/Gui-Yom)

## TODO ideas

The base is solid. I plan to implement various tools to help the process of making encodes (except
encoding itself) from pre-filtering to validation. In no particular order or priority :

### Tools & workflows

- GUI with plots and interactive usage
- GUI for interactive inspection of error maps
- Hull generation, by running a command automatically (e.g.
  `turbo-metrics --ssimulacra2 --hull --ref ref.png -- avifenc ref.png --crf @`)

### Algorithms implementations

- XPSNR
- Butteraugli
- VMAF (using both libvmaf and a custom CUDA impl)
- Scene detection (histogram based should be easy)
- Scene detection like the one used in rav1e (not even sure that's possible on a GPU)
- Denoising algorithms (the usual ones in vapoursynth are fucking slow, maybe putting the whole
  processing chain on the GPU can help, needs more research)
- New ssimulacra2 implementation, without relying on NPP and with separate planes computations.
- NVflip
- Audio metrics ? I don't know much about those

### Inputs

- Region selection
- More video containers (mp4)
- Raw bitstreams
- More codecs
- Finish implementing useful colorspaces
- libavcodec input so everything is supported
- CPU decoder fallback
- Integrations with other tools ?

### Outputs

- Parseable stdout
- JSON output
- CSV output
- Plot output

### Platform support

Currently, we're locked to Nvidia hardware. However, nothing here explicitly requires CUDA.

- Other hardware video decoding API.
- Other accelerated compute platforms (krnl, cubecl, Vulkan).
- libavcodec input might help a lot since everything is already implemented.

## About video hardware acceleration

Processing videos efficiently is a 2 parts problem :

### Video decoding

So you want a cross-platform way to decode videos on every possible platform ? Sadge. This is a
mess, there are nearly as many different api as there are hw vendors, os and gpu apis.

Recap table :

| API          | Windows  | Linux | Nvidia    | Intel | AMD | AV1 | HEVC | AVC | MPEG2 | VC1 |
|--------------|----------|-------|-----------|-------|-----|-----|------|-----|-------|-----|
| NVDEC        | ✅        | ✅     | ✅         | ❌     | ❌   | ✅   | ✅    | ✅   | ✅     | ✅   |
| VPL          | ✅        | ✅     | ❌         | ✅     | ❌   | ✅   | ✅    | ✅   |       |     |
| AMF          | ✅        | ✅     | ❌         | ❌     | ✅   | ✅   | ✅    | ✅   |       |     |
| DXVA         | ✅        |       | ✅         | ✅     | ✅   |     |      | ✅   | ✅     |     |
| Vulkan Video | ✅        | ✅     | ✅         | ✅     | ✅   | ✅   | ✅    | ✅   |       |     |
| VAAPI        | 🟦vaon12 | ✅     | 🟦Nouveau | ✅     |     | ✅   | ✅    | ✅   |       |     |
| VDPAU        |          | ✅     | ✅         |       |     |     |      |     |       |     |

There is still the option to decode video on the CPU and stream frames to the GPU for computations.
This is still faster than doing all processing on the CPU alone.

### Compute

Your GPU will blow your CPU on any image processing task. Processing frames on the GPU is the best
thing that can be done for speed.

Recap table :

| API      | Windows | Linux | Intel | AMD     | Nvidia | NVDEC | VPL | AMF | Vulkan Video | CPU-side Rust | GPU-side Rust |
|----------|---------|-------|-------|---------|--------|-------|-----|-----|--------------|---------------|---------------|
| CUDA     | ✅       | ✅     |       | 🟦ZLUDA | ✅      | ✅     |     |     |              | ✅             | ✅llvm ptx     |
| Vulkan   | ✅       | ✅     | ✅     | ✅       | ✅      | ✅     |     | ✅   | ✅            | ✅             | ✅Spir-V       |
| OpenCL   | ✅       | ✅     | ✅     | ✅       | ✅      |       |     |     |              | ✅             | ✅Spir-V       |
| ROCm/HIP | ✅       | ✅     |       | ✅       |        |       |     |     |              |               |               |
| WGPU     | ✅       | ✅     | ✅     | ✅       | ✅      |       |     |     | ✅            | ✅             | ✅Spir-V       |

From both those tables, it seems Vulkan and Vulkan Video are the way forward but well, it's Vulkan.
