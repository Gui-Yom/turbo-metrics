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

An attempt at computing the ssimulacra2 metric with hardware acceleration on NVidia GPU.

## Planned

- Bindings to oneVPL for hardware accelerated video decoding on Intel platforms.
- Vulkan video decoding library.
- ssimulacra2 on WebGPU or vulkan.
- Bindings to NVENC
