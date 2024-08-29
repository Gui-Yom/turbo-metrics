# cudarse

General purpose (understand no ML) CUDA bindings for the Driver API and other libraries.

## Libraries

### cudarse-driver

Raw and safeish bindings to the CUDA Driver API. The safeish bindings are designed to not get in
your way, at the expense of dubious safety.

### cudarse-npp

Raw and safe bindings to CUDA NPP libraries. Currently only for image functions, but it comes with a
nice type system.

### cudarse-video

Bindings to Video Codec SDK. Decode video with hardware acceleration on NVidia GPUs.

