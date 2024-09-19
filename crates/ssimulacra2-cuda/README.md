# ssimulacra2-cuda

An implementation of ssimulacra2 using CUDA.

## Features

- Close to the original implementation, and with close results.
- Leverages many custom kernels written in Rust and a few CUDA NPP primitives.
- Uses CUDA graphs to alleviate the cost of launching the 200+ kernels per image pair.

## TODO

- Investigate if it is possible to change some computations to accelerate processing without
  deviating from the original implementation too much. Maybe making it configurable.
- More custom kernels, is it possible to run the whole computation in a single fused kernel launch ?
- Use less memory (currently 500MB for 1080p), might be possible by using a single fused kernel.

## Credits

Original reference implementation : https://github.com/cloudinary/ssimulacra2

With inspiration from : https://github.com/rust-av/ssimulacra2
