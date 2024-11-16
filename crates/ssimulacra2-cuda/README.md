# ssimulacra2-cuda

An implementation of ssimulacra2 using CUDA.

## Features

- Tries to stay close to the original implementation for comparable results.
- Leverages custom kernels written in Rust and CUDA NPP primitives.
- Uses CUDA graphs to reduce the cost of launching 200+ kernels per image pair.

## TODO

- Investigate if it is possible to change some computations to accelerate processing without
  deviating from the original implementation too much. Maybe making it configurable.
- More custom kernels, is it possible to run the whole computation in a single fused kernel launch ?
- Use less memory (currently 500MB for 1080p), might be possible by using fused kernels.

## Credits

Original reference implementation : https://github.com/cloudinary/ssimulacra2

With inspiration from : https://github.com/rust-av/ssimulacra2

## Computing ssimulacra2

ssimu2 was designed for still images, there is no temporal dependency. Scores are computed for each
reference-distorted pair independently.

1. Get a reference-distorted pair
2. Convert the frames to linear rgb (with f32 samples)
3. Then for each scale (6) :
    1. Downscale by the scale number (or downscale the previous frames). The first iteration
       computes at full scale.
    2. Convert the frames to XYB
    3. Blur ref \* ref, dis \* dis, ref \* dis, ref, dis using a recursive Gaussian blur
    4. Compute 1-ssim, artifact and detail_loss error scores from the 5 blurred frames and the
       original frame pair. We get 3 error maps.
    5. Error maps are reduced to a single number using the 1-norm and 4-norm.
4. From this, we get 6 scores for each scale (6) and color component (3). (total=108)
5. Each score is weighted and the resulting values are summed.
6. The final value is processed through a non-linear function and clipped to render a score between
   100 and -inf.

## Differences from the reference implementation

The steps outlined above are respected, but there are notable differences between the way the
individual computations are written. The scores differences can be explained by :

- The conversion from yuv to linear rgb might not yield the same results as other tools (garbage in,
  garbage out)

- I used explicit FMA (mul_add in rust) when possible, the GPU likes that.
- The order of operations might not be the same, some calculations have been rearranged
- floating point operations might not yield the same result at all (especially with approximated
  math functions like powf)
- Weights were computed by fitting the error scores to MOS, deviation in the error scores is
  amplified by weights.
- The final non-linear function is rather steep (x^3), score variations are amplified again.

## How to do better ?

Outside the scope of this impl (that's the user responsibility) : validate yuv to linear rgb
conversion for correctness compared to other implementations.

I don't think it's possible to achieve close similarity in the scores so that it becomes
negligible for all purposes (this is inherent to an implementation on another platform).
I also think that trying to get better scores get in the way of trying to optimize the
implementation for speed.

I think the way forward is another implementation purely optimized for GPU computations with
redesigned weights (using the same method as the original impl).

The Cuda implementation can probably be sped up by a good bit (something around 2-3x faster).

- Separated planes for sparse computations (do not compute for planes where the weight is 0)
- Designing weights to minimize the number of computations (see previous point)
- Fused compute kernels
- Blur algorithm optimized for GPU (using convolutions instead of the recursive blurring)
- Approximations / fast math (this is fine if the weights are designed for it)
- Launch parameters optimization (cuda grid and block layout) per architecture
