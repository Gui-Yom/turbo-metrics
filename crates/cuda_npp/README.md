# cuda_npp

Not that safe bindings to the CUDA NPP library.

## Safety

The API is relatively safe to use, as long as you remember cuda calls are asynchronous. I try to always use stream
ordered functions, but you must remember to synchronize the stream before the end of your function (or before things can
get dropped).

## Links

https://docs.nvidia.com/cuda/npp/image_color_conversion.html#color-gamma-correction

https://docs.nvidia.com/cuda/npp/introduction.html

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
