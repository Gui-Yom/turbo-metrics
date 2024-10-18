# turbo-metrics

CLI to process a pair of videos or images and compute various metrics and statistics.

---

Included metrics :

- PSNR
- SSIM
- MSSSIM
- SSIMULACRA2

Supported video containers :

- MKV
- IVF

Supported video codecs :

- AV1
- AVC/H.264
- MPEG-2 Part 2/H.262

Supported image codecs :

- PNG
- JPEG
- JPEG-XL
- AVIF* (8 bits only, requires libdav1d)
- Webp*
- QOI*
- GIF*
- TIFF*

\* _feature turned off by default_

Build a release binary with `cargo build --release -p turbo-metrics --features static`. Start with
`turbo-metrics --help`.

## Usage

```text
Usage: turbo-metrics [OPTIONS] <REFERENCE> <DISTORTED>

Arguments:
  <REFERENCE>
          Reference media

  <DISTORTED>
          Distorted media. Use '-' to read from stdin

Options:
  -m, --metrics <METRICS>
          Select the metrics to compute, selecting many at once will reduce overhead because the video will only be decoded once

          Possible values:
          - psnr:        PSNR computed with NPP in linear RGB
          - ssim:        SSIM computed with NPP in linear RGB
          - msssim:      MSSSIM computed with NPP in linear RGB
          - ssimulacra2: SSIMULACRA2 computed with CUDA

      --every <EVERY>
          Only compute metrics every few frames, effectively down-sampling the measurements. Still, this tool will decode all frames, hence increasing overhead. Check Mpx/s to see what I mean.

          E.g. 8 invocations with --every 8 will perform around 50% worse than a single pass computing every frame.

          [default: 0]

      --skip <SKIP>
          Index of the first frame to start computing at. Useful for overlaying separate computations with `every`

          [default: 0]

      --skip-ref <SKIP_REF>
          Index of the first frame to start computing at the reference frame. Additive with `skip`

          [default: 0]

      --skip-dis <SKIP_DIS>
          Index of the first frame to start computing at the distorted frame. Additive with `skip`

          [default: 0]

      --frame-count <FRAME_COUNT>
          Amount of frames to compute. Useful for computing subsets with `skip`, `skip-ref`, and `skip-dis`

          [default: 0]

      --output <OUTPUT>
          Choose the CLI stdout format. Omit the option for the default. Status messages will be printed to stderr in all cases

          Possible values:
          - default: Default classic output for human reading. This won't print the score for each individual frames
          - json:    Json object output. Contains both per-frame scores and aggregated stats
          - csv:     CSV output. Only contains per-frame scores

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

## Example output

```shell
$ turbo-metrics.exe --ssimulacra2 ref.mkv dis.mkv
Using device NVIDIA GeForce RTX 4070 with CUDA version 12060
Reference: H262, 720x576, CP: BT601_625, MC: BT601_625, TC: BT709, Full range: false
Distorted: AV1, 720x576, CP: BT601_625, MC: BT601_625, TC: BT709, Full range: false
Initializing SSIMULACRA2
Initialized, now processing ...
Decoded: 109935, processed: 109935 frame pairs in 164314 ms (669 fps) (Mpx/s: 277.470)
Stats :
ssimulacra2: Stats {
    min: 35.734578404505434,
    max: 99.9939529862985,
    mean: 80.1701991776592,
    var: 23.09975783182373,
    sample_var: 23.099967955696524,
    stddev: 4.806220743143591,
    sample_stddev: 4.806242602667548,
    p1: 65.69003946045632,
    p5: 71.74814773321043,
    p50: 80.70284796735427,
    p95: 86.67674900306754,
    p99: 91.01382044601559,
}
```
