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

## Example

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
