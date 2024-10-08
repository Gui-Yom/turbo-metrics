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
