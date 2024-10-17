use crate::kernel::Kernel;
use cudarse_driver::CuStream;
use cudarse_npp::image::idei::Convert;
use cudarse_npp::image::if_::Filter;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};
use cudarse_npp::sys::NppiSize;
use cudarse_npp::{get_stream_ctx, ScratchBuffer};
use std::ops::Add;

mod kernel;

pub struct Xpsnr {
    kernel: Kernel,
    /// Number of blocks along width
    blocks_w: usize,
    /// Number of blocks along height
    blocks_h: usize,
    /// Reference image
    ref_: Image<u8, C<1>>,
    /// Reference image converted to 16bit
    ref_16: Image<i16, C<1>>,
    /// Previous reference image
    prev: Image<u8, C<1>>,
    /// Distorted image
    dis: Image<u8, C<1>>,
    /// Reference image after a highpass filter
    highpass: Image<i16, C<1>>,
    /// Device buffer to store sse values computed per block
    sse_dev: ScratchBuffer,
    /// Device buffer to store spatial activity values computed per block
    sact_dev: ScratchBuffer,
    tact_dev: ScratchBuffer,
    wsse_dev: ScratchBuffer,
    sse: Vec<u32>,
    sact: Vec<u32>,
    tact: Vec<u32>,
    highpass_coeffs: ScratchBuffer,
    weights: Vec<f64>,
}

impl Xpsnr {
    pub fn new(width: u32, height: u32, stream: &CuStream) -> Self {
        // let block_size = 4
        //     * (32.0 * (width as f64 * height as f64 / (3840.0 * 2160.0)).sqrt() + 0.5).floor()
        //         as u32;
        let block_size = 16;
        let num_blocks_w = ((width + block_size - 1) / block_size) as usize;
        let num_blocks_h = ((height + block_size - 1) / block_size) as usize;
        let num_blocks = num_blocks_w * num_blocks_h;
        dbg!(num_blocks);

        let ref_ = Image::malloc(width, height).unwrap();
        let ref_16 = ref_.malloc_same_size().unwrap();
        let prev = ref_.malloc_same_size().unwrap();
        let dis = ref_.malloc_same_size().unwrap();
        let highpass = ref_.malloc_same_size().unwrap();
        let sse =
            ScratchBuffer::alloc_len(size_of::<u32>() * num_blocks, stream.inner() as _).unwrap();
        let sact =
            ScratchBuffer::alloc_len(size_of::<u32>() * num_blocks, stream.inner() as _).unwrap();
        let tact =
            ScratchBuffer::alloc_len(size_of::<u32>() * num_blocks, stream.inner() as _).unwrap();

        let wsse_dev = ScratchBuffer::alloc_from_host(&0.0, stream.inner() as _).unwrap();

        let highpass_coeffs = [-1, -2, -1, -2, 12, -2, -1, -2, -1];
        let highpass_coeffs =
            ScratchBuffer::alloc_from_host(&highpass_coeffs, stream.inner() as _).unwrap();

        Self {
            kernel: Kernel::load(),
            blocks_w: num_blocks_w,
            blocks_h: num_blocks_h,
            ref_,
            ref_16,
            prev,
            dis,
            highpass,
            sse_dev: sse,
            sact_dev: sact,
            tact_dev: tact,
            sse: vec![0; num_blocks],
            sact: vec![0; num_blocks],
            tact: vec![0; num_blocks],
            highpass_coeffs,
            weights: vec![0.0; num_blocks],
            wsse_dev,
        }
    }

    pub fn compute(&mut self, stream: &CuStream) {
        let ctx = get_stream_ctx().unwrap().with_stream(stream.inner() as _);
        self.ref_.convert(&mut self.ref_16, ctx).unwrap();
        self.ref_16
            .filter(
                &mut self.highpass,
                &self.highpass_coeffs,
                NppiSize {
                    width: 3,
                    height: 3,
                },
                ctx,
            )
            .unwrap();
        self.kernel.xpsnr_support_8(
            stream,
            &self.ref_,
            &mut self.prev,
            &self.dis,
            &self.highpass,
            &mut self.sse_dev,
            &mut self.sact_dev,
            &mut self.tact_dev,
        );
        let num_blocks = self.blocks_w * self.blocks_h;

        let width = self.ref_.width();
        let height = self.ref_.height();
        let block_weight_smoothing = width * height <= 640 * 480;
        const BITDEPTH: u32 = 8;

        let wsse = if block_weight_smoothing {
            self.sse_dev
                .copy_to_cpu_buf(&mut self.sse, stream.inner() as _)
                .unwrap();
            self.sact_dev
                .copy_to_cpu_buf(&mut self.sact, stream.inner() as _)
                .unwrap();
            self.tact_dev
                .copy_to_cpu_buf(&mut self.tact, stream.inner() as _)
                .unwrap();
            stream.sync().unwrap();

            for blk in 0..num_blocks {
                let mut msact = 1.0 + self.sact[blk] as f64 / (16 * 16) as f64;
                msact += 2.0 * self.tact[blk] as f64 / (16 * 16) as f64;
                msact = msact.max((1 << (BITDEPTH - 2)) as f64);
                msact *= msact;
                self.weights[blk] = msact.sqrt().recip();
                let mut msact_prev = if blk % self.blocks_w == 0 {
                    // First column
                    if blk > 1 {
                        self.weights[blk - 2]
                    } else {
                        0.0
                    }
                } else {
                    if blk % self.blocks_w > 1 {
                        self.weights[blk - 2].max(self.weights[blk])
                    } else {
                        self.weights[blk]
                    }
                };
                if blk > self.blocks_w {
                    // First row
                    msact_prev = msact_prev.max(self.weights[blk - 1 - self.blocks_w]);
                }
                if blk > 0 && self.weights[blk - 1] > msact_prev {
                    self.weights[blk - 1] = msact_prev;
                }
                if blk == num_blocks - 1 && blk > 0 {
                    msact_prev = self.weights[blk - 1].max(self.weights[blk - self.blocks_w]);
                    self.weights[blk] = self.weights[blk].min(msact_prev);
                }
            }
            self.sse
                .iter()
                .zip(&self.weights)
                .map(|(&sse, &w)| w * sse as f64)
                .reduce(Add::add)
                .unwrap()
        } else {
            self.kernel.xpsnr_postprocess(
                stream,
                &self.sse_dev,
                &self.sact_dev,
                &self.tact_dev,
                &mut self.wsse_dev,
            );
            let mut tmp = 0.0f32;
            self.wsse_dev
                .copy_to_cpu(&mut tmp, stream.inner() as _)
                .unwrap();
            tmp as f64
        };

        let wsse = if wsse.is_sign_negative() {
            0
        } else {
            let r = width as f64 * height as f64 / (3840.0 * 2160.0);
            let avgact =
                (16.0 * (1 << (2 * BITDEPTH - 9)) as f64 / 0.00001f64.max(r).sqrt()).sqrt();
            (wsse * avgact + 0.5) as u64
        };
        dbg!(wsse);
    }
}

#[cfg(test)]
mod tests {
    use crate::Xpsnr;
    use cudarse_driver::{init_cuda_and_primary_ctx, CuStream};
    use cudarse_npp::image::ImgMut;

    #[test]
    fn test() {
        init_cuda_and_primary_ctx().unwrap();
        let stream = CuStream::new().unwrap();
        let mut xpsnr = Xpsnr::new(4, 4, &stream);
        xpsnr
            .ref_
            .copy_from_cpu(&[16; 16], stream.inner() as _)
            .unwrap();
        xpsnr
            .prev
            .copy_from_cpu(&[16; 16], stream.inner() as _)
            .unwrap();
        xpsnr
            .dis
            .copy_from_cpu(&[14; 16], stream.inner() as _)
            .unwrap();
        xpsnr.compute(&stream);
        let mut sse = vec![0u32; 1];
        xpsnr
            .sse_dev
            .copy_to_cpu_buf(&mut sse, stream.inner() as _)
            .unwrap();
        stream.sync().unwrap();
        dbg!(sse);
    }
}
