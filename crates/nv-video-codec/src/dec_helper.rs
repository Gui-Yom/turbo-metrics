use cuda_driver::CuCtx;
use nv_video_codec_sys::{
    cudaVideoSurfaceFormat, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPARSERPARAMS, CUVIDPICPARAMS,
};

use crate::dec::{query_caps, CuVideoCtxLock, CuVideoDecoder, VideoParserCb};

pub struct DecoderState<F: FnMut(&CuVideoDecoder, &CUVIDEOFORMAT, &CUVIDPARSERDISPINFO)> {
    decoder: Option<CuVideoDecoder>,
    format: Option<CUVIDEOFORMAT>,
    on_decode: F,
}

impl<F: FnMut(&CuVideoDecoder, &CUVIDEOFORMAT, &CUVIDPARSERDISPINFO)> DecoderState<F> {
    pub fn new(cb: F) -> Self {
        Self {
            decoder: None,
            format: None,
            on_decode: cb,
        }
    }
}

impl<F: FnMut(&CuVideoDecoder, &CUVIDEOFORMAT, &CUVIDPARSERDISPINFO)> VideoParserCb
    for DecoderState<F>
{
    fn sequence_callback(&mut self, format: &CUVIDEOFORMAT) -> i32 {
        self.format = Some(format.clone());
        let caps = query_caps(
            format.codec,
            format.chroma_format,
            format.bit_depth_luma_minus8 as u32 + 8,
        )
        .unwrap();
        if caps.bIsSupported == 0 {
            println!("Unsupported codec/chroma/bitdepth");
            return 0;
        }

        assert!(
            caps.is_output_format_supported(cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12)
        );

        let lock = CuVideoCtxLock::new(&CuCtx::get_current().unwrap()).unwrap();

        self.decoder = Some(
            CuVideoDecoder::new(
                dbg!(format),
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12,
                &lock,
            )
            .unwrap(),
        );

        let surfaces = format.min_num_decode_surfaces;
        dbg!(surfaces.max(1) as i32)
    }

    fn decode_picture(&mut self, pic: &CUVIDPICPARAMS) -> i32 {
        if let Some(decoder) = &self.decoder {
            decoder.decode(pic).unwrap();
            1
        } else {
            eprintln!("Decoder isn't initialized but decode_picture has been called !");
            0
        }
    }

    fn display_picture(&mut self, disp: &CUVIDPARSERDISPINFO) -> i32 {
        (self.on_decode)(
            self.decoder.as_ref().unwrap(),
            self.format.as_ref().unwrap(),
            disp,
        );
        1
    }
}
