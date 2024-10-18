use crate::dec::{
    query_caps, select_output_format, CuVideoCtxLock, CuVideoDecoder, CuvidParserCallbacks,
    FrameMapping,
};
use cudarse_driver::sys::CuResult;
use cudarse_driver::CuStream;
use cudarse_video_sys::{
    cudaVideoSurfaceFormat, CUVIDEOFORMAT, CUVIDPARSERDISPINFO, CUVIDPICPARAMS,
    CUVIDRECONFIGUREDECODERINFO, CUVIDSEIMESSAGEINFO,
};
use std::cell::{OnceCell, RefCell, RefMut};
use std::collections::VecDeque;
use std::iter::repeat_with;
use std::ops::DerefMut;
use std::slice;
use tracing::{debug, error, trace, warn};

/// Simple decoder for single threaded use.
pub struct NvDecoderSimple<'a> {
    extra_decode_surfaces: u32,
    lock: Option<&'a CuVideoCtxLock>,
    decoder: OnceCell<(CuVideoDecoder<'a>, CUVIDEOFORMAT, cudaVideoSurfaceFormat)>,
    frames: RefCell<VecDeque<Option<CUVIDPARSERDISPINFO>>>,
}

impl<'a> NvDecoderSimple<'a> {
    pub fn new(extra_decode_surfaces: u32, lock_for_cuda: Option<&'a CuVideoCtxLock>) -> Self {
        Self {
            extra_decode_surfaces,
            lock: lock_for_cuda,
            decoder: OnceCell::new(),
            frames: RefCell::new(VecDeque::with_capacity(4)),
        }
    }

    pub fn format(&self) -> Option<&CUVIDEOFORMAT> {
        self.decoder.get().map(|(_, f, _)| f)
    }

    pub fn display_queue_len(&self) -> usize {
        self.frames.borrow().len()
    }

    pub fn has_frames(&self) -> bool {
        !self.frames.borrow().is_empty()
    }

    pub fn frames<'b>(&'b self) -> impl Iterator<Item = Option<CUVIDPARSERDISPINFO>> + 'b {
        const MIN_QUEUE_LEN: usize = 0;
        // Try to maintain a long enough queue for throughput.
        let mut frames = self.frames.borrow_mut();
        let len = frames.len();
        let count = if frames.iter().any(|i| i.is_none()) {
            // Immediately return remaining frames if sequence has ended
            len
        } else if len > MIN_QUEUE_LEN {
            len - MIN_QUEUE_LEN
        } else {
            // Return nothing, wait a bit for the queue to fill up
            0
        };
        repeat_with(move || RefMut::deref_mut(&mut frames).pop_front().unwrap()).take(count)
    }

    pub fn frames_sync<'b>(
        &'b self,
        other: &'b Self,
    ) -> impl Iterator<Item = (Option<CUVIDPARSERDISPINFO>, Option<CUVIDPARSERDISPINFO>)> + 'b {
        let mut frames = self.frames.borrow_mut();
        let mut frames2 = other.frames.borrow_mut();
        let len = frames.len().min(frames2.len());
        repeat_with(move || {
            (
                RefMut::deref_mut(&mut frames).pop_front().unwrap(),
                RefMut::deref_mut(&mut frames2).pop_front().unwrap(),
            )
        })
        .take(len)
    }

    pub fn map<'map>(
        &'map self,
        disp: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<FrameMapping<'map>> {
        self.decoder.get().unwrap().0.map(disp, stream)
    }

    #[cfg(feature = "npp")]
    pub fn map_npp_nv12<'map>(
        &'map self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<crate::dec::npp::NvDecNV12<'map>> {
        if let crate::dec::npp::NvDecFrame::NV12(frame) = self.map_npp(info, stream)? {
            Ok(frame)
        } else {
            panic!("actual surface format is not NV12")
        }
    }

    #[cfg(feature = "npp")]
    pub fn map_npp<'map>(
        &'map self,
        info: &CUVIDPARSERDISPINFO,
        stream: &CuStream,
    ) -> CuResult<crate::dec::npp::NvDecFrame<'map>> {
        if let Some((decoder, format, surface_format)) = self.decoder.get() {
            let mapping = decoder.map(info, stream)?;
            match surface_format {
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12 => {
                    Ok(crate::dec::npp::NvDecFrame::NV12(
                        crate::dec::npp::NvDecNV12::from_mapping(mapping, format),
                    ))
                }
                cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_P016 => {
                    Ok(crate::dec::npp::NvDecFrame::P016(
                        crate::dec::npp::NvDecP016::from_mapping(mapping, format),
                    ))
                }
                other => todo!("Unsupported surface format '{other:?}' in map_npp"),
            }
        } else {
            panic!("map_npp called but instance is not initialized")
        }
    }
}

impl CuvidParserCallbacks for NvDecoderSimple<'_> {
    fn sequence_callback(&self, format: &CUVIDEOFORMAT) -> Result<u32, ()> {
        let caps = query_caps(
            format.codec,
            format.chroma_format,
            format.bit_depth_luma_minus8 as u32 + 8,
        )
        .map_err(|e| error!("{e}"))?;
        if caps.bIsSupported == 0 {
            error!("Unsupported codec/chroma/bitdepth");
            return Err(());
        }

        let surface_format = select_output_format(&format, &caps);

        assert!(format.min_num_decode_surfaces > 1);
        let surfaces = format.min_num_decode_surfaces as u32 + self.extra_decode_surfaces;
        debug!(surfaces);
        if let Some((decoder, _, old_surface_format)) = self.decoder.get() {
            // Decoder needs to be reconfigured
            warn!("Decoder reconfiguration is only supported for the same codec params (only size can change)");
            if surface_format != *old_surface_format {
                panic!("Can't change surface format when reconfiguring");
            }
            // TODO handle complete reconfiguring
            decoder
                .reconfigure(&format)
                .expect("Could not reconfigure the decoder");
        } else {
            // Initialize new
            let decoder = CuVideoDecoder::new(format, surface_format, surfaces, self.lock)
                .map_err(|e| error!("{e}"))?;
            self.decoder
                .set((decoder, format.clone(), surface_format))
                .unwrap();
        }

        Ok(surfaces as _)
    }

    fn decode_picture(&self, pic: &CUVIDPICPARAMS) -> Result<(), ()> {
        // TODO investigate if having a decode queue here would allow us to feed many nalus at once without fearing a 'still in decode queue' bug
        trace!(
            "decode {} ({} bytes)",
            pic.CurrPicIdx,
            pic.nBitstreamDataLen
        );
        if let Some((decoder, _, _)) = self.decoder.get() {
            if self
                .frames
                .borrow()
                .iter()
                .find(|p| {
                    p.map(|p| p.picture_index == pic.CurrPicIdx)
                        .unwrap_or(false)
                })
                .is_some()
            {
                panic!("decode {} : still in decode queue", pic.CurrPicIdx);
            }
            decoder.decode(pic).unwrap();
            Ok(())
        } else {
            error!("decode_picture called before any decoder has been initialized");
            Err(())
        }
    }

    fn display_picture(&self, disp: Option<&CUVIDPARSERDISPINFO>) -> Result<(), ()> {
        if let Some(idx) = disp.map(|d| d.picture_index) {
            trace!("display {}", idx);
        }
        self.frames.borrow_mut().push_back(disp.cloned());
        Ok(())
    }

    fn get_sei_msg(&self, sei: &CUVIDSEIMESSAGEINFO) -> i32 {
        if sei.sei_message_count > 0 {
            let messages =
                unsafe { slice::from_raw_parts(sei.pSEIMessage, sei.sei_message_count as usize) };
            for msg in messages {
                trace!(
                    "SEI {} ({} bytes)",
                    msg.sei_message_type,
                    msg.sei_message_size
                );
            }
        }
        1
    }
}

#[cfg(test)]
mod tests {
    use crate::dec::CuVideoParser;
    use crate::dec_simple::NvDecoderSimple;
    use codec_bitstream::h264::{NalReader, NaluType};
    use codec_bitstream::{h264, Codec};
    use core::slice;
    use cudarse_driver::CuStream;
    use cudarse_video_sys::cudaVideoCodec;
    use matroska_demuxer::{Frame, MatroskaFile};
    use std::collections::VecDeque;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    use tracing::debug;
    use tracing_subscriber::EnvFilter;

    #[test]
    fn it_works() {
        tracing_subscriber::fmt()
            .compact()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
        cudarse_driver::init_cuda_and_primary_ctx().unwrap();
        let cb = NvDecoderSimple::new(3, None);
        let mut parser =
            CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &cb, None, None).unwrap();
        let mut reader = NalReader::new(File::open("../../data/raw.h264").unwrap());
        'main: loop {
            if let Ok(pkt) = reader.read_nalu() {
                debug!(
                    "parse {:?} ({} bytes)",
                    NaluType::from_nalu_header(pkt[4]),
                    pkt.len()
                );
                parser.parse_data(&pkt, 0).unwrap();
            } else {
                parser.flush().unwrap();
            }
            for frame in cb.frames() {
                let Some(frame) = frame else {
                    break 'main;
                };
                debug!("got frame {}", frame.picture_index);
                let Ok(mapping) = cb.map(&frame, &CuStream::DEFAULT) else {
                    panic!("Can't map")
                };
            }
        }
    }

    #[test]
    fn it_works_dual() {
        tracing_subscriber::fmt()
            .compact()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
        cudarse_driver::init_cuda_and_primary_ctx().unwrap();
        let cb = NvDecoderSimple::new(3, None);
        let mut parser =
            CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &cb, None, None).unwrap();
        let mut reader = NalReader::new(File::open("../../data/raw.h264").unwrap());

        let cb2 = NvDecoderSimple::new(3, None);
        let mut parser2 =
            CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &cb2, None, None).unwrap();
        let mut reader2 = NalReader::new(File::open("../../data/raw.h264").unwrap());

        'main: loop {
            while !cb.has_frames() || !cb2.has_frames() {
                if let Ok(pkt) = reader.read_nalu() {
                    // println!(
                    //     "parse {:?} ({} bytes)",
                    //     NaluType::from_nalu_header(pkt[4]),
                    //     pkt.len()
                    // );
                    parser.parse_data(&pkt, 0).unwrap();
                } else {
                    parser.flush().unwrap();
                }
                if let Ok(pkt) = reader2.read_nalu() {
                    // println!(
                    //     "parse {:?} ({} bytes)",
                    //     NaluType::from_nalu_header(pkt[4]),
                    //     pkt.len()
                    // );
                    parser2.parse_data(&pkt, 0).unwrap();
                } else {
                    parser2.flush().unwrap();
                }
            }
            for (frame, frame2) in cb.frames_sync(&cb2) {
                let Some(frame) = frame else {
                    break 'main;
                };
                let Some(frame2) = frame2 else {
                    break 'main;
                };
                // println!("got frame {}", frame.picture_index);
                let Ok(mapping) = cb.map(&frame, &CuStream::DEFAULT) else {
                    panic!("Can't map")
                };
                let Ok(mapping2) = cb2.map(&frame2, &CuStream::DEFAULT) else {
                    panic!("Can't map")
                };
            }
        }
    }

    #[test]
    fn repro_dpbbug_mkv() {
        tracing_subscriber::fmt()
            .compact()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
        cudarse_driver::init_cuda_and_primary_ctx().unwrap();
        let cb = NvDecoderSimple::new(3, None);
        let mut parser =
            CuVideoParser::new(cudaVideoCodec::cudaVideoCodec_H264, &cb, None, None).unwrap();
        let mut mkv = MatroskaFile::open(BufReader::new(
            File::open("../../../data/dpbbug.mkv").unwrap(),
        ))
        .unwrap();
        let track = &mkv.tracks()[0];
        let mut frame = Frame::default();
        let mut packet = Vec::new();

        let (nal_length_size, sps_pps) =
            h264::avcc_extradata_to_annexb(track.codec_private().unwrap());
        // dbg!(nal_length_size);
        parser.parse_data(&sps_pps, 0).unwrap();

        let mut offset = 0;

        'main: loop {
            if frame.data.len() - offset == 0 {
                if let Ok(true) = mkv.next_frame(&mut frame) {
                    offset = 0;
                } else {
                    parser.flush().unwrap();
                }
            }
            if frame.data.len() - offset > 0 {
                if let Ok(read) =
                    h264::avcc_into_annexb(&frame.data[offset..], nal_length_size, &mut packet)
                {
                    offset += read;
                } else {
                    panic!("incomplete nalu");
                }
                debug!(
                    "parse {:?} ({} bytes)",
                    NaluType::from_nalu_header(packet[4]),
                    packet.len()
                );
                parser.parse_data(&packet, frame.timestamp as i64).unwrap();
            }

            for frame in cb.frames() {
                let Some(frame) = frame else {
                    break 'main;
                };
                debug!("got frame {}", frame.picture_index);
                let Ok(mapping) = cb.map(&frame, &CuStream::DEFAULT) else {
                    panic!("Can't map")
                };
            }
        }
    }
}
