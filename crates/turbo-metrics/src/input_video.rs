use crate::color::{color_characteristics_from_format, ColorRange};
use crate::input_image::PROBE_LEN;
use crate::{codec_to_nvdec, FormatIdentifier, FrameSource, HwFrame};
use codec_bitstream::h264::NaluType;
use codec_bitstream::{av1, h264, ivf, Codec, ColorCharacteristics};
use cudarse_driver::CuStream;
use cudarse_video::dec_simple::NvDecoderSimple;
use cudarse_video::parser::CuvidParser;
use matroska_demuxer::{Frame as MkvFrame, MatroskaFile, TrackEntry};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, ErrorKind, Read, Seek};
use std::rc::Rc;
use tracing::trace;

#[derive(Debug, Copy, Clone)]
pub enum Container {
    Mkv,
    Ivf,
}

#[derive(Debug)]
pub enum ProbeError {
    /// Could not find a video track in the mkv file
    MKVNoVideo,
    /// Unknown codec in MKV
    MkvUnknownCodec(String),
    /// Unknown codec in IVF, with fourcc as argument
    IvfUnknownCodec([u8; 4]),
    /// Could not even parse the container
    UnknownContainer,
}

impl Display for ProbeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for ProbeError {}

pub type Matroska = MatroskaFile<BufReader<File>>;

pub enum VideoProbe {
    Mkv(Matroska),
    Ivf(ivf::Header, Box<dyn Read>),
}

impl VideoProbe {
    /// Probe by peeking at a [BufRead] without advancing the read pointer.
    /// Returns the reader if probe is unsuccessful
    fn probe_stream_inner<R: BufRead + 'static>(
        mut r: R,
    ) -> io::Result<Result<Self, (ProbeError, R)>> {
        let mut start = r.fill_buf()?;
        if start.len() < PROBE_LEN {
            return Err(io::Error::from(ErrorKind::UnexpectedEof));
        }
        match ivf::read_header(&mut start) {
            Ok((header, len)) if header.codec().is_some() => {
                r.consume(len);
                Ok(Ok(Self::Ivf(header, Box::new(r))))
            }
            Ok((header, _)) => Ok(Err((ProbeError::IvfUnknownCodec(header.fourcc), r))),
            Err(e) if e.kind() == ErrorKind::InvalidData => {
                Ok(Err((ProbeError::UnknownContainer, r)))
            }
            Err(e) => Err(e),
        }
    }

    /// Probe by peeking at a [BufRead] without advancing the read pointer.
    pub fn probe_stream(r: BufReader<impl Read + 'static>) -> io::Result<Result<Self, ProbeError>> {
        Self::probe_stream_inner(r).map(|i| i.map_err(|(e, _)| e))
    }

    pub fn probe_file(r: BufReader<File>) -> io::Result<Result<Self, ProbeError>> {
        let ret = Self::probe_stream_inner(r)?;

        match ret {
            // First try could not parse anything
            Err((ProbeError::UnknownContainer, r)) => {
                if let Ok(matroska) = Matroska::open(r) {
                    if let Some((_, track)) = mkv_find_video_track(&matroska) {
                        if mkv_codec_id_to_codec(track.codec_id()).is_some() {
                            Ok(Ok(Self::Mkv(matroska)))
                        } else {
                            Ok(Err(ProbeError::MkvUnknownCodec(
                                track.codec_id().to_string(),
                            )))
                        }
                    } else {
                        Ok(Err(ProbeError::MKVNoVideo))
                    }
                } else {
                    // Can't parse
                    Ok(Err(ProbeError::UnknownContainer))
                }
            }
            other => Ok(other.map_err(|(e, _)| e)),
        }
    }

    pub fn make_demuxer(self) -> DynDemuxer {
        match self {
            VideoProbe::Mkv(mkv) => Box::new(MkvDemuxer::new(mkv)),
            VideoProbe::Ivf(header, stream) => Box::new(IvfDemuxer::new(stream, header)),
        }
    }
}

fn mkv_find_video_track(matroska: &MatroskaFile<impl Read + Seek>) -> Option<(usize, &TrackEntry)> {
    matroska
        .tracks()
        .iter()
        .enumerate()
        .find(|(_, t)| t.video().is_some())
}

pub trait Demuxer {
    fn container(&self) -> Container;
    fn codec(&self) -> Codec;
    fn clock_rate(&self) -> u32;
    fn frame_count(&self) -> usize;
    /// Kickstart parsing by feeding out of band data in the parser
    fn init(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>);
    /// Feed a single packet into the parser, return false if there is no more data
    fn demux(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) -> bool;
}

pub type DynDemuxer = Box<dyn Demuxer>;

impl Demuxer for DynDemuxer {
    fn container(&self) -> Container {
        Box::as_ref(self).container()
    }

    fn codec(&self) -> Codec {
        Box::as_ref(self).codec()
    }

    fn clock_rate(&self) -> u32 {
        Box::as_ref(self).clock_rate()
    }

    fn frame_count(&self) -> usize {
        Box::as_ref(self).frame_count()
    }

    fn init(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) {
        Box::as_mut(self).init(parser)
    }

    fn demux(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) -> bool {
        Box::as_mut(self).demux(parser)
    }
}

pub struct IvfDemuxer {
    header: ivf::Header,
    stream: Box<dyn Read>,
    buf: Vec<u8>,
    frame: u32,
}

impl IvfDemuxer {
    fn new(stream: Box<dyn Read>, header: ivf::Header) -> Self {
        // dbg!(header.frames);
        Self {
            stream,
            header,
            buf: Vec::with_capacity(65535),
            frame: 0,
        }
    }
}

impl Demuxer for IvfDemuxer {
    fn container(&self) -> Container {
        Container::Ivf
    }

    fn codec(&self) -> Codec {
        // This can't fail at the moment this is called
        self.header.codec().unwrap()
    }

    fn clock_rate(&self) -> u32 {
        // (dbg!(self.header.timebase_num) as f32 / dbg!(self.header.timebase_den) as f32 * 1000.0)
        //     as u32
        1000000
    }

    fn frame_count(&self) -> usize {
        self.header.frames as _
    }

    fn init(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) {
        let _ = ivf::read_packet(&mut self.stream, &mut self.buf).unwrap();
        parser.parse_data(&self.buf, 0).unwrap()
    }

    fn demux(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) -> bool {
        if let Ok(pts) = ivf::read_packet(&mut self.stream, &mut self.buf) {
            self.frame += 1;
            if self.buf.is_empty() {
                // This should not happen
                false
            } else {
                parser.parse_data(&self.buf, pts as i64).unwrap();
                true
            }
        } else {
            parser.flush().unwrap();
            false
        }
    }
}

pub struct MkvDemuxer {
    mkv: Matroska,
    nal_length_size: usize,
    frame: MkvFrame,
    /// Offset after the data we have already read in the frame
    frame_offset: usize,
    packet: Vec<u8>,
    track_id: u64,
    codec: Codec,
}

impl MkvDemuxer {
    pub fn new(mkv: Matroska) -> Self {
        let (id, v_track) = mkv_find_video_track(&mkv).expect("No video track in mkv file");
        let codec =
            mkv_codec_id_to_codec(v_track.codec_id()).expect("Unsupported video codec in mkv");

        Self {
            mkv,
            nal_length_size: 0,
            frame: Default::default(),
            frame_offset: 0,
            packet: vec![],
            track_id: id as u64,
            codec,
        }
    }
}

impl Demuxer for MkvDemuxer {
    fn container(&self) -> Container {
        Container::Mkv
    }

    fn codec(&self) -> Codec {
        self.codec
    }

    fn clock_rate(&self) -> u32 {
        self.mkv.info().timestamp_scale().get() as _
    }

    fn frame_count(&self) -> usize {
        0
    }

    fn init(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) {
        let track = &self.mkv.tracks()[self.track_id as usize];
        match self.codec {
            Codec::MPEG2 => {
                // dbg!(v_track.codec_private());
                if let Some(private) = track.codec_private() {
                    parser.parse_data(private, 0).unwrap();
                }
            }
            Codec::H264 => {
                let (nls, sps_pps_bitstream) =
                    h264::avcc_extradata_to_annexb(track.codec_private().unwrap());
                // dbg!(nal_length_size);
                parser.parse_data(&sps_pps_bitstream, 0).unwrap();
                self.nal_length_size = nls;
            }
            Codec::AV1 => {
                let extradata =
                    av1::extract_seq_hdr_from_mkv_codec_private(track.codec_private().unwrap())
                        .to_vec();
                parser.parse_data(&extradata, 0).unwrap();
            }
        }
    }

    fn demux(&mut self, parser: &mut CuvidParser<Rc<NvDecoderSimple<'static>>>) -> bool {
        // Pull a frame if we're empty
        if self.frame.data.len() - self.frame_offset == 0 {
            // Loop because the next frame might not be from the video track we expect
            loop {
                if let Ok(true) = self.mkv.next_frame(&mut self.frame) {
                    if self.frame.track - 1 == self.track_id {
                        self.frame_offset = 0;
                        break;
                    }
                } else {
                    parser.flush().unwrap();
                    return false;
                }
            }
        }

        // There should be data available in the frame at this point
        if self.frame.data.len() - self.frame_offset > 0 {
            match self.codec {
                Codec::AV1 | Codec::MPEG2 => {
                    // TODO investigate if feeding a full frame can cause a problem with those codecs
                    parser
                        .parse_data(&self.frame.data, self.frame.timestamp as i64)
                        .unwrap();
                    // This triggers a new frame to be pulled next time
                    self.frame_offset = self.frame.data.len();
                    true
                }
                Codec::H264 => {
                    // With H.264, we must feed a single NALU at a time to cuvid
                    // Feeding more than one will cause it to fire many decode callbacks at the same time, causing problems with the dpb.
                    if let Ok(read) = h264::avcc_into_annexb(
                        &self.frame.data[self.frame_offset..],
                        self.nal_length_size,
                        &mut self.packet,
                    ) {
                        self.frame_offset += read;
                        trace!(
                            "parse {:?} ({} bytes)",
                            NaluType::from_nalu_header(self.packet[4]),
                            self.packet.len()
                        );
                        parser
                            .parse_data(&self.packet, self.frame.timestamp as i64)
                            .unwrap();
                        true
                    } else {
                        panic!("incomplete nalu");
                    }
                }
            }
        } else {
            false
        }
    }
}

fn mkv_codec_id_to_codec(id: &str) -> Option<Codec> {
    match id {
        "V_MPEG4/ISO/AVC" => Some(Codec::H264),
        "V_AV1" => Some(Codec::AV1),
        "V_MPEG2" => Some(Codec::MPEG2),
        // Unsupported
        _ => None,
    }
}

pub struct VideoFrameSource<D: Demuxer> {
    demuxer: D,
    decoder: Rc<NvDecoderSimple<'static>>,
    parser: CuvidParser<Rc<NvDecoderSimple<'static>>>,
}

impl<D: Demuxer> VideoFrameSource<D> {
    pub fn new(mut demuxer: D) -> Self {
        let decoder = Rc::new(NvDecoderSimple::new(3, None));
        let mut parser = CuvidParser::new(
            codec_to_nvdec(demuxer.codec()),
            Rc::clone(&decoder),
            Some(demuxer.clock_rate()),
            None,
        )
        .unwrap();

        demuxer.init(&mut parser);
        while decoder.format().is_none() {
            if !demuxer.demux(&mut parser) {
                break;
            }
        }
        assert!(decoder.format().is_some());

        Self {
            demuxer,
            decoder,
            parser,
        }
    }
}

impl<D: Demuxer> FrameSource for VideoFrameSource<D> {
    fn format_id(&self) -> FormatIdentifier {
        FormatIdentifier {
            container: Some(format!("{:?}", self.demuxer.container())),
            codec: self.demuxer.codec().to_string(),
            decoder: "NVDEC".to_string(),
        }
    }

    fn width(&self) -> u32 {
        self.decoder.format().unwrap().display_width()
    }

    fn height(&self) -> u32 {
        self.decoder.format().unwrap().display_height()
    }

    fn color_characteristics(&self) -> (ColorCharacteristics, ColorRange) {
        color_characteristics_from_format(self.decoder.format().unwrap())
    }

    fn frame_count(&self) -> usize {
        self.demuxer.frame_count()
    }

    fn skip_frames(&mut self, mut n: u32) {
        while n > 0 {
            if self.decoder.frames().next().is_some() {
                n -= 1;
            } else {
                self.demuxer.demux(&mut self.parser);
            }
        }
    }

    fn next_frame(&mut self, stream: &CuStream) -> Result<Option<HwFrame>, Box<dyn Error>> {
        while !self.decoder.has_frames() {
            if !self.demuxer.demux(&mut self.parser) {
                break;
            }
        }
        if let Some(Some(d)) = self.decoder.frames().next() {
            Ok(Some(HwFrame::NvDec(self.decoder.map_npp(&d, stream)?)))
        } else {
            Ok(None)
        }
    }
}
