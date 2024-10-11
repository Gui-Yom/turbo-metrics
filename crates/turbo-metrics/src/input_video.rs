use crate::input_image::PROBE_LEN;
use codec_bitstream::{av1, h264, ivf, Codec};
use cudarse_video::dec::CuVideoParser;
use matroska_demuxer::{Frame, MatroskaFile, TrackEntry};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, ErrorKind, Read, Seek};

pub type Matroska = MatroskaFile<BufReader<File>>;

pub enum VideoProbe {
    MKV(Matroska),
    IVF(ivf::Header, Box<dyn Read>),
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
                Ok(Ok(Self::IVF(header, Box::new(r))))
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
                            Ok(Ok(Self::MKV(matroska)))
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
}

impl From<VideoProbe> for Box<dyn Demuxer> {
    fn from(value: VideoProbe) -> Self {
        match value {
            VideoProbe::MKV(mkv) => Box::new(MkvDemuxer::new(mkv)),
            VideoProbe::IVF(header, stream) => Box::new(IvfDemuxer::new(stream, header)),
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
    fn codec(&self) -> Codec;
    fn clock_rate(&self) -> u32;
    /// Kickstart parsing
    fn init(&mut self, parser: &mut CuVideoParser);
    /// Return false if there is no more data
    fn demux(&mut self, parser: &mut CuVideoParser) -> bool;
}

impl Demuxer for Box<dyn Demuxer> {
    fn codec(&self) -> Codec {
        Box::as_ref(self).codec()
    }

    fn clock_rate(&self) -> u32 {
        Box::as_ref(self).clock_rate()
    }

    fn init(&mut self, parser: &mut CuVideoParser) {
        Box::as_mut(self).init(parser)
    }

    fn demux(&mut self, parser: &mut CuVideoParser) -> bool {
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
    fn codec(&self) -> Codec {
        // This can't fail at the moment this is called
        self.header.codec().unwrap()
    }

    fn clock_rate(&self) -> u32 {
        // (dbg!(self.header.timebase_num) as f32 / dbg!(self.header.timebase_den) as f32 * 1000.0)
        //     as u32
        1000000
    }

    fn init(&mut self, parser: &mut CuVideoParser) {
        let _ = ivf::read_packet(&mut self.stream, &mut self.buf).unwrap();
        parser.parse_data(&self.buf, 0).unwrap()
    }

    fn demux(&mut self, parser: &mut CuVideoParser) -> bool {
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
    frame: Frame,
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
            packet: vec![],
            track_id: id as u64,
            codec,
        }
    }
}

impl Demuxer for MkvDemuxer {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn clock_rate(&self) -> u32 {
        self.mkv.info().timestamp_scale().get() as _
    }

    fn init(&mut self, parser: &mut CuVideoParser) {
        let track = &self.mkv.tracks()[self.track_id as usize];
        match self.codec {
            Codec::H262 => {
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

    fn demux(&mut self, parser: &mut CuVideoParser) -> bool {
        loop {
            if let Ok(true) = self.mkv.next_frame(&mut self.frame) {
                if self.frame.track - 1 == self.track_id {
                    match self.codec {
                        Codec::H264 => {
                            h264::packet_to_annexb(
                                &mut self.packet,
                                &self.frame.data,
                                self.nal_length_size,
                            );
                            parser
                                .parse_data(&self.packet, self.frame.timestamp as i64)
                                .unwrap();
                        }
                        Codec::AV1 | Codec::H262 => {
                            parser
                                .parse_data(&self.frame.data, self.frame.timestamp as i64)
                                .unwrap();
                        }
                    }
                    return true;
                } else {
                    continue;
                }
            } else {
                parser.flush().unwrap();
                return false;
            }
        }
    }
}

fn mkv_codec_id_to_codec(id: &str) -> Option<Codec> {
    match id {
        "V_MPEG4/ISO/AVC" => Some(Codec::H264),
        "V_AV1" => Some(Codec::AV1),
        "V_MPEG2" => Some(Codec::H262),
        // Unsupported
        _ => None,
    }
}
