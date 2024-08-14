use std::collections::VecDeque;
use std::io::{Cursor, ErrorKind, Read};
use std::iter::repeat;
use std::{io, mem};

use bitstream_io::{BigEndian, BitRead, BitReader};
use h264_reader::annexb::AnnexBReader;
use h264_reader::avcc::AvccError;
use h264_reader::nal::{Nal, RefNal};
use h264_reader::push::{AccumulatedNalHandler, NalAccumulator, NalInterest};

pub const NALU_DELIMITER: [u8; 4] = [0, 0, 0, 1];

/*
0      Unspecified                                                    non-VCL
1      Coded slice of a non-IDR picture                               VCL
2      Coded slice data partition A                                   VCL
3      Coded slice data partition B                                   VCL
4      Coded slice data partition C                                   VCL
5      Coded slice of an IDR picture                                  VCL
6      Supplemental enhancement information (SEI)                     non-VCL
7      Sequence parameter set                                         non-VCL
8      Picture parameter set                                          non-VCL
9      Access unit delimiter                                          non-VCL
10     End of sequence                                                non-VCL
11     End of stream                                                  non-VCL
12     Filler data                                                    non-VCL
13     Sequence parameter set extension                               non-VCL
14     Prefix NAL unit                                                non-VCL
15     Subset sequence parameter set                                  non-VCL
16     Depth parameter set                                            non-VCL
17..18 Reserved                                                       non-VCL
19     Coded slice of an auxiliary coded picture without partitioning non-VCL
20     Coded slice extension                                          non-VCL
21     Coded slice extension for depth view components                non-VCL
22..23 Reserved                                                       non-VCL
24..31 Unspecified                                                    non-VCL
 */
#[derive(Debug)]
#[repr(u8)]
pub enum NaluType {
    Unspecified = 0,
    Slice = 1,
    SliceA = 2,
    SliceB = 3,
    SliceC = 4,
    SliceIDR = 5,
    SEI = 6,
    SPS = 7,
    PPS = 8,
    AUD = 9,
    EOSeq = 10,
    EOStream = 11,
}

impl NaluType {
    pub fn from_nalu_header(header: u8) -> Self {
        let value = header & 0b00011111;
        assert!(value <= NaluType::EOStream as u8);
        unsafe { mem::transmute::<u8, NaluType>(value) }
    }
}

/// Get NAL units with SPS and PPS from the avc decoder configuration record.
pub fn avcc_extradata_to_annexb(codec_private: &[u8]) -> (usize, Vec<u8>) {
    let mut reader = BitReader::endian(Cursor::new(codec_private), BigEndian);
    let version: u8 = reader.read(8).unwrap();
    let profile: u8 = reader.read(8).unwrap();
    let profile_compat: u8 = reader.read(8).unwrap();
    let level: u8 = reader.read(8).unwrap();
    reader.read::<u8>(6).unwrap(); // Reserved
    let nal_size: u8 = reader.read::<u8>(2).unwrap() + 1;
    reader.read::<u8>(3).unwrap(); // Reserved
    let num_sps: u8 = reader.read(5).unwrap();

    let mut nalus = Vec::new();
    for _ in 0..num_sps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&NALU_DELIMITER);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }
    let num_pps: u8 = reader.read(8).unwrap();
    for _ in 0..num_pps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&NALU_DELIMITER);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }

    (nal_size as usize, nalus)
}

pub fn extract_sps_pps_clean(codec_private: &[u8]) -> Result<Vec<u8>, AvccError> {
    let dcr = h264_reader::avcc::AvcDecoderConfigurationRecord::try_from(codec_private)?;
    let mut nalus = Vec::new();
    for sps in dcr.sequence_parameter_sets() {
        let sps = sps.map_err(AvccError::ParamSet)?;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        nalus.extend_from_slice(sps);
    }
    for pps in dcr.picture_parameter_sets() {
        let pps = pps.map_err(AvccError::ParamSet)?;
        nalus.extend_from_slice(&[0, 0, 0, 1]);
        nalus.extend_from_slice(pps);
    }
    Ok(nalus)
}

pub fn packet_to_annexb(buffer: &mut Vec<u8>, mut packet: &[u8], nal_length_size: usize) {
    buffer.clear();
    while packet.len() > nal_length_size {
        let mut len = 0usize;
        /// Read a variable number of bytes in big endian format
        for i in 0..nal_length_size {
            len = (len << 8) | packet[i] as usize;
        }
        packet = &packet[nal_length_size..];
        buffer.extend_from_slice(&NALU_DELIMITER);
        buffer.extend_from_slice(&packet[..len]);
        packet = &packet[len..];
    }
    assert!(packet.is_empty());
}

pub struct NalReader<R: Read> {
    annexb: AnnexBReader<NalAccumulator<Handler<R>>>,
}

struct Handler<R: Read> {
    r: R,
    packets: VecDeque<Vec<u8>>,
}

impl<R: Read> AccumulatedNalHandler for Handler<R> {
    fn nal(&mut self, nal: RefNal<'_>) -> NalInterest {
        if nal.is_complete() {
            let mut reader = nal.reader();
            let mut pkt = NALU_DELIMITER.to_vec();
            reader.read_to_end(&mut pkt).unwrap();
            self.packets.push_back(pkt);
        }
        NalInterest::Buffer
    }
}

impl<R: Read> NalReader<R> {
    pub fn new(r: R) -> Self {
        Self {
            annexb: AnnexBReader::accumulate(Handler {
                r,
                packets: VecDeque::new(),
            }),
        }
    }

    pub fn read_nalu(&mut self) -> io::Result<Vec<u8>> {
        while self.annexb.nal_handler_ref().packets.is_empty() {
            let mut buf = [0; 2048];
            let len = self.annexb.nal_handler_mut().r.read(&mut buf)?;
            if len == 0 {
                return Err(io::Error::from(ErrorKind::UnexpectedEof));
            }
            self.annexb.push(&buf[..len]);
        }
        Ok(self.annexb.nal_handler_mut().packets.pop_front().unwrap())
    }
}

// pub struct NalReader<R: Read> {
//     r: R,
//     buf: VecDeque<u8>,
//     pop: Option<usize>,
// }
//
// impl<R: Read> NalReader<R> {
//     pub fn new(r: R) -> Self {
//         Self {
//             r,
//             buf: VecDeque::from(vec![0; 8 * 1024]),
//             pop: None,
//         }
//     }
//
//     pub fn read_nalu(&mut self) -> Result<&[u8], ()> {
//         const READ_AMOUNT: usize = 2048;
//         if let Some(idx) = self.pop {
//             self.buf.drain(..idx);
//         }
//         loop {
//             let buf = self.buf.make_contiguous();
//             for (i, w) in buf.windows(4).enumerate() {
//                 if i == 0 {
//                     assert_eq!(w, NALU_HEADER);
//                     continue;
//                 }
//                 if w == NALU_HEADER {
//                     self.pop = Some(i);
//                     return Ok(&buf[..i]);
//                 }
//             }
//             let old_len = self.buf.len();
//             self.buf.extend(repeat(0).take(READ_AMOUNT));
//             let buf = self.buf.make_contiguous();
//             let len = self.r.read(&mut buf[old_len..]).unwrap();
//             assert!(len > 0);
//             self.buf.drain(old_len + len..);
//         }
//
//         Err(())
//     }
// }

#[cfg(test)]
mod tests {
    use matroska_demuxer::{Frame, MatroskaFile};
    use std::fs::File;
    use std::io::BufReader;

    use crate::h264::{avcc_extradata_to_annexb, extract_sps_pps_clean, NalReader, NaluType};

    const RAW_H264: &'static str = "../../data/raw.h264";
    const RAW_H264_SOURCE: &'static str = "../../data/dummy_encode.mkv";
    const TEST_MKV: &'static str = "../../data/chainsaw_man_s01e01_v.mkv";

    #[test]
    fn sps_pps_extraction() {
        let mut matroska =
            MatroskaFile::open(BufReader::new(File::open(TEST_MKV).unwrap())).unwrap();

        let (_, bitstream) =
            avcc_extradata_to_annexb(matroska.tracks()[0].codec_private().unwrap());
        // dbg!(bitstream);

        let mut matroska =
            MatroskaFile::open(BufReader::new(File::open(TEST_MKV).unwrap())).unwrap();

        let extradata2 =
            extract_sps_pps_clean(matroska.tracks()[0].codec_private().unwrap()).unwrap();
        // dbg!(extradata2);

        assert_eq!(bitstream, extradata2);
    }

    #[test]
    fn test_nalus() {
        let mut r = NalReader::new(File::open(RAW_H264).unwrap());
        let mut i = 0;
        while let Ok(pkt) = r.read_nalu() {
            println!(
                "{i:>3}: {:?} ({} bytes)",
                NaluType::from_nalu_header(pkt[4]),
                pkt.len()
            );
            i += 1;
        }
    }

    #[test]
    fn test_nal_reader() {
        let mut r = NalReader::new(File::open(RAW_H264).unwrap());
        while let Ok(pkt) = r.read_nalu() {
            dbg!(pkt.len());
        }

        let mut matroska =
            MatroskaFile::open(BufReader::new(File::open(RAW_H264_SOURCE).unwrap())).unwrap();
        let (nal_length_size, sps_pps) =
            avcc_extradata_to_annexb(matroska.tracks()[0].codec_private().unwrap());
        dbg!(sps_pps.len());
        // let mut pkt = Vec::new();
        let mut frame = Frame::default();
        while let Ok(remaining) = matroska.next_frame(&mut frame) {
            if !remaining {
                break;
            }
            dbg!(frame.data.len());
        }
        // matroska.next_frame(&mut frame).unwrap();
        // packet_to_annexb(&mut pkt, &frame.data, nal_length_size);

        // dbg!(apkt.len());
        // dbg!(pkt.len());
        // assert_eq!(&pkt, &apkt);
    }
}
