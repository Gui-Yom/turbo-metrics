use std::io::Cursor;
use std::iter::repeat;

use bitstream_io::{BigEndian, BitRead, BitReader};
use h264_reader::avcc::AvccError;

pub const NALU_HEADER: [u8; 4] = [0, 0, 0, 1];

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
        nalus.extend_from_slice(&NALU_HEADER);
        let start = nalus.len();
        nalus.extend(repeat(0).take(len));
        reader.read_bytes(&mut nalus[start..]).unwrap();
    }
    let num_pps: u8 = reader.read(8).unwrap();
    for _ in 0..num_pps {
        let len = reader.read::<u16>(16).unwrap() as usize;
        nalus.extend_from_slice(&NALU_HEADER);
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
        for i in 0..nal_length_size {
            len = (len << 8) | packet[i] as usize;
        }
        packet = &packet[nal_length_size..];
        buffer.extend_from_slice(&NALU_HEADER);
        buffer.extend_from_slice(&packet[..len]);
        packet = &packet[len..];
    }
    assert!(packet.is_empty());
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;

    use matroska_demuxer::MatroskaFile;

    use crate::h264::{avcc_extradata_to_annexb, extract_sps_pps_clean};

    #[test]
    fn it_works() {
        let mut matroska = MatroskaFile::open(BufReader::new(
            File::open("../../data/chainsaw_man_s01e01_v.mkv").unwrap(),
        ))
        .unwrap();

        let (_, bitstream) =
            avcc_extradata_to_annexb(matroska.tracks()[0].codec_private().unwrap());
        dbg!(bitstream);

        let mut matroska = MatroskaFile::open(BufReader::new(
            File::open("../../data/chainsaw_man_s01e01_v.mkv").unwrap(),
        ))
        .unwrap();

        let extradata2 =
            extract_sps_pps_clean(matroska.tracks()[0].codec_private().unwrap()).unwrap();
        dbg!(extradata2);

        assert_eq!(bitstream, extradata2);
    }
}
