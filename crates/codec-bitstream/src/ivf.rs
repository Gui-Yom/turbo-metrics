use crate::Codec;
use bitstream_io::{ByteRead, ByteReader, LittleEndian};
use std::io;
use std::iter::repeat;

#[derive(Debug, Clone)]
pub struct Header {
    pub fourcc: [u8; 4],
    pub w: u16,
    pub h: u16,
    pub timebase_num: u32,
    pub timebase_den: u32,
    pub frames: u32,
}

impl Header {
    pub fn codec(&self) -> Option<Codec> {
        Codec::from_fourcc(self.fourcc)
    }
}

pub fn read_header(r: impl io::Read) -> io::Result<(Header, usize)> {
    let mut br = ByteReader::endian(r, LittleEndian);

    let mut signature = [0u8; 4];
    br.read_bytes(&mut signature)?;

    if &signature != b"DKIF" {
        return Err(io::ErrorKind::InvalidData.into());
    }

    let _version: u16 = br.read()?;
    let length: u16 = br.read()?;
    let mut fourcc = [0u8; 4];
    br.read_bytes(&mut fourcc)?;

    let w: u16 = br.read()?;
    let h: u16 = br.read()?;

    let timebase_den: u32 = br.read()?;
    let timebase_num: u32 = br.read()?;

    let frames: u32 = br.read()?;
    let _unused: u32 = br.read()?;

    Ok((
        Header {
            fourcc,
            w,
            h,
            timebase_num,
            timebase_den,
            frames,
        },
        length as usize,
    ))
}

/// Read a packet in `buf` and returns the packet presentation timestamp.
pub fn read_packet(r: impl io::Read, buf: &mut Vec<u8>) -> io::Result<u64> {
    let mut br = ByteReader::endian(r, LittleEndian);

    let len: u32 = br.read()?;
    let pts: u64 = br.read()?;
    // buf.clear();
    buf.reserve(len as usize);
    if buf.len() < len as usize {
        buf.extend(repeat(0).take(len as usize - buf.len()));
    } else if buf.len() > len as usize {
        buf.truncate(len as usize);
    }

    br.read_bytes(buf)?;

    Ok(pts)
}
