use std::mem;

/// Extract sequence header obu from mkv CodecPrivate
pub fn extract_seq_hdr_from_mkv_codec_private(codec_private: &[u8]) -> &[u8] {
    &codec_private[4..]
}

#[derive(Debug, Copy, Clone)]
pub struct ColorCharacteristics {
    pub cp: ColourPrimaries,
    pub mc: MatrixCoefficients,
    pub tc: TransferCharacteristic,
}

impl ColorCharacteristics {
    pub fn or(self, other: Self) -> Self {
        Self {
            cp: if matches!(self.cp, ColourPrimaries::Unspecified) {
                other.cp
            } else {
                self.cp
            },
            mc: if matches!(self.mc, MatrixCoefficients::Unspecified) {
                other.mc
            } else {
                self.mc
            },
            tc: if matches!(self.tc, TransferCharacteristic::Unspecified) {
                other.tc
            } else {
                self.tc
            },
        }
    }
}

/// defined by the “Color primaries” section of ISO/IEC 23091-4/ITU-T H.273
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum ColourPrimaries {
    BT709 = 1,
    Unspecified = 2,
    BT601 = 6,
    // TODO complete colour primaries
}

impl ColourPrimaries {
    pub fn from_byte(byte: u8) -> Self {
        assert!(byte <= ColourPrimaries::BT601 as u8);
        unsafe { mem::transmute::<u8, _>(byte) }
    }
}

/// defined by the “Transfer characteristics” section of ISO/IEC 23091-4/ITU-T H.273
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum TransferCharacteristic {
    Reserved = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved2 = 3,
    BT601 = 6,
    // TODO complete transfer characteristic
}

impl TransferCharacteristic {
    pub fn from_byte(byte: u8) -> Self {
        assert!(byte <= TransferCharacteristic::BT601 as u8);
        unsafe { mem::transmute::<u8, _>(byte) }
    }
}

/// defined by the “Matrix coefficients” section of ISO/IEC 23091-4/ITU-T H.273
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved = 3,
    FCC = 4,
    BT601 = 6,
    // TODO complete matrix coefficients
}

impl MatrixCoefficients {
    pub fn from_byte(byte: u8) -> Self {
        assert!(byte <= MatrixCoefficients::BT601 as u8);
        unsafe { mem::transmute::<u8, _>(byte) }
    }
}
