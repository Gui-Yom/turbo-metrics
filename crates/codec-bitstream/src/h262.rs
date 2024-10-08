use std::mem;

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

/// As defined in "Table 6-7 – Colour primaries" of the H.262 specification.
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum ColourPrimaries {
    Forbidden = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved = 3,
    FCC = 4,
    BT601_625 = 5,
    BT601_525 = 6,
    Smpte240m = 7,
    // rest is reserved
}

impl ColourPrimaries {
    pub fn from_byte(byte: u8) -> Self {
        if byte > ColourPrimaries::Smpte240m as u8 {
            Self::Reserved
        } else {
            unsafe { mem::transmute::<u8, ColourPrimaries>(byte) }
        }
    }
}

/// As defined in "Table 6-8 – Transfer characteristics" of the H.262 specification.
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum TransferCharacteristic {
    Forbidden = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved = 3,
    Gamma22 = 4,
    Gamma28 = 5,
    BT601 = 6,
    // TODO complete transfer characteristic
}

impl TransferCharacteristic {
    pub fn from_byte(byte: u8) -> Self {
        assert!(byte <= TransferCharacteristic::BT601 as u8);
        unsafe { mem::transmute::<u8, TransferCharacteristic>(byte) }
    }
}

/// As defined in "Table 6-9 – Matrix coefficients" of the H.262 specification.
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Forbidden = 0,
    BT709 = 1,
    Unspecified = 2,
    Reserved = 3,
    FCC = 4,
    BT601_625 = 5,
    BT601_525 = 6,
    Smpte240m = 7,
    YCgCo = 8,
    // rest is reserved
}

impl MatrixCoefficients {
    pub fn from_byte(byte: u8) -> Self {
        if byte > MatrixCoefficients::YCgCo as u8 {
            Self::Reserved
        } else {
            unsafe { mem::transmute::<u8, MatrixCoefficients>(byte) }
        }
    }
}
