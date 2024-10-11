use std::fmt::{Display, Formatter};

pub mod av1;
pub mod h262;
pub mod h264;
pub mod ivf;

#[derive(Debug, Copy, Clone)]
pub enum Codec {
    AV1,
    H264,
    H262,
}

impl Codec {
    pub fn from_fourcc(fourcc: [u8; 4]) -> Option<Self> {
        match &fourcc {
            b"AV01" => Some(Self::AV1),
            b"AVC1" => Some(Self::H264),
            _ => None,
        }
    }
}

impl Display for Codec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:width$}",
            match self {
                Codec::AV1 => "AV1",
                Codec::H264 => "H.264/AVC",
                Codec::H262 => "H.262/MPEG2",
            },
            width = f.width().unwrap_or(0)
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ColorCharacteristics {
    pub cp: ColourPrimaries,
    pub mc: MatrixCoefficients,
    pub tc: TransferCharacteristic,
}

impl ColorCharacteristics {
    pub fn from_codec_bytes(codec: Codec, cp: u8, mc: u8, tc: u8) -> Self {
        match codec {
            Codec::AV1 => Self {
                cp: av1::ColourPrimaries::from_byte(cp).into(),
                mc: av1::MatrixCoefficients::from_byte(mc).into(),
                tc: av1::TransferCharacteristic::from_byte(tc).into(),
            },
            Codec::H264 => Self {
                cp: h264::ColourPrimaries::from_byte(cp).into(),
                mc: h264::MatrixCoefficients::from_byte(mc).into(),
                tc: h264::TransferCharacteristic::from_byte(tc).into(),
            },
            Codec::H262 => Self {
                cp: h262::ColourPrimaries::from_byte(cp).into(),
                mc: h262::MatrixCoefficients::from_byte(mc).into(),
                tc: h262::TransferCharacteristic::from_byte(tc).into(),
            },
        }
    }

    pub fn or(self, other: Self) -> Self {
        Self {
            cp: if matches!(
                self.cp,
                ColourPrimaries::Unspecified | ColourPrimaries::Invalid
            ) {
                other.cp
            } else {
                self.cp
            },
            mc: if matches!(
                self.mc,
                MatrixCoefficients::Unspecified | MatrixCoefficients::Invalid
            ) {
                other.mc
            } else {
                self.mc
            },
            tc: if matches!(
                self.tc,
                TransferCharacteristic::Unspecified | TransferCharacteristic::Invalid
            ) {
                other.tc
            } else {
                self.tc
            },
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ColourPrimaries {
    Invalid,
    Unspecified,
    Unsupported,
    BT709,
    BT601_525,
    BT601_625,
}

impl From<av1::ColourPrimaries> for ColourPrimaries {
    fn from(value: av1::ColourPrimaries) -> Self {
        match value {
            av1::ColourPrimaries::Unspecified => Self::Unspecified,
            av1::ColourPrimaries::BT709 => Self::BT709,
            av1::ColourPrimaries::BT601 => Self::BT601_625,
        }
    }
}

impl From<h264::ColourPrimaries> for ColourPrimaries {
    fn from(value: h264::ColourPrimaries) -> Self {
        match value {
            h264::ColourPrimaries::Reserved | h264::ColourPrimaries::Reserved2 => Self::Invalid,
            h264::ColourPrimaries::Unspecified => Self::Unspecified,
            h264::ColourPrimaries::FCC => Self::Unsupported,
            h264::ColourPrimaries::BT709 => Self::BT709,
            h264::ColourPrimaries::BT601_525 => Self::BT601_525,
            h264::ColourPrimaries::BT601_625 => Self::BT601_625,
        }
    }
}

impl From<h262::ColourPrimaries> for ColourPrimaries {
    fn from(value: h262::ColourPrimaries) -> Self {
        match value {
            h262::ColourPrimaries::Reserved | h262::ColourPrimaries::Forbidden => Self::Invalid,
            h262::ColourPrimaries::Unspecified => Self::Unspecified,
            h262::ColourPrimaries::FCC | h262::ColourPrimaries::Smpte240m => Self::Unsupported,
            h262::ColourPrimaries::BT709 => Self::BT709,
            h262::ColourPrimaries::BT601_525 => Self::BT601_525,
            h262::ColourPrimaries::BT601_625 => Self::BT601_625,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum TransferCharacteristic {
    Invalid,
    Unspecified,
    Unsupported,
    BT709,
}

impl From<av1::TransferCharacteristic> for TransferCharacteristic {
    fn from(value: av1::TransferCharacteristic) -> Self {
        match value {
            av1::TransferCharacteristic::Reserved | av1::TransferCharacteristic::Reserved2 => {
                Self::Invalid
            }
            av1::TransferCharacteristic::Unspecified => Self::Unspecified,
            av1::TransferCharacteristic::BT709 | av1::TransferCharacteristic::BT601 => Self::BT709,
        }
    }
}

impl From<h264::TransferCharacteristic> for TransferCharacteristic {
    fn from(value: h264::TransferCharacteristic) -> Self {
        match value {
            h264::TransferCharacteristic::Reserved | h264::TransferCharacteristic::Reserved2 => {
                Self::Invalid
            }
            h264::TransferCharacteristic::Unspecified => Self::Unspecified,
            h264::TransferCharacteristic::Gamma22 | h264::TransferCharacteristic::Gamma28 => {
                Self::Unsupported
            }
            h264::TransferCharacteristic::BT709 | h264::TransferCharacteristic::BT601 => {
                Self::BT709
            }
        }
    }
}

impl From<h262::TransferCharacteristic> for TransferCharacteristic {
    fn from(value: h262::TransferCharacteristic) -> Self {
        match value {
            h262::TransferCharacteristic::Reserved | h262::TransferCharacteristic::Forbidden => {
                Self::Invalid
            }
            h262::TransferCharacteristic::Unspecified => Self::Unspecified,
            h262::TransferCharacteristic::Gamma22 | h262::TransferCharacteristic::Gamma28 => {
                Self::Unsupported
            }
            h262::TransferCharacteristic::BT709 | h262::TransferCharacteristic::BT601 => {
                Self::BT709
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MatrixCoefficients {
    Invalid,
    Unspecified,
    Unsupported,
    BT709,
    BT601_525,
    BT601_625,
}

impl From<av1::MatrixCoefficients> for MatrixCoefficients {
    fn from(value: av1::MatrixCoefficients) -> Self {
        match value {
            av1::MatrixCoefficients::Reserved => Self::Invalid,
            av1::MatrixCoefficients::Unspecified => Self::Unspecified,
            av1::MatrixCoefficients::Identity | av1::MatrixCoefficients::FCC => Self::Unsupported,
            av1::MatrixCoefficients::BT709 => Self::BT709,
            av1::MatrixCoefficients::BT601 => Self::BT601_625,
        }
    }
}

impl From<h264::MatrixCoefficients> for MatrixCoefficients {
    fn from(value: h264::MatrixCoefficients) -> Self {
        match value {
            h264::MatrixCoefficients::Reserved => Self::Invalid,
            h264::MatrixCoefficients::Unspecified => Self::Unspecified,
            h264::MatrixCoefficients::Identity | h264::MatrixCoefficients::FCC => Self::Unsupported,
            h264::MatrixCoefficients::BT709 => Self::BT709,
            h264::MatrixCoefficients::BT601_525 => Self::BT601_525,
            h264::MatrixCoefficients::BT601_625 => Self::BT601_625,
        }
    }
}

impl From<h262::MatrixCoefficients> for MatrixCoefficients {
    fn from(value: h262::MatrixCoefficients) -> Self {
        match value {
            h262::MatrixCoefficients::Reserved | h262::MatrixCoefficients::Forbidden => {
                Self::Invalid
            }
            h262::MatrixCoefficients::Unspecified => Self::Unspecified,
            h262::MatrixCoefficients::FCC
            | h262::MatrixCoefficients::Smpte240m
            | h262::MatrixCoefficients::YCgCo => Self::Unsupported,
            h262::MatrixCoefficients::BT709 => Self::BT709,
            h262::MatrixCoefficients::BT601_525 => Self::BT601_525,
            h262::MatrixCoefficients::BT601_625 => Self::BT601_625,
        }
    }
}
