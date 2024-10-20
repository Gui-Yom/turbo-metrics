use crate::{nvdec_to_codec, HwFrame};
use codec_bitstream::{
    ColorCharacteristics, ColourPrimaries, MatrixCoefficients, TransferCharacteristic,
};
use cuda_colorspace::{ColorMatrix, ColorspaceConversion, Transfer};
use cudarse_driver::CuStream;
use cudarse_npp::image::{ImgMut, C};
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::sys::CUVIDEOFORMAT;

#[derive(Debug, Copy, Clone)]
pub enum ColorRange {
    Limited,
    Full,
}

impl From<bool> for ColorRange {
    fn from(value: bool) -> Self {
        if value {
            ColorRange::Full
        } else {
            ColorRange::Limited
        }
    }
}

impl Into<bool> for ColorRange {
    fn into(self) -> bool {
        match self {
            ColorRange::Limited => false,
            ColorRange::Full => true,
        }
    }
}

pub fn color_characteristics_from_format(
    format: &CUVIDEOFORMAT,
) -> (ColorCharacteristics, ColorRange) {
    (
        ColorCharacteristics::from_codec_bytes(
            nvdec_to_codec(format.codec),
            format.video_signal_description.color_primaries,
            format.video_signal_description.matrix_coefficients,
            format.video_signal_description.transfer_characteristics,
        )
        .or(color_characteristics_fallback(format)),
        format.video_signal_description.full_range().into(),
    )
}

fn color_characteristics_fallback(format: &CUVIDEOFORMAT) -> ColorCharacteristics {
    let height = format.display_height();
    if height <= 525 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT601_525,
            mc: MatrixCoefficients::BT601_525,
            tc: TransferCharacteristic::BT709,
        }
    } else if height <= 625 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT601_625,
            mc: MatrixCoefficients::BT601_625,
            tc: TransferCharacteristic::BT709,
        }
    } else if height <= 1080 {
        ColorCharacteristics {
            cp: ColourPrimaries::BT709,
            mc: MatrixCoefficients::BT709,
            tc: TransferCharacteristic::BT709,
        }
    } else {
        ColorCharacteristics {
            cp: ColourPrimaries::BT709,
            mc: MatrixCoefficients::BT709,
            tc: TransferCharacteristic::BT709,
        }
    }
}

pub fn get_color_matrix(colors: &ColorCharacteristics) -> ColorMatrix {
    match (colors.cp, colors.mc) {
        (ColourPrimaries::BT709, MatrixCoefficients::BT709) => ColorMatrix::BT709,
        (ColourPrimaries::BT601_525, MatrixCoefficients::BT601_525) => ColorMatrix::BT601_525,
        (ColourPrimaries::BT601_625, MatrixCoefficients::BT601_625) => ColorMatrix::BT601_625,
        _ => todo!(),
    }
}

pub fn get_transfer(colors: &ColorCharacteristics) -> Transfer {
    match colors.tc {
        TransferCharacteristic::BT709 => Transfer::BT709,
        _ => todo!(),
    }
}

pub fn convert_frame_to_linearrgb(
    frame: HwFrame<'_>,
    colors: (ColorCharacteristics, ColorRange),
    colorspace: &ColorspaceConversion,
    dst: impl ImgMut<f32, C<3>>,
    stream: &CuStream,
) {
    let color_matrix = get_color_matrix(&colors.0);
    let transfer = get_transfer(&colors.0);
    match frame {
        HwFrame::NvDec(NvDecFrame::NV12(f)) => colorspace
            .biplanaryuv420_to_linearrgb_8(color_matrix, transfer, colors.1.into(), f, dst, stream)
            .unwrap(),
        HwFrame::NvDec(NvDecFrame::P016(f)) => colorspace
            .biplanaryuv420_to_linearrgb_16(color_matrix, transfer, colors.1.into(), f, dst, stream)
            .unwrap(),
        HwFrame::Npp8(f) => colorspace.srgb_to_linear_u8(f, dst, &stream).unwrap(),
        HwFrame::Npp16(f) => colorspace.srgb_to_linear_u16(f, dst, &stream).unwrap(),
        HwFrame::Npp32(f) => colorspace.srgb_to_linear_f32(f, dst, &stream).unwrap(),
    }
}
