use codec_bitstream::{
    Codec, ColorCharacteristics, ColourPrimaries, MatrixCoefficients, TransferCharacteristic,
};
use cuda_colorspace::{ColorMatrix, Transfer};
pub use cudarse_driver;
pub use cudarse_npp;
pub use cudarse_video;
use cudarse_video::sys::{cudaVideoCodec, cudaVideoCodec_enum, CUVIDEOFORMAT};
pub use stats;

pub fn color_characteristics_from_format(format: &CUVIDEOFORMAT) -> (ColorCharacteristics, bool) {
    (
        ColorCharacteristics::from_codec_bytes(
            cuda_codec_to_codec(format.codec),
            format.video_signal_description.color_primaries,
            format.video_signal_description.matrix_coefficients,
            format.video_signal_description.transfer_characteristics,
        )
        .or(color_characteristics_fallback(format)),
        format.video_signal_description.full_range(),
    )
}

fn cuda_codec_to_codec(codec: cudaVideoCodec) -> Codec {
    match codec {
        cudaVideoCodec_enum::cudaVideoCodec_MPEG2 => Codec::H262,
        cudaVideoCodec_enum::cudaVideoCodec_H264 => Codec::H264,
        cudaVideoCodec_enum::cudaVideoCodec_AV1 => Codec::AV1,
        _ => todo!(),
    }
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
