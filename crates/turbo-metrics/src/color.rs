use crate::img::{reinterpret_slice, CpuImg, SampleType};
use crate::npp;
use codec_bitstream::{
    Codec, ColorCharacteristics, ColourPrimaries, MatrixCoefficients, TransferCharacteristic,
};
use cuda_colorspace::{ColorMatrix, ColorspaceConversion, Transfer};
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{ImgMut, C};
use cudarse_video::dec::npp::NvDecFrame;
use cudarse_video::sys::{cudaVideoCodec, cudaVideoCodec_enum, CUVIDEOFORMAT};
use std::fmt::Display;

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

pub fn video_color_print(format: &CUVIDEOFORMAT) -> impl Display {
    let (colors, full_range) = color_characteristics_from_format(format);
    format!(
        "CP: {:?}, MC: {:?}, TC: {:?}, Full range: {}",
        colors.cp, colors.mc, colors.tc, full_range
    )
}

pub fn convert_frame_to_linearrgb(
    frame: NvDecFrame<'_>,
    colors: (ColorCharacteristics, bool),
    colorspace: &ColorspaceConversion,
    dst: impl ImgMut<f32, C<3>>,
    stream: &CuStream,
) {
    let color_matrix = get_color_matrix(&colors.0);
    let transfer = get_transfer(&colors.0);
    match frame {
        NvDecFrame::NV12(frame) => colorspace
            .biplanaryuv420_to_linearrgb_8(color_matrix, transfer, colors.1, frame, dst, stream)
            .unwrap(),
        NvDecFrame::P016(frame) => colorspace
            .biplanaryuv420_to_linearrgb_16(color_matrix, transfer, colors.1, frame, dst, stream)
            .unwrap(),
    };
}

pub(crate) fn cpu_to_linear(
    src: &CpuImg,
    dst: impl ImgMut<f32, C<3>>,
    conversion: &ColorspaceConversion,
    stream: &CuStream,
) {
    // TODO assume srgb
    match src.sample_type {
        SampleType::U8 => {
            let mut tmp = npp::image::Image::malloc(src.width, src.height).unwrap();
            tmp.copy_from_cpu(&src.data, stream.inner() as _).unwrap();
            conversion.srgb_to_linear_u8(&tmp, dst, &stream).unwrap();
        }
        SampleType::U16 => {
            let mut tmp = npp::image::Image::malloc(src.width, src.height).unwrap();
            tmp.copy_from_cpu(reinterpret_slice(&src.data), stream.inner() as _)
                .unwrap();
            conversion.srgb_to_linear_u16(&tmp, dst, &stream).unwrap();
        }
        SampleType::F32 => {
            let mut tmp = npp::image::Image::malloc(src.width, src.height).unwrap();
            tmp.copy_from_cpu(reinterpret_slice(&src.data), stream.inner() as _)
                .unwrap();
            conversion.srgb_to_linear_f32(&tmp, dst, &stream).unwrap();
        }
    }
}
