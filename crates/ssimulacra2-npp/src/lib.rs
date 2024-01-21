use cuda_npp::safe::icc::GammaFwdIP;
use cuda_npp::safe::ig::Resize;
use cuda_npp::safe::Image;
use cuda_npp::safe::Result;
use cuda_npp::{get_stream_ctx, C};

pub fn ssimulacra2(mut source: Image<u8, C<3>>, mut distorted: Image<u8, C<3>>) -> Result<f32> {
    let dev = cudarc::driver::safe::CudaDevice::new(0).unwrap();
    let ctx = get_stream_ctx()?;
    source.gamma_fwd_ip(ctx)?;
    distorted.gamma_fwd_ip(ctx)?;

    // TODO convert to linear

    for scale in 0..6 {
        if scale > 0 {
            let new_width = (source.width + 1) / 2;
            let new_height = (source.height + 1) / 2;
            source = source.resize(new_width, new_height, ctx)?;
            distorted = distorted.resize(new_width, new_height, ctx)?;
        }

        // TODO convert to xyb

        // TODO mul planes
    }

    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use cuda_npp::safe::isu::Malloc;

    use super::*;

    #[test]
    fn it_works() {
        let source = zune_image::image::Image::open("../ssimulacra2-cuda/source.png").unwrap();
        assert_eq!(source.channels_ref(true).len(), 3);
        let source_bytes = &source.flatten_to_u8()[0];
        let mut source_img =
            Image::<u8, C<3>>::malloc(source.dimensions().0 as u32, source.dimensions().1 as u32)
                .unwrap();
        source_img.copy_from_cpu(&source_bytes).unwrap();

        let dis = zune_image::image::Image::open("../ssimulacra2-cuda/distorted.png").unwrap();
        assert_eq!(dis.channels_ref(true).len(), 3);
        let dis_bytes = &dis.flatten_to_u8()[0];
        let mut dis_img =
            Image::<u8, C<3>>::malloc(dis.dimensions().0 as u32, dis.dimensions().1 as u32)
                .unwrap();
        dis_img.copy_from_cpu(&dis_bytes).unwrap();

        dbg!(ssimulacra2(source_img, dis_img));
    }
}
