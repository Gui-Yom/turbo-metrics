use std::time::Instant;

use zune_image::codecs::png::zune_core::options::DecoderOptions;

use cuda_driver::CuDevice;
use ssimulacra2_cuda::cpu::CpuImg;
use ssimulacra2_cuda::{cpu, Ssimulacra2};

fn main() {
    cuda_driver::init_cuda().expect("Could not initialize the CUDA API");
    let dev = CuDevice::get(0).unwrap();
    println!(
        "Using device {} with CUDA version {}",
        dev.name().unwrap(),
        cuda_driver::cuda_driver_version().unwrap()
    );
    // Bind to main thread
    dev.retain_primary_ctx().unwrap().set_current().unwrap();

    let ref_img = zune_image::image::Image::open_with_options(
        "crates/ssimulacra2-cuda/source.png",
        DecoderOptions::new_fast(),
    )
    .unwrap();
    let dis_img = zune_image::image::Image::open_with_options(
        "crates/ssimulacra2-cuda/distorted.png",
        DecoderOptions::new_fast(),
    )
    .unwrap();

    // Upload to gpu
    let (width, height) = ref_img.dimensions();
    let ref_bytes = &ref_img.flatten_to_u8()[0];

    let (dwidth, dheight) = dis_img.dimensions();
    assert_eq!((width, height), (dwidth, dheight));
    let dis_bytes = &dis_img.flatten_to_u8()[0];

    let mut ssimulacra2 = Ssimulacra2::new(width as u32, height as u32).unwrap();
    println!(
        "Approximate computed memory usage : {} MB",
        ssimulacra2.mem_usage() / 1024 / 1024
    );
    let (free, total) = cuda_driver::mem_info().unwrap();
    println!(
        "Reported memory usage : {} MB",
        (total - free) / 1024 / 1024
    );
    let start = Instant::now();
    let gpu_score = dbg!(ssimulacra2.compute(ref_bytes, dis_bytes)).unwrap();
    println!(
        "GPU: Finished computing a single frame in {} ms",
        start.elapsed().as_millis()
    );

    let c_score = 17.398_505_f64;
    assert!(
        (gpu_score - c_score).abs() < 0.25f64,
        "GPU result {gpu_score:.6} not equal to expected {c_score:.6}",
    );

    let start = Instant::now();
    let ref_img = CpuImg::from_srgb(ref_bytes, width, height);
    let dis_img = CpuImg::from_srgb(dis_bytes, width, height);
    let cpu_score = dbg!(cpu::compute_frame_ssimulacra2(&ref_img, &dis_img));
    println!(
        "CPU: Finished computing a single frame in {} ms",
        start.elapsed().as_millis()
    );

    assert!(
        (cpu_score - c_score).abs() < 0.25f64,
        "CPU result {cpu_score:.6} not equal to expected {c_score:.6}",
    );

    println!(
        "Error between CPU and GPU : {}",
        (cpu_score - gpu_score).abs()
    );
}
