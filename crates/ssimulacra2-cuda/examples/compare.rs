use std::env;
use std::time::Instant;

use zune_image::codecs::png::zune_core::options::DecoderOptions;

use cpu::CpuImg;
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img};
use cudarse_npp::set_stream;
use ssimulacra2_cuda::Ssimulacra2;

mod cpu;

fn main() {
    cudarse_driver::init_cuda_and_primary_ctx().expect("Could not initialize CUDA API");

    let mut args = env::args().skip(1);
    let ref_path = args.next().unwrap();
    let dis_path = args.next().unwrap();

    let ref_img =
        zune_image::image::Image::open_with_options(&ref_path, DecoderOptions::new_fast()).unwrap();
    let dis_img =
        zune_image::image::Image::open_with_options(&dis_path, DecoderOptions::new_fast()).unwrap();

    // Upload to gpu
    assert_eq!(ref_img.dimensions(), dis_img.dimensions());
    let (width, height) = ref_img.dimensions();

    let ref_bytes = &ref_img.flatten_to_u8()[0];
    let dis_bytes = &dis_img.flatten_to_u8()[0];

    let stream = CuStream::new().unwrap();
    set_stream(stream.inner() as _).unwrap();

    let mut tmp_ref = Image::malloc(width as _, height as _).unwrap();
    let mut tmp_dis = tmp_ref.malloc_same_size().unwrap();
    let mut ref_linear = tmp_ref.malloc_same_size().unwrap();
    let mut dis_linear = tmp_ref.malloc_same_size().unwrap();

    let mut ssimulacra2 = Ssimulacra2::new(&ref_linear, &dis_linear, &stream).unwrap();
    println!(
        "Approximate computed memory usage : {} MB",
        ssimulacra2.mem_usage() / 1024 / 1024
    );
    let (free, total) = cudarse_driver::mem_info().unwrap();
    println!(
        "Reported memory usage : {} MB",
        (total - free) / 1024 / 1024
    );
    let start = Instant::now();
    let gpu_score = dbg!(ssimulacra2.compute_from_cpu_srgb_sync(
        ref_bytes,
        dis_bytes,
        &mut tmp_ref,
        &mut tmp_dis,
        &mut ref_linear,
        &mut dis_linear,
        &stream
    ))
    .unwrap();
    let elapsed = start.elapsed().as_nanos();
    println!(
        "GPU: Finished computing a single frame in {:.2} ms ({:.1} fps)",
        elapsed as f64 / 1000_000.0,
        1000_000_000.0 / elapsed as f64
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
    let elapsed = start.elapsed().as_millis();
    println!(
        "CPU: Finished computing a single frame in {} ms ({:.1} fps)",
        elapsed,
        1000.0 / elapsed as f64
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
