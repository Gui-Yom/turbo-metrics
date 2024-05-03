use std::env;

fn main() {
    let cuda_path =
        env::var("CUDA_PATH").expect("CUDA_PATH must be set to the path of your CUDA installation");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-arg={cuda_path}/nvvm/libdevice/libdevice.10.bc");
}
