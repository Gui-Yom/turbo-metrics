use std::env;
use std::path::{Path, PathBuf};

use bindgen::EnumVariation;

fn include(path: &Path, header: &str) -> String {
    path.join(header).to_str().unwrap().to_string()
}

fn main() {
    let vmaf_path = PathBuf::from("C:\\Code\\C\\vmaf\\libvmaf\\build\\install");
    let vmaf_include_path = vmaf_path.join("include");
    let vmaf_link_path = vmaf_path.join("lib");
    let cuda_path = PathBuf::from(env::var("CUDA_PATH").expect(
        "environment variable CUDA_PATH must be set for nv-video-codec-sys to find the CUDA SDK",
    ));
    let cuda_include_path = cuda_path.join("include");
    let cuda_link_path = cuda_path.join("lib/x64");

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=static:+whole-archive=vmaf");
    println!("cargo:rustc-link-lib=static=pthreadVC3");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_link_path.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        vmaf_link_path.display()
    );

    let bindgen = bindgen::Builder::default()
        .clang_args([
            "-I",
            vmaf_include_path.to_str().unwrap(),
            "-I",
            cuda_include_path.to_str().unwrap(),
        ])
        .header(include(&vmaf_include_path, "libvmaf/libvmaf.h"))
        .header(include(&vmaf_include_path, "libvmaf/libvmaf_cuda.h"))
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .allowlist_function("^vmaf.*")
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(true),
        ));

    let bindings = bindgen.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
