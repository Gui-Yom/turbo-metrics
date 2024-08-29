use std::env;
use std::path::{Path, PathBuf};

use bindgen::EnumVariation;

fn link_lib(lib: &str) {
    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static={lib}");
    } else {
        println!("cargo:rustc-link-lib={lib}");
    }
}

fn include(path: &Path, header: &str) -> String {
    path.join(header).to_str().unwrap().to_string()
}

fn main() {
    let sdk_path = PathBuf::from(env::var("NV_VIDEO_CODEC_SDK").expect(
        "environment variable NV_VIDEO_CODEC_SDK must be set for nv-video-codec-sys to find the video codec sdk path.",
    ));
    let sdk_include = sdk_path.join("Interface");
    let sdk_lib = sdk_path.join("Lib/x64");
    let cuda_path = PathBuf::from(env::var("CUDA_PATH").expect(
        "environment variable CUDA_PATH must be set for nv-video-codec-sys to find the CUDA SDK",
    ));
    let cuda_include_path = cuda_path.join("include");
    let cuda_link_path = cuda_path.join("lib/x64");

    // println!("cargo:rustc-link-lib=cuda");
    link_lib("cuda");
    link_lib("nvcuvid");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_link_path.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_lib.display());
    let bindgen = bindgen::Builder::default()
        .clang_args([
            "-I",
            cuda_include_path.to_str().unwrap(),
            "-I",
            sdk_include.to_str().unwrap(),
        ])
        .header(include(&sdk_include, "nvcuvid.h"))
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .allowlist_function("^cuvid.*")
        .allowlist_type("^CUvideo.*")
        .allowlist_type("^CUVIDEO.*")
        .allowlist_type("^cudaAudio.*")
        .allowlist_type("^cudaVideo.*")
        .allowlist_type("^CUAUDIO.*")
        .allowlist_type("^CUVID.*")
        .blocklist_type("^CUresult")
        .blocklist_type("^CUcontext")
        .blocklist_item("CUcontext")
        .bitfield_enum("CUvideopacketflags")
        .must_use_type("^CUresult")
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
