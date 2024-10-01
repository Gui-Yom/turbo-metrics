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
    let cuda_path = if cfg!(target_os = "windows") {
        PathBuf::from(env::var("CUDA_PATH").expect(
            "environment variable CUDA_PATH must be set for nv-video-codec-sys to find the CUDA SDK",
        ))
    } else if cfg!(target_os = "linux") {
        PathBuf::from(env::var("CUDA_PATH").unwrap_or("/usr/local/cuda".to_string()))
    } else {
        todo!("Unsupported platform")
    };
    if !cuda_path.exists() || !cuda_path.is_dir() {
        panic!(
            "Path to the CUDA SDK is invalid or inaccessible : {}",
            cuda_path.display()
        );
    }

    let video_sdk_path = PathBuf::from(env::var("NV_VIDEO_CODEC_SDK").expect(
        "environment variable NV_VIDEO_CODEC_SDK must be set for nv-video-codec-sys to find the video codec sdk path.",
    ));
    if !video_sdk_path.exists() || !video_sdk_path.is_dir() {
        panic!(
            "Path to the Video Codec SDK is invalid or inaccessible : {}",
            video_sdk_path.display()
        );
    }
    #[cfg(target_os = "windows")]
    let cuda_link = cuda_path.join("lib/x64");
    #[cfg(target_os = "linux")]
    let cuda_link = cuda_path.join("lib64");

    println!("cargo:rustc-link-search=native={}", cuda_link.display());

    // On Windows, we need to link to the lib in the video codec sdk dir.
    // Linux uses the standard /usr/lib

    // On Windows we link to the .lib files
    #[cfg(target_os = "windows")]
    {
        let sdk_link = video_sdk_path.join("Lib/x64");
        println!("cargo:rustc-link-search=native={}", sdk_link.display());
        link_lib("cuda");
        link_lib("nvcuvid");
    }

    // On Linux, we link to the .so
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=nvcuvid");
    }
    let bindgen = bindgen::Builder::default()
        .clang_args([
            "-I",
            cuda_path.join("include").to_str().unwrap(),
            "-I",
            video_sdk_path.join("Interface").to_str().unwrap(),
        ])
        .header(include(&video_sdk_path.join("Interface"), "nvcuvid.h"))
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
