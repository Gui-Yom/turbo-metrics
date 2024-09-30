use std::env;
use std::path::{Path, PathBuf};

use bindgen::EnumVariation;

fn include(path: &Path, header: &str) -> String {
    path.join(header).to_str().unwrap().to_string()
}

fn main() {
    let cuda_path = if cfg!(target_os = "windows") {
        PathBuf::from(env::var("CUDA_PATH").expect(
            "environment variable CUDA_PATH must be set for nv-video-codec-sys to find the CUDA SDK",
        ))
    } else if cfg!(target_os = "linux") {
        PathBuf::from(env::var("CUDA_PATH").unwrap_or(
            "/usr/local/cuda".to_string()
        ))
    } else {
        todo!("Unsupported platform")
    };
    if !cuda_path.exists() || !cuda_path.is_dir() {
        panic!("Path to the CUDA SDK is invalid or inaccessible : {}", cuda_path.display());
    }
    let cuda_include = cuda_path.join("include");
    #[cfg(target_os = "windows")]
    let cuda_link_path = cuda_path.join("lib/x64");

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-search={}", cuda_link.display());
        if cfg!(feature = "static") {
            println!("cargo:rustc-link-lib=static={lib}");
        } else {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=cuda");
    let bindgen = bindgen::Builder::default()
        .clang_args(["-I", cuda_include.to_str().unwrap()])
        .header(include(&cuda_include, "cuda.h"))
        .header(include(&cuda_include, "cudaProfiler.h"))
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .allowlist_function("^cu.*")
        .allowlist_var("^CU.*")
        .allowlist_type("^CU.*")
        .allowlist_type("^cuda.*")
        .allowlist_type("^cudaError_enum")
        .allowlist_type("^cuuint(32|64)_t")
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
