use std::env;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use bindgen::callbacks::{DeriveInfo, ParseCallbacks};
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
    let cuda_path = PathBuf::from(env::var("CUDA_PATH").expect(
        "environment variable CUDA_PATH must be set for cuda_npp_sys to find the CUDA SDK",
    ));
    let include_path = cuda_path.join("include");
    let link_path = cuda_path.join("lib/x64");

    // println!("cargo:rustc-link-lib=cuda");
    link_lib("cuda");
    println!("cargo:rustc-link-search={}", link_path.display());
    let mut bindgen = bindgen::Builder::default()
        .clang_args(["-I", include_path.to_str().unwrap()])
        .header(include(&include_path, "cuda.h"))
        .header(include(&include_path, "cudaProfiler.h"))
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
