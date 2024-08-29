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
        "environment variable CUDA_PATH must be set for cuda-npp-sys to find the CUDA SDK",
    ));
    let include_path = cuda_path.join("include");
    let link_path = cuda_path.join("lib/x64");

    println!("cargo:rustc-link-lib=cuda");
    link_lib("cudart");
    println!("cargo:rustc-link-search={}", link_path.display());
    let mut bindgen = bindgen::Builder::default()
        .clang_args(["-I", include_path.to_str().unwrap()])
        .header(include(&include_path, "nppdefs.h"))
        .header(include(&include_path, "nppcore.h"))
        .must_use_type("NppStatus")
        //.blocklist_type("(Npp8u)|(Npp8s)|(Npp16u)|(Npp16s)|(Npp32s)|(Npp32f)")
        .generate_comments(false)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .use_core()
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .allowlist_function("^npp.*")
        .allowlist_function("^cudaMallocAsync")
        .allowlist_function("^cudaFreeAsync")
        .allowlist_function("^cudaMemcpy2DAsync")
        .allowlist_function("^cudaMemcpyAsync")
        .allowlist_var("^CU.*")
        .allowlist_type("^Npp.*")
        .allowlist_type("^cuda.*")
        .allowlist_type("^cuuint(32|64)_t")
        .must_use_type("NppStatus")
        .must_use_type("cudaError")
        .parse_callbacks(Box::new(CustomCallbacks))
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(true),
        ));

    // npp core
    link_lib("nppc");
    #[cfg(all(feature = "static", target_os = "linux"))]
    link_lib("culibos");

    #[cfg(feature = "ial")]
    {
        link_lib("nppial");
        bindgen = bindgen.header(include(
            &include_path,
            "nppi_arithmetic_and_logical_operations.h",
        ));
    }
    #[cfg(feature = "icc")]
    {
        link_lib("nppicc");
        bindgen = bindgen.header(include(&include_path, "nppi_color_conversion.h"));
    }
    #[cfg(feature = "idei")]
    {
        link_lib("nppidei");
        bindgen = bindgen.header(include(
            &include_path,
            "nppi_data_exchange_and_initialization.h",
        ));
    }
    #[cfg(feature = "if")]
    {
        link_lib("nppif");
        bindgen = bindgen.header(include(&include_path, "nppi_filtering_functions.h"));
    }
    #[cfg(feature = "ig")]
    {
        link_lib("nppig");
        bindgen = bindgen.header(include(&include_path, "nppi_geometry_transforms.h"));
    }
    #[cfg(feature = "ist")]
    {
        link_lib("nppist");
        bindgen = bindgen.header(include(&include_path, "nppi_statistics_functions.h"));
    }
    #[cfg(feature = "isu")]
    {
        link_lib("nppisu");
        bindgen = bindgen.header(include(&include_path, "nppi_support_functions.h"));
    }
    let bindings = bindgen.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[derive(Debug)]
struct CustomCallbacks;

impl ParseCallbacks for CustomCallbacks {
    fn add_derives(&self, info: &DeriveInfo<'_>) -> Vec<String> {
        if info.name == "NppiSize" {
            vec![String::from("PartialEq"), String::from("Eq")]
        } else {
            Vec::new()
        }
    }
}
