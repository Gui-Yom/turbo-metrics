use cargo_metadata::MetadataCommand;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// Build a crate in the workspace as ptx and copy the resulting ptx to this build script `OUT_DIR`
///
/// You would then include the ptx code at compile time with :
/// ```rust,ignore
/// include_str!(concat!(env!("OUT_DIR"),"/{package}.ptx"))
/// ```
pub fn build_ptx_crate(package: &str, profile: &str, validate: bool) {
    let meta = MetadataCommand::new().no_deps().exec().unwrap();
    let p = meta.packages.iter().find(|p| p.name == package).unwrap();
    println!("cargo:rerun-if-changed={}", p.manifest_path);
    for target in &p.targets {
        if target.is_custom_build() {
            println!("cargo:rerun-if-changed={}", target.src_path);
        } else {
            println!(
                "cargo:rerun-if-changed={}",
                target.src_path.parent().unwrap()
            );
        }
    }

    // We need to filter environment variables, or it breaks cargo when called from a build script
    let envs = env::vars().filter(|(k, _)| {
        !k.starts_with("CARGO_")
            && k != "CARGO"
            && k != "RUSTC"
            && k != "RUSTDOC"
            && k != "RUSTUP_TOOLCHAIN"
            && k != "OPT_LEVEL"
            && k != "OUT_DIR"
            || k == "CARGO_MAKEFLAGS"
            || k == "CARGO_HOME"
    }); //.collect::<Vec<_>>();
        // Convert current stable toolchain to nightly toolchain.
    let toolchain = env::var("RUSTUP_TOOLCHAIN").unwrap();
    let toolchain = toolchain.split_once('-').unwrap().1;
    //dbg!(&toolchain);
    let mut cmd = Command::new("cargo")
        .env_clear()
        .envs(envs)
        .env("RUSTUP_TOOLCHAIN", format!("nightly-{toolchain}"))
        .args([
            "+nightly",
            //"-vv",
            "build",
            "--package",
            package,
            "--target",
            "nvptx64-nvidia-cuda",
            "--profile",
            profile,
        ])
        //.current_dir(format!("../{package}"))
        .spawn()
        .unwrap();
    assert!(cmd.wait().unwrap().success());
    copy_file(
        meta.target_directory.as_std_path(),
        package,
        profile,
        validate,
    );
}

fn copy_file(target_dir: &Path, package: &str, profile: &str, validate: bool) {
    let package = package.replace('-', "_");
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // target/{TARGET}/{PROFILE}/{CRATE}.ptx
    let mut src_path = target_dir.join("nvptx64-nvidia-cuda");
    src_path.push(profile);
    src_path.push(&package);
    src_path.set_extension("ptx");

    if validate {
        assert!(
            Command::new("ptxas")
                .args([
                    "--compile-only",
                    "--warn-on-spills",
                    "--warning-as-error",
                    "--verbose",
                    "--output-file",
                ])
                .arg(if cfg!(target_os = "linux") {
                    "/dev/null"
                } else if cfg!(target_os = "windows") {
                    "nul"
                } else {
                    todo!()
                })
                .arg(&src_path)
                .status()
                .is_ok_and(|s| s.success()),
            "ptxas validation failed"
        );
    }

    fs::copy(dbg!(src_path), dir.join(&package).with_extension("ptx")).unwrap();
}

pub fn link_libdevice() {
    let cuda_path =
        env::var("CUDA_PATH").expect("CUDA_PATH must be set to the path of your CUDA installation");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-link-arg={cuda_path}/nvvm/libdevice/libdevice.10.bc");
}

// See: https://github.com/rust-lang/rust/blob/564758c4c329e89722454dd2fbb35f1ac0b8b47c/src/bootstrap/dist.rs#L2334-L2341
fn rustlib() -> PathBuf {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(rustc)
        .arg("--print")
        .arg("sysroot")
        .output()
        .unwrap();
    let sysroot = String::from_utf8(output.stdout).unwrap().trim().to_owned();
    let mut pathbuf = PathBuf::from(sysroot);
    pathbuf.push("lib");
    pathbuf.push("rustlib");
    pathbuf.push(env::var("HOST").expect("No HOST set for host triple"));
    pathbuf.push("bin");
    pathbuf
}

/// Compile and link a llvm .ll file
pub fn link_llvm_ir_file(file: impl AsRef<Path>) {
    let path = file.as_ref();
    println!("cargo:rerun-if-changed={}", path.display());
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap())
        .join(path.file_name().unwrap())
        .with_extension("bc");
    assert!(Command::new(rustlib().join("llvm-as"))
        .arg("-o")
        .arg(&out_path)
        .arg(path)
        .status()
        .unwrap()
        .success());
    println!("cargo:rustc-link-arg={}", out_path.display());
}

/// Compile and link a llvm .ll file
pub fn link_llvm_ir(bitcode: &str) {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut hash = DefaultHasher::new();
    bitcode.hash(&mut hash);
    let hash = hash.finish();
    let path = out.join(&format!("{:x}.ll", hash));
    fs::write(&path, bitcode).unwrap();
    link_llvm_ir_file(&path);
}
