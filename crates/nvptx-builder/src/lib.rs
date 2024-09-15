use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

/// Build a crate in the workspace as ptx and copy the resulting ptx to this build script `OUT_DIR`
///
/// You would then include the ptx code at compile time with :
/// ```rust,ignore
/// include_str!(concat!(env!("OUT_DIR"),"/{package}.ptx"))
/// ```
pub fn build_ptx_crate(package: &str, profile: &str) {
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
    copy_file(package, profile);
}

fn copy_file(package: &str, profile: &str) {
    let package = package.replace('-', "_");
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Dubious way of getting the workspace target dir.
    let target_dir = dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    // target/{TARGET}/{PROFILE}/{CRATE}.ptx
    let mut src_path = target_dir.join("nvptx64-nvidia-cuda");
    src_path.push(profile);
    src_path.push(&package);
    src_path.set_extension("ptx");
    fs::copy(dbg!(src_path), dir.join(&package).with_extension("ptx")).unwrap();
}
