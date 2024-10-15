fn main() {
    nvptx_builder::build_ptx_crate("xpsnr-cuda-kernel", "release-nvptx", true);
}
