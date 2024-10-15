fn main() {
    nvptx_builder::build_ptx_crate("cuda-colorspace-kernel", "release-nvptx", true);
}
