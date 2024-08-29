fn main() {
    nvptx_builder::build_ptx_crate("vmaf-cuda-kernel", "release-nvptx");
}
