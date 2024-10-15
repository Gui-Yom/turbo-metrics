fn main() {
    nvptx_builder::link_libdevice();
    //     nvptx_builder::link_bitcode(
    //         r#"
    // target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
    // target triple = "nvptx64-nvidia-cuda"
    //
    // ; Minimal bitcode to include a cuda shared memory in a Rust kernel.
    // ; Originally compiled from shared.cu, then cleaned by hand.
    //
    // @RING = dso_local addrspace(3) global [16 x [16 x float]] undef, align 4
    //     "#,
    //     );
}
