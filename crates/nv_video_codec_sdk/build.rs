fn main() {
    println!("cargo:rustc-link-lib=nvcuvid");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-search=C:\\apps\\Video_Codec_SDK\\Lib\\x64");
    println!("cargo:rustc-link-search=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\lib\\x64");
    // bindgen wrapper.h -o src/bindings.rs --default-enum-style rust --bitfield-enum CUvideopacketflags --with-derive-default --use-core --sort-semantically --merge-extern-blocks -- -IC:\apps\Video_Codec_SDK\Interface -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include"
}
