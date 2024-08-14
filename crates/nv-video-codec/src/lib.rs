pub use nv_video_codec_sys as sys;

/// Thin layer over the raw bindings for a working decoder.
pub mod dec;
/// Multithreaded, complex and probably full of errors implementation of a decoder.
pub mod dec_mt;
/// Simpler decoder implementation using no additional thread.
pub mod dec_simple;
pub mod enc;
