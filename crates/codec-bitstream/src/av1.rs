/// Extract sequence header obu from mkv CodecPrivate
pub fn extract_seq_hdr_from_mkv_codec_private(codec_private: &[u8]) -> &[u8] {
    let obu = &codec_private[4..];
    obu
}
