pub enum VideoCodec {
    H264,
}

pub struct FrameNV12 {
    pub width: usize,
    pub height: usize,
    //
    pub buf: Vec<u8>,
}
