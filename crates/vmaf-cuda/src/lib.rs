const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vmaf_cuda_kernel.ptx"));

#[cfg(test)]
mod tests {
    use crate::PTX;

    #[test]
    fn ptx_is_included() {
        dbg!(PTX);
    }
}
