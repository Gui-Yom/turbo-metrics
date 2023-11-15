use crate::{generic, ChannelLayout, Sample};
use cuda_npp_sys::NppStatus;

use crate::result::Result;

/// returned device pointer is guaranteed to not be null.
/// Step is in bytes
pub fn malloc<S: Sample + 'static, C: ChannelLayout + 'static>(
    width: i32,
    height: i32,
) -> Result<(*const S, i32)> {
    let mut step = 0;
    let ptr = unsafe { generic::malloc::<S, C>(width, height, &mut step) };
    if ptr.is_null() {
        Err(NppStatus::NPP_MEMORY_ALLOCATION_ERR)
    } else {
        Ok((ptr, step))
    }
}
