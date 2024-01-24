#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl NppStatus {
    pub fn result(self) -> Result<(), NppStatus> {
        if self == NppStatus::NPP_NO_ERROR {
            Ok(())
        } else {
            Err(self)
        }
    }
}
