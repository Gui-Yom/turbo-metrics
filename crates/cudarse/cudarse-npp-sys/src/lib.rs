#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::fmt::{Debug, Display, Formatter};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl NppiRect {
    pub fn transpose(self) -> Self {
        Self {
            width: self.height,
            height: self.width,
            x: self.x,
            y: self.y,
        }
    }

    pub fn norm(self) -> f64 {
        1.0 / ((self.width - self.x) * (self.height - self.y)) as f64
    }
}

impl NppStatus {
    pub fn result(self) -> Result<()> {
        if self == NppStatus::NPP_NO_ERROR {
            Ok(())
        } else {
            Err(Error::from(self))
        }
    }

    pub fn result_with<T>(self, value: T) -> Result<T> {
        self.result().map(|_| value)
    }
}

impl cudaError {
    pub fn result(self) -> Result<()> {
        if self == cudaError::cudaSuccess {
            Ok(())
        } else {
            Err(Error::from(self))
        }
    }

    pub fn result_with<T>(self, value: T) -> Result<T> {
        self.result().map(|_| value)
    }
}

#[derive(Debug)]
pub enum Error {
    NppError(NppStatus),
    CudaError(cudaError),
}

impl From<NppStatus> for Error {
    fn from(value: NppStatus) -> Self {
        Error::NppError(value)
    }
}

impl From<cudaError> for Error {
    fn from(value: cudaError) -> Self {
        Error::CudaError(value)
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

impl NppStreamContext {
    pub fn with_stream(&self, stream: cudaStream_t) -> Self {
        Self {
            hStream: stream,
            ..*self
        }
    }
}
