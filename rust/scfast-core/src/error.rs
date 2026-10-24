/// Errors returned by scfast kernels. Bindings turn these into R / Python errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Inputs with inconsistent shapes or invalid sparse structure.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A parameter outside its valid range (e.g. more components than genes).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Failure inside a dependency (linear algebra, clustering library).
    #[error("{context}: {message}")]
    Backend { context: &'static str, message: String },
    /// A CUDA device was requested but could not be used: not compiled in, no driver or
    /// device, out of device memory, or a kernel / library failure. Callers that resolved
    /// `device = "auto"` may fall back to the CPU path on this variant (and say so).
    #[error("GPU: {0}")]
    Gpu(String),
}

impl Error {
    pub(crate) fn gpu(msg: impl Into<String>) -> Self {
        Error::Gpu(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
