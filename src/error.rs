//! Error types for ssm-latent-model.

/// Result type alias using [`ModelError`].
pub type Result<T> = std::result::Result<T, ModelError>;

/// Errors that can occur in the SSM latent model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// I/O error during model save/load.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration validation error.
    #[error("Configuration error: {message}")]
    Config {
        /// Human-readable description of the validation failure.
        message: String,
    },

    /// Tensor shape mismatch.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Error during model serialization/deserialization.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Backend-specific error.
    #[error("Backend error: {0}")]
    Backend(String),
}
