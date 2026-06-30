//! Core configuration, device, dtype, error, and trait types shared by Aarambh AI crates.
#![deny(missing_docs)]

/// Model and training configuration schemas.
pub mod config;
/// Device selection helpers.
pub mod device;
/// Numeric precision and dtype helpers.
pub mod dtype;
/// Shared error and result types.
pub mod error;
/// Common traits implemented by models, tokenizers, and serializable components.
pub mod traits;

pub use config::ModelConfig;
pub use config::TrainConfig;
pub use device::Device;
pub use dtype::DType;
pub use dtype::Precision;
pub use error::AarambhError;
pub use error::Result;
pub use traits::Configurable;
pub use traits::Forward;
pub use traits::Loadable;
pub use traits::Saveable;
pub use traits::TokenizerLike;
