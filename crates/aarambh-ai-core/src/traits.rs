use std::path::Path;

use candle_core::Tensor;

use crate::config::ModelConfig;
use crate::device::Device;
use crate::error::Result;

/// Minimal forward-pass interface for tensor modules.
pub trait Forward {
    /// Run the module on input tensor `xs`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

/// Interface for components that can persist themselves to disk.
pub trait Saveable {
    /// Save the component to `path`.
    fn save(&self, path: &Path) -> Result<()>;
}

/// Interface for components that can be loaded from disk.
pub trait Loadable: Sized {
    /// Load the component from `path` onto `device`.
    fn load(path: &Path, device: &Device) -> Result<Self>;
}

/// Interface for components that expose their model configuration.
pub trait Configurable {
    /// Return the model configuration backing this component.
    fn config(&self) -> &ModelConfig;
}

/// Common tokenizer interface used by data, training, and inference crates.
pub trait TokenizerLike {
    /// Encode text into token ids.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    /// Decode token ids into text.
    fn decode(&self, ids: &[u32]) -> Result<String>;
    /// Return the tokenizer vocabulary size.
    fn vocab_size(&self) -> usize;
    /// Return the end-of-sequence token id.
    fn eos_token_id(&self) -> u32;
    /// Return the optional beginning-of-sequence token id.
    fn bos_token_id(&self) -> Option<u32>;
}
