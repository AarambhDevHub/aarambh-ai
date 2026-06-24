use std::path::Path;

use candle_core::Tensor;

use crate::config::ModelConfig;
use crate::device::Device;
use crate::error::Result;

pub trait Forward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub trait Saveable {
    fn save(&self, path: &Path) -> Result<()>;
}

pub trait Loadable: Sized {
    fn load(path: &Path, device: &Device) -> Result<Self>;
}

pub trait Configurable {
    fn config(&self) -> &ModelConfig;
}

pub trait TokenizerLike {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> u32;
    fn bos_token_id(&self) -> Option<u32>;
}
