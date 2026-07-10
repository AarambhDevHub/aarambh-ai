use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Result};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};

const IMAGE_TOKENS: &str = "image_tokens";

/// Persistent cache for frozen projected image tokens used by vision replay.
#[derive(Debug, Clone)]
pub struct VisionCache {
    state_dir: PathBuf,
}

impl VisionCache {
    /// Create a cache rooted at a self-learning state directory.
    pub fn new(state_dir: impl Into<PathBuf>) -> Self {
        Self {
            state_dir: state_dir.into(),
        }
    }

    /// Return the cache directory.
    pub fn cache_dir(&self) -> PathBuf {
        self.state_dir.join("vision_cache")
    }

    /// Return a stable replay reference for an image and model-specific salt.
    pub fn image_ref(&self, image_path: impl AsRef<Path>, salt: &str) -> Result<PathBuf> {
        let image_path = image_path.as_ref();
        let bytes = fs::read(image_path).map_err(|err| {
            AarambhError::Io(std::io::Error::new(
                err.kind(),
                format!(
                    "failed to read image {} for cache key: {err}",
                    image_path.display()
                ),
            ))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(salt.as_bytes());
        hasher.update(b"\0");
        hasher.update(image_path.display().to_string().as_bytes());
        hasher.update(b"\0");
        hasher.update(&bytes);
        let hash = hasher.finalize();
        Ok(PathBuf::from("vision_cache").join(format!("{hash:x}.safetensors")))
    }

    /// Resolve a replay image reference to an absolute cache path.
    pub fn absolute_path(&self, image_ref: impl AsRef<Path>) -> PathBuf {
        let image_ref = image_ref.as_ref();
        if image_ref.is_absolute() {
            image_ref.to_path_buf()
        } else {
            self.state_dir.join(image_ref)
        }
    }

    /// Load projected image tokens if the cache entry exists.
    pub fn load_projected_tokens(
        &self,
        image_ref: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let path = self.absolute_path(image_ref);
        if !path.exists() {
            return Ok(None);
        }
        let mut tensors = candle_core::safetensors::load(&path, device)?;
        tensors.remove(IMAGE_TOKENS).map(Some).ok_or_else(|| {
            AarambhError::Checkpoint(format!(
                "vision cache {} is missing tensor {IMAGE_TOKENS}",
                path.display()
            ))
        })
    }

    /// Save projected image tokens under a replay image reference.
    pub fn save_projected_tokens(
        &self,
        image_ref: impl AsRef<Path>,
        image_tokens: &Tensor,
    ) -> Result<()> {
        let path = self.absolute_path(image_ref);
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let tensors = [(IMAGE_TOKENS.to_string(), image_tokens.detach())]
            .into_iter()
            .collect();
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Return how many projected-token cache files are present.
    pub fn cached_entry_count(&self) -> usize {
        let Ok(read_dir) = fs::read_dir(self.cache_dir()) else {
            return 0;
        };
        read_dir
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "safetensors")
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn projected_token_cache_roundtrips() {
        let root =
            std::env::temp_dir().join(format!("aarambh_vision_cache_{}", std::process::id()));
        let image = root.join("image.bin");
        fs::create_dir_all(&root).unwrap();
        fs::write(&image, b"fake-image").unwrap();
        let cache = VisionCache::new(&root);
        let image_ref = cache.image_ref(&image, "salt").unwrap();
        let device = Device::Cpu;
        let tokens = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &device).unwrap();
        cache.save_projected_tokens(&image_ref, &tokens).unwrap();
        let loaded = cache
            .load_projected_tokens(&image_ref, &device)
            .unwrap()
            .unwrap();
        let _ = fs::remove_dir_all(root);
        assert_eq!(loaded.dims(), &[1, 2, 2]);
        assert_eq!(
            loaded.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1., 2.], vec![3., 4.]]]
        );
    }
}
