use aarambh_ai_core::{AarambhError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder};
use serde::{Deserialize, Serialize};

/// Temporal position encoding applied to frame patch features.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TemporalEncodingKind {
    /// Train one additive vector per frame position.
    #[default]
    Learned,
    /// Use deterministic sinusoidal frame positions.
    Sinusoidal,
}

/// Temporal encoding dimensions and implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalEncodingConfig {
    /// Maximum supported number of sampled frames.
    pub max_frames: usize,
    /// Vision feature width before projection.
    pub hidden_dim: usize,
    /// Position encoding implementation.
    pub kind: TemporalEncodingKind,
}

impl TemporalEncodingConfig {
    /// Validate temporal encoding dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.max_frames == 0 || self.hidden_dim == 0 {
            return Err(AarambhError::Config(
                "temporal max_frames and hidden_dim must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// Additive temporal position encoder for frame patch tokens.
#[derive(Debug, Clone)]
pub struct TemporalEncoder {
    config: TemporalEncodingConfig,
    learned: Option<Tensor>,
}

impl TemporalEncoder {
    /// Build an encoder, allocating learned positions through the supplied variable builder.
    pub fn new(config: TemporalEncodingConfig, vb: Option<VarBuilder<'_>>) -> Result<Self> {
        config.validate()?;
        let learned = match config.kind {
            TemporalEncodingKind::Learned => {
                let vb = vb.ok_or_else(|| {
                    AarambhError::Config(
                        "learned temporal encoding requires a variable builder".into(),
                    )
                })?;
                Some(vb.get_with_hints(
                    (config.max_frames, config.hidden_dim),
                    "position_embedding",
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.01,
                    },
                )?)
            }
            TemporalEncodingKind::Sinusoidal => None,
        };
        Ok(Self { config, learned })
    }

    /// Return temporal encoding configuration.
    pub fn config(&self) -> &TemporalEncodingConfig {
        &self.config
    }

    /// Add frame positions to `[frames, patch_tokens, hidden_dim]` features.
    ///
    /// Position zero is always an exact zero offset, preserving single-image behavior.
    pub fn forward(&self, frame_features: &Tensor) -> Result<Tensor> {
        let dims = frame_features.dims();
        if dims.len() != 3 || dims[2] != self.config.hidden_dim {
            return Err(AarambhError::Shape(format!(
                "frame_features must be [frames, patches, {}], got {dims:?}",
                self.config.hidden_dim
            )));
        }
        let frames = dims[0];
        if frames == 0 || frames > self.config.max_frames {
            return Err(AarambhError::Config(format!(
                "frame count {frames} must be in 1..={}",
                self.config.max_frames
            )));
        }
        if frames == 1 {
            return Ok(frame_features.clone());
        }
        let offsets = self.offsets(frames, frame_features.device(), frame_features.dtype())?;
        Ok(frame_features.broadcast_add(&offsets.unsqueeze(1)?)?)
    }

    /// Return `[frames, hidden_dim]` offsets with an exact-zero first row.
    pub fn offsets(&self, frames: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        if frames == 0 || frames > self.config.max_frames {
            return Err(AarambhError::Config(format!(
                "frame count {frames} must be in 1..={}",
                self.config.max_frames
            )));
        }
        match self.config.kind {
            TemporalEncodingKind::Learned => {
                let learned = self
                    .learned
                    .as_ref()
                    .expect("learned tensor is initialized");
                let tail = if frames > 1 {
                    learned.narrow(0, 1, frames - 1)?.to_dtype(dtype)?
                } else {
                    Tensor::zeros((0, self.config.hidden_dim), dtype, device)?
                };
                let zero = Tensor::zeros((1, self.config.hidden_dim), dtype, device)?;
                Ok(Tensor::cat(&[&zero, &tail], 0)?)
            }
            TemporalEncodingKind::Sinusoidal => {
                let mut values = Vec::with_capacity(frames * self.config.hidden_dim);
                for position in 0..frames {
                    for channel in 0..self.config.hidden_dim {
                        let pair = channel / 2;
                        let angle = position as f32
                            / 10_000f32.powf(2.0 * pair as f32 / self.config.hidden_dim as f32);
                        let value = if channel % 2 == 0 {
                            angle.sin()
                        } else {
                            angle.cos() - 1.0
                        };
                        values.push(value);
                    }
                }
                Ok(
                    Tensor::from_vec(values, (frames, self.config.hidden_dim), device)?
                        .to_dtype(dtype)?,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn sinusoidal_position_zero_is_exactly_zero() {
        let encoder = TemporalEncoder::new(
            TemporalEncodingConfig {
                max_frames: 8,
                hidden_dim: 6,
                kind: TemporalEncodingKind::Sinusoidal,
            },
            None,
        )
        .unwrap();
        let offsets = encoder.offsets(3, &Device::Cpu, DType::F32).unwrap();
        let offsets = offsets.to_vec2::<f32>().unwrap();
        assert_eq!(offsets[0], vec![0.0; 6]);
        assert_ne!(offsets[1], offsets[2]);
    }

    #[test]
    fn one_frame_is_an_exact_identity() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let encoder = TemporalEncoder::new(
            TemporalEncodingConfig {
                max_frames: 4,
                hidden_dim: 3,
                kind: TemporalEncodingKind::Learned,
            },
            Some(vb),
        )
        .unwrap();
        let input = Tensor::ones((1, 2, 3), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            encoder.forward(&input).unwrap().to_vec3::<f32>().unwrap(),
            input.to_vec3::<f32>().unwrap()
        );
    }
}
