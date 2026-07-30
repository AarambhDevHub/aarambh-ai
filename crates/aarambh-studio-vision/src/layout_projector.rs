use aarambh_studio_core::{AarambhError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::VisionProjector;

/// Row/column position implementation used by the document projector.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayoutEncodingKind {
    /// Train independent row and column embedding tables.
    #[default]
    Learned,
    /// Use deterministic sinusoidal row and column offsets.
    Sinusoidal,
}

/// Layout position dimensions for a document projector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutProjectorConfig {
    /// Number of patch rows on every preprocessed page.
    pub patch_rows: usize,
    /// Number of patch columns on every preprocessed page.
    pub patch_cols: usize,
    /// Language-model hidden width emitted by the base projector.
    pub hidden_dim: usize,
    /// Learned or sinusoidal row/column positions.
    pub encoding: LayoutEncodingKind,
}

impl LayoutProjectorConfig {
    /// Validate patch-grid and hidden dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.patch_rows == 0 || self.patch_cols == 0 || self.hidden_dim == 0 {
            return Err(AarambhError::Config(
                "layout patch rows, columns, and hidden_dim must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Return the number of patch tokens on one page.
    pub fn patch_count(&self) -> usize {
        self.patch_rows * self.patch_cols
    }
}

/// Vision projector augmented with independent two-dimensional patch positions.
#[derive(Debug, Clone)]
pub struct LayoutAwareProjector {
    base: VisionProjector,
    config: LayoutProjectorConfig,
    row_embedding: Option<Tensor>,
    col_embedding: Option<Tensor>,
}

impl LayoutAwareProjector {
    /// Build a layout-aware wrapper around an existing vision projector.
    pub fn new(
        base: VisionProjector,
        config: LayoutProjectorConfig,
        vb: Option<VarBuilder<'_>>,
    ) -> Result<Self> {
        config.validate()?;
        if base.config().llm_d_model != config.hidden_dim {
            return Err(AarambhError::Config(format!(
                "layout hidden_dim {} does not match projector output {}",
                config.hidden_dim,
                base.config().llm_d_model
            )));
        }
        let (row_embedding, col_embedding) = match config.encoding {
            LayoutEncodingKind::Learned => {
                let vb = vb.ok_or_else(|| {
                    AarambhError::Config(
                        "learned document layout encoding requires a variable builder".into(),
                    )
                })?;
                let init = Init::Randn {
                    mean: 0.0,
                    stdev: 0.01,
                };
                (
                    Some(vb.get_with_hints(
                        (config.patch_rows, config.hidden_dim),
                        "row_embedding",
                        init,
                    )?),
                    Some(vb.get_with_hints(
                        (config.patch_cols, config.hidden_dim),
                        "col_embedding",
                        init,
                    )?),
                )
            }
            LayoutEncodingKind::Sinusoidal => (None, None),
        };
        Ok(Self {
            base,
            config,
            row_embedding,
            col_embedding,
        })
    }

    /// Return the layout configuration.
    pub fn config(&self) -> &LayoutProjectorConfig {
        &self.config
    }

    /// Return the unchanged base vision projector.
    pub fn base_projector(&self) -> &VisionProjector {
        &self.base
    }

    /// Project `[pages, patches, vision_hidden]` and add row/column positions.
    pub fn forward(&self, patches: &Tensor, patch_grid: (usize, usize)) -> Result<Tensor> {
        if patch_grid != (self.config.patch_rows, self.config.patch_cols) {
            return Err(AarambhError::Shape(format!(
                "document patch grid {patch_grid:?} does not match configured ({}, {})",
                self.config.patch_rows, self.config.patch_cols
            )));
        }
        let projected = self.base.forward(patches)?;
        let dims = projected.dims();
        if dims[1] != self.config.patch_count() {
            return Err(AarambhError::Shape(format!(
                "projected document has {} patches, expected {}",
                dims[1],
                self.config.patch_count()
            )));
        }
        let offsets = self.offsets(projected.device(), projected.dtype())?;
        Ok(projected.broadcast_add(&offsets.unsqueeze(0)?)?)
    }

    /// Return flattened `[patches, hidden_dim]` row-plus-column offsets.
    pub fn offsets(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let rows = self.axis_offsets(
            self.config.patch_rows,
            self.row_embedding.as_ref(),
            true,
            device,
            dtype,
        )?;
        let cols = self.axis_offsets(
            self.config.patch_cols,
            self.col_embedding.as_ref(),
            false,
            device,
            dtype,
        )?;
        let row_ids = (0..self.config.patch_rows)
            .flat_map(|row| std::iter::repeat_n(row as u32, self.config.patch_cols))
            .collect::<Vec<_>>();
        let col_ids = (0..self.config.patch_rows)
            .flat_map(|_| (0..self.config.patch_cols).map(|col| col as u32))
            .collect::<Vec<_>>();
        let row_ids = Tensor::from_vec(row_ids, self.config.patch_count(), device)?;
        let col_ids = Tensor::from_vec(col_ids, self.config.patch_count(), device)?;
        Ok((rows.index_select(&row_ids, 0)? + cols.index_select(&col_ids, 0)?)?)
    }

    fn axis_offsets(
        &self,
        positions: usize,
        learned: Option<&Tensor>,
        row_axis: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        match self.config.encoding {
            LayoutEncodingKind::Learned => Ok(learned
                .expect("learned layout tensor is initialized")
                .to_dtype(dtype)?),
            LayoutEncodingKind::Sinusoidal => {
                let mut values = Vec::with_capacity(positions * self.config.hidden_dim);
                let row_width = self.config.hidden_dim.div_ceil(2);
                let (start, width) = if row_axis {
                    (0, row_width)
                } else {
                    (row_width, self.config.hidden_dim - row_width)
                };
                for position in 0..positions {
                    for channel in 0..self.config.hidden_dim {
                        if channel < start || channel >= start + width {
                            values.push(0.0);
                            continue;
                        }
                        let local_channel = channel - start;
                        let pair = local_channel / 2;
                        let angle = position as f32
                            / 10_000f32.powf(2.0 * pair as f32 / width.max(1) as f32);
                        values.push(if local_channel % 2 == 0 {
                            angle.sin()
                        } else {
                            angle.cos() - 1.0
                        });
                    }
                }
                Ok(
                    Tensor::from_vec(values, (positions, self.config.hidden_dim), device)?
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
    fn sinusoidal_offsets_distinguish_rows_and_columns() {
        let device = Device::Cpu;
        let base_vars = VarMap::new();
        let base = VisionProjector::new(
            crate::ProjectorConfig {
                vit_d_model: 4,
                llm_d_model: 6,
                hidden_mult: 1,
            },
            VarBuilder::from_varmap(&base_vars, DType::F32, &device),
        )
        .unwrap();
        let projector = LayoutAwareProjector::new(
            base,
            LayoutProjectorConfig {
                patch_rows: 2,
                patch_cols: 2,
                hidden_dim: 6,
                encoding: LayoutEncodingKind::Sinusoidal,
            },
            None,
        )
        .unwrap();
        let offsets = projector
            .offsets(&device, DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_ne!(offsets[0], offsets[1]);
        assert_ne!(offsets[0], offsets[2]);
        assert_ne!(offsets[1], offsets[2]);
    }

    #[test]
    fn invalid_patch_grid_is_rejected() {
        let device = Device::Cpu;
        let base_vars = VarMap::new();
        let base = VisionProjector::new(
            crate::ProjectorConfig {
                vit_d_model: 4,
                llm_d_model: 6,
                hidden_mult: 1,
            },
            VarBuilder::from_varmap(&base_vars, DType::F32, &device),
        )
        .unwrap();
        let projector = LayoutAwareProjector::new(
            base,
            LayoutProjectorConfig {
                patch_rows: 2,
                patch_cols: 2,
                hidden_dim: 6,
                encoding: LayoutEncodingKind::Sinusoidal,
            },
            None,
        )
        .unwrap();
        let input = Tensor::zeros((1, 4, 4), DType::F32, &device).unwrap();
        assert!(projector.forward(&input, (1, 4)).is_err());
    }
}
