use std::collections::HashMap;

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_data::dataset::TextDataset;
use aarambh_ai_model::AarambhModel;
use candle_core::{Device, Tensor};

use crate::gptq::compute_hessian;
use crate::types::{tensor_from_f32_vec, tensor_to_f32_vec};

#[derive(Debug, Default)]
pub struct CalibrationStats {
    features: HashMap<String, usize>,
    rows: HashMap<String, usize>,
    abs_sums: HashMap<String, Vec<f32>>,
    hessian_sums: HashMap<String, Vec<f32>>,
}

impl CalibrationStats {
    pub fn observe(&mut self, name: &str, activations: &Tensor, with_hessian: bool) -> Result<()> {
        let dims = activations.dims();
        let features = *dims.last().ok_or_else(|| {
            AarambhError::Shape("calibration activations must have rank >= 1".into())
        })?;
        let rows = activations.elem_count() / features;
        let values = tensor_to_f32_vec(activations)?;

        let entry_features = self.features.entry(name.to_string()).or_insert(features);
        if *entry_features != features {
            return Err(AarambhError::Shape(format!(
                "calibration feature mismatch for {name}: expected {}, got {features}",
                *entry_features
            )));
        }

        let abs_sums = self
            .abs_sums
            .entry(name.to_string())
            .or_insert_with(|| vec![0.0; features]);
        for row in values.chunks(features) {
            for (idx, value) in row.iter().enumerate() {
                abs_sums[idx] += value.abs();
            }
        }
        *self.rows.entry(name.to_string()).or_insert(0) += rows;

        if with_hessian {
            let h = compute_hessian(activations)?;
            let h_values = tensor_to_f32_vec(&h)?;
            let h_sums = self
                .hessian_sums
                .entry(name.to_string())
                .or_insert_with(|| vec![0.0; features * features]);
            for (dst, src) in h_sums.iter_mut().zip(h_values.iter()) {
                *dst += *src;
            }
        }

        Ok(())
    }

    pub fn activation_scales(&self, name: &str, device: &Device) -> Result<Tensor> {
        let rows = self.rows.get(name).copied().unwrap_or(0);
        let sums = self
            .abs_sums
            .get(name)
            .ok_or_else(|| AarambhError::Config(format!("no calibration stats for {name}")))?;
        let mut scales = sums
            .iter()
            .map(|sum| (*sum / rows.max(1) as f32).max(1e-6).sqrt())
            .collect::<Vec<_>>();
        let mean = scales.iter().sum::<f32>() / scales.len() as f32;
        if mean.is_finite() && mean > 0.0 {
            for scale in &mut scales {
                *scale /= mean;
            }
        }
        tensor_from_f32_vec(scales, &[sums.len()], device)
    }

    pub fn hessian(&self, name: &str, device: &Device) -> Result<Tensor> {
        let features = self
            .features
            .get(name)
            .copied()
            .ok_or_else(|| AarambhError::Config(format!("no calibration stats for {name}")))?;
        let values = self
            .hessian_sums
            .get(name)
            .ok_or_else(|| AarambhError::Config(format!("no hessian stats for {name}")))?
            .clone();
        tensor_from_f32_vec(values, &[features, features], device)
    }

    pub fn layer_names(&self) -> Vec<String> {
        let mut names = self.features.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }
}

pub fn run_calibration(
    model: &AarambhModel,
    tokenizer: &dyn TokenizerLike,
    dataset: &dyn TextDataset,
    n_samples: usize,
    max_seq_len: usize,
    device: &Device,
    with_hessian: bool,
) -> Result<CalibrationStats> {
    if n_samples == 0 {
        return Err(AarambhError::Config(
            "calibration n_samples must be non-zero".into(),
        ));
    }
    let mut stats = CalibrationStats::default();
    let mut seen = 0usize;
    for idx in 0..dataset.len() {
        if seen >= n_samples {
            break;
        }
        let mut ids = tokenizer.encode(dataset.get(idx))?;
        if ids.len() < 2 {
            continue;
        }
        ids.truncate(max_seq_len.max(1));
        let seq_len = ids.len();
        let input = Tensor::from_vec(ids, (1, seq_len), device)?;
        let captures = model.linear_inputs(&input)?;
        for (name, activations) in captures {
            stats.observe(&name, &activations, with_hessian)?;
        }
        seen += 1;
    }
    if seen == 0 {
        return Err(AarambhError::Config(
            "calibration dataset produced no usable samples".into(),
        ));
    }
    Ok(stats)
}
