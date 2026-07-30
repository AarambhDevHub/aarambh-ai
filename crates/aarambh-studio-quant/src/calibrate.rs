use std::collections::HashMap;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{Device, Tensor};

use crate::gptq::compute_hessian;
use crate::types::{tensor_from_f32_vec, tensor_to_f32_vec};

#[derive(Debug, Default)]
/// Aggregated activation statistics used for AWQ and GPTQ.
pub struct CalibrationStats {
    features: HashMap<String, usize>,
    rows: HashMap<String, usize>,
    abs_sums: HashMap<String, Vec<f32>>,
    hessian_sums: HashMap<String, Vec<f32>>,
}

impl CalibrationStats {
    /// Observe one layer's activation tensor.
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

    /// Return normalized activation scales for a layer.
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

    /// Return the accumulated Hessian approximation for a layer.
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

    /// Return sorted layer names present in the calibration statistics.
    pub fn layer_names(&self) -> Vec<String> {
        let mut names = self.features.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }
}
