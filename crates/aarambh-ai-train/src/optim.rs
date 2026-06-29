use std::collections::HashMap;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result, TrainConfig};
use candle_core::{DType, Tensor, Var};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

pub type GradMap = HashMap<String, Tensor>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
}

impl From<&TrainConfig> for AdamWConfig {
    fn from(config: &TrainConfig) -> Self {
        Self {
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
            weight_decay: config.weight_decay,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainableParameter {
    name: String,
    var: Var,
    decoupled_weight_decay: bool,
}

impl TrainableParameter {
    pub fn new(name: String, var: Var) -> Self {
        let decoupled_weight_decay = uses_weight_decay(&name);
        Self {
            name,
            var,
            decoupled_weight_decay,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tensor(&self) -> &Tensor {
        self.var.as_tensor()
    }

    pub fn uses_weight_decay(&self) -> bool {
        self.decoupled_weight_decay
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    config: AdamWConfig,
    step: usize,
    params: Vec<TrainableParameter>,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamW {
    pub fn new(params: Vec<TrainableParameter>, config: AdamWConfig) -> Result<Self> {
        validate_config(&config)?;
        let mut m = HashMap::with_capacity(params.len());
        let mut v = HashMap::with_capacity(params.len());
        for param in &params {
            m.insert(param.name.clone(), f32_zeros_like(param.tensor())?);
            v.insert(param.name.clone(), f32_zeros_like(param.tensor())?);
        }
        Ok(Self {
            config,
            step: 0,
            params,
            m,
            v,
        })
    }

    pub fn from_varmap(varmap: &VarMap, config: AdamWConfig) -> Result<Self> {
        let mut params = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| TrainableParameter::new(name.clone(), var.clone()))
                .collect::<Vec<_>>()
        };
        params.sort_by(|a, b| a.name.cmp(&b.name));
        Self::new(params, config)
    }

    pub fn parameters(&self) -> &[TrainableParameter] {
        &self.params
    }

    pub fn step_num(&self) -> usize {
        self.step
    }

    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    pub fn step(&mut self, grads: &GradMap, lr: f64) -> Result<()> {
        if grads.is_empty() {
            return Err(AarambhError::Config(
                "optimizer step received no gradients".into(),
            ));
        }

        self.step += 1;
        let bias_correction1 = 1.0 - self.config.beta1.powf(self.step as f64);
        let bias_correction2 = 1.0 - self.config.beta2.powf(self.step as f64);

        for param in &self.params {
            let Some(grad) = grads.get(&param.name) else {
                continue;
            };
            let grad = grad.to_dtype(DType::F32)?;
            let param_f32 = param.tensor().to_dtype(DType::F32)?;
            let m_prev = self
                .m
                .get(&param.name)
                .ok_or_else(|| AarambhError::Config(format!("missing AdamW m for {}", param.name)))?
                .to_dtype(DType::F32)?;
            let v_prev = self
                .v
                .get(&param.name)
                .ok_or_else(|| AarambhError::Config(format!("missing AdamW v for {}", param.name)))?
                .to_dtype(DType::F32)?;

            let m_new = (m_prev.affine(self.config.beta1, 0.0)?
                + grad.affine(1.0 - self.config.beta1, 0.0)?)?;
            let grad_sq = grad.sqr()?;
            let v_new = (v_prev.affine(self.config.beta2, 0.0)?
                + grad_sq.affine(1.0 - self.config.beta2, 0.0)?)?;

            let m_hat = m_new.affine(1.0 / bias_correction1, 0.0)?;
            let v_hat = v_new.affine(1.0 / bias_correction2, 0.0)?;
            let denom = v_hat.sqrt()?.affine(1.0, self.config.epsilon)?;
            let adam_update = m_hat.broadcast_div(&denom)?;
            let update = if param.decoupled_weight_decay && self.config.weight_decay > 0.0 {
                (adam_update + param_f32.affine(self.config.weight_decay, 0.0)?)?
            } else {
                adam_update
            };
            let update = update.affine(lr, 0.0)?;
            let new_value = (param_f32 - update)?;
            let new_value = new_value.to_dtype(param.tensor().dtype())?;
            param.var.set(&new_value.detach())?;

            self.m.insert(param.name.clone(), m_new.detach());
            self.v.insert(param.name.clone(), v_new.detach());
        }

        Ok(())
    }

    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut tensors = HashMap::with_capacity(self.params.len() * 2);
        for param in &self.params {
            let m = self.m.get(&param.name).ok_or_else(|| {
                AarambhError::Config(format!("missing AdamW m for {}", param.name))
            })?;
            let v = self.v.get(&param.name).ok_or_else(|| {
                AarambhError::Config(format!("missing AdamW v for {}", param.name))
            })?;
            tensors.insert(format!("m.{}", param.name), m.clone());
            tensors.insert(format!("v.{}", param.name), v.clone());
        }
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    pub fn load_state(
        &mut self,
        path: impl AsRef<Path>,
        device: &candle_core::Device,
    ) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, device)?;
        for param in &self.params {
            let m_name = format!("m.{}", param.name);
            let v_name = format!("v.{}", param.name);
            let m = tensors.get(&m_name).ok_or_else(|| {
                AarambhError::Checkpoint(format!("missing optimizer state tensor {m_name}"))
            })?;
            let v = tensors.get(&v_name).ok_or_else(|| {
                AarambhError::Checkpoint(format!("missing optimizer state tensor {v_name}"))
            })?;
            self.m.insert(param.name.clone(), m.to_dtype(DType::F32)?);
            self.v.insert(param.name.clone(), v.to_dtype(DType::F32)?);
        }
        Ok(())
    }
}

fn f32_zeros_like(tensor: &Tensor) -> candle_core::Result<Tensor> {
    Tensor::zeros(tensor.shape(), DType::F32, tensor.device())
}

pub fn global_grad_norm(grads: &GradMap) -> Result<f64> {
    let mut sum_sq = 0f64;
    for grad in grads.values() {
        let grad = grad.to_dtype(DType::F32)?;
        sum_sq += grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    }
    Ok(sum_sq.sqrt())
}

pub fn clip_gradients(grads: &mut GradMap, max_norm: f64) -> Result<f64> {
    let norm = global_grad_norm(grads)?;
    if max_norm > 0.0 && norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for grad in grads.values_mut() {
            *grad = grad.affine(scale, 0.0)?.detach();
        }
    }
    Ok(norm)
}

fn uses_weight_decay(name: &str) -> bool {
    name.ends_with(".weight")
        && name != "embedding.weight"
        && !name.contains(".norm")
        && !name.starts_with("final_norm.")
        && !name.ends_with(".bias")
}

fn validate_config(config: &AdamWConfig) -> Result<()> {
    if !(0.0..1.0).contains(&config.beta1) {
        return Err(AarambhError::Config("AdamW beta1 must be in [0, 1)".into()));
    }
    if !(0.0..1.0).contains(&config.beta2) {
        return Err(AarambhError::Config("AdamW beta2 must be in [0, 1)".into()));
    }
    if config.epsilon <= 0.0 {
        return Err(AarambhError::Config(
            "AdamW epsilon must be positive".into(),
        ));
    }
    if config.weight_decay < 0.0 {
        return Err(AarambhError::Config(
            "AdamW weight_decay must be non-negative".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor, Var};

    #[test]
    fn beta2_default_matches_architecture() {
        let train = TrainConfig::default();
        let config = AdamWConfig::from(&train);
        assert_eq!(config.beta2, 0.95);
    }

    #[test]
    fn weight_decay_policy_excludes_embeddings_and_norms() {
        let device = Device::Cpu;
        let var = Var::ones((2, 2), DType::F32, &device).unwrap();

        assert!(
            !TrainableParameter::new("embedding.weight".into(), var.clone()).uses_weight_decay()
        );
        assert!(
            !TrainableParameter::new("blocks.0.norm1.weight".into(), var.clone())
                .uses_weight_decay()
        );
        assert!(
            !TrainableParameter::new("final_norm.weight".into(), var.clone()).uses_weight_decay()
        );
        assert!(TrainableParameter::new("blocks.0.attn.wq.weight".into(), var).uses_weight_decay());
    }

    #[test]
    fn gradient_clipping_caps_norm() {
        let device = Device::Cpu;
        let mut grads = GradMap::from([(
            "w".to_string(),
            Tensor::from_vec(vec![3f32, 4.], (2,), &device).unwrap(),
        )]);
        let before = clip_gradients(&mut grads, 1.0).unwrap();
        let after = global_grad_norm(&grads).unwrap();
        assert!((before - 5.0).abs() < 1e-5);
        assert!(after <= 1.00001, "after norm was {after}");
    }

    #[test]
    fn optimizer_state_uses_f32_for_lower_precision_params() {
        let device = Device::Cpu;
        let Ok(var) = Var::ones((2, 2), DType::BF16, &device) else {
            return;
        };
        let optimizer = AdamW::new(
            vec![TrainableParameter::new("w.weight".into(), var)],
            AdamWConfig {
                beta1: 0.9,
                beta2: 0.95,
                epsilon: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        assert_eq!(optimizer.m["w.weight"].dtype(), DType::F32);
        assert_eq!(optimizer.v["w.weight"].dtype(), DType::F32);
    }
}
