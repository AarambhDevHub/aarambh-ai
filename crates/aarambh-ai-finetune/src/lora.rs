use aarambh_ai_core::{AarambhError, Result};
use aarambh_ai_quant::{PackedInt4Tensor, dequantise_i4, quantise_affine_i4};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, Linear, Module, VarMap};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub group_size: usize,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
            target_modules: vec![
                "attn.wq".into(),
                "attn.wk".into(),
                "attn.wv".into(),
                "attn.wo".into(),
            ],
            group_size: 64,
        }
    }
}

impl LoraConfig {
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(AarambhError::Config(
                "LoRA rank must be greater than zero".into(),
            ));
        }
        if self.alpha <= 0.0 {
            return Err(AarambhError::Config("LoRA alpha must be positive".into()));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(AarambhError::Config(
                "LoRA dropout must be in [0, 1)".into(),
            ));
        }
        if self.target_modules.is_empty() {
            return Err(AarambhError::Config(
                "LoRA target_modules must not be empty".into(),
            ));
        }
        if self.group_size == 0 {
            return Err(AarambhError::Config(
                "QLoRA group_size must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    pub fn from_target_csv(value: &str) -> Vec<String> {
        value
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect()
    }

    pub fn targets_weight(&self, weight_name: &str) -> bool {
        let module_name = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
        self.target_modules
            .iter()
            .any(|target| module_name.ends_with(target))
    }
}

#[derive(Debug, Clone)]
pub enum BaseLinear {
    F32(Tensor),
    I4(PackedInt4Tensor),
}

impl BaseLinear {
    pub fn from_tensor(weight: &Tensor, quantized: bool, group_size: usize) -> Result<Self> {
        if quantized {
            Ok(Self::I4(quantise_affine_i4(weight, group_size)?))
        } else {
            Ok(Self::F32(weight.detach()))
        }
    }

    pub fn weight(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::F32(weight) => Ok(weight.clone()),
            Self::I4(weight) => Ok(dequantise_i4(weight, device)?.detach()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(weight) => weight.dims(),
            Self::I4(weight) => &weight.shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoraLinear {
    name: String,
    base: BaseLinear,
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
    scale: f64,
    dropout: f32,
    device: Device,
}

impl LoraLinear {
    pub fn new(
        name: impl Into<String>,
        base_weight: &Tensor,
        config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let name = name.into();
        let dims = base_weight.dims();
        if dims.len() != 2 {
            return Err(AarambhError::Shape(format!(
                "LoRA linear {name} expected 2D base weight, got {dims:?}"
            )));
        }
        config.validate()?;
        let base = BaseLinear::from_tensor(base_weight, quantized_base, config.group_size)?;
        let should_adapt = config.targets_weight(&name);
        let (lora_a, lora_b) = if should_adapt {
            let out_dim = dims[0];
            let in_dim = dims[1];
            let a_name = adapter_tensor_name(&name, "lora_a");
            let b_name = adapter_tensor_name(&name, "lora_b");
            let a = varmap.get(
                (config.rank, in_dim),
                &a_name,
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.01,
                },
                DType::F32,
                device,
            )?;
            let b = varmap.get(
                (out_dim, config.rank),
                &b_name,
                Init::Const(0.0),
                DType::F32,
                device,
            )?;
            (Some(a), Some(b))
        } else {
            (None, None)
        };

        Ok(Self {
            name,
            base,
            lora_a,
            lora_b,
            scale: config.alpha / config.rank as f64,
            dropout: config.dropout,
            device: device.clone(),
        })
    }

    pub fn from_adapter(
        name: impl Into<String>,
        base_weight: &Tensor,
        config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        Self::new(name, base_weight, config, varmap, quantized_base, device)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn has_adapter(&self) -> bool {
        self.lora_a.is_some() && self.lora_b.is_some()
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let base_weight = self.base.weight(&self.device)?;
        let base_out = linear_forward(x, &base_weight)?;
        let (Some(lora_a), Some(lora_b)) = (&self.lora_a, &self.lora_b) else {
            return Ok(base_out);
        };

        let x = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(x, self.dropout)?
        } else {
            x.clone()
        };
        let down = linear_forward(&x, lora_a)?;
        let up = linear_forward(&down, lora_b)?;
        let up = up.affine(self.scale, 0.0)?;
        Ok((base_out + up)?)
    }

    pub fn merged_weight(&self) -> Result<Tensor> {
        let base = self.base.weight(&self.device)?;
        let (Some(lora_a), Some(lora_b)) = (&self.lora_a, &self.lora_b) else {
            return Ok(base);
        };
        let delta = lora_b.matmul(lora_a)?.affine(self.scale, 0.0)?;
        Ok((base + delta)?.detach())
    }

    pub fn adapter_param_count(&self) -> usize {
        let mut count = 0;
        if let Some(a) = &self.lora_a {
            count += a.elem_count();
        }
        if let Some(b) = &self.lora_b {
            count += b.elem_count();
        }
        count
    }
}

pub fn linear_forward(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let layer = Linear::new(weight.clone(), None);
    Ok(layer.forward(x)?)
}

pub fn adapter_tensor_name(weight_name: &str, suffix: &str) -> String {
    weight_name
        .strip_suffix(".weight")
        .map(|name| format!("{name}.{suffix}"))
        .unwrap_or_else(|| format!("{weight_name}.{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn target_matching_uses_weight_suffixes() {
        let config = LoraConfig {
            target_modules: vec!["attn.wq".into(), "ffn.w_down".into()],
            ..Default::default()
        };
        assert!(config.targets_weight("blocks.0.attn.wq.weight"));
        assert!(config.targets_weight("blocks.1.ffn.w_down.weight"));
        assert!(!config.targets_weight("blocks.0.attn.wk.weight"));
    }

    #[test]
    fn zero_lora_matches_base_forward() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        let x = Tensor::from_vec(vec![10f32, 100.], (1, 2), &device).unwrap();
        let varmap = VarMap::new();
        let config = LoraConfig {
            rank: 1,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec!["w".into()],
            ..Default::default()
        };
        let lora = LoraLinear::new("w.weight", &weight, &config, &varmap, false, &device).unwrap();
        let base = linear_forward(&x, &weight)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let out = lora.forward(&x, true).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, base);
    }

    #[test]
    fn lora_backward_reaches_adapter_params() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        let x = Tensor::from_vec(vec![10f32, 100.], (1, 2), &device).unwrap();
        let varmap = VarMap::new();
        let config = LoraConfig {
            rank: 1,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec!["w".into()],
            ..Default::default()
        };
        let lora = LoraLinear::new("w.weight", &weight, &config, &varmap, false, &device).unwrap();
        let loss = lora.forward(&x, true).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(
            data.values()
                .any(|var| grads.get(var.as_tensor()).is_some())
        );
    }
}
