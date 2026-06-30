use std::collections::HashMap;

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
use aarambh_ai_nn::RopeCache;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarMap};

use crate::lora::{LoraConfig, LoraLinear, linear_forward};

#[derive(Debug, Clone)]
/// Aarambh model with LoRA adapters attached to selected linear layers.
pub struct LoraAarambhModel {
    config: ModelConfig,
    embedding_weight: Tensor,
    blocks: Vec<LoraBlock>,
    final_norm_weight: Tensor,
    lm_head: Option<LoraLinear>,
    rope_cache: RopeCache,
    causal_mask: Tensor,
    adapter_param_count: usize,
    base_param_count: usize,
}

impl LoraAarambhModel {
    /// Build a LoRA model from base checkpoint tensors.
    pub fn from_tensors(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        quantized_base: bool,
        device: &Device,
    ) -> Result<(Self, VarMap)> {
        AarambhModel::validate_config(config)?;
        lora_config.validate()?;
        let varmap = VarMap::new();
        let embedding_weight = required_tensor(tensors, "embedding.weight")?;
        let mut blocks = Vec::with_capacity(config.n_layers);

        for layer_idx in 0..config.n_layers {
            blocks.push(LoraBlock::new(
                layer_idx,
                config,
                tensors,
                lora_config,
                &varmap,
                quantized_base,
                device,
            )?);
        }

        let final_norm_weight = required_tensor(tensors, "final_norm.weight")?;
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(LoraLinear::new(
                "lm_head.weight",
                required_ref(tensors, "lm_head.weight")?,
                lora_config,
                &varmap,
                quantized_base,
                device,
            )?)
        };
        let dtype = embedding_weight.dtype();
        let rope_cache = RopeCache::new(
            config.max_seq_len,
            config.head_dim(),
            config.rope_theta,
            dtype,
            device,
        )?;
        let causal_mask = create_causal_mask(config.max_seq_len, dtype, device)?;
        let adapter_param_count = adapter_param_count(&blocks, lm_head.as_ref());
        let base_param_count = tensors.values().map(tensor_elem_count).sum();

        let model = Self {
            config: config.clone(),
            embedding_weight,
            blocks,
            final_norm_weight,
            lm_head,
            rope_cache,
            causal_mask,
            adapter_param_count,
            base_param_count,
        };
        Ok((model, varmap))
    }

    /// Return the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return the number of adapter parameters.
    pub fn adapter_param_count(&self) -> usize {
        self.adapter_param_count
    }

    /// Return the number of base model parameters.
    pub fn base_param_count(&self) -> usize {
        self.base_param_count
    }

    /// Return adapter parameters divided by base parameters.
    pub fn trainable_ratio(&self) -> f64 {
        if self.base_param_count == 0 {
            0.0
        } else {
            self.adapter_param_count as f64 / self.base_param_count as f64
        }
    }

    /// Run the training forward path with adapters enabled.
    pub fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward(token_ids, true)
    }

    /// Run the evaluation forward path with adapters enabled.
    pub fn forward_eval(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward(token_ids, false)
    }

    /// Return checkpoint tensors with adapters merged into base weights.
    pub fn merged_tensors(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "embedding.weight".to_string(),
            self.embedding_weight.detach(),
        );

        for (idx, block) in self.blocks.iter().enumerate() {
            tensors.insert(
                format!("blocks.{idx}.norm1.weight"),
                block.norm1_weight.detach(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wq.weight"),
                block.attn.wq.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wk.weight"),
                block.attn.wk.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wv.weight"),
                block.attn.wv.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wo.weight"),
                block.attn.wo.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.norm2.weight"),
                block.norm2_weight.detach(),
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_gate.weight"),
                block.ffn.w_gate.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_up.weight"),
                block.ffn.w_up.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_down.weight"),
                block.ffn.w_down.merged_weight()?,
            );
        }

        tensors.insert(
            "final_norm.weight".to_string(),
            self.final_norm_weight.detach(),
        );
        if let Some(lm_head) = &self.lm_head {
            tensors.insert("lm_head.weight".to_string(), lm_head.merged_weight()?);
        }
        Ok(tensors)
    }

    fn forward(&self, token_ids: &Tensor, train: bool) -> Result<Tensor> {
        let (_, seq_len) = self.check_token_ids(token_ids)?;
        let mask = self.causal_mask(seq_len)?;
        let embedding = Embedding::new(self.embedding_weight.clone(), self.config.hidden_dim);
        let mut x = embedding.forward(token_ids)?;

        for block in &self.blocks {
            x = block.forward(&x, &self.rope_cache, Some(&mask), train)?;
        }

        let x = candle_nn::ops::rms_norm_slow(
            &x,
            &self.final_norm_weight,
            self.config.norm_eps as f32,
        )?;
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(&x, train),
            None => linear_forward(&x, &self.embedding_weight),
        }
    }

    fn check_token_ids(&self, token_ids: &Tensor) -> Result<(usize, usize)> {
        let dims = token_ids.dims();
        if dims.len() != 2 {
            return Err(AarambhError::Shape(format!(
                "token_ids must have shape [batch, seq], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let seq_len = dims[1];
        if batch == 0 || seq_len == 0 {
            return Err(AarambhError::Shape(
                "batch and sequence length must be non-zero".into(),
            ));
        }
        if seq_len > self.config.max_seq_len {
            return Err(AarambhError::Shape(format!(
                "sequence length {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        Ok((batch, seq_len))
    }

    fn causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        Ok(self
            .causal_mask
            .narrow(0, 0, seq_len)?
            .narrow(1, 0, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?)
    }
}

#[derive(Debug, Clone)]
struct LoraBlock {
    norm1_weight: Tensor,
    attn: LoraAttention,
    norm2_weight: Tensor,
    ffn: LoraFfn,
    norm_eps: f32,
}

impl LoraBlock {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}");
        Ok(Self {
            norm1_weight: required_tensor(tensors, &format!("{prefix}.norm1.weight"))?,
            attn: LoraAttention::new(
                layer_idx,
                config,
                tensors,
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            norm2_weight: required_tensor(tensors, &format!("{prefix}.norm2.weight"))?,
            ffn: LoraFfn::new(
                layer_idx,
                tensors,
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            norm_eps: config.norm_eps as f32,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = candle_nn::ops::rms_norm_slow(x, &self.norm1_weight, self.norm_eps)?;
        let hidden = self.attn.forward(&hidden, rope, mask, train)?;
        let x = (residual + hidden)?;

        let residual = x.clone();
        let hidden = candle_nn::ops::rms_norm_slow(&x, &self.norm2_weight, self.norm_eps)?;
        let hidden = self.ffn.forward(&hidden, train)?;
        Ok((residual + hidden)?)
    }
}

#[derive(Debug, Clone)]
struct LoraAttention {
    wq: LoraLinear,
    wk: LoraLinear,
    wv: LoraLinear,
    wo: LoraLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl LoraAttention {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}.attn");
        let head_dim = config.head_dim();
        Ok(Self {
            wq: make_linear(
                tensors,
                &format!("{prefix}.wq.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wk: make_linear(
                tensors,
                &format!("{prefix}.wk.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wv: make_linear(
                tensors,
                &format!("{prefix}.wv.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wo: make_linear(
                tensors,
                &format!("{prefix}.wo.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let b = dims[0];
        let seq_len = dims[1];

        let q = self.wq.forward(x, train)?;
        let k = self.wk.forward(x, train)?;
        let v = self.wv.forward(x, train)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let (q, k) = rope.apply(&q, &k, 0)?;

        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(&k, n_repeats)?;
        let v = repeat_heads(&v, n_repeats)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let out =
            aarambh_ai_kernel::dispatch::attention_forward_candle(&q, &k, &v, mask, self.scale)?;

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;
        self.wo.forward(&out, train)
    }
}

#[derive(Debug, Clone)]
struct LoraFfn {
    w_gate: LoraLinear,
    w_up: LoraLinear,
    w_down: LoraLinear,
}

impl LoraFfn {
    fn new(
        layer_idx: usize,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}.ffn");
        Ok(Self {
            w_gate: make_linear(
                tensors,
                &format!("{prefix}.w_gate.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_up: make_linear(
                tensors,
                &format!("{prefix}.w_up.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_down: make_linear(
                tensors,
                &format!("{prefix}.w_down.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x, train)?)?;
        let up = self.w_up.forward(x, train)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden, train)
    }
}

fn make_linear(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    lora_config: &LoraConfig,
    varmap: &VarMap,
    quantized_base: bool,
    device: &Device,
) -> Result<LoraLinear> {
    LoraLinear::new(
        name,
        required_ref(tensors, name)?,
        lora_config,
        varmap,
        quantized_base,
        device,
    )
}

fn required_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    Ok(required_ref(tensors, name)?.detach())
}

fn required_ref<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| AarambhError::Checkpoint(format!("missing tensor {name}")))
}

fn create_causal_mask(
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let tril = Tensor::tril2(seq_len, DType::U32, device)?;
    let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
    tril.where_cond(&zeros, &neg_inf)?.to_dtype(dtype)
}

fn repeat_heads(x: &Tensor, n_repeats: usize) -> Result<Tensor> {
    if n_repeats == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    let b = dims[0];
    let seq = dims[1];
    let n_kv = dims[2];
    let head_dim = dims[3];
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, seq, n_repeats, n_kv, head_dim))?;
    Ok(x.reshape((b, seq, n_kv * n_repeats, head_dim))?
        .contiguous()?)
}

fn adapter_param_count(blocks: &[LoraBlock], lm_head: Option<&LoraLinear>) -> usize {
    let mut count = 0;
    for block in blocks {
        count += block.attn.wq.adapter_param_count();
        count += block.attn.wk.adapter_param_count();
        count += block.attn.wv.adapter_param_count();
        count += block.attn.wo.adapter_param_count();
        count += block.ffn.w_gate.adapter_param_count();
        count += block.ffn.w_up.adapter_param_count();
        count += block.ffn.w_down.adapter_param_count();
    }
    if let Some(lm_head) = lm_head {
        count += lm_head.adapter_param_count();
    }
    count
}

fn tensor_elem_count(tensor: &Tensor) -> usize {
    tensor.dims().iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn lora_model_trainable_ratio_is_small() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, _varmap) =
            LoraAarambhModel::from_tensors(&config, &base.named_tensors(), &lora, false, &device)
                .unwrap();
        assert!(model.adapter_param_count() > 0);
        assert!(model.trainable_ratio() < 0.2);
    }

    #[test]
    fn lora_model_backward_reaches_adapter_params() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, varmap) =
            LoraAarambhModel::from_tensors(&config, &base.named_tensors(), &lora, false, &device)
                .unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let loss = model.forward_train(&ids).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(
            data.values()
                .any(|var| grads.get(var.as_tensor()).is_some())
        );
    }
}
