use std::collections::HashMap;

use aarambh_ai_core::{AarambhError, Configurable, Forward, ModelConfig, Result};
use aarambh_ai_nn::{
    GroupedQueryAttention, KVCache, RMSNorm, RopeCache, SwiGluFfn, TransformerBlock,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, linear_no_bias};

use crate::embedding::TokenEmbedding;
use crate::head::LmHead;

#[derive(Debug, Clone)]
pub struct AarambhModel {
    config: ModelConfig,
    embedding: TokenEmbedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    lm_head: LmHead,
    rope_cache: RopeCache,
    causal_mask: Tensor,
}

impl AarambhModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Self::validate_config(cfg)?;

        let embedding = TokenEmbedding::new(cfg.vocab_size, cfg.hidden_dim, vb.pp("embedding"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);

        for layer_idx in 0..cfg.n_layers {
            let block_vb = vb.pp("blocks").pp(layer_idx);
            let norm1 = RMSNorm::new(
                block_vb
                    .pp("norm1")
                    .get_with_hints(cfg.hidden_dim, "weight", Init::Const(1.0))?,
                cfg.norm_eps as f32,
            );
            let norm2 = RMSNorm::new(
                block_vb
                    .pp("norm2")
                    .get_with_hints(cfg.hidden_dim, "weight", Init::Const(1.0))?,
                cfg.norm_eps as f32,
            );

            let attn_vb = block_vb.pp("attn");
            let head_dim = cfg.head_dim();
            let attn = GroupedQueryAttention::new(
                linear_no_bias(cfg.hidden_dim, cfg.n_heads * head_dim, attn_vb.pp("wq"))?,
                linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, attn_vb.pp("wk"))?,
                linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, attn_vb.pp("wv"))?,
                linear_no_bias(cfg.n_heads * head_dim, cfg.hidden_dim, attn_vb.pp("wo"))?,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
            );

            let ffn_vb = block_vb.pp("ffn");
            let ffn = SwiGluFfn::new(
                linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, ffn_vb.pp("w_gate"))?,
                linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, ffn_vb.pp("w_up"))?,
                linear_no_bias(cfg.ffn_dim, cfg.hidden_dim, ffn_vb.pp("w_down"))?,
            );

            blocks.push(TransformerBlock::new(norm1, attn, norm2, ffn));
        }

        let final_norm = RMSNorm::new(
            vb.pp("final_norm")
                .get_with_hints(cfg.hidden_dim, "weight", Init::Const(1.0))?,
            cfg.norm_eps as f32,
        );
        let lm_head = if cfg.tie_embeddings {
            LmHead::tied(embedding.weight())
        } else {
            LmHead::untied(cfg.hidden_dim, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let dtype = embedding.weight().dtype();
        let rope_cache = RopeCache::new(
            cfg.max_seq_len,
            cfg.head_dim(),
            cfg.rope_theta,
            dtype,
            vb.device(),
        )?;
        let causal_mask = create_causal_mask(cfg.max_seq_len, dtype, vb.device())?;

        Ok(Self {
            config: cfg.clone(),
            embedding,
            blocks,
            final_norm,
            lm_head,
            rope_cache,
            causal_mask,
        })
    }

    pub fn validate_config(cfg: &ModelConfig) -> Result<()> {
        if cfg.vocab_size == 0 {
            return Err(AarambhError::Config("vocab_size must be non-zero".into()));
        }
        if cfg.hidden_dim == 0 || cfg.ffn_dim == 0 || cfg.n_layers == 0 {
            return Err(AarambhError::Config(
                "hidden_dim, ffn_dim, and n_layers must be non-zero".into(),
            ));
        }
        if cfg.n_heads == 0 || cfg.n_kv_heads == 0 {
            return Err(AarambhError::Config(
                "n_heads and n_kv_heads must be non-zero".into(),
            ));
        }
        if cfg.max_seq_len == 0 {
            return Err(AarambhError::Config("max_seq_len must be non-zero".into()));
        }
        if !cfg.hidden_dim.is_multiple_of(cfg.n_heads) {
            return Err(AarambhError::Config(
                "hidden_dim must be divisible by n_heads".into(),
            ));
        }
        if !cfg.n_heads.is_multiple_of(cfg.n_kv_heads) {
            return Err(AarambhError::Config(
                "n_heads must be divisible by n_kv_heads".into(),
            ));
        }
        if cfg.head_dim() != 64 {
            return Err(AarambhError::Config(
                "head_dim must be 64 for aarambh-ai Phase 3 model scales".into(),
            ));
        }
        Ok(())
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = self.check_token_ids(token_ids, 0)?;
        let mask = self.causal_mask(seq_len, 0)?;
        let mut x = self.embedding.forward(token_ids)?;

        for block in &self.blocks {
            x = block.forward(&x, &self.rope_cache, Some(&mask), None, 0)?;
        }

        let x = self.final_norm.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    pub fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = self.check_token_ids(token_ids, 0)?;
        let mask = self.causal_mask(seq_len, 0)?;
        let mut x = self.embedding.forward(token_ids)?;

        for block in &self.blocks {
            x = block.forward_train(&x, &self.rope_cache, Some(&mask), 0)?;
        }

        let x = self.final_norm.forward_train(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    pub fn linear_inputs(&self, token_ids: &Tensor) -> Result<HashMap<String, Tensor>> {
        let (_, seq_len) = self.check_token_ids(token_ids, 0)?;
        let mask = self.causal_mask(seq_len, 0)?;
        let mut capture = HashMap::new();
        let mut x = self.embedding.forward(token_ids)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_capture(
                &x,
                &self.rope_cache,
                Some(&mask),
                layer_idx,
                &mut capture,
            )?;
        }

        let x = self.final_norm.forward(&x)?;
        if !self.lm_head.is_tied() {
            capture.insert("lm_head.weight".to_string(), x);
        }
        Ok(capture)
    }

    pub fn forward_with_cache(
        &self,
        token_ids: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut [KVCache],
    ) -> Result<Tensor> {
        if kv_caches.len() != self.blocks.len() {
            return Err(AarambhError::Shape(format!(
                "expected {} KV caches, got {}",
                self.blocks.len(),
                kv_caches.len()
            )));
        }

        let (_, seq_len) = self.check_token_ids(token_ids, seqlen_offset)?;
        let mask = self.causal_mask(seq_len, seqlen_offset)?;
        let mut x = self.embedding.forward(token_ids)?;

        for (block, cache) in self.blocks.iter().zip(kv_caches.iter_mut()) {
            x = block.forward(
                &x,
                &self.rope_cache,
                Some(&mask),
                Some(cache),
                seqlen_offset,
            )?;
        }

        let x = self.final_norm.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    pub fn empty_kv_cache(&self) -> Vec<KVCache> {
        (0..self.blocks.len()).map(|_| KVCache::new()).collect()
    }

    pub fn named_tensors(&self) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "embedding.weight".to_string(),
            self.embedding.weight().clone(),
        );

        for (idx, block) in self.blocks.iter().enumerate() {
            tensors.insert(
                format!("blocks.{idx}.norm1.weight"),
                block.norm1().weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wq.weight"),
                block.attn().wq_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wk.weight"),
                block.attn().wk_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wv.weight"),
                block.attn().wv_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wo.weight"),
                block.attn().wo_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.norm2.weight"),
                block.norm2().weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_gate.weight"),
                block.ffn().w_gate_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_up.weight"),
                block.ffn().w_up_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_down.weight"),
                block.ffn().w_down_weight().clone(),
            );
        }

        tensors.insert(
            "final_norm.weight".to_string(),
            self.final_norm.weight().clone(),
        );
        if !self.lm_head.is_tied() {
            tensors.insert("lm_head.weight".to_string(), self.lm_head.weight().clone());
        }
        tensors
    }

    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        if name == "embedding.weight" {
            return Some(self.embedding.weight());
        }
        if name == "final_norm.weight" {
            return Some(self.final_norm.weight());
        }
        if name == "lm_head.weight" {
            return Some(self.lm_head.weight());
        }

        for (idx, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{idx}.");
            let Some(suffix) = name.strip_prefix(&prefix) else {
                continue;
            };
            return match suffix {
                "norm1.weight" => Some(block.norm1().weight()),
                "attn.wq.weight" => Some(block.attn().wq_weight()),
                "attn.wk.weight" => Some(block.attn().wk_weight()),
                "attn.wv.weight" => Some(block.attn().wv_weight()),
                "attn.wo.weight" => Some(block.attn().wo_weight()),
                "norm2.weight" => Some(block.norm2().weight()),
                "ffn.w_gate.weight" => Some(block.ffn().w_gate_weight()),
                "ffn.w_up.weight" => Some(block.ffn().w_up_weight()),
                "ffn.w_down.weight" => Some(block.ffn().w_down_weight()),
                _ => None,
            };
        }

        None
    }

    pub fn embedding(&self) -> &TokenEmbedding {
        &self.embedding
    }

    pub fn blocks(&self) -> &[TransformerBlock] {
        &self.blocks
    }

    pub fn lm_head(&self) -> &LmHead {
        &self.lm_head
    }

    fn check_token_ids(&self, token_ids: &Tensor, seqlen_offset: usize) -> Result<(usize, usize)> {
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
        if seqlen_offset + seq_len > self.config.max_seq_len {
            return Err(AarambhError::Shape(format!(
                "sequence length {} with offset {} exceeds max_seq_len {}",
                seq_len, seqlen_offset, self.config.max_seq_len
            )));
        }
        Ok((batch, seq_len))
    }

    fn causal_mask(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let total_len = seqlen_offset + seq_len;
        let mask = self
            .causal_mask
            .narrow(0, seqlen_offset, seq_len)?
            .narrow(1, 0, total_len)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }
}

impl Configurable for AarambhModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl Forward for AarambhModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        AarambhModel::forward(self, xs)
    }
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
