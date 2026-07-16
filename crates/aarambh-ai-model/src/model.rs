use std::collections::HashMap;

use aarambh_ai_core::{AarambhError, AttentionKind, Configurable, Forward, ModelConfig, Result};
use aarambh_ai_nn::{
    DeltaNetState, DsaAttention, DsaForwardStats, DsaKvCache, DsaTeacherOutput, FeedForwardLayer,
    GatedDeltaNetLayer, GroupedQueryAttention, HybridKvCache, KVCache, MoeFfn, MoeForwardStats,
    MtpHead, RMSNorm, RopeCache, SharedExpertPath, SwiGluFfn, TokenMixer, TransformerBlock,
};
use candle_core::{DType, Tensor};
use candle_nn::{Init, VarBuilder, linear_no_bias};

use crate::embedding::TokenEmbedding;
use crate::head::LmHead;

#[derive(Debug, Clone)]
/// Full Aarambh decoder-only causal language model.
pub struct AarambhModel {
    config: ModelConfig,
    embedding: TokenEmbedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    lm_head: LmHead,
    mtp_heads: Vec<MtpHead>,
    rope_cache: RopeCache,
}

#[derive(Debug)]
/// Logits plus optional MoE auxiliary metadata from a training forward pass.
pub struct ModelForwardOutput {
    /// Language-model logits.
    pub logits: Tensor,
    /// Final normalized trunk hidden states used by optional auxiliary heads.
    pub final_hidden_states: Tensor,
    /// Average MoE load-balancing auxiliary loss when MoE layers are active.
    pub moe_aux_loss: Option<Tensor>,
    /// Average per-expert utilization across active MoE layers.
    pub expert_utilization: Vec<f32>,
    /// Average periodic DSA indexer teacher loss when requested.
    pub dsa_indexer_loss: Option<Tensor>,
    /// Average top-k block recall against the dense teacher.
    pub dsa_top_k_recall: Option<f32>,
    /// Sparse attention selection and fallback counters.
    pub dsa_stats: DsaForwardStats,
}

#[derive(Debug)]
/// Logits and final hidden states from a cache-mutating model forward.
pub struct CachedModelOutput {
    /// Language-model logits for every supplied token.
    pub logits: Tensor,
    /// Final normalized trunk hidden states for every supplied token.
    pub final_hidden_states: Tensor,
}

#[derive(Debug)]
/// Logits produced by one multi-token prediction auxiliary head.
pub struct MtpPrediction {
    /// Future-token offset represented by these logits.
    pub offset: usize,
    /// Vocabulary logits shaped `[batch, valid_anchors, vocab]`.
    pub logits: Tensor,
}

impl AarambhModel {
    /// Build a model from configuration and a Candle variable builder.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Self::validate_config(cfg)?;

        let embedding = TokenEmbedding::new(cfg.vocab_size, cfg.hidden_dim, vb.pp("embedding"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        let deltanet_config = cfg
            .attention_schedule
            .as_ref()
            .map(|schedule| schedule.validate(cfg.n_layers, cfg.hidden_dim, cfg.n_heads))
            .transpose()?;

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

            let attention_kind = cfg.attention_kind_for_layer(layer_idx);
            let mixer = match attention_kind {
                AttentionKind::Full => {
                    TokenMixer::Attention(build_attention(cfg, block_vb.clone())?)
                }
                AttentionKind::Sparse => {
                    let index_dim = cfg.head_dim();
                    let dsa_vb = block_vb.pp("dsa");
                    TokenMixer::Sparse(DsaAttention::new(
                        build_attention(cfg, block_vb.clone())?,
                        linear_no_bias(cfg.hidden_dim, index_dim, dsa_vb.pp("index_q"))?,
                        linear_no_bias(cfg.hidden_dim, index_dim, dsa_vb.pp("index_k"))?,
                        cfg.dsa_config
                            .as_ref()
                            .expect("validated DSA config is present")
                            .clone(),
                        index_dim,
                    ))
                }
                AttentionKind::GatedDeltaNet => TokenMixer::GatedDelta(build_gated_deltanet(
                    cfg.hidden_dim,
                    cfg.norm_eps as f32,
                    deltanet_config
                        .as_ref()
                        .expect("validated hybrid config has Gated DeltaNet settings"),
                    block_vb.pp("deltanet"),
                )?),
            };

            let ffn_vb = block_vb.pp("ffn");
            let ffn = match cfg
                .moe
                .as_ref()
                .filter(|moe| moe.applies_to_layer(layer_idx))
            {
                Some(moe) => {
                    let routed_expert_count = moe.routed_expert_count()?;
                    let expert_dim = moe.fine_grained_expert_dim()?;
                    let experts = (0..routed_expert_count)
                        .map(|expert_idx| {
                            let expert_vb = ffn_vb.pp("experts").pp(expert_idx);
                            build_swiglu(cfg.hidden_dim, expert_dim, expert_vb)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let shared_experts = (0..moe.num_shared_experts)
                        .map(|expert_idx| {
                            build_swiglu(
                                cfg.hidden_dim,
                                expert_dim,
                                ffn_vb.pp("shared_experts").pp(expert_idx),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    FeedForwardLayer::Moe(MoeFfn::new_with_shared(
                        moe.clone(),
                        linear_no_bias(cfg.hidden_dim, routed_expert_count, ffn_vb.pp("router"))?,
                        experts,
                        SharedExpertPath::new(shared_experts),
                    ))
                }
                None => FeedForwardLayer::Dense(SwiGluFfn::new(
                    linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, ffn_vb.pp("w_gate"))?,
                    linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, ffn_vb.pp("w_up"))?,
                    linear_no_bias(cfg.ffn_dim, cfg.hidden_dim, ffn_vb.pp("w_down"))?,
                )),
            };

            blocks.push(TransformerBlock::new_with_mixer(norm1, mixer, norm2, ffn));
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

        let mtp_heads = match &cfg.mtp {
            Some(mtp) => (2..=mtp.num_future_tokens)
                .enumerate()
                .map(|(head_index, offset)| {
                    MtpHead::new(
                        offset,
                        cfg,
                        lm_head.weight(),
                        vb.pp("mtp").pp("heads").pp(head_index),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            None => Vec::new(),
        };

        let dtype = embedding.weight().dtype();
        let rope_cache = RopeCache::from_config(cfg, dtype, vb.device())?;

        Ok(Self {
            config: cfg.clone(),
            embedding,
            blocks,
            final_norm,
            lm_head,
            mtp_heads,
            rope_cache,
        })
    }

    /// Validate model dimensions before construction.
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
        if let Some(rope_scaling) = &cfg.rope_scaling {
            rope_scaling.validate(cfg.max_seq_len, cfg.head_dim())?;
        }
        if let Some(moe) = &cfg.moe {
            moe.validate(cfg.n_layers)?;
        }
        if let Some(schedule) = &cfg.attention_schedule {
            schedule.validate(cfg.n_layers, cfg.hidden_dim, cfg.n_heads)?;
        }
        if let Some(dsa) = &cfg.dsa_config {
            dsa.validate()?;
            if cfg.attention_schedule.is_none() {
                return Err(AarambhError::Config(
                    "dsa_config requires attention_schedule so sparse and Gated DeltaNet layers are explicit"
                        .into(),
                ));
            }
            if !(0..cfg.n_layers)
                .any(|layer| cfg.attention_kind_for_layer(layer) == AttentionKind::Sparse)
            {
                return Err(AarambhError::Config(
                    "dsa_config does not select any sparse attention layer".into(),
                ));
            }
        }
        if let Some(mtp) = &cfg.mtp {
            mtp.validate(cfg.max_seq_len)?;
        }
        Ok(())
    }

    /// Run a full causal forward pass over token ids.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.check_token_ids(token_ids, 0)?;
        let x = self.embedding.forward(token_ids)?;
        self.forward_embeddings(&x)
    }

    /// Convert token ids into model hidden states.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.check_token_ids(token_ids, 0)?;
        Ok(self.embedding.forward(token_ids)?)
    }

    /// Run a full causal forward pass over precomputed token embeddings.
    pub fn forward_embeddings(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.check_embeddings(embeddings, 0)?;
        let mut x = embeddings.clone();
        for block in &self.blocks {
            x = block.forward(&x, &self.rope_cache, None, None, 0)?;
        }

        let x = self.final_norm.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    /// Run the training forward path over token ids.
    pub fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.check_token_ids(token_ids, 0)?;
        let x = self.embedding.forward(token_ids)?;
        self.forward_embeddings_train(&x)
    }

    /// Run the training forward path over precomputed token embeddings.
    pub fn forward_embeddings_train(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.check_embeddings(embeddings, 0)?;
        let mut x = embeddings.clone();
        for block in &self.blocks {
            x = block.forward_train(&x, &self.rope_cache, None, 0)?;
        }

        let x = self.final_norm.forward_train(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    /// Run the training forward path and collect MoE auxiliary metadata.
    pub fn forward_train_with_aux(&self, token_ids: &Tensor) -> Result<ModelForwardOutput> {
        self.forward_train_with_aux_and_dsa_teacher(token_ids, false)
    }

    /// Run training and optionally collect the periodic dense DSA teacher loss.
    pub fn forward_train_with_aux_and_dsa_teacher(
        &self,
        token_ids: &Tensor,
        collect_dsa_teacher: bool,
    ) -> Result<ModelForwardOutput> {
        self.check_token_ids(token_ids, 0)?;
        let x = self.embedding.forward(token_ids)?;
        self.forward_embeddings_train_with_aux_and_dsa_teacher(&x, collect_dsa_teacher)
    }

    /// Run the training forward path over embeddings and collect MoE auxiliary metadata.
    pub fn forward_embeddings_train_with_aux(
        &self,
        embeddings: &Tensor,
    ) -> Result<ModelForwardOutput> {
        self.forward_embeddings_train_with_aux_and_dsa_teacher(embeddings, false)
    }

    /// Run embedding training and optionally collect the DSA teacher objective.
    pub fn forward_embeddings_train_with_aux_and_dsa_teacher(
        &self,
        embeddings: &Tensor,
        collect_dsa_teacher: bool,
    ) -> Result<ModelForwardOutput> {
        self.check_embeddings(embeddings, 0)?;
        let mut stats = MoeForwardStats::default();
        let mut dsa_stats = DsaForwardStats::default();
        let mut dsa_teachers = Vec::new();
        let mut x = embeddings.clone();
        for block in &self.blocks {
            x = block.forward_train_with_stats(
                &x,
                &self.rope_cache,
                None,
                0,
                &mut stats,
                &mut dsa_stats,
                collect_dsa_teacher,
                &mut dsa_teachers,
            )?;
        }

        let x = self.final_norm.forward_train(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(ModelForwardOutput {
            logits,
            final_hidden_states: x,
            moe_aux_loss: stats.aux_loss()?,
            expert_utilization: stats.expert_utilization(),
            dsa_indexer_loss: average_dsa_loss(&dsa_teachers)?,
            dsa_top_k_recall: (!dsa_teachers.is_empty()).then(|| {
                dsa_teachers
                    .iter()
                    .map(|teacher| teacher.top_k_recall)
                    .sum::<f32>()
                    / dsa_teachers.len() as f32
            }),
            dsa_stats,
        })
    }

    /// Capture inputs to linear layers for calibration and quantisation.
    pub fn linear_inputs(&self, token_ids: &Tensor) -> Result<HashMap<String, Tensor>> {
        self.check_token_ids(token_ids, 0)?;
        let mut capture = HashMap::new();
        let mut x = self.embedding.forward(token_ids)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_capture(&x, &self.rope_cache, None, layer_idx, &mut capture)?;
        }

        let x = self.final_norm.forward(&x)?;
        if !self.lm_head.is_tied() {
            capture.insert("lm_head.weight".to_string(), x.clone());
        }
        for head_index in 0..self.mtp_heads.len() {
            if token_ids.dim(1)? >= self.mtp_heads[head_index].offset() {
                let _ =
                    self.forward_mtp_head_with_capture(head_index, &x, token_ids, &mut capture)?;
            }
        }
        Ok(capture)
    }

    /// Run incremental inference using per-layer KV caches.
    pub fn forward_with_cache(
        &self,
        token_ids: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut [HybridKvCache],
    ) -> Result<Tensor> {
        if kv_caches.len() != self.blocks.len() {
            return Err(AarambhError::Shape(format!(
                "expected {} hybrid caches, got {}",
                self.blocks.len(),
                kv_caches.len()
            )));
        }

        self.check_token_ids(token_ids, seqlen_offset)?;
        Ok(self
            .forward_with_cache_output(token_ids, seqlen_offset, kv_caches)?
            .logits)
    }

    /// Run incremental inference and return both logits and final hidden states.
    pub fn forward_with_cache_output(
        &self,
        token_ids: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut [HybridKvCache],
    ) -> Result<CachedModelOutput> {
        if kv_caches.len() != self.blocks.len() {
            return Err(AarambhError::Shape(format!(
                "expected {} hybrid caches, got {}",
                self.blocks.len(),
                kv_caches.len()
            )));
        }
        self.check_token_ids(token_ids, seqlen_offset)?;
        let x = self.embedding.forward(token_ids)?;
        self.forward_embeddings_with_cache_output(&x, seqlen_offset, kv_caches)
    }

    /// Run incremental inference using precomputed embeddings and per-layer KV caches.
    pub fn forward_embeddings_with_cache(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut [HybridKvCache],
    ) -> Result<Tensor> {
        Ok(self
            .forward_embeddings_with_cache_output(embeddings, seqlen_offset, kv_caches)?
            .logits)
    }

    /// Run cached inference from embeddings and retain final hidden states.
    pub fn forward_embeddings_with_cache_output(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_caches: &mut [HybridKvCache],
    ) -> Result<CachedModelOutput> {
        if kv_caches.len() != self.blocks.len() {
            return Err(AarambhError::Shape(format!(
                "expected {} hybrid caches, got {}",
                self.blocks.len(),
                kv_caches.len()
            )));
        }

        self.check_embeddings(embeddings, seqlen_offset)?;
        let mut x = embeddings.clone();

        for (block, cache) in self.blocks.iter().zip(kv_caches.iter_mut()) {
            x = block.forward(&x, &self.rope_cache, None, Some(cache), seqlen_offset)?;
        }

        let final_hidden_states = self.final_norm.forward(&x)?;
        let logits = self.lm_head.forward(&final_hidden_states)?;
        Ok(CachedModelOutput {
            logits,
            final_hidden_states,
        })
    }

    /// Decode one token for multiple independent sequences in a shared model pass.
    pub fn forward_decode_batch(
        &self,
        token_ids: &Tensor,
        seqlen_offsets: &[usize],
        caches: &mut [&mut [HybridKvCache]],
    ) -> Result<Tensor> {
        let dims = token_ids.dims();
        if dims.len() != 2 || dims[1] != 1 {
            return Err(AarambhError::Shape(format!(
                "batched decode token ids must have shape [batch, 1], got {dims:?}"
            )));
        }
        let batch = dims[0];
        if seqlen_offsets.len() != batch || caches.len() != batch {
            return Err(AarambhError::Shape(format!(
                "batched decode received batch {batch}, {} offsets, and {} caches",
                seqlen_offsets.len(),
                caches.len()
            )));
        }
        for (row, (&offset, cache)) in seqlen_offsets.iter().zip(caches.iter()).enumerate() {
            if offset >= self.config.max_seq_len {
                return Err(AarambhError::Shape(format!(
                    "batch row {row} offset {offset} exceeds max_seq_len {}",
                    self.config.max_seq_len
                )));
            }
            if cache.len() != self.blocks.len() {
                return Err(AarambhError::Shape(format!(
                    "batch row {row} has {} cache layers, expected {}",
                    cache.len(),
                    self.blocks.len()
                )));
            }
        }

        let mut x = self.embedding.forward(token_ids)?;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mut layer_caches = caches
                .iter_mut()
                .map(|cache| &mut cache[layer_idx])
                .collect::<Vec<_>>();
            x = block.forward_decode_batch(
                &x,
                &self.rope_cache,
                &mut layer_caches,
                seqlen_offsets,
            )?;
        }
        let x = self.final_norm.forward(&x)?;
        Ok(self.lm_head.forward(&x)?)
    }

    /// Create one correctly typed empty cache per transformer block.
    pub fn empty_kv_cache(&self) -> Vec<HybridKvCache> {
        self.empty_kv_cache_with_capacity(self.config.max_seq_len)
    }

    /// Create one correctly typed cache with a requested full-attention capacity.
    pub fn empty_kv_cache_with_capacity(&self, capacity: usize) -> Vec<HybridKvCache> {
        self.blocks
            .iter()
            .map(|block| match block.mixer() {
                TokenMixer::Attention(_) => HybridKvCache::Full(KVCache::with_capacity(capacity)),
                TokenMixer::Sparse(attn) => HybridKvCache::Sparse(DsaKvCache::with_capacity(
                    capacity,
                    attn.config().block_size,
                )),
                TokenMixer::GatedDelta(_) => HybridKvCache::Linear(DeltaNetState::new()),
            })
            .collect()
    }

    /// Return a map of model tensor names to tensors.
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
            insert_mixer_tensors(&mut tensors, idx, block.mixer());
            tensors.insert(
                format!("blocks.{idx}.norm2.weight"),
                block.norm2().weight().clone(),
            );
            insert_ffn_tensors(&mut tensors, idx, block.ffn());
        }

        tensors.insert(
            "final_norm.weight".to_string(),
            self.final_norm.weight().clone(),
        );
        if !self.lm_head.is_tied() {
            tensors.insert("lm_head.weight".to_string(), self.lm_head.weight().clone());
        }
        for (head_index, head) in self.mtp_heads.iter().enumerate() {
            for (name, tensor) in head.named_tensors() {
                tensors.insert(format!("mtp.heads.{head_index}.{name}"), tensor);
            }
        }
        tensors
    }

    /// Return a named weight tensor by checkpoint name.
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
        if let Some(suffix) = name.strip_prefix("mtp.heads.") {
            let (head_index, name) = suffix.split_once('.')?;
            return self
                .mtp_heads
                .get(head_index.parse::<usize>().ok()?)?
                .get_weight(name);
        }

        for (idx, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{idx}.");
            let Some(suffix) = name.strip_prefix(&prefix) else {
                continue;
            };
            return match suffix {
                "norm1.weight" => Some(block.norm1().weight()),
                "attn.wq.weight" => block.mixer().as_attention().map(|v| v.wq_weight()),
                "attn.wk.weight" => block.mixer().as_attention().map(|v| v.wk_weight()),
                "attn.wv.weight" => block.mixer().as_attention().map(|v| v.wv_weight()),
                "attn.wo.weight" => block.mixer().as_attention().map(|v| v.wo_weight()),
                "dsa.index_q.weight" => block.mixer().as_sparse().map(DsaAttention::index_q_weight),
                "dsa.index_k.weight" => block.mixer().as_sparse().map(DsaAttention::index_k_weight),
                "norm2.weight" => Some(block.norm2().weight()),
                "ffn.w_gate.weight" => block.ffn().as_dense().map(SwiGluFfn::w_gate_weight),
                "ffn.w_up.weight" => block.ffn().as_dense().map(SwiGluFfn::w_up_weight),
                "ffn.w_down.weight" => block.ffn().as_dense().map(SwiGluFfn::w_down_weight),
                "ffn.router.weight" => block.ffn().as_moe().map(MoeFfn::router_weight),
                _ if suffix.starts_with("deltanet.") => block
                    .mixer()
                    .as_gated_delta()
                    .and_then(|layer| layer.get_weight(&suffix[9..])),
                _ => get_moe_expert_weight(block.ffn(), suffix),
            };
        }

        None
    }

    /// Return the token embedding layer.
    pub fn embedding(&self) -> &TokenEmbedding {
        &self.embedding
    }

    /// Return all transformer blocks.
    pub fn blocks(&self) -> &[TransformerBlock] {
        &self.blocks
    }

    /// Return the language-model head.
    pub fn lm_head(&self) -> &LmHead {
        &self.lm_head
    }

    /// Return all configured multi-token prediction auxiliary heads.
    pub fn mtp_heads(&self) -> &[MtpHead] {
        &self.mtp_heads
    }

    /// Run one MTP head over every valid training anchor.
    pub fn forward_mtp_head_train(
        &self,
        head_index: usize,
        final_hidden_states: &Tensor,
        token_ids: &Tensor,
    ) -> Result<MtpPrediction> {
        self.forward_mtp_head_impl(head_index, final_hidden_states, token_ids, true, None)
    }

    /// Run one MTP head for a single inference anchor and proposed prefix.
    pub fn forward_mtp_head(
        &self,
        head_index: usize,
        anchor_hidden: &Tensor,
        intervening_token_ids: &Tensor,
    ) -> Result<MtpPrediction> {
        let head = self.mtp_heads.get(head_index).ok_or_else(|| {
            AarambhError::Config(format!("MTP head index {head_index} is not configured"))
        })?;
        let (batch, anchors, hidden) = anchor_hidden.dims3()?;
        if anchors != 1 || hidden != self.config.hidden_dim {
            return Err(AarambhError::Shape(format!(
                "MTP inference anchor must have shape [batch, 1, {}], got {:?}",
                self.config.hidden_dim,
                anchor_hidden.dims()
            )));
        }
        let (token_batch, intervening) = intervening_token_ids.dims2()?;
        if token_batch != batch || intervening + 1 != head.offset() {
            return Err(AarambhError::Shape(format!(
                "MTP offset {} expects token ids [batch, {}], got {:?}",
                head.offset(),
                head.offset() - 1,
                intervening_token_ids.dims()
            )));
        }
        let embeddings = self
            .embedding
            .forward(intervening_token_ids)?
            .unsqueeze(1)?;
        Ok(MtpPrediction {
            offset: head.offset(),
            logits: head.forward(anchor_hidden, &embeddings, &self.rope_cache)?,
        })
    }

    fn forward_mtp_head_with_capture(
        &self,
        head_index: usize,
        final_hidden_states: &Tensor,
        token_ids: &Tensor,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<MtpPrediction> {
        self.forward_mtp_head_impl(
            head_index,
            final_hidden_states,
            token_ids,
            false,
            Some(capture),
        )
    }

    fn forward_mtp_head_impl(
        &self,
        head_index: usize,
        final_hidden_states: &Tensor,
        token_ids: &Tensor,
        training: bool,
        capture: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<MtpPrediction> {
        let head = self.mtp_heads.get(head_index).ok_or_else(|| {
            AarambhError::Config(format!("MTP head index {head_index} is not configured"))
        })?;
        let (batch, seq_len) = self.check_token_ids(token_ids, 0)?;
        let hidden_dims = final_hidden_states.dims();
        if hidden_dims != [batch, seq_len, self.config.hidden_dim] {
            return Err(AarambhError::Shape(format!(
                "MTP final hidden states must have shape [{batch}, {seq_len}, {}], got {hidden_dims:?}",
                self.config.hidden_dim
            )));
        }
        if seq_len < head.offset() {
            return Err(AarambhError::Shape(format!(
                "sequence length {seq_len} is shorter than MTP offset {}",
                head.offset()
            )));
        }
        let anchors = seq_len - head.offset() + 1;
        let trunk = final_hidden_states.narrow(1, 0, anchors)?;
        let mut embeddings = Vec::with_capacity(head.offset() - 1);
        for shift in 1..head.offset() {
            let ids = token_ids.narrow(1, shift, anchors)?;
            embeddings.push(self.embedding.forward(&ids)?.unsqueeze(2)?);
        }
        let refs = embeddings.iter().collect::<Vec<_>>();
        let embeddings = Tensor::cat(&refs, 2)?;
        let logits = match capture {
            Some(capture) => head.forward_with_capture(
                &trunk,
                &embeddings,
                &self.rope_cache,
                head_index,
                capture,
            )?,
            None if training => head.forward_train(&trunk, &embeddings, &self.rope_cache)?,
            None => head.forward(&trunk, &embeddings, &self.rope_cache)?,
        };
        Ok(MtpPrediction {
            offset: head.offset(),
            logits,
        })
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

    fn check_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(usize, usize)> {
        let dims = embeddings.dims();
        if dims.len() != 3 {
            return Err(AarambhError::Shape(format!(
                "embeddings must have shape [batch, seq, hidden_dim], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let seq_len = dims[1];
        let hidden_dim = dims[2];
        if batch == 0 || seq_len == 0 {
            return Err(AarambhError::Shape(
                "batch and sequence length must be non-zero".into(),
            ));
        }
        if hidden_dim != self.config.hidden_dim {
            return Err(AarambhError::Shape(format!(
                "embedding hidden dim {hidden_dim} does not match model hidden_dim {}",
                self.config.hidden_dim
            )));
        }
        if seqlen_offset + seq_len > self.config.max_seq_len {
            return Err(AarambhError::Shape(format!(
                "sequence length {} with offset {} exceeds max_seq_len {}",
                seq_len, seqlen_offset, self.config.max_seq_len
            )));
        }
        Ok((batch, seq_len))
    }
}

fn insert_ffn_tensors(
    tensors: &mut HashMap<String, Tensor>,
    layer_idx: usize,
    ffn: &FeedForwardLayer,
) {
    match ffn {
        FeedForwardLayer::Dense(ffn) => {
            tensors.insert(
                format!("blocks.{layer_idx}.ffn.w_gate.weight"),
                ffn.w_gate_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{layer_idx}.ffn.w_up.weight"),
                ffn.w_up_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{layer_idx}.ffn.w_down.weight"),
                ffn.w_down_weight().clone(),
            );
        }
        FeedForwardLayer::Moe(ffn) => {
            tensors.insert(
                format!("blocks.{layer_idx}.ffn.router.weight"),
                ffn.router_weight().clone(),
            );
            for (expert_idx, expert) in ffn.experts().iter().enumerate() {
                let prefix = format!("blocks.{layer_idx}.ffn.experts.{expert_idx}");
                tensors.insert(
                    format!("{prefix}.w_gate.weight"),
                    expert.w_gate_weight().clone(),
                );
                tensors.insert(
                    format!("{prefix}.w_up.weight"),
                    expert.w_up_weight().clone(),
                );
                tensors.insert(
                    format!("{prefix}.w_down.weight"),
                    expert.w_down_weight().clone(),
                );
            }
            for (expert_idx, expert) in ffn.shared_experts().experts().iter().enumerate() {
                let prefix = format!("blocks.{layer_idx}.ffn.shared_experts.{expert_idx}");
                tensors.insert(
                    format!("{prefix}.w_gate.weight"),
                    expert.w_gate_weight().clone(),
                );
                tensors.insert(
                    format!("{prefix}.w_up.weight"),
                    expert.w_up_weight().clone(),
                );
                tensors.insert(
                    format!("{prefix}.w_down.weight"),
                    expert.w_down_weight().clone(),
                );
            }
        }
    }
}

fn insert_mixer_tensors(
    tensors: &mut HashMap<String, Tensor>,
    layer_idx: usize,
    mixer: &TokenMixer,
) {
    match mixer {
        TokenMixer::Attention(attn) => {
            for (name, tensor) in [
                ("wq", attn.wq_weight()),
                ("wk", attn.wk_weight()),
                ("wv", attn.wv_weight()),
                ("wo", attn.wo_weight()),
            ] {
                tensors.insert(
                    format!("blocks.{layer_idx}.attn.{name}.weight"),
                    tensor.clone(),
                );
            }
        }
        TokenMixer::Sparse(sparse) => {
            let attn = sparse.attention();
            for (name, tensor) in [
                ("wq", attn.wq_weight()),
                ("wk", attn.wk_weight()),
                ("wv", attn.wv_weight()),
                ("wo", attn.wo_weight()),
            ] {
                tensors.insert(
                    format!("blocks.{layer_idx}.attn.{name}.weight"),
                    tensor.clone(),
                );
            }
            tensors.insert(
                format!("blocks.{layer_idx}.dsa.index_q.weight"),
                sparse.index_q_weight().clone(),
            );
            tensors.insert(
                format!("blocks.{layer_idx}.dsa.index_k.weight"),
                sparse.index_k_weight().clone(),
            );
        }
        TokenMixer::GatedDelta(layer) => {
            for (name, tensor) in layer.named_tensors() {
                tensors.insert(
                    format!("blocks.{layer_idx}.deltanet.{name}"),
                    tensor.clone(),
                );
            }
        }
    }
}

fn build_attention(cfg: &ModelConfig, block_vb: VarBuilder<'_>) -> Result<GroupedQueryAttention> {
    let attn_vb = block_vb.pp("attn");
    let head_dim = cfg.head_dim();
    Ok(GroupedQueryAttention::new(
        linear_no_bias(cfg.hidden_dim, cfg.n_heads * head_dim, attn_vb.pp("wq"))?,
        linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, attn_vb.pp("wk"))?,
        linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, attn_vb.pp("wv"))?,
        linear_no_bias(cfg.n_heads * head_dim, cfg.hidden_dim, attn_vb.pp("wo"))?,
        cfg.n_heads,
        cfg.n_kv_heads,
        head_dim,
    ))
}

fn build_swiglu(hidden_dim: usize, ffn_dim: usize, vb: VarBuilder<'_>) -> Result<SwiGluFfn> {
    Ok(SwiGluFfn::new(
        linear_no_bias(hidden_dim, ffn_dim, vb.pp("w_gate"))?,
        linear_no_bias(hidden_dim, ffn_dim, vb.pp("w_up"))?,
        linear_no_bias(ffn_dim, hidden_dim, vb.pp("w_down"))?,
    ))
}

fn average_dsa_loss(teachers: &[DsaTeacherOutput]) -> Result<Option<Tensor>> {
    let Some(first) = teachers.first() else {
        return Ok(None);
    };
    let mut loss = first.loss.clone();
    for teacher in &teachers[1..] {
        loss = (loss + &teacher.loss)?;
    }
    Ok(Some(loss.affine(1.0 / teachers.len() as f64, 0.0)?))
}

fn build_gated_deltanet(
    hidden_dim: usize,
    _norm_eps: f32,
    config: &aarambh_ai_core::GatedDeltaNetConfig,
    vb: VarBuilder<'_>,
) -> Result<GatedDeltaNetLayer> {
    let key_dim = config.n_heads * config.key_head_dim;
    let value_dim = config.n_heads * config.value_head_dim;
    let conv_init = Init::Uniform {
        lo: -(1.0 / config.conv_kernel_size as f64).sqrt(),
        up: (1.0 / config.conv_kernel_size as f64).sqrt(),
    };
    Ok(GatedDeltaNetLayer::new(
        linear_no_bias(hidden_dim, key_dim, vb.pp("q_proj"))?,
        linear_no_bias(hidden_dim, key_dim, vb.pp("k_proj"))?,
        linear_no_bias(hidden_dim, value_dim, vb.pp("v_proj"))?,
        linear_no_bias(hidden_dim, config.n_heads, vb.pp("beta_proj"))?,
        linear_no_bias(hidden_dim, config.n_heads, vb.pp("alpha_proj"))?,
        linear_no_bias(hidden_dim, value_dim, vb.pp("gate_proj"))?,
        linear_no_bias(value_dim, hidden_dim, vb.pp("out_proj"))?,
        vb.pp("q_conv")
            .get_with_hints((key_dim, config.conv_kernel_size), "weight", conv_init)?,
        vb.pp("k_conv")
            .get_with_hints((key_dim, config.conv_kernel_size), "weight", conv_init)?,
        vb.pp("v_conv").get_with_hints(
            (value_dim, config.conv_kernel_size),
            "weight",
            conv_init,
        )?,
        vb.pp("out_norm")
            .get_with_hints(config.value_head_dim, "weight", Init::Const(1.0))?,
        vb.get_with_hints_dtype(
            config.n_heads,
            "A_log",
            Init::Uniform {
                lo: -6.907_755,
                up: 2.772_589,
            },
            DType::F32,
        )?,
        vb.get_with_hints_dtype(
            config.n_heads,
            "dt_bias",
            Init::Uniform {
                lo: -6.907_255,
                up: -2.252_168,
            },
            DType::F32,
        )?,
        config.clone(),
    ))
}

fn get_moe_expert_weight<'a>(ffn: &'a FeedForwardLayer, suffix: &str) -> Option<&'a Tensor> {
    let ffn = ffn.as_moe()?;
    let (experts, suffix) = if let Some(suffix) = suffix.strip_prefix("ffn.experts.") {
        (ffn.experts(), suffix)
    } else {
        (
            ffn.shared_experts().experts(),
            suffix.strip_prefix("ffn.shared_experts.")?,
        )
    };
    let (expert_idx, name) = suffix.split_once('.')?;
    let expert = experts.get(expert_idx.parse::<usize>().ok()?)?;
    match name {
        "w_gate.weight" => Some(expert.w_gate_weight()),
        "w_up.weight" => Some(expert.w_up_weight()),
        "w_down.weight" => Some(expert.w_down_weight()),
        _ => None,
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
