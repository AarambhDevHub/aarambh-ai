use std::collections::HashMap;

use aarambh_ai_core::{AarambhError, ModelConfig, QatTarget, Result};
use aarambh_ai_quant::{QatContext, QatLinear};
use candle_core::Tensor;
use candle_nn::{Init, Linear, VarBuilder, linear_no_bias};

use crate::{GroupedQueryAttention, RMSNorm, RopeCache, SwiGluFfn, TransformerBlock};

#[derive(Debug, Clone)]
/// One auxiliary future-token prediction head.
///
/// The head refines a short causal sequence containing one contextual trunk
/// state followed by the real or proposed intervening token embeddings. Its
/// output projection shares the main model's LM-head weight.
pub struct MtpHead {
    offset: usize,
    trunk_norm: RMSNorm,
    token_norm: RMSNorm,
    refine_block: TransformerBlock,
    output_norm: RMSNorm,
    lm_head: QatLinear,
}

impl MtpHead {
    /// Build an auxiliary head for a future-token offset of at least two.
    pub fn new(
        offset: usize,
        cfg: &ModelConfig,
        lm_head_weight: &Tensor,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        Self::new_with_qat(offset, cfg, lm_head_weight, vb, None)
    }

    /// Build an auxiliary head with optional QAT projection coverage.
    pub fn new_with_qat(
        offset: usize,
        cfg: &ModelConfig,
        lm_head_weight: &Tensor,
        vb: VarBuilder<'_>,
        qat: Option<QatContext>,
    ) -> Result<Self> {
        if offset < 2 {
            return Err(AarambhError::Config(
                "an MTP auxiliary head offset must be at least 2".into(),
            ));
        }
        if offset > cfg.max_seq_len {
            return Err(AarambhError::Config(format!(
                "MTP head offset {offset} exceeds max_seq_len {}",
                cfg.max_seq_len
            )));
        }

        let trunk_norm = build_norm(cfg, vb.pp("trunk_norm"))?;
        let token_norm = build_norm(cfg, vb.pp("token_norm"))?;
        let refine_vb = vb.pp("refine");
        let refine_block = TransformerBlock::new(
            build_norm(cfg, refine_vb.pp("norm1"))?,
            build_attention(cfg, refine_vb.pp("attn"), qat.clone())?,
            build_norm(cfg, refine_vb.pp("norm2"))?,
            build_swiglu(cfg, refine_vb.pp("ffn"), qat.clone())?,
        );
        let output_norm = build_norm(cfg, vb.pp("output_norm"))?;

        Ok(Self {
            offset,
            trunk_norm,
            token_norm,
            refine_block,
            output_norm,
            lm_head: QatLinear::new(
                Linear::new(lm_head_weight.clone(), None),
                QatTarget::Mtp,
                qat,
            ),
        })
    }

    /// Return the future-token offset predicted by this head.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Run the differentiable training path.
    ///
    /// `trunk_hidden` has shape `[batch, anchors, hidden]` and
    /// `intervening_embeddings` has shape
    /// `[batch, anchors, offset - 1, hidden]`.
    pub fn forward_train(
        &self,
        trunk_hidden: &Tensor,
        intervening_embeddings: &Tensor,
        rope: &RopeCache,
    ) -> Result<Tensor> {
        let (local, batch, anchors) =
            self.local_sequence(trunk_hidden, intervening_embeddings, true)?;
        let refined = self.refine_block.forward_train(&local, rope, None, 0)?;
        self.project_last(&refined, batch, anchors, true)
    }

    /// Run the inference path for real or proposed intervening embeddings.
    pub fn forward(
        &self,
        trunk_hidden: &Tensor,
        intervening_embeddings: &Tensor,
        rope: &RopeCache,
    ) -> Result<Tensor> {
        let (local, batch, anchors) =
            self.local_sequence(trunk_hidden, intervening_embeddings, false)?;
        let refined = self.refine_block.forward(&local, rope, None, None, 0)?;
        self.project_last(&refined, batch, anchors, false)
    }

    /// Run the inference path while recording linear-layer calibration inputs.
    pub fn forward_with_capture(
        &self,
        trunk_hidden: &Tensor,
        intervening_embeddings: &Tensor,
        rope: &RopeCache,
        head_index: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let (local, batch, anchors) =
            self.local_sequence(trunk_hidden, intervening_embeddings, false)?;
        let mut block_capture = HashMap::new();
        let refined = self.refine_block.forward_with_capture(
            &local,
            rope,
            None,
            head_index,
            &mut block_capture,
        )?;
        let old_prefix = format!("blocks.{head_index}.");
        let new_prefix = format!("mtp.heads.{head_index}.refine.");
        for (name, value) in block_capture {
            let suffix = name.strip_prefix(&old_prefix).unwrap_or(&name);
            capture.insert(format!("{new_prefix}{suffix}"), value);
        }
        self.project_last(&refined, batch, anchors, false)
    }

    /// Return this head's uniquely owned tensors using suffix-only names.
    pub fn named_tensors(&self) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "trunk_norm.weight".to_string(),
            self.trunk_norm.weight().clone(),
        );
        tensors.insert(
            "token_norm.weight".to_string(),
            self.token_norm.weight().clone(),
        );
        tensors.insert(
            "refine.norm1.weight".to_string(),
            self.refine_block.norm1().weight().clone(),
        );
        let attention = self
            .refine_block
            .mixer()
            .as_attention()
            .expect("MTP refinement always uses grouped-query attention");
        for (name, tensor) in [
            ("wq", attention.wq_weight()),
            ("wk", attention.wk_weight()),
            ("wv", attention.wv_weight()),
            ("wo", attention.wo_weight()),
        ] {
            tensors.insert(format!("refine.attn.{name}.weight"), tensor.clone());
        }
        tensors.insert(
            "refine.norm2.weight".to_string(),
            self.refine_block.norm2().weight().clone(),
        );
        let ffn = self
            .refine_block
            .ffn()
            .as_dense()
            .expect("MTP refinement always uses a dense SwiGLU FFN");
        tensors.insert(
            "refine.ffn.w_gate.weight".to_string(),
            ffn.w_gate_weight().clone(),
        );
        tensors.insert(
            "refine.ffn.w_up.weight".to_string(),
            ffn.w_up_weight().clone(),
        );
        tensors.insert(
            "refine.ffn.w_down.weight".to_string(),
            ffn.w_down_weight().clone(),
        );
        tensors.insert(
            "output_norm.weight".to_string(),
            self.output_norm.weight().clone(),
        );
        tensors
    }

    /// Return a uniquely owned tensor by its suffix-only checkpoint name.
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        match name {
            "trunk_norm.weight" => Some(self.trunk_norm.weight()),
            "token_norm.weight" => Some(self.token_norm.weight()),
            "refine.norm1.weight" => Some(self.refine_block.norm1().weight()),
            "refine.attn.wq.weight" => self
                .refine_block
                .mixer()
                .as_attention()
                .map(GroupedQueryAttention::wq_weight),
            "refine.attn.wk.weight" => self
                .refine_block
                .mixer()
                .as_attention()
                .map(GroupedQueryAttention::wk_weight),
            "refine.attn.wv.weight" => self
                .refine_block
                .mixer()
                .as_attention()
                .map(GroupedQueryAttention::wv_weight),
            "refine.attn.wo.weight" => self
                .refine_block
                .mixer()
                .as_attention()
                .map(GroupedQueryAttention::wo_weight),
            "refine.norm2.weight" => Some(self.refine_block.norm2().weight()),
            "refine.ffn.w_gate.weight" => self
                .refine_block
                .ffn()
                .as_dense()
                .map(SwiGluFfn::w_gate_weight),
            "refine.ffn.w_up.weight" => self
                .refine_block
                .ffn()
                .as_dense()
                .map(SwiGluFfn::w_up_weight),
            "refine.ffn.w_down.weight" => self
                .refine_block
                .ffn()
                .as_dense()
                .map(SwiGluFfn::w_down_weight),
            "output_norm.weight" => Some(self.output_norm.weight()),
            _ => None,
        }
    }

    fn local_sequence(
        &self,
        trunk_hidden: &Tensor,
        intervening_embeddings: &Tensor,
        training: bool,
    ) -> Result<(Tensor, usize, usize)> {
        let (batch, anchors, hidden) = trunk_hidden.dims3()?;
        let (embed_batch, embed_anchors, intervening, embed_hidden) =
            intervening_embeddings.dims4()?;
        if batch != embed_batch || anchors != embed_anchors || hidden != embed_hidden {
            return Err(AarambhError::Shape(format!(
                "MTP trunk shape {:?} is incompatible with intervening embeddings {:?}",
                trunk_hidden.dims(),
                intervening_embeddings.dims()
            )));
        }
        if intervening + 1 != self.offset {
            return Err(AarambhError::Shape(format!(
                "MTP offset {} requires {} intervening embeddings, got {intervening}",
                self.offset,
                self.offset - 1
            )));
        }
        if anchors == 0 {
            return Err(AarambhError::Shape(
                "MTP requires at least one valid anchor position".into(),
            ));
        }

        let trunk = if training {
            self.trunk_norm.forward_train(trunk_hidden)?
        } else {
            self.trunk_norm.forward(trunk_hidden)?
        }
        .unsqueeze(2)?;
        let intervening = if training {
            self.token_norm.forward_train(intervening_embeddings)?
        } else {
            self.token_norm.forward(intervening_embeddings)?
        };
        let local = Tensor::cat(&[&trunk, &intervening], 2)?.reshape((
            batch * anchors,
            self.offset,
            hidden,
        ))?;
        Ok((local, batch, anchors))
    }

    fn project_last(
        &self,
        refined: &Tensor,
        batch: usize,
        anchors: usize,
        training: bool,
    ) -> Result<Tensor> {
        let hidden = refined.dim(2)?;
        let last = refined
            .narrow(1, self.offset - 1, 1)?
            .reshape((batch, anchors, hidden))?;
        let last = if training {
            self.output_norm.forward_train(&last)?
        } else {
            self.output_norm.forward(&last)?
        };
        Ok(self.lm_head.forward(&last)?)
    }
}

fn build_norm(cfg: &ModelConfig, vb: VarBuilder<'_>) -> Result<RMSNorm> {
    Ok(RMSNorm::new(
        vb.get_with_hints(cfg.hidden_dim, "weight", Init::Const(1.0))?,
        cfg.norm_eps as f32,
    ))
}

fn build_attention(
    cfg: &ModelConfig,
    vb: VarBuilder<'_>,
    qat: Option<QatContext>,
) -> Result<GroupedQueryAttention> {
    let head_dim = cfg.head_dim();
    Ok(GroupedQueryAttention::new(
        QatLinear::new(
            linear_no_bias(cfg.hidden_dim, cfg.n_heads * head_dim, vb.pp("wq"))?,
            QatTarget::Mtp,
            qat.clone(),
        ),
        QatLinear::new(
            linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, vb.pp("wk"))?,
            QatTarget::Mtp,
            qat.clone(),
        ),
        QatLinear::new(
            linear_no_bias(cfg.hidden_dim, cfg.n_kv_heads * head_dim, vb.pp("wv"))?,
            QatTarget::Mtp,
            qat.clone(),
        ),
        QatLinear::new(
            linear_no_bias(cfg.n_heads * head_dim, cfg.hidden_dim, vb.pp("wo"))?,
            QatTarget::Mtp,
            qat,
        ),
        cfg.n_heads,
        cfg.n_kv_heads,
        head_dim,
    ))
}

fn build_swiglu(
    cfg: &ModelConfig,
    vb: VarBuilder<'_>,
    qat: Option<QatContext>,
) -> Result<SwiGluFfn> {
    Ok(SwiGluFfn::new(
        QatLinear::new(
            linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, vb.pp("w_gate"))?,
            QatTarget::Mtp,
            qat.clone(),
        ),
        QatLinear::new(
            linear_no_bias(cfg.hidden_dim, cfg.ffn_dim, vb.pp("w_up"))?,
            QatTarget::Mtp,
            qat.clone(),
        ),
        QatLinear::new(
            linear_no_bias(cfg.ffn_dim, cfg.hidden_dim, vb.pp("w_down"))?,
            QatTarget::Mtp,
            qat,
        ),
    ))
}
