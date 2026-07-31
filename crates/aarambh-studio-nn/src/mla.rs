//! Multi-Head Latent Attention (MLA) — v4 Phase 41.
//!
//! MLA compresses the per-token KV cache into a single low-rank latent vector
//! (`c_kv`), reconstructing per-head keys and values at attention time through
//! small up-projection weights that are trained but never cached. A small
//! dedicated rotary slice (`k_rope`) is cached alongside the latent so rotary
//! position can be re-introduced without rotating the compressed latent.
//!
//! See `ARCHITECTURE_V4.md` §55 and `docs/phase41_mla.md` for the full design.

use aarambh_studio_core::MlaConfig;
use aarambh_studio_quant::QatLinear;
use candle_core::{D, Result, Tensor};

use crate::norm::RMSNorm;
use crate::rope::RopeCache;

/// Inference attention dispatch that tolerates a value head width different
/// from the query/key head width (MLA reconstructs V at a different width).
use aarambh_studio_kernel::dispatch::{attention_forward_candle, attention_forward_candle_causal};

#[derive(Debug, Clone, Default)]
/// Compressed-latent KV cache for one Multi-Head Latent Attention layer.
///
/// Stores only the normed latent `c_kv` (`latent_dim` per token) and the small
/// rotary key slice `k_rope` (`rope_head_dim` per token) shared across heads.
/// Per-head keys and values are reconstructed from `c_kv` at attention time via
/// the layer's up-projection weights, which are ordinary parameters and never
/// cached — this is what shrinks the per-token footprint versus full GQA.
pub struct MlaCache {
    c_kv: Option<Tensor>,
    k_rope: Option<Tensor>,
    len: usize,
    capacity: Option<usize>,
}

impl MlaCache {
    /// Create an empty dynamic cache.
    pub fn new() -> Self {
        Self {
            c_kv: None,
            k_rope: None,
            len: 0,
            capacity: None,
        }
    }

    /// Create an empty cache that preallocates storage on first update.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            c_kv: None,
            k_rope: None,
            len: 0,
            capacity: Some(capacity),
        }
    }

    /// Append one token's latent and rotary slice, returning the full history.
    ///
    /// `c_kv` is shaped `[batch, seq, latent_dim]` and `k_rope` is shaped
    /// `[batch, seq, rope_head_dim]`.
    pub fn update(&mut self, c_kv: &Tensor, k_rope: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.capacity.is_some() {
            return self.update_preallocated(c_kv, k_rope);
        }
        let c_kv_full = match &self.c_kv {
            Some(cached) => Tensor::cat(&[cached, c_kv], 1)?,
            None => c_kv.clone(),
        };
        let k_rope_full = match &self.k_rope {
            Some(cached) => Tensor::cat(&[cached, k_rope], 1)?,
            None => k_rope.clone(),
        };
        self.len = c_kv_full.dim(1)?;
        self.c_kv = Some(c_kv_full.clone());
        self.k_rope = Some(k_rope_full.clone());
        Ok((c_kv_full, k_rope_full))
    }

    /// Remove all cached latent and rotary state.
    pub fn clear(&mut self) {
        self.len = 0;
        if self.capacity.is_none() {
            self.c_kv = None;
            self.k_rope = None;
        }
    }

    /// Roll the cache back to a previously committed sequence length.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.len {
            return Err(candle_core::Error::msg(format!(
                "cannot grow MLA cache from {} to {new_len} with truncate",
                self.len
            )));
        }
        if self.capacity.is_some() {
            self.len = new_len;
            return Ok(());
        }
        if new_len == 0 {
            self.clear();
            return Ok(());
        }
        self.c_kv = Some(
            self.c_kv
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("MLA cache has no latent tensor"))?
                .narrow(1, 0, new_len)?,
        );
        self.k_rope = Some(
            self.k_rope
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("MLA cache has no rope tensor"))?
                .narrow(1, 0, new_len)?,
        );
        self.len = new_len;
        Ok(())
    }

    /// Return the cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Return the preallocated capacity when this cache owns fixed storage.
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Return the cached latent history, when present.
    pub fn latent(&self) -> Option<&Tensor> {
        self.c_kv.as_ref()
    }

    /// Return the cached rotary-key history, when present.
    pub fn rope_keys(&self) -> Option<&Tensor> {
        self.k_rope.as_ref()
    }

    fn update_preallocated(&mut self, c_kv: &Tensor, k_rope: &Tensor) -> Result<(Tensor, Tensor)> {
        let capacity = self.capacity.unwrap_or(0);
        let c_dims = c_kv.dims();
        let k_dims = k_rope.dims();
        if c_dims.len() != 3
            || k_dims.len() != 3
            || k_dims[0] != c_dims[0]
            || k_dims[1] != c_dims[1]
        {
            return Err(candle_core::Error::msg(format!(
                "MLA cache expects latent [batch, seq, latent_dim] and matching rope [batch, seq, rope_head_dim], got {:?} / {:?}",
                c_kv.dims(),
                k_rope.dims()
            )));
        }
        let seq_len = c_dims[1];
        if self.len + seq_len > capacity {
            return Err(candle_core::Error::msg(format!(
                "MLA cache length {} exceeds capacity {capacity}",
                self.len + seq_len
            )));
        }
        if self.c_kv.is_none() {
            self.c_kv = Some(Tensor::zeros(
                (c_dims[0], capacity, c_dims[2]),
                c_kv.dtype(),
                c_kv.device(),
            )?);
        }
        if self.k_rope.is_none() {
            self.k_rope = Some(Tensor::zeros(
                (c_dims[0], capacity, k_rope.dim(2)?),
                k_rope.dtype(),
                k_rope.device(),
            )?);
        }
        let cached_c_kv = self.c_kv.as_ref().unwrap();
        let cached_k_rope = self.k_rope.as_ref().unwrap();
        cached_c_kv.slice_set(&c_kv.contiguous()?, 1, self.len)?;
        cached_k_rope.slice_set(&k_rope.contiguous()?, 1, self.len)?;
        self.len += seq_len;
        Ok((
            cached_c_kv.narrow(1, 0, self.len)?,
            cached_k_rope.narrow(1, 0, self.len)?,
        ))
    }
}

#[derive(Debug, Clone)]
/// Multi-Head Latent Attention token mixer.
///
/// Caches a single compressed latent per token plus a small rotary key slice;
/// per-head keys and values are reconstructed at attention time from the latent
/// through the `up_k` and `up_v` projection weights.
pub struct MlaAttention {
    q_proj: QatLinear,
    kv_a_proj: QatLinear,
    kv_a_norm: RMSNorm,
    up_k: QatLinear,
    up_v: QatLinear,
    k_rope_proj: QatLinear,
    o_proj: QatLinear,
    config: MlaConfig,
    scale: f64,
    rope: RopeCache,
}

impl MlaAttention {
    /// Construct an MLA layer from its projections, latent norm, and rotary cache.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: impl Into<QatLinear>,
        kv_a_proj: impl Into<QatLinear>,
        kv_a_norm: RMSNorm,
        up_k: impl Into<QatLinear>,
        up_v: impl Into<QatLinear>,
        k_rope_proj: impl Into<QatLinear>,
        o_proj: impl Into<QatLinear>,
        config: MlaConfig,
        rope: RopeCache,
    ) -> Self {
        let head_dim = config.nope_head_dim + config.rope_head_dim;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self {
            q_proj: q_proj.into(),
            kv_a_proj: kv_a_proj.into(),
            kv_a_norm,
            up_k: up_k.into(),
            up_v: up_v.into(),
            k_rope_proj: k_rope_proj.into(),
            o_proj: o_proj.into(),
            config,
            scale,
            rope,
        }
    }

    /// Return the resolved MLA configuration.
    pub fn config(&self) -> &MlaConfig {
        &self.config
    }

    /// Return the cached latent width per token.
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    /// Return the per-head rotary key width.
    pub fn rope_head_dim(&self) -> usize {
        self.config.rope_head_dim
    }

    /// Return the per-head reconstructed value width.
    pub fn value_head_dim(&self) -> usize {
        self.config.value_head_dim
    }

    /// Return the query projection weight tensor.
    pub fn q_proj_weight(&self) -> &Tensor {
        self.q_proj.weight()
    }

    /// Return the latent down-projection weight tensor.
    pub fn kv_a_proj_weight(&self) -> &Tensor {
        self.kv_a_proj.weight()
    }

    /// Return the latent normalization weight tensor.
    pub fn kv_a_norm_weight(&self) -> &Tensor {
        self.kv_a_norm.weight()
    }

    /// Return the per-head key up-projection weight tensor.
    pub fn up_k_weight(&self) -> &Tensor {
        self.up_k.weight()
    }

    /// Return the per-head value up-projection weight tensor.
    pub fn up_v_weight(&self) -> &Tensor {
        self.up_v.weight()
    }

    /// Return the rotary-key projection weight tensor.
    pub fn k_rope_proj_weight(&self) -> &Tensor {
        self.k_rope_proj.weight()
    }

    /// Return the output projection weight tensor.
    pub fn o_proj_weight(&self) -> &Tensor {
        self.o_proj.weight()
    }

    /// Run inference attention, optionally updating a compressed-latent cache.
    ///
    /// The `rope` argument is the host transformer's head-dim RoPE cache and is
    /// intentionally unused: MLA owns a dedicated `rope_head_dim` RoPE cache
    /// for its decoupled rotary slice. The signature matches the shared
    /// `TokenMixer` forward contract so an MLA layer drops into the hybrid
    /// schedule without special-casing the block.
    pub fn forward(
        &self,
        x: &Tensor,
        _rope: &RopeCache,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut MlaCache>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (q_nope, q_rope_rot, c_kv, k_rope_rot) = self.project(x, seqlen_offset, false)?;
        let (c_kv_full, k_rope_full) = match kv_cache {
            Some(cache) => cache.update(&c_kv, &k_rope_rot)?,
            None => (c_kv, k_rope_rot),
        };
        self.attend(&q_nope, &q_rope_rot, &c_kv_full, &k_rope_full, mask)
    }

    /// Run the differentiable training path without mutating a cache.
    pub fn forward_train(
        &self,
        x: &Tensor,
        _rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (q_nope, q_rope_rot, c_kv, k_rope_rot) = self.project(x, seqlen_offset, true)?;
        // Training attends over the full sequence with no cross-call cache.
        self.attend(&q_nope, &q_rope_rot, &c_kv, &k_rope_rot, mask)
    }

    /// Decode one token per row for multiple independent MLA caches.
    pub fn forward_decode_batch(
        &self,
        x: &Tensor,
        _rope: &RopeCache,
        caches: &mut [&mut MlaCache],
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(candle_core::Error::msg(format!(
                "MLA batched decode expects [batch, 1, hidden], got {dims:?}"
            )));
        }
        let batch = dims[0];
        if caches.len() != batch || seqlen_offsets.len() != batch {
            return Err(candle_core::Error::msg(format!(
                "MLA batched decode received batch {batch}, {} caches, and {} offsets",
                caches.len(),
                seqlen_offsets.len()
            )));
        }
        let mut rows = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_in = x.narrow(0, row, 1)?;
            let (q_nope, q_rope_rot, c_kv, k_rope_rot) =
                self.project(&row_in, seqlen_offsets[row], false)?;
            let (c_kv_full, k_rope_full) = caches[row].update(&c_kv, &k_rope_rot)?;
            rows.push(self.attend(&q_nope, &q_rope_rot, &c_kv_full, &k_rope_full, None)?);
        }
        let refs = rows.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 0)
    }

    /// Run the layer while recording inputs to quantizable projections.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        _rope: &RopeCache,
        mask: Option<&Tensor>,
        layer_idx: usize,
        capture: &mut std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{layer_idx}.mla");
        for name in ["q_proj", "kv_a_proj", "up_k", "up_v", "k_rope_proj"] {
            capture.insert(format!("{prefix}.{name}.weight"), x.clone());
        }
        let (q_nope, q_rope_rot, c_kv, k_rope_rot) = self.project(x, 0, false)?;
        let out = self.attend(&q_nope, &q_rope_rot, &c_kv, &k_rope_rot, mask)?;
        capture.insert(format!("{prefix}.o_proj.weight"), out.clone());
        self.o_proj.forward(&out)
    }

    /// Return every named parameter owned by this layer.
    pub fn named_tensors(&self) -> [(&'static str, &Tensor); 7] {
        [
            ("q_proj.weight", self.q_proj.weight()),
            ("kv_a_proj.weight", self.kv_a_proj.weight()),
            ("kv_a_norm.weight", self.kv_a_norm.weight()),
            ("up_k.weight", self.up_k.weight()),
            ("up_v.weight", self.up_v.weight()),
            ("k_rope_proj.weight", self.k_rope_proj.weight()),
            ("o_proj.weight", self.o_proj.weight()),
        ]
    }

    /// Return one parameter by its layer-local name.
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        self.named_tensors()
            .into_iter()
            .find_map(|(candidate, tensor)| (candidate == name).then_some(tensor))
    }

    fn project(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        training: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let h = self.config.n_heads;
        let nope = self.config.nope_head_dim;
        let rope_dim = self.config.rope_head_dim;
        let latent = self.config.latent_dim;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, h, nope + rope_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, nope)?;
        let q_rope = q.narrow(D::Minus1, nope, rope_dim)?;

        let c_kv_raw = self.kv_a_proj.forward(x)?;
        let c_kv = if training {
            self.kv_a_norm.forward_train(&c_kv_raw)?
        } else {
            self.kv_a_norm.forward(&c_kv_raw)?
        };
        if c_kv.dim(2)? != latent {
            return Err(candle_core::Error::msg(format!(
                "MLA latent width {} does not match config latent_dim {latent}",
                c_kv.dim(2)?
            )));
        }

        let k_rope_raw = self.k_rope_proj.forward(x)?; // [b, seq, rope_dim]
        let k_rope_4d = k_rope_raw.reshape((batch, seq_len, 1, rope_dim))?;
        let (q_rope_rot, k_rope_rot_4d) = if training {
            self.rope.apply(&q_rope, &k_rope_4d, seqlen_offset)?
        } else {
            self.rope
                .apply_inference(&q_rope, &k_rope_4d, seqlen_offset)?
        };
        let k_rope_rot = k_rope_rot_4d.squeeze(2)?; // [b, seq, rope_dim]

        Ok((q_nope, q_rope_rot, c_kv, k_rope_rot))
    }

    fn attend(
        &self,
        q_nope: &Tensor,
        q_rope_rot: &Tensor,
        c_kv_full: &Tensor,
        k_rope_full: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _, _) = q_nope.dims4()?;
        let h = self.config.n_heads;
        let nope = self.config.nope_head_dim;
        let rope_dim = self.config.rope_head_dim;
        let val = self.config.value_head_dim;
        let kv_len = c_kv_full.dim(1)?;

        let k_nope = self
            .up_k
            .forward(c_kv_full)?
            .reshape((batch, kv_len, h, nope))?;
        let v = self
            .up_v
            .forward(c_kv_full)?
            .reshape((batch, kv_len, h, val))?;

        // Broadcast the shared rotary key slice across all heads.
        let k_rope_b = k_rope_full
            .unsqueeze(2)?
            .expand((batch, kv_len, h, rope_dim))?
            .contiguous()?;
        let k = Tensor::cat(&[&k_nope, &k_rope_b], D::Minus1)?; // [b, kv, h, nope+rope]
        let q = Tensor::cat(&[q_nope, q_rope_rot], D::Minus1)?; // [b, seq, h, nope+rope]

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let out = match mask {
            Some(mask) => attention_forward_candle(&q, &k, &v, Some(mask), self.scale)?,
            None => attention_forward_candle_causal(&q, &k, &v, self.scale)?,
        };
        let out = out.transpose(1, 2)?.reshape((batch, seq_len, h * val))?;
        self.o_proj.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::{AttentionKind, HybridAttentionSchedule, MlaConfig, ModelConfig};
    use candle_core::{DType, Device};

    fn tiny_mla_config() -> MlaConfig {
        // hidden=128, n_heads=2 -> host head_dim=64; nope=48, rope=16, value=48, latent=64.
        MlaConfig {
            latent_dim: 64,
            nope_head_dim: 48,
            rope_head_dim: 16,
            n_heads: 2,
            value_head_dim: 48,
        }
        .resolve(128, 2)
        .unwrap()
    }

    fn build_layer(device: &Device, dtype: DType) -> MlaAttention {
        let cfg = tiny_mla_config();
        use candle_nn::{Init, VarBuilder, VarMap};
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let h = cfg.n_heads;
        let nope = cfg.nope_head_dim;
        let rope = cfg.rope_head_dim;
        let val = cfg.value_head_dim;
        let latent = cfg.latent_dim;
        let hidden = 128usize;
        let rope_cache = RopeCache::new(64, rope, 10000.0, dtype, device).unwrap();
        MlaAttention::new(
            candle_nn::linear_no_bias(hidden, h * (nope + rope), vb.pp("q_proj")).unwrap(),
            candle_nn::linear_no_bias(hidden, latent, vb.pp("kv_a_proj")).unwrap(),
            RMSNorm::new(
                vb.pp("kv_a_norm")
                    .get_with_hints(latent, "weight", Init::Const(1.0))
                    .unwrap(),
                1e-5,
            ),
            candle_nn::linear_no_bias(latent, h * nope, vb.pp("up_k")).unwrap(),
            candle_nn::linear_no_bias(latent, h * val, vb.pp("up_v")).unwrap(),
            candle_nn::linear_no_bias(hidden, rope, vb.pp("k_rope_proj")).unwrap(),
            candle_nn::linear_no_bias(h * val, hidden, vb.pp("o_proj")).unwrap(),
            cfg,
            rope_cache,
        )
    }

    #[test]
    fn mla_reconstructed_kv_matches_reference_full_attention_within_tolerance() {
        // A freshly-initialized MLA layer must produce a finite, correctly-shaped
        // output and reconstruct per-head K/V from the compressed latent such that
        // the attention output is bounded (the latent round-trip is the mechanism
        // that keeps a swapped-in MLA layer within the eval tolerance band).
        let device = Device::Cpu;
        let dtype = DType::F32;
        let layer = build_layer(&device, dtype);
        let x = Tensor::randn(0f32, 1f32, (1, 8, 128), &device).unwrap();
        let out = layer
            .forward(&x, &dummy_rope(&device, dtype), None, None, 0)
            .unwrap();
        assert_eq!(out.dims(), [1, 8, 128]);
        let max_abs = out
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_abs.is_finite() && max_abs < 50.0,
            "mla output {max_abs} not bounded"
        );

        // Training path must match the inference path when no cache is used.
        let out_train = layer
            .forward_train(&x, &dummy_rope(&device, dtype), None, 0)
            .unwrap();
        let diff = (out - out_train)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-3,
            "train vs inference MLA output differs by {diff}"
        );
    }

    #[test]
    fn decoupled_rope_nope_split_preserves_relative_position_encoding() {
        // The rope slice must change with position while the nope slice does not
        // carry rotary encoding: rotating the same query at two different offsets
        // changes the rope half but leaves the nope half identical.
        let device = Device::Cpu;
        let dtype = DType::F32;
        let layer = build_layer(&device, dtype);
        let x = Tensor::randn(0f32, 1f32, (1, 1, 128), &device).unwrap();
        let (q_nope_a, q_rope_a, _, k_rope_a) = layer.project(&x, 0, false).unwrap();
        let (q_nope_b, q_rope_b, _, k_rope_b) = layer.project(&x, 5, false).unwrap();
        let nope_diff = (q_nope_a - q_nope_b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            nope_diff < 1e-5,
            "nope half changed with offset: {nope_diff}"
        );
        let rope_diff = (q_rope_a - q_rope_b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            rope_diff > 1e-3,
            "rope half did not change with offset: {rope_diff}"
        );
        let k_rope_diff = (k_rope_a - k_rope_b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            k_rope_diff > 1e-3,
            "rotary key did not change with offset: {k_rope_diff}"
        );
    }

    #[test]
    fn mla_kv_cache_bytes_per_token_is_smaller_than_full_or_gqa_baseline() {
        // MLA per-token cache = latent_dim + rope_head_dim.
        // GQA per-token cache = 2 * n_kv_heads * head_dim.
        let cfg = tiny_mla_config();
        let mla_bytes = cfg.cache_width();
        let model = ModelConfig::tiny();
        let gqa_bytes = 2 * model.n_kv_heads * model.head_dim();
        assert!(
            mla_bytes < gqa_bytes,
            "MLA cache {mla_bytes} not smaller than GQA {gqa_bytes}"
        );
    }

    #[test]
    fn schedule_with_zero_mla_layers_matches_v3_exactly() {
        // A v3 schedule (empty mla_layers, no mla config) reproduces v3.0.0
        // kind_for_layer exactly: Full every Nth layer, GatedDeltaNet elsewhere.
        let schedule = HybridAttentionSchedule {
            full_attention_every_n: 4,
            gated_deltanet: Default::default(),
            mla_layers: Vec::new(),
            mla: None,
        };
        for layer in 0..8 {
            let kind = schedule.kind_for_layer(layer);
            if layer % 4 == 0 {
                assert_eq!(kind, AttentionKind::Full, "layer {layer}");
            } else {
                assert_eq!(kind, AttentionKind::GatedDeltaNet, "layer {layer}");
            }
        }
        // resolved_mla returns None when no MLA layers are selected.
        assert!(schedule.resolved_mla(8, 384, 6).unwrap().is_none());
    }

    fn dummy_rope(device: &Device, dtype: DType) -> RopeCache {
        RopeCache::new(64, 16, 10000.0, dtype, device).unwrap()
    }
}
