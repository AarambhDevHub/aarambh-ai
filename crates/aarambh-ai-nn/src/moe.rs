use aarambh_ai_core::MoeConfig;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::dispatch::dense_weighted_dispatch;
use crate::ffn::SwiGluFfn;

#[derive(Debug)]
/// Router output after selecting top-k experts per token.
pub struct GatingOutput {
    /// Selected expert indices with shape `[batch, seq, top_k]`.
    pub indices: Tensor,
    /// Selected expert probabilities with shape `[batch, seq, top_k]`.
    pub weights: Tensor,
    /// Dense expert dispatch weights with shape `[batch, seq, num_experts]`.
    pub dispatch_weights: Tensor,
    /// Differentiable load-balancing auxiliary loss.
    pub aux_loss: Tensor,
    /// Per-expert selected-token fraction, normalized to sum to 1.0.
    pub expert_utilization: Vec<f32>,
}

#[derive(Debug, Default)]
/// Aggregated MoE metadata collected during a model forward pass.
pub struct MoeForwardStats {
    aux_losses: Vec<Tensor>,
    expert_utilization: Vec<f32>,
}

impl MoeForwardStats {
    /// Add one MoE layer's auxiliary loss and expert-utilization summary.
    pub fn record(&mut self, aux_loss: Tensor, expert_utilization: &[f32]) -> Result<()> {
        if self.expert_utilization.is_empty() {
            self.expert_utilization = vec![0.0; expert_utilization.len()];
        }
        if self.expert_utilization.len() != expert_utilization.len() {
            candle_core::bail!("cannot aggregate expert utilization with different expert counts");
        }
        for (dst, src) in self
            .expert_utilization
            .iter_mut()
            .zip(expert_utilization.iter())
        {
            *dst += *src;
        }
        self.aux_losses.push(aux_loss);
        Ok(())
    }

    /// Return the average auxiliary loss across recorded MoE layers.
    pub fn aux_loss(&self) -> Result<Option<Tensor>> {
        let Some((first, rest)) = self.aux_losses.split_first() else {
            return Ok(None);
        };
        let mut sum = first.clone();
        for loss in rest {
            sum = (&sum + loss)?;
        }
        Ok(Some(sum.affine(1.0 / self.aux_losses.len() as f64, 0.0)?))
    }

    /// Return average selected-token fraction per expert.
    pub fn expert_utilization(&self) -> Vec<f32> {
        if self.aux_losses.is_empty() {
            return Vec::new();
        }
        self.expert_utilization
            .iter()
            .map(|value| *value / self.aux_losses.len() as f32)
            .collect()
    }

    /// Return true when no MoE layers have recorded stats.
    pub fn is_empty(&self) -> bool {
        self.aux_losses.is_empty()
    }
}

#[derive(Debug, Clone)]
/// Router followed by independent SwiGLU experts.
pub struct MoeFfn {
    config: MoeConfig,
    router: Linear,
    experts: Vec<SwiGluFfn>,
}

impl MoeFfn {
    /// Create a Mixture-of-Experts feed-forward layer.
    pub fn new(config: MoeConfig, router: Linear, experts: Vec<SwiGluFfn>) -> Self {
        Self {
            config,
            router,
            experts,
        }
    }

    /// Run inference through top-k expert routing.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, false, None)
    }

    /// Run training through top-k expert routing.
    pub fn forward_train(&self, x: &Tensor, stats: Option<&mut MoeForwardStats>) -> Result<Tensor> {
        self.forward_inner(x, true, stats)
    }

    /// Run the MoE layer while recording calibration activations.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut std::collections::HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        capture.insert(format!("blocks.{layer_idx}.ffn.router.weight"), x.clone());
        self.forward_inner_with_expert_capture(x, false, layer_idx, capture, None)
    }

    /// Return the router weight tensor.
    pub fn router_weight(&self) -> &Tensor {
        self.router.weight()
    }

    /// Return the expert layers.
    pub fn experts(&self) -> &[SwiGluFfn] {
        &self.experts
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        train: bool,
        stats: Option<&mut MoeForwardStats>,
    ) -> Result<Tensor> {
        self.forward_inner_with_expert_capture(
            x,
            train,
            0,
            &mut std::collections::HashMap::new(),
            stats,
        )
    }

    fn forward_inner_with_expert_capture(
        &self,
        x: &Tensor,
        train: bool,
        layer_idx: usize,
        capture: &mut std::collections::HashMap<String, Tensor>,
        stats: Option<&mut MoeForwardStats>,
    ) -> Result<Tensor> {
        if self.experts.len() != self.config.num_experts {
            candle_core::bail!(
                "MoE expert count mismatch: config has {}, layer has {}",
                self.config.num_experts,
                self.experts.len()
            );
        }
        let logits = self.router.forward(x)?;
        let gating = top_k_gating(&logits, self.config.top_k)?;
        if let Some(stats) = stats {
            stats.record(gating.aux_loss.clone(), &gating.expert_utilization)?;
        }

        let mut outputs = Vec::with_capacity(self.experts.len());
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let output = if capture.is_empty() {
                if train {
                    expert.forward_train(x)?
                } else {
                    expert.forward(x)?
                }
            } else {
                let prefix = format!("blocks.{layer_idx}.ffn.experts.{expert_idx}");
                capture.insert(format!("{prefix}.w_gate.weight"), x.clone());
                capture.insert(format!("{prefix}.w_up.weight"), x.clone());
                let gate = expert.w_gate_forward(x)?;
                let up = expert.w_up_forward(x)?;
                let hidden =
                    aarambh_ai_kernel::fused_ffn::fused_swiglu(&gate, &up).or_else(|_| {
                        let gate = candle_nn::ops::silu(&gate)?;
                        gate * up
                    })?;
                capture.insert(format!("{prefix}.w_down.weight"), hidden.clone());
                expert.w_down_forward(&hidden)?
            };
            outputs.push(output);
        }
        dense_weighted_dispatch(&outputs, &gating.dispatch_weights)
    }
}

/// Select top-k experts and produce dense dispatch weights.
pub fn top_k_gating(logits: &Tensor, top_k: usize) -> Result<GatingOutput> {
    let dims = logits.dims();
    if dims.len() != 3 {
        candle_core::bail!(
            "router logits must have shape [batch, seq, num_experts], got {:?}",
            dims
        );
    }
    let (batch, seq_len, num_experts) = (dims[0], dims[1], dims[2]);
    if top_k == 0 || top_k > num_experts {
        candle_core::bail!("top_k must be in 1..={num_experts}, got {top_k}");
    }

    let logits_f32 = logits.to_dtype(DType::F32)?.contiguous()?;
    let sorted_indices = logits_f32.arg_sort_last_dim(false)?;
    let indices = sorted_indices.narrow(D::Minus1, 0, top_k)?.contiguous()?;
    let selected_logits = logits_f32.gather(&indices, D::Minus1)?;
    let weights = candle_nn::ops::softmax(&selected_logits, D::Minus1)?;

    let dispatch_weights = Tensor::zeros(
        (batch, seq_len, num_experts),
        DType::F32,
        logits.device(),
    )?
    .scatter_add(&indices, &weights, D::Minus1)?;
    let selected_mask = Tensor::zeros((batch, seq_len, num_experts), DType::F32, logits.device())?
        .scatter_add(
            &indices,
            &Tensor::ones((batch, seq_len, top_k), DType::F32, logits.device())?,
            D::Minus1,
        )?;

    let router_probs = candle_nn::ops::softmax(&logits_f32, D::Minus1)?;
    let token_count = (batch * seq_len) as f64;
    let router_prob_mean = router_probs.sum((0, 1))?.affine(1.0 / token_count, 0.0)?;
    let dispatch_fraction = selected_mask
        .sum((0, 1))?
        .affine(1.0 / (token_count * top_k as f64), 0.0)?;
    let aux_loss = load_balancing_loss_from_stats(&router_prob_mean, &dispatch_fraction)?;
    let expert_utilization = dispatch_fraction.to_vec1::<f32>()?;

    Ok(GatingOutput {
        indices,
        weights,
        dispatch_weights,
        aux_loss,
        expert_utilization,
    })
}

/// Compute shifted Switch-style load-balancing loss from per-expert means.
pub fn load_balancing_loss_from_stats(
    router_prob_mean: &Tensor,
    dispatch_fraction: &Tensor,
) -> Result<Tensor> {
    if router_prob_mean.dims().len() != 1 || dispatch_fraction.dims() != router_prob_mean.dims() {
        candle_core::bail!(
            "router probability and dispatch stats must be same rank-1 shape, got {:?} and {:?}",
            router_prob_mean.dims(),
            dispatch_fraction.dims()
        );
    }
    let num_experts = router_prob_mean.dims()[0] as f64;
    (router_prob_mean * dispatch_fraction)?
        .sum_all()?
        .affine(num_experts, -1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarBuilder, VarMap, linear_no_bias};

    #[test]
    fn top_k_gating_selects_correct_number_of_experts_per_token() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![0.0f32, 3.0, 1.0, 2.0, 9.0, 8.0, 7.0, 6.0],
            (1, 2, 4),
            &device,
        )
        .unwrap();
        let gating = top_k_gating(&logits, 2).unwrap();
        assert_eq!(gating.indices.dims(), &[1, 2, 2]);
        assert_eq!(gating.weights.dims(), &[1, 2, 2]);
        assert_eq!(gating.dispatch_weights.dims(), &[1, 2, 4]);
        let sums = gating
            .weights
            .sum(D::Minus1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for row in sums.into_iter().flatten() {
            assert!((row - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn load_balancing_loss_is_zero_at_perfectly_uniform_routing() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.25f32; 4], (4,), &device).unwrap();
        let dispatch = Tensor::from_vec(vec![0.25f32; 4], (4,), &device).unwrap();
        let loss = load_balancing_loss_from_stats(&probs, &dispatch)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn moe_ffn_output_shape_matches_dense_ffn_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 16,
            aux_loss_weight: 0.01,
            every_n_layers: 1,
        };
        let router = linear_no_bias(8, 2, vb.pp("router")).unwrap();
        let experts = (0..2)
            .map(|idx| {
                let expert_vb = vb.pp("experts").pp(idx);
                SwiGluFfn::new(
                    linear_no_bias(8, 16, expert_vb.pp("w_gate")).unwrap(),
                    linear_no_bias(8, 16, expert_vb.pp("w_up")).unwrap(),
                    linear_no_bias(16, 8, expert_vb.pp("w_down")).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let moe = MoeFfn::new(cfg, router, experts);
        let x = vb
            .get_with_hints(
                (2, 3, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )
            .unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 3, 8]);
    }
}
