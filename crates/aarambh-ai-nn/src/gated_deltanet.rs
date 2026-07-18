use std::collections::HashMap;

use aarambh_ai_core::GatedDeltaNetConfig;
use aarambh_ai_quant::QatLinear;
use candle_core::{DType, Result, Tensor};

const NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Execution form used by a Gated DeltaNet layer.
pub enum DeltaNetForm {
    /// Sequential recurrence for autoregressive inference.
    Sequential,
    /// Sequence chunks used by training and long-prompt prefill.
    ChunkParallel {
        /// Number of tokens grouped into one execution chunk.
        chunk_size: usize,
    },
}

#[derive(Debug, Clone, Default)]
/// Fixed-size recurrent and short-convolution state for one Gated DeltaNet layer.
pub struct DeltaNetState {
    recurrent: Option<Tensor>,
    q_history: Option<Tensor>,
    k_history: Option<Tensor>,
    v_history: Option<Tensor>,
    len: usize,
}

impl DeltaNetState {
    /// Create an empty recurrent state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove recurrent and convolution history.
    pub fn clear(&mut self) {
        self.recurrent = None;
        self.q_history = None;
        self.k_history = None;
        self.v_history = None;
        self.len = 0;
    }

    /// Return the number of tokens represented by this state.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Return the number of recurrent elements, independent of sequence length.
    pub fn recurrent_elements(&self) -> usize {
        self.recurrent.as_ref().map(Tensor::elem_count).unwrap_or(0)
    }

    /// Return all recurrent-matrix and convolution-history elements.
    pub fn state_elements(&self) -> usize {
        self.recurrent_elements()
            + self.q_history.as_ref().map(Tensor::elem_count).unwrap_or(0)
            + self.k_history.as_ref().map(Tensor::elem_count).unwrap_or(0)
            + self.v_history.as_ref().map(Tensor::elem_count).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
/// Gated DeltaNet linear-attention token mixer.
pub struct GatedDeltaNetLayer {
    q_proj: QatLinear,
    k_proj: QatLinear,
    v_proj: QatLinear,
    beta_proj: QatLinear,
    alpha_proj: QatLinear,
    gate_proj: QatLinear,
    out_proj: QatLinear,
    q_conv: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    out_norm: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    config: GatedDeltaNetConfig,
}

impl GatedDeltaNetLayer {
    /// Construct a Gated DeltaNet layer from projections and recurrent parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: impl Into<QatLinear>,
        k_proj: impl Into<QatLinear>,
        v_proj: impl Into<QatLinear>,
        beta_proj: impl Into<QatLinear>,
        alpha_proj: impl Into<QatLinear>,
        gate_proj: impl Into<QatLinear>,
        out_proj: impl Into<QatLinear>,
        q_conv: Tensor,
        k_conv: Tensor,
        v_conv: Tensor,
        out_norm: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        config: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            q_proj: q_proj.into(),
            k_proj: k_proj.into(),
            v_proj: v_proj.into(),
            beta_proj: beta_proj.into(),
            alpha_proj: alpha_proj.into(),
            gate_proj: gate_proj.into(),
            out_proj: out_proj.into(),
            q_conv,
            k_conv,
            v_conv,
            out_norm,
            a_log,
            dt_bias,
            config,
        }
    }

    /// Run a full sequence without retaining recurrent state.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_form(
            x,
            None,
            DeltaNetForm::ChunkParallel {
                chunk_size: self.config.chunk_size,
            },
        )
    }

    /// Run the differentiable training path.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_form(
            x,
            None,
            DeltaNetForm::ChunkParallel {
                chunk_size: self.config.chunk_size,
            },
        )
    }

    /// Run incremental inference and update fixed-size recurrent state.
    pub fn forward_cached(&self, x: &Tensor, state: &mut DeltaNetState) -> Result<Tensor> {
        self.forward_form(x, Some(state), DeltaNetForm::Sequential)
    }

    /// Run independently cached rows from one decode batch.
    pub fn forward_decode_batch(
        &self,
        x: &Tensor,
        states: &mut [&mut DeltaNetState],
    ) -> Result<Tensor> {
        if x.rank() != 3 || x.dim(1)? != 1 || x.dim(0)? != states.len() {
            return Err(candle_core::Error::msg(format!(
                "Gated DeltaNet batched decode expects [batch, 1, hidden] and one state per row, got {:?} and {} states",
                x.dims(),
                states.len()
            )));
        }
        let mut rows = Vec::with_capacity(states.len());
        for (row, state) in states.iter_mut().enumerate() {
            rows.push(self.forward_cached(&x.narrow(0, row, 1)?, state)?);
        }
        Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)
    }

    /// Run the layer while recording inputs to quantizable projections.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let prefix = format!("blocks.{layer_idx}.deltanet");
        for name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "beta_proj",
            "alpha_proj",
            "gate_proj",
        ] {
            capture.insert(format!("{prefix}.{name}.weight"), x.clone());
        }
        let projected = self.forward_projected(x, None, DeltaNetForm::Sequential)?;
        capture.insert(format!("{prefix}.out_proj.weight"), projected.clone());
        self.out_proj.forward(&projected)
    }

    fn forward_form(
        &self,
        x: &Tensor,
        state: Option<&mut DeltaNetState>,
        form: DeltaNetForm,
    ) -> Result<Tensor> {
        let projected = self.forward_projected(x, state, form)?;
        self.out_proj.forward(&projected)
    }

    fn forward_projected(
        &self,
        x: &Tensor,
        state: Option<&mut DeltaNetState>,
        form: DeltaNetForm,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let h = self.config.n_heads;
        let dk = self.config.key_head_dim;
        let dv = self.config.value_head_dim;
        let input_dtype = x.dtype();

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;
        let use_optimized_recurrence = state.is_some() && matches!(form, DeltaNetForm::Sequential);
        let chunk_size = match form {
            DeltaNetForm::Sequential => 1,
            DeltaNetForm::ChunkParallel { chunk_size } => chunk_size,
        };
        let mut owned_state = DeltaNetState::new();
        let state = state.unwrap_or(&mut owned_state);
        let q = causal_depthwise_conv(&q_raw, &self.q_conv, &mut state.q_history)?;
        let k = causal_depthwise_conv(&k_raw, &self.k_conv, &mut state.k_history)?;
        let v = causal_depthwise_conv(&v_raw, &self.v_conv, &mut state.v_history)?;
        let q = l2_normalize(&candle_nn::ops::silu(&q)?.reshape((batch, seq_len, h, dk))?)?
            .to_dtype(DType::F32)?;
        let k = l2_normalize(&candle_nn::ops::silu(&k)?.reshape((batch, seq_len, h, dk))?)?
            .to_dtype(DType::F32)?;
        let v = candle_nn::ops::silu(&v)?
            .reshape((batch, seq_len, h, dv))?
            .to_dtype(DType::F32)?;
        let beta = candle_nn::ops::sigmoid(&self.beta_proj.forward(x)?.to_dtype(DType::F32)?)?;
        let alpha_input = self.alpha_proj.forward(x)?.to_dtype(DType::F32)?;
        let alpha_input = alpha_input.broadcast_add(&self.dt_bias)?;
        let alpha = softplus(&alpha_input)?
            .broadcast_mul(&self.a_log.exp()?)?
            .neg()?
            .exp()?;
        let gate = self
            .gate_proj
            .forward(x)?
            .reshape((batch, seq_len, h, dv))?
            .to_dtype(DType::F32)?;

        let mut recurrent = match &state.recurrent {
            Some(recurrent) => recurrent.clone(),
            None => Tensor::zeros((batch, h, dk, dv), DType::F32, x.device())?,
        };
        if recurrent.dims() != [batch, h, dk, dv] {
            return Err(candle_core::Error::msg(format!(
                "Gated DeltaNet state shape {:?} does not match [{batch}, {h}, {dk}, {dv}]",
                recurrent.dims()
            )));
        }

        let norm_weight = self.out_norm.to_dtype(DType::F32)?;
        let mut outputs = Vec::with_capacity(seq_len);
        for chunk_start in (0..seq_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            for token_idx in chunk_start..chunk_end {
                let q_t = q.narrow(1, token_idx, 1)?.squeeze(1)?;
                let k_t = k.narrow(1, token_idx, 1)?.squeeze(1)?;
                let v_t = v.narrow(1, token_idx, 1)?.squeeze(1)?;
                let alpha_head = alpha.narrow(1, token_idx, 1)?.squeeze(1)?;
                let beta_head = beta.narrow(1, token_idx, 1)?.squeeze(1)?;

                let optimized = if use_optimized_recurrence {
                    let packed = Tensor::cat(
                        &[
                            &q_t,
                            &k_t,
                            &v_t,
                            &alpha_head.unsqueeze(2)?,
                            &beta_head.unsqueeze(2)?,
                        ],
                        2,
                    )?;
                    aarambh_ai_kernel::gated_delta_recurrent(&packed, &recurrent, dk, dv).ok()
                } else {
                    None
                };
                let output = match optimized {
                    Some(result) => {
                        recurrent = result
                            .narrow(2, 0, dk * dv)?
                            .contiguous()?
                            .reshape((batch, h, dk, dv))?;
                        result.narrow(2, dk * dv, dv)?
                    }
                    None => {
                        let alpha_t = alpha_head.unsqueeze(2)?.unsqueeze(3)?;
                        let beta_t = beta_head.unsqueeze(2)?.unsqueeze(3)?;
                        let decayed = recurrent.broadcast_mul(&alpha_t)?;
                        let predicted = k_t.unsqueeze(2)?.matmul(&decayed)?.squeeze(2)?;
                        let error = (v_t - predicted)?;
                        let update = k_t.unsqueeze(3)?.matmul(&error.unsqueeze(2)?)?;
                        recurrent = (decayed + update.broadcast_mul(&beta_t)?)?;
                        q_t.unsqueeze(2)?.matmul(&recurrent)?.squeeze(2)?
                    }
                };
                let variance = (output.sqr()?.sum_keepdim(2)? / dv as f64)?;
                let output = output.broadcast_div(&(variance + NORM_EPS)?.sqrt()?)?;
                let output = output.broadcast_mul(&norm_weight)?;
                let gate_t = gate.narrow(1, token_idx, 1)?.squeeze(1)?;
                let output = (output * candle_nn::ops::silu(&gate_t)?)?;
                outputs.push(output.unsqueeze(1)?);
            }
        }
        state.recurrent = Some(recurrent);
        state.len += seq_len;
        Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch, seq_len, h * dv))?
            .to_dtype(input_dtype)
    }

    /// Return this layer's resolved configuration.
    pub fn config(&self) -> &GatedDeltaNetConfig {
        &self.config
    }

    /// Return every named parameter owned by this layer.
    pub fn named_tensors(&self) -> [(&'static str, &Tensor); 13] {
        [
            ("q_proj.weight", self.q_proj.weight()),
            ("k_proj.weight", self.k_proj.weight()),
            ("v_proj.weight", self.v_proj.weight()),
            ("beta_proj.weight", self.beta_proj.weight()),
            ("alpha_proj.weight", self.alpha_proj.weight()),
            ("gate_proj.weight", self.gate_proj.weight()),
            ("out_proj.weight", self.out_proj.weight()),
            ("q_conv.weight", &self.q_conv),
            ("k_conv.weight", &self.k_conv),
            ("v_conv.weight", &self.v_conv),
            ("out_norm.weight", &self.out_norm),
            ("A_log", &self.a_log),
            ("dt_bias", &self.dt_bias),
        ]
    }

    /// Return one parameter by its layer-local name.
    pub fn get_weight(&self, name: &str) -> Option<&Tensor> {
        self.named_tensors()
            .into_iter()
            .find_map(|(candidate, tensor)| (candidate == name).then_some(tensor))
    }
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(3)? + NORM_EPS)?.sqrt()?;
    x.broadcast_div(&norm)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let correction = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    x.relu()? + correction
}

fn causal_depthwise_conv(
    x: &Tensor,
    weight: &Tensor,
    history: &mut Option<Tensor>,
) -> Result<Tensor> {
    let (batch, seq_len, channels) = x.dims3()?;
    let (weight_channels, kernel) = weight.dims2()?;
    if channels != weight_channels {
        return Err(candle_core::Error::msg(format!(
            "depthwise convolution channels {channels} do not match weight channels {weight_channels}"
        )));
    }
    let history_len = kernel - 1;
    let prior = match history.as_ref() {
        Some(prior) => prior.clone(),
        None => Tensor::zeros((batch, history_len, channels), x.dtype(), x.device())?,
    };
    if prior.dims() != [batch, history_len, channels] {
        return Err(candle_core::Error::msg(format!(
            "depthwise convolution history shape {:?} does not match [{batch}, {history_len}, {channels}]",
            prior.dims()
        )));
    }
    let joined = Tensor::cat(&[&prior, x], 1)?;
    let mut outputs = Vec::with_capacity(seq_len);
    for token_idx in 0..seq_len {
        let mut sum: Option<Tensor> = None;
        for tap in 0..kernel {
            let source_idx = history_len + token_idx - tap;
            let source = joined.narrow(1, source_idx, 1)?;
            let coefficient = weight.narrow(1, tap, 1)?.transpose(0, 1)?.unsqueeze(0)?;
            let term = source.broadcast_mul(&coefficient)?;
            sum = Some(match sum {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        outputs.push(sum.expect("convolution kernel is non-empty"));
    }
    *history = Some(joined.narrow(1, seq_len, history_len)?);
    Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)
}
