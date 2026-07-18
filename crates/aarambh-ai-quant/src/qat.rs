use std::collections::HashSet;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use aarambh_ai_core::{AarambhError, QatConfig, QatTarget, QuantBits, QuantGranularity, Result};
use candle_core::{DType, Error, Tensor, TensorId};
use candle_nn::Linear;

const Q4_K_M_BLOCK_SIZE: usize = 256;

/// Device-native fake quantizer used by quantization-aware training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FakeQuantize {
    bits: QuantBits,
    granularity: QuantGranularity,
}

impl FakeQuantize {
    /// Create a fake quantizer for a bit width and scaling granularity.
    pub const fn new(bits: QuantBits, granularity: QuantGranularity) -> Self {
        Self { bits, granularity }
    }

    /// Return the configured integer bit width.
    pub const fn bits(&self) -> QuantBits {
        self.bits
    }

    /// Return the configured scaling granularity.
    pub const fn granularity(&self) -> QuantGranularity {
        self.granularity
    }

    /// Quantize then dequantize a tensor without copying it to the host.
    ///
    /// The returned tensor is detached from autograd. Use [`Self::forward`] when
    /// the fake-quantized value participates in training.
    pub fn simulate(&self, weight: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = weight.dtype();
        let weight = weight.to_dtype(DType::F32)?.detach();
        let quantized = match (self.granularity, self.bits) {
            (QuantGranularity::ExportAligned, QuantBits::Int4) => {
                fake_quant_export_q4_k_m(&weight)?
            }
            (QuantGranularity::ExportAligned, QuantBits::Int8) => {
                fake_quant_symmetric(&weight, 127.0, ScaleAxes::Tensor)?
            }
            (QuantGranularity::PerTensor, QuantBits::Int4) => {
                fake_quant_affine(&weight, 15.0, ScaleAxes::Tensor, false)?
            }
            (QuantGranularity::PerTensor, QuantBits::Int8) => {
                fake_quant_symmetric(&weight, 127.0, ScaleAxes::Tensor)?
            }
            (QuantGranularity::PerOutputChannel, QuantBits::Int4) => {
                fake_quant_affine(&weight, 15.0, ScaleAxes::OutputChannel, false)?
            }
            (QuantGranularity::PerOutputChannel, QuantBits::Int8) => {
                fake_quant_symmetric(&weight, 127.0, ScaleAxes::OutputChannel)?
            }
        };
        quantized.to_dtype(dtype)
    }

    /// Apply fake quantization with an identity straight-through gradient.
    pub fn forward(&self, weight: &Tensor) -> candle_core::Result<Tensor> {
        let quantized = self.simulate(weight)?;
        let correction = (&quantized - &weight.detach())?.detach();
        weight + correction
    }
}

#[derive(Debug, Clone, Copy)]
enum ScaleAxes {
    Tensor,
    OutputChannel,
}

fn fake_quant_export_q4_k_m(weight: &Tensor) -> candle_core::Result<Tensor> {
    let original_shape = weight.dims().to_vec();
    let flat = weight.flatten_all()?;
    let values = flat.elem_count();
    let padded_values = values.div_ceil(Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_SIZE;
    let padded = if padded_values == values {
        flat
    } else {
        let tail = Tensor::zeros(padded_values - values, DType::F32, weight.device())?;
        Tensor::cat(&[&flat, &tail], 0)?
    };
    let blocks = padded.reshape((padded_values / Q4_K_M_BLOCK_SIZE, Q4_K_M_BLOCK_SIZE))?;
    let min = blocks.min_keepdim(1)?;
    let max = blocks.max_keepdim(1)?;
    let scale = safe_scale(&(&max - &min)?, 15.0)?;
    let codes = blocks
        .broadcast_sub(&min)?
        .broadcast_div(&scale)?
        .round()?
        .clamp(0.0, 15.0)?;

    // GGUF stores both values as f16, but derives the 4-bit codes using the
    // original f32 range. Reproduce that order exactly for export parity.
    let stored_scale = scale.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
    let stored_min = min.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
    let dequantized = codes
        .broadcast_mul(&stored_scale)?
        .broadcast_add(&stored_min)?
        .flatten_all()?
        .narrow(0, 0, values)?;
    dequantized.reshape(original_shape)
}

fn fake_quant_affine(
    weight: &Tensor,
    qmax: f64,
    axes: ScaleAxes,
    store_params_as_f16: bool,
) -> candle_core::Result<Tensor> {
    ensure_matrix_for_channel_scales(weight, axes)?;
    let (min, max) = min_max(weight, axes)?;
    let scale = safe_scale(&(&max - &min)?, qmax)?;
    let codes = weight
        .broadcast_sub(&min)?
        .broadcast_div(&scale)?
        .round()?
        .clamp(0.0, qmax)?;
    let (scale, min) = if store_params_as_f16 {
        (
            scale.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
            min.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
        )
    } else {
        (scale, min)
    };
    codes.broadcast_mul(&scale)?.broadcast_add(&min)
}

fn fake_quant_symmetric(
    weight: &Tensor,
    qmax: f64,
    axes: ScaleAxes,
) -> candle_core::Result<Tensor> {
    ensure_matrix_for_channel_scales(weight, axes)?;
    let max_abs = match axes {
        ScaleAxes::Tensor => weight.abs()?.max_all()?,
        ScaleAxes::OutputChannel => weight.abs()?.max_keepdim(1)?,
    };
    let scale = safe_scale(&max_abs, qmax)?;
    weight
        .broadcast_div(&scale)?
        .round()?
        .clamp(-qmax, qmax)?
        .broadcast_mul(&scale)
}

fn min_max(weight: &Tensor, axes: ScaleAxes) -> candle_core::Result<(Tensor, Tensor)> {
    match axes {
        ScaleAxes::Tensor => Ok((weight.min_all()?, weight.max_all()?)),
        ScaleAxes::OutputChannel => Ok((weight.min_keepdim(1)?, weight.max_keepdim(1)?)),
    }
}

fn safe_scale(range: &Tensor, qmax: f64) -> candle_core::Result<Tensor> {
    let scale = (range / qmax)?;
    let ones = scale.ones_like()?;
    range.le(f32::EPSILON)?.where_cond(&ones, &scale)
}

fn ensure_matrix_for_channel_scales(weight: &Tensor, axes: ScaleAxes) -> candle_core::Result<()> {
    if matches!(axes, ScaleAxes::OutputChannel) && weight.rank() != 2 {
        return Err(Error::Msg(format!(
            "per-output-channel fake quantization requires rank-2 weights, got {:?}",
            weight.dims()
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct QatRuntime {
    config: QatConfig,
    generation: AtomicU64,
    wrapped_tensors: AtomicUsize,
    wrapped_parameters: AtomicUsize,
    cache_refreshes: AtomicUsize,
    registered_weights: Mutex<HashSet<TensorId>>,
}

/// Shared QAT runtime state for one training model.
#[derive(Debug, Clone)]
pub struct QatContext(Arc<QatRuntime>);

impl QatContext {
    /// Validate a QAT configuration and create its runtime context.
    pub fn new(config: QatConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self(Arc::new(QatRuntime {
            config,
            generation: AtomicU64::new(0),
            wrapped_tensors: AtomicUsize::new(0),
            wrapped_parameters: AtomicUsize::new(0),
            cache_refreshes: AtomicUsize::new(0),
            registered_weights: Mutex::new(HashSet::new()),
        })))
    }

    /// Return the immutable QAT configuration.
    pub fn config(&self) -> &QatConfig {
        &self.0.config
    }

    /// Return the current optimizer generation.
    pub fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }

    /// Invalidate cached fake-quantized weights after an optimizer update.
    pub fn advance_generation(&self) -> u64 {
        self.0.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Return model-wide QAT runtime counters.
    pub fn stats(&self) -> QatStats {
        QatStats {
            generation: self.generation(),
            wrapped_tensors: self.0.wrapped_tensors.load(Ordering::Relaxed),
            wrapped_parameters: self.0.wrapped_parameters.load(Ordering::Relaxed),
            cache_refreshes: self.0.cache_refreshes.load(Ordering::Relaxed),
        }
    }

    fn register(&self, weight: &Tensor) {
        let Ok(mut registered) = self.0.registered_weights.lock() else {
            return;
        };
        if !registered.insert(weight.id()) {
            return;
        }
        self.0.wrapped_tensors.fetch_add(1, Ordering::Relaxed);
        self.0
            .wrapped_parameters
            .fetch_add(weight.elem_count(), Ordering::Relaxed);
    }

    fn refreshed(&self) {
        self.0.cache_refreshes.fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot of QAT runtime and coverage counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QatStats {
    /// Optimizer generation used by the weight cache.
    pub generation: u64,
    /// Number of linear weight tensors covered by QAT.
    pub wrapped_tensors: usize,
    /// Number of scalar model parameters covered by QAT.
    pub wrapped_parameters: usize,
    /// Number of fake-quantized cache materializations.
    pub cache_refreshes: usize,
}

/// Linear projection with optional cached, straight-through fake quantization.
#[derive(Debug, Clone)]
pub struct QatLinear {
    inner: Linear,
    target: QatTarget,
    context: Option<QatContext>,
    cache: Arc<Mutex<Option<(u64, Tensor)>>>,
}

impl QatLinear {
    /// Wrap a linear projection for a QAT target.
    pub fn new(inner: Linear, target: QatTarget, context: Option<QatContext>) -> Self {
        let context = context.filter(|context| context.config().applies_to(target));
        if let Some(context) = &context {
            context.register(inner.weight());
        }
        Self {
            inner,
            target,
            context,
            cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Wrap a linear projection without enabling fake quantization.
    pub fn disabled(inner: Linear, target: QatTarget) -> Self {
        Self::new(inner, target, None)
    }

    /// Return the underlying trainable weight tensor.
    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }

    /// Return the optional bias tensor.
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }

    /// Return whether this projection is covered by active QAT.
    pub fn is_quantized(&self) -> bool {
        self.context.is_some()
    }

    /// Return the target category associated with this projection.
    pub const fn target(&self) -> QatTarget {
        self.target
    }

    /// Apply the linear projection using a generation-cached effective weight.
    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let weight = self.effective_weight()?;
        let output = match *input.dims() {
            [b1, b2, m, k] => {
                if input.is_contiguous() {
                    let weight = weight.t()?;
                    input
                        .reshape((b1 * b2 * m, k))?
                        .matmul(&weight)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let weight = weight.broadcast_left((b1, b2))?.t()?;
                    input.matmul(&weight)?
                }
            }
            [batch, m, k] => {
                if input.is_contiguous() {
                    let weight = weight.t()?;
                    input
                        .reshape((batch * m, k))?
                        .matmul(&weight)?
                        .reshape((batch, m, ()))?
                } else {
                    let weight = weight.broadcast_left(batch)?.t()?;
                    input.matmul(&weight)?
                }
            }
            _ => input.matmul(&weight.t()?)?,
        };
        match self.inner.bias() {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output),
        }
    }

    fn effective_weight(&self) -> candle_core::Result<Tensor> {
        let Some(context) = &self.context else {
            return Ok(self.inner.weight().clone());
        };
        let generation = context.generation();
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| Error::Msg("QAT weight cache lock poisoned".into()))?;
        if let Some((cached_generation, weight)) = cache.as_ref()
            && *cached_generation == generation
        {
            return Ok(weight.clone());
        }
        let quantizer = FakeQuantize::new(
            context.config().effective_bits(self.target),
            context.config().granularity,
        );
        let weight = quantizer.forward(self.inner.weight())?;
        *cache = Some((generation, weight.clone()));
        context.refreshed();
        Ok(weight)
    }
}

impl From<Linear> for QatLinear {
    fn from(inner: Linear) -> Self {
        Self::disabled(inner, QatTarget::Attention)
    }
}

impl candle_nn::Module for QatLinear {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        QatLinear::forward(self, input)
    }
}

#[derive(Debug, Clone, Copy)]
/// Compatibility fake-quantization node retained for existing callers.
pub struct FakeQuantNode {
    /// Number of quantization bits.
    pub bits: u8,
    /// Whether to use symmetric quantization around zero.
    pub symmetric: bool,
}

impl FakeQuantNode {
    /// Create a fake-quantization node after validating bit width.
    pub fn new(bits: u8, symmetric: bool) -> Result<Self> {
        validate_bits(bits)?;
        Ok(Self { bits, symmetric })
    }

    /// Apply device-native fake quantization to a tensor.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        validate_bits(self.bits)?;
        let qmax = if self.symmetric {
            ((1u32 << (self.bits - 1)) - 1) as f64
        } else {
            ((1u32 << self.bits) - 1) as f64
        };
        let input_f32 = input.to_dtype(DType::F32)?;
        let output = if self.symmetric {
            fake_quant_symmetric(&input_f32, qmax, ScaleAxes::Tensor)?
        } else {
            fake_quant_affine(&input_f32, qmax, ScaleAxes::Tensor, false)?
        };
        Ok(output.to_dtype(input.dtype())?)
    }
}

/// Apply asymmetric device-native fake quantization to a tensor.
pub fn fake_quantise(input: &Tensor, bits: u8) -> Result<Tensor> {
    FakeQuantNode::new(bits, false)?.forward(input)
}

fn validate_bits(bits: u8) -> Result<()> {
    if !(2..=8).contains(&bits) {
        return Err(AarambhError::Config(format!(
            "fake quant bits must be in 2..=8, got {bits}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    use crate::{dequantise_block_q4_k_m, quantise_absmax_i8, quantise_block_q4_k_m};

    fn values() -> Vec<f32> {
        (0..Q4_K_M_BLOCK_SIZE)
            .map(|index| (index as f32 * 0.117).sin() * 1.7 - 0.13)
            .collect()
    }

    #[test]
    fn export_aligned_q4_matches_gguf_block_encoding() {
        let values = values();
        let input = Tensor::from_vec(values.clone(), (16, 16), &Device::Cpu).unwrap();
        let actual = FakeQuantize::new(QuantBits::Int4, QuantGranularity::ExportAligned)
            .simulate(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = dequantise_block_q4_k_m(&quantise_block_q4_k_m(&values));
        assert_eq!(actual, expected);
    }

    #[test]
    fn export_aligned_q4_matches_padded_gguf_tail() {
        let values = values().into_iter().take(173).collect::<Vec<_>>();
        let input = Tensor::from_vec(values.clone(), (173,), &Device::Cpu).unwrap();
        let actual = FakeQuantize::new(QuantBits::Int4, QuantGranularity::ExportAligned)
            .simulate(&input)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut padded = values;
        padded.resize(Q4_K_M_BLOCK_SIZE, 0.0);
        let expected = dequantise_block_q4_k_m(&quantise_block_q4_k_m(&padded));
        assert_eq!(actual, expected[..173]);
    }

    #[test]
    fn export_aligned_q8_matches_absmax_export() {
        let input = Tensor::from_vec(values(), (16, 16), &Device::Cpu).unwrap();
        let actual = FakeQuantize::new(QuantBits::Int8, QuantGranularity::ExportAligned)
            .simulate(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let exported = quantise_absmax_i8(&input).unwrap();
        let expected = exported
            .data
            .iter()
            .map(|value| *value as f32 * exported.scale)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn straight_through_estimator_has_identity_gradient() {
        let weight = Var::from_vec(values(), (16, 16), &Device::Cpu).unwrap();
        let output = FakeQuantize::new(QuantBits::Int4, QuantGranularity::ExportAligned)
            .forward(&weight)
            .unwrap();
        let loss = output.sum_all().unwrap();
        let gradients = loss.backward().unwrap();
        let gradient = gradients.get(&weight).unwrap().to_vec2::<f32>().unwrap();
        assert!(gradient.iter().flatten().all(|value| *value == 1.0));
    }

    #[test]
    fn linear_cache_is_reused_until_generation_advances() {
        let weight = Var::from_vec(values(), (16, 16), &Device::Cpu).unwrap();
        let context = QatContext::new(QatConfig::default()).unwrap();
        let linear = QatLinear::new(
            Linear::new(weight.as_tensor().clone(), None),
            QatTarget::Attention,
            Some(context.clone()),
        );
        let input = Tensor::ones((2, 16), DType::F32, &Device::Cpu).unwrap();
        linear.forward(&input).unwrap();
        linear.forward(&input).unwrap();
        assert_eq!(context.stats().cache_refreshes, 1);
        context.advance_generation();
        linear.forward(&input).unwrap();
        assert_eq!(context.stats().cache_refreshes, 2);
    }
}
