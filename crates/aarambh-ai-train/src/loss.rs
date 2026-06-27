use aarambh_ai_core::{AarambhError, Result};
use candle_core::{DType, Tensor};

pub fn cross_entropy_loss(
    logits: &Tensor,
    labels: &Tensor,
    padding_mask: &Tensor,
) -> Result<Tensor> {
    let logits_dims = logits.dims();
    if logits_dims.len() != 3 {
        return Err(AarambhError::Shape(format!(
            "logits must have shape [batch, seq, vocab], got {logits_dims:?}"
        )));
    }
    let (batch, seq_len, vocab) = (logits_dims[0], logits_dims[1], logits_dims[2]);
    if labels.dims() != [batch, seq_len] {
        return Err(AarambhError::Shape(format!(
            "labels must have shape [{batch}, {seq_len}], got {:?}",
            labels.dims()
        )));
    }
    if padding_mask.dims() != [batch, seq_len] {
        return Err(AarambhError::Shape(format!(
            "padding_mask must have shape [{batch}, {seq_len}], got {:?}",
            padding_mask.dims()
        )));
    }

    let tokens = batch * seq_len;
    let flat_logits = logits.reshape((tokens, vocab))?;
    let flat_labels = labels.reshape((tokens,))?;
    let flat_mask = padding_mask.reshape((tokens,))?.to_dtype(DType::F32)?;
    let denom = flat_mask.sum_all()?;
    let denom_value = denom.to_scalar::<f32>()?;
    if denom_value <= 0.0 {
        return Err(AarambhError::Shape(
            "padding_mask must contain at least one non-padding token".into(),
        ));
    }

    let log_probs = candle_nn::ops::log_softmax(&flat_logits, 1)?;
    let picked = log_probs.gather(&flat_labels.unsqueeze(1)?, 1)?;
    let nll = picked.reshape((tokens,))?.affine(-1.0, 0.0)?;
    let masked = nll.broadcast_mul(&flat_mask)?;
    Ok(masked.sum_all()?.broadcast_div(&denom)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn cross_entropy_is_low_for_confident_correct_logits() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![10f32, -10., -10., -10., 10., -10.], (1, 2, 3), &device).unwrap();
        let labels = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u32, 1], (1, 2), &device).unwrap();

        let loss = cross_entropy_loss(&logits, &labels, &mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss < 1e-3, "loss was {loss}");
    }

    #[test]
    fn padding_mask_excludes_positions() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![10f32, -10., -10., 10., -10., -10.], (1, 2, 3), &device).unwrap();
        let labels = Tensor::from_vec(vec![0u32, 2], (1, 2), &device).unwrap();
        let full_mask = Tensor::from_vec(vec![1u32, 1], (1, 2), &device).unwrap();
        let partial_mask = Tensor::from_vec(vec![1u32, 0], (1, 2), &device).unwrap();

        let full = cross_entropy_loss(&logits, &labels, &full_mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let partial = cross_entropy_loss(&logits, &labels, &partial_mask)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(full > 5.0, "full loss was {full}");
        assert!(partial < 1e-3, "partial loss was {partial}");
    }
}
