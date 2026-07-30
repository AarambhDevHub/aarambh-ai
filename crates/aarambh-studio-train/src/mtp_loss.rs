use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_model::MtpPrediction;
use candle_core::Tensor;

use crate::loss::cross_entropy_loss;

#[derive(Debug)]
/// Scalar loss tensor for one auxiliary future-token offset.
pub struct MtpHeadLoss {
    /// Future-token offset represented by the loss.
    pub offset: usize,
    /// Masked cross-entropy loss for this offset.
    pub loss: Tensor,
}

#[derive(Debug)]
/// Main, auxiliary, and combined multi-token prediction losses.
pub struct MtpLossOutput {
    /// Unmodified next-token cross-entropy loss.
    pub main_loss: Tensor,
    /// Mean auxiliary-head loss before applying its configured weight.
    pub auxiliary_loss: Option<Tensor>,
    /// Individual auxiliary losses in increasing offset order.
    pub head_losses: Vec<MtpHeadLoss>,
    /// Main loss plus the weighted mean auxiliary loss.
    pub total_loss: Tensor,
}

/// Compute the target-aligned loss for one MTP prediction.
pub fn mtp_head_loss(
    prediction: MtpPrediction,
    labels: &Tensor,
    padding_mask: &Tensor,
) -> Result<MtpHeadLoss> {
    if prediction.offset < 2 {
        return Err(AarambhError::Config(
            "MTP auxiliary prediction offsets must be at least 2".into(),
        ));
    }
    let (batch, seq_len) = labels.dims2()?;
    if padding_mask.dims() != [batch, seq_len] {
        return Err(AarambhError::Shape(format!(
            "MTP padding mask must have shape [{batch}, {seq_len}], got {:?}",
            padding_mask.dims()
        )));
    }
    let shift = prediction.offset - 1;
    if shift >= seq_len {
        return Err(AarambhError::Shape(format!(
            "MTP offset {} has no valid target in sequence length {seq_len}",
            prediction.offset
        )));
    }
    let anchors = seq_len - shift;
    if prediction.logits.dims().len() != 3
        || prediction.logits.dim(0)? != batch
        || prediction.logits.dim(1)? != anchors
    {
        return Err(AarambhError::Shape(format!(
            "MTP offset {} logits must have leading shape [{batch}, {anchors}], got {:?}",
            prediction.offset,
            prediction.logits.dims()
        )));
    }
    let targets = labels.narrow(1, shift, anchors)?;
    let mask = padding_mask.narrow(1, shift, anchors)?;
    Ok(MtpHeadLoss {
        offset: prediction.offset,
        loss: cross_entropy_loss(&prediction.logits, &targets, &mask)?,
    })
}

/// Combine the main loss with the weighted mean of auxiliary head losses.
pub fn combine_mtp_losses(
    main_loss: Tensor,
    head_losses: Vec<MtpHeadLoss>,
    auxiliary_weight: f64,
) -> Result<MtpLossOutput> {
    if auxiliary_weight < 0.0 || !auxiliary_weight.is_finite() {
        return Err(AarambhError::Config(
            "MTP auxiliary loss weight must be finite and non-negative".into(),
        ));
    }
    let auxiliary_loss = if let Some(first) = head_losses.first() {
        let mut sum = first.loss.clone();
        for head in &head_losses[1..] {
            sum = (sum + &head.loss)?;
        }
        Some(sum.affine(1.0 / head_losses.len() as f64, 0.0)?)
    } else {
        None
    };
    let total_loss = match &auxiliary_loss {
        Some(auxiliary) if auxiliary_weight > 0.0 => {
            (&main_loss + &auxiliary.affine(auxiliary_weight, 0.0)?)?
        }
        _ => main_loss.clone(),
    };
    Ok(MtpLossOutput {
        main_loss,
        auxiliary_loss,
        head_losses,
        total_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn offset_two_uses_labels_starting_at_index_one() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                -10f32, 10., -10., -10., -10., -10., 10., -10., 10., -10., -10., -10.,
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap();
        let labels = Tensor::from_vec(vec![3u32, 1, 2, 0], (1, 4), &device).unwrap();
        let mask = Tensor::ones((1, 4), DType::U32, &device).unwrap();
        let loss = mtp_head_loss(MtpPrediction { offset: 2, logits }, &labels, &mask)
            .unwrap()
            .loss
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss < 1e-3, "loss was {loss}");
    }

    #[test]
    fn auxiliary_heads_are_averaged_before_weighting() {
        let device = Device::Cpu;
        let main = Tensor::new(2f32, &device).unwrap();
        let losses = vec![
            MtpHeadLoss {
                offset: 2,
                loss: Tensor::new(4f32, &device).unwrap(),
            },
            MtpHeadLoss {
                offset: 3,
                loss: Tensor::new(6f32, &device).unwrap(),
            },
        ];
        let output = combine_mtp_losses(main, losses, 0.3).unwrap();
        let total = output.total_loss.to_scalar::<f32>().unwrap();
        assert!((total - 3.5).abs() < 1e-6, "total was {total}");
    }

    #[test]
    fn disabled_mtp_returns_main_loss_exactly() {
        let device = Device::Cpu;
        let main = Tensor::new(2f32, &device).unwrap();
        let output = combine_mtp_losses(main.clone(), Vec::new(), 0.0).unwrap();
        assert!(output.auxiliary_loss.is_none());
        assert_eq!(
            output.total_loss.to_scalar::<f32>().unwrap(),
            main.to_scalar::<f32>().unwrap()
        );
    }
}
