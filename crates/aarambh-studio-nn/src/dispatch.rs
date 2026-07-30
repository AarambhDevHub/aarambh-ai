use candle_core::{D, Result, Tensor};

/// Combine per-expert outputs using dense masked expert weights.
pub fn dense_weighted_dispatch(
    expert_outputs: &[Tensor],
    dispatch_weights: &Tensor,
) -> Result<Tensor> {
    if expert_outputs.is_empty() {
        candle_core::bail!("dense_weighted_dispatch requires at least one expert output");
    }
    let first_dims = expert_outputs[0].dims().to_vec();
    if first_dims.len() != 3 {
        candle_core::bail!(
            "expert outputs must have shape [batch, seq, hidden], got {:?}",
            first_dims
        );
    }
    for output in expert_outputs.iter().skip(1) {
        if output.dims() != first_dims {
            candle_core::bail!(
                "all expert outputs must have the same shape, expected {:?}, got {:?}",
                first_dims,
                output.dims()
            );
        }
    }

    let expected_weights = [first_dims[0], first_dims[1], expert_outputs.len()];
    if dispatch_weights.dims() != expected_weights {
        candle_core::bail!(
            "dispatch weights must have shape {:?}, got {:?}",
            expected_weights,
            dispatch_weights.dims()
        );
    }

    let stacked = Tensor::stack(expert_outputs, 2)?;
    let weights = dispatch_weights
        .to_dtype(stacked.dtype())?
        .unsqueeze(D::Minus1)?;
    stacked.broadcast_mul(&weights)?.sum(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn dense_dispatch_weighted_sum_matches_shape() {
        let device = Device::Cpu;
        let a = Tensor::ones((1, 2, 4), DType::F32, &device).unwrap();
        let b = a.affine(2.0, 0.0).unwrap();
        let weights = Tensor::from_vec(vec![1.0f32, 0.0, 0.25, 0.75], (1, 2, 2), &device).unwrap();
        let out = dense_weighted_dispatch(&[a, b], &weights).unwrap();
        assert_eq!(out.dims(), &[1, 2, 4]);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values[..4].iter().all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(values[4..].iter().all(|value| (*value - 1.75).abs() < 1e-6));
    }
}
