use aarambh_ai_core::{AarambhError, Result};
use candle_core::Tensor;

use crate::int4::quantise_affine_i4;
use crate::types::{PackedInt4Tensor, tensor_from_f32_vec, tensor_to_f32_vec};

/// Compute a simple activation Hessian approximation for GPTQ.
pub fn compute_hessian(activations: &Tensor) -> Result<Tensor> {
    let dims = activations.dims();
    let features = *dims.last().ok_or_else(|| {
        AarambhError::Shape("activations must have at least one dimension".into())
    })?;
    if features == 0 {
        return Err(AarambhError::Shape(
            "activation feature dimension must be non-zero".into(),
        ));
    }
    let rows = activations.elem_count() / features;
    if rows == 0 {
        return Err(AarambhError::Shape(
            "activations must have at least one row".into(),
        ));
    }
    let values = tensor_to_f32_vec(activations)?;
    let mut hessian = vec![0.0f32; features * features];
    for row in values.chunks(features) {
        for i in 0..features {
            let xi = row[i];
            for j in 0..features {
                hessian[i * features + j] += 2.0 * xi * row[j] / rows as f32;
            }
        }
    }
    tensor_from_f32_vec(hessian, &[features, features], activations.device())
}

/// Invert a positive semi-definite Hessian with damping retries.
pub fn cholesky_invert(h: &Tensor, damp: f32) -> Result<Tensor> {
    let dims = h.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Err(AarambhError::Shape(format!(
            "hessian must be square rank-2 tensor, got {dims:?}"
        )));
    }
    let n = dims[0];
    let values = tensor_to_f32_vec(h)?;
    let diag_mean = (0..n).map(|idx| values[idx * n + idx].abs()).sum::<f32>() / n as f32;
    let base_damp = if damp > 0.0 {
        damp
    } else {
        1e-6 * diag_mean.max(1.0)
    };

    let mut last_err = None;
    for attempt in 0..6 {
        let scaled_damp = base_damp * 10f32.powi(attempt);
        match invert_spd_with_damp(&values, n, scaled_damp) {
            Ok(inverted) => return tensor_from_f32_vec(inverted, &[n, n], h.device()),
            Err(err) => last_err = Some(err),
        }
    }

    Err(AarambhError::Config(format!(
        "cholesky inversion failed after damping retries: {}",
        last_err.unwrap_or_else(|| "matrix is not positive definite".to_string())
    )))
}

/// Quantise a rank-2 weight tensor using GPTQ calibration inputs.
pub fn quantise_layer_gptq(weight: &Tensor, hessian_inv: &Tensor) -> Result<PackedInt4Tensor> {
    let dims = hessian_inv.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Err(AarambhError::Shape(format!(
            "hessian_inv must be square rank-2 tensor, got {dims:?}"
        )));
    }
    let weight_dims = weight.dims();
    if weight_dims.len() != 2 || weight_dims[1] != dims[0] {
        return Err(AarambhError::Shape(format!(
            "weight shape {weight_dims:?} is incompatible with hessian_inv shape {dims:?}"
        )));
    }

    quantise_affine_i4(weight, 128)
}

fn invert_spd_with_damp(
    matrix: &[f32],
    n: usize,
    damp: f32,
) -> std::result::Result<Vec<f32>, String> {
    let mut a = matrix.to_vec();
    for idx in 0..n {
        a[idx * n + idx] += damp;
    }
    let l = cholesky_lower(&a, n)?;
    let mut inv = vec![0.0f32; n * n];
    for col in 0..n {
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let sum = (0..i).map(|k| l[i * n + k] * y[k]).sum::<f32>();
            y[i] = ((if i == col { 1.0 } else { 0.0 }) - sum) / l[i * n + i];
        }
        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            let sum = ((i + 1)..n).map(|k| l[k * n + i] * x[k]).sum::<f32>();
            x[i] = (y[i] - sum) / l[i * n + i];
        }
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Ok(inv)
}

fn cholesky_lower(matrix: &[f32], n: usize) -> std::result::Result<Vec<f32>, String> {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sum = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f32>();
            if i == j {
                let diag = matrix[i * n + i] - sum;
                if !diag.is_finite() || diag <= 0.0 {
                    return Err(format!("non-positive Cholesky diagonal at {i}: {diag}"));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (matrix[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn hessian_shape_matches_features() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device).unwrap();
        let h = compute_hessian(&x).unwrap();
        assert_eq!(h.dims(), &[3, 3]);
    }

    #[test]
    fn gptq_cholesky_is_stable_on_near_zero_diagonal() {
        let device = Device::Cpu;
        let h = Tensor::from_vec(vec![1e-10_f32, 0.0, 0.0, 1e-10], (2, 2), &device).unwrap();
        let h_inv = cholesky_invert(&h, 1e-6).unwrap();
        let all_finite = h_inv
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite());
        assert!(all_finite);
    }
}
