use aarambh_ai_core::{AarambhError, ModelConfig, Result, RopeScalingConfig, RopeScalingMethod};

/// Return unscaled RoPE inverse frequencies for one attention head.
pub fn base_inverse_frequencies(head_dim: usize, theta: f64) -> Result<Vec<f32>> {
    validate_rope_dims(head_dim, theta)?;
    Ok((0..head_dim / 2)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect())
}

/// Return NTK-aware theta for a target context scaling factor.
pub fn ntk_aware_theta(theta: f64, factor: f64, head_dim: usize) -> Result<f64> {
    validate_rope_dims(head_dim, theta)?;
    if factor <= 1.0 || !factor.is_finite() {
        return Err(AarambhError::Config(
            "NTK RoPE scaling factor must be finite and greater than 1.0".into(),
        ));
    }
    Ok(theta * factor.powf(head_dim as f64 / (head_dim as f64 - 2.0)))
}

/// Return inverse frequencies and cos/sin attention scale for a model config.
pub fn inverse_frequencies_for_config(cfg: &ModelConfig) -> Result<(Vec<f32>, f32)> {
    let head_dim = cfg.head_dim();
    match &cfg.rope_scaling {
        Some(scaling) => {
            scaled_inverse_frequencies(head_dim, cfg.rope_theta, cfg.max_seq_len, scaling)
        }
        None => Ok((base_inverse_frequencies(head_dim, cfg.rope_theta)?, 1.0)),
    }
}

/// Return inverse frequencies and cos/sin attention scale for a scaling config.
pub fn scaled_inverse_frequencies(
    head_dim: usize,
    theta: f64,
    max_seq_len: usize,
    scaling: &RopeScalingConfig,
) -> Result<(Vec<f32>, f32)> {
    scaling.validate(max_seq_len, head_dim)?;
    let attn_factor = scaling.attn_factor as f32;
    match scaling.method {
        RopeScalingMethod::Linear => {
            let inv = base_inverse_frequencies(head_dim, theta)?
                .into_iter()
                .map(|freq| freq / scaling.factor as f32)
                .collect();
            Ok((inv, attn_factor))
        }
        RopeScalingMethod::Ntk => {
            let theta = ntk_aware_theta(theta, scaling.factor, head_dim)?;
            Ok((base_inverse_frequencies(head_dim, theta)?, attn_factor))
        }
        RopeScalingMethod::Yarn => Ok((yarn_frequencies(head_dim, theta, scaling)?, attn_factor)),
    }
}

/// Return YaRN-scaled inverse frequencies for one attention head.
pub fn yarn_frequencies(
    head_dim: usize,
    theta: f64,
    scaling: &RopeScalingConfig,
) -> Result<Vec<f32>> {
    validate_rope_dims(head_dim, theta)?;
    if !matches!(scaling.method, RopeScalingMethod::Yarn) {
        return Err(AarambhError::Config(
            "yarn_frequencies requires method = yarn".into(),
        ));
    }
    scaling.validate(scaling.original_max_seq_len, head_dim)?;

    let half_dim = head_dim / 2;
    let (low, high) = correction_range(
        scaling.beta_fast,
        scaling.beta_slow,
        head_dim,
        theta,
        scaling.original_max_seq_len,
    );

    let mut inv_freq = Vec::with_capacity(half_dim);
    for i in 0..half_dim {
        let pos_freq = theta.powf(2.0 * i as f64 / head_dim as f64);
        let extrapolated = 1.0 / pos_freq;
        let interpolated = extrapolated / scaling.factor;
        let ramp = linear_ramp(i as f64, low, high);
        let extrapolation_weight = 1.0 - ramp;
        let blended = interpolated * ramp + extrapolated * extrapolation_weight;
        inv_freq.push(blended as f32);
    }
    Ok(inv_freq)
}

fn validate_rope_dims(head_dim: usize, theta: f64) -> Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(AarambhError::Config(
            "RoPE head_dim must be non-zero and even".into(),
        ));
    }
    if theta <= 0.0 || !theta.is_finite() {
        return Err(AarambhError::Config(
            "RoPE theta must be finite and positive".into(),
        ));
    }
    Ok(())
}

fn correction_range(
    beta_fast: f64,
    beta_slow: f64,
    head_dim: usize,
    theta: f64,
    original_max_seq_len: usize,
) -> (f64, f64) {
    let low = correction_dim(beta_fast, head_dim, theta, original_max_seq_len).floor();
    let high = correction_dim(beta_slow, head_dim, theta, original_max_seq_len).ceil();
    let max_dim = (head_dim / 2).saturating_sub(1) as f64;
    (low.clamp(0.0, max_dim), high.clamp(0.0, max_dim))
}

fn correction_dim(rotations: f64, head_dim: usize, theta: f64, original_max_seq_len: usize) -> f64 {
    let numerator = head_dim as f64
        * ((original_max_seq_len as f64) / (rotations * 2.0 * std::f64::consts::PI)).ln();
    numerator / (2.0 * theta.ln())
}

fn linear_ramp(dim: f64, low: f64, high: f64) -> f64 {
    let high = if (high - low).abs() < f64::EPSILON {
        high + 0.001
    } else {
        high
    };
    ((dim - low) / (high - low)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntk_theta_scales_up() {
        let theta = ntk_aware_theta(10000.0, 4.0, 64).unwrap();
        assert!(theta > 10000.0);
    }

    #[test]
    fn yarn_frequencies_interpolate_correctly_at_boundary() {
        let scaling = RopeScalingConfig {
            method: RopeScalingMethod::Yarn,
            factor: 4.0,
            original_max_seq_len: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.138629436,
        };
        let base = base_inverse_frequencies(64, 500000.0).unwrap();
        let yarn = yarn_frequencies(64, 500000.0, &scaling).unwrap();

        assert!((base[0] - yarn[0]).abs() < 1e-6);
        let last = base.len() - 1;
        assert!((base[last] / scaling.factor as f32 - yarn[last]).abs() < 1e-8);
        for pair in yarn.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }
}
