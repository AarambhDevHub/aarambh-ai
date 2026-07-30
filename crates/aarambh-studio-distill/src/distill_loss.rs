use aarambh_studio_core::{AarambhError, Result};
use candle_core::{DType, Tensor};

/// Forward-KL loss and diagnostic value for one replay batch.
#[derive(Debug)]
pub struct SoftKlLossOutput {
    /// Differentiable mean forward-KL loss.
    pub loss: Tensor,
    /// Number of completion positions included in the loss.
    pub token_count: usize,
}

/// Reward-weighted policy loss and normalized advantages.
#[derive(Debug)]
pub struct RewardLossOutput {
    /// Differentiable reward policy loss.
    pub loss: Tensor,
    /// Group-normalized and clipped advantage for every rollout.
    pub advantages: Vec<f32>,
}

/// Compute teacher-to-student forward KL on packed completion logits.
pub fn soft_kl_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f64,
) -> Result<SoftKlLossOutput> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(AarambhError::Config(
            "soft-KL temperature must be finite and greater than zero".into(),
        ));
    }
    let (tokens, vocab) = student_logits.dims2()?;
    if tokens == 0 || teacher_logits.dims() != [tokens, vocab] {
        return Err(AarambhError::Shape(format!(
            "teacher logits {:?} must match non-empty student logits {:?}",
            teacher_logits.dims(),
            student_logits.dims()
        )));
    }
    let student = student_logits
        .to_dtype(DType::F32)?
        .affine(1.0 / temperature, 0.0)?;
    let teacher = teacher_logits
        .detach()
        .to_dtype(DType::F32)?
        .affine(1.0 / temperature, 0.0)?;
    let student_log_probs = candle_nn::ops::log_softmax(&student, 1)?;
    let teacher_log_probs = candle_nn::ops::log_softmax(&teacher, 1)?;
    let teacher_probs = candle_nn::ops::softmax(&teacher, 1)?;
    let token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)?)?.sum(1)?;
    let loss = token_kl
        .sum_all()?
        .affine(temperature * temperature / tokens as f64, 0.0)?;
    Ok(SoftKlLossOutput {
        loss,
        token_count: tokens,
    })
}

/// Normalize scalar scores independently within each prompt rollout group.
pub fn group_normalized_advantages(
    scores: &[f32],
    group_size: usize,
    clip: f64,
) -> Result<Vec<f32>> {
    if group_size < 2 || scores.is_empty() || !scores.len().is_multiple_of(group_size) {
        return Err(AarambhError::Config(
            "reward scores must contain complete groups of at least two rollouts".into(),
        ));
    }
    if clip <= 0.0 || !clip.is_finite() || scores.iter().any(|score| !score.is_finite()) {
        return Err(AarambhError::Config(
            "reward scores and advantage clip must be finite".into(),
        ));
    }
    let clip = clip as f32;
    let mut advantages = Vec::with_capacity(scores.len());
    for group in scores.chunks(group_size) {
        let mean = group.iter().sum::<f32>() / group.len() as f32;
        let variance = group
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f32>()
            / group.len() as f32;
        let std = variance.sqrt();
        if std <= 1e-6 {
            advantages.extend(std::iter::repeat_n(0.0, group.len()));
        } else {
            advantages.extend(
                group
                    .iter()
                    .map(|score| ((score - mean) / (std + 1e-6)).clamp(-clip, clip)),
            );
        }
    }
    Ok(advantages)
}

/// Compute an advantage-weighted policy loss from packed student logits.
pub fn reward_policy_loss(
    packed_student_logits: &Tensor,
    packed_labels: &Tensor,
    completion_counts: &[usize],
    scores: &[f32],
    group_size: usize,
    advantage_clip: f64,
) -> Result<RewardLossOutput> {
    let (tokens, _vocab) = packed_student_logits.dims2()?;
    if tokens == 0
        || packed_labels.dims() != [tokens]
        || completion_counts.len() != scores.len()
        || completion_counts.iter().sum::<usize>() != tokens
    {
        return Err(AarambhError::Shape(
            "reward policy tensors, completion counts, and scores are inconsistent".into(),
        ));
    }
    let advantages = group_normalized_advantages(scores, group_size, advantage_clip)?;
    let selected = candle_nn::ops::log_softmax(&packed_student_logits.to_dtype(DType::F32)?, 1)?
        .gather(&packed_labels.reshape((tokens, 1))?, 1)?
        .reshape(tokens)?;
    let mut offset = 0usize;
    let mut weighted = Vec::with_capacity(completion_counts.len());
    for (&count, &advantage) in completion_counts.iter().zip(&advantages) {
        let mean_log_prob = selected
            .narrow(0, offset, count)?
            .sum_all()?
            .affine(1.0 / count as f64, 0.0)?;
        weighted.push(mean_log_prob.affine(-(advantage as f64), 0.0)?);
        offset += count;
    }
    let loss = Tensor::stack(&weighted.iter().collect::<Vec<_>>(), 0)?
        .sum_all()?
        .affine(1.0 / weighted.len() as f64, 0.0)?;
    Ok(RewardLossOutput { loss, advantages })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn identical_distributions_have_near_zero_soft_kl() {
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0], (2, 3), &Device::Cpu).unwrap();
        let loss = soft_kl_loss(&logits, &logits, 1.0)
            .unwrap()
            .loss
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.abs() < 1e-6, "loss={loss}");
    }

    #[test]
    fn soft_kl_gradient_flows_only_through_student() {
        let device = Device::Cpu;
        let student = Var::from_tensor(
            &Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2), &device).unwrap(),
        )
        .unwrap();
        let teacher = Var::from_tensor(
            &Tensor::from_vec(vec![0.4f32, 0.3, 0.2, 0.1], (2, 2), &device).unwrap(),
        )
        .unwrap();
        let grads = soft_kl_loss(student.as_tensor(), teacher.as_tensor(), 1.0)
            .unwrap()
            .loss
            .backward()
            .unwrap();
        assert!(grads.get(student.as_tensor()).is_some());
        assert!(grads.get(teacher.as_tensor()).is_none());
    }

    #[test]
    fn distillation_updates_reduce_teacher_student_kl_proxy() {
        let device = Device::Cpu;
        let student =
            Var::from_tensor(&Tensor::zeros((2, 3), DType::F32, &device).unwrap()).unwrap();
        let teacher =
            Tensor::from_vec(vec![4.0f32, 0.0, -4.0, -4.0, 0.0, 4.0], (2, 3), &device).unwrap();
        let before = soft_kl_loss(student.as_tensor(), &teacher, 1.0)
            .unwrap()
            .loss
            .to_scalar::<f32>()
            .unwrap();
        for _ in 0..8 {
            let loss = soft_kl_loss(student.as_tensor(), &teacher, 1.0)
                .unwrap()
                .loss;
            let grads = loss.backward().unwrap();
            let gradient = grads.get(student.as_tensor()).unwrap();
            let next = (student.as_tensor() - gradient.affine(0.5, 0.0).unwrap()).unwrap();
            student.set(&next.detach()).unwrap();
        }
        let after = soft_kl_loss(student.as_tensor(), &teacher, 1.0)
            .unwrap()
            .loss
            .to_scalar::<f32>()
            .unwrap();
        assert!(after < before, "before={before} after={after}");
    }

    #[test]
    fn reward_advantages_are_group_local_centered_and_clipped() {
        let advantages = group_normalized_advantages(&[1.0, 3.0, 10.0, 10.0], 2, 0.5).unwrap();
        assert_eq!(advantages.len(), 4);
        assert!((advantages[0] + 0.5).abs() < 1e-6);
        assert!((advantages[1] - 0.5).abs() < 1e-6);
        assert_eq!(&advantages[2..], &[0.0, 0.0]);
    }

    #[test]
    fn reward_policy_loss_prefers_the_higher_scored_rollout() {
        let device = Device::Cpu;
        let logits = Var::from_tensor(
            &Tensor::from_vec(vec![3.0f32, 0.0, 0.0, 3.0], (2, 2), &device).unwrap(),
        )
        .unwrap();
        let labels = Tensor::from_vec(vec![0u32, 1], 2, &device).unwrap();
        let output =
            reward_policy_loss(logits.as_tensor(), &labels, &[1, 1], &[0.0, 1.0], 2, 5.0).unwrap();
        assert!(output.loss.to_scalar::<f32>().unwrap().abs() < 1e-5);
        assert!(
            output
                .loss
                .backward()
                .unwrap()
                .get(logits.as_tensor())
                .is_some()
        );
    }
}
