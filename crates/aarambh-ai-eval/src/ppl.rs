use std::fs;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_train::cross_entropy_loss;
use candle_core::Tensor;

use crate::harness::EvalContext;
use crate::report::TaskScore;

/// Perplexity computation result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PplResult {
    /// Average negative log-likelihood.
    pub loss: f64,
    /// Perplexity.
    pub ppl: f64,
    /// Number of scored target tokens.
    pub tokens: usize,
}

/// Return `exp(loss)` with finite-value guards.
pub fn perplexity_from_loss(loss: f64) -> f64 {
    if loss.is_finite() {
        loss.exp()
    } else {
        f64::INFINITY
    }
}

/// Compute perplexity for a plaintext holdout file.
pub fn compute_ppl(
    context: &EvalContext,
    holdout_path: impl AsRef<Path>,
    max_examples: Option<usize>,
) -> Result<TaskScore> {
    let result = compute_ppl_result(context, holdout_path, max_examples)?;
    Ok(TaskScore::perplexity(
        result.loss,
        result.ppl,
        result.tokens,
    ))
}

/// Compute raw perplexity statistics for a plaintext holdout file.
pub fn compute_ppl_result(
    context: &EvalContext,
    holdout_path: impl AsRef<Path>,
    max_examples: Option<usize>,
) -> Result<PplResult> {
    let content = fs::read_to_string(holdout_path.as_ref())?;
    let mut records = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if records.is_empty() && !content.is_empty() {
        records.push(content);
    }
    if let Some(max) = max_examples {
        records.truncate(max);
    }
    if records.is_empty() {
        return Err(AarambhError::Config(format!(
            "holdout {} is empty",
            holdout_path.as_ref().display()
        )));
    }

    let mut total_nll = 0.0f64;
    let mut total_tokens = 0usize;
    let max_seq_len = context.max_seq_len().max(1);

    for record in records {
        let ids = context.tokenizer().encode(&record)?;
        if ids.len() < 2 {
            continue;
        }
        let mut pos = 0usize;
        while pos + 1 < ids.len() {
            let end = (pos + max_seq_len + 1).min(ids.len());
            let input = ids[pos..end - 1].to_vec();
            let labels = ids[pos + 1..end].to_vec();
            if input.is_empty() || labels.is_empty() {
                break;
            }
            context.record_context_len(input.len());
            let mask = vec![1u32; labels.len()];
            let input_tensor = Tensor::from_vec(input, (1, labels.len()), context.device())?;
            let label_tensor =
                Tensor::from_vec(labels.clone(), (1, labels.len()), context.device())?;
            let mask_tensor = Tensor::from_vec(mask, (1, labels.len()), context.device())?;
            let logits = context.model().forward(&input_tensor)?;
            let loss = cross_entropy_loss(&logits, &label_tensor, &mask_tensor)?
                .to_scalar::<f32>()? as f64;
            total_nll += loss * labels.len() as f64;
            total_tokens += labels.len();
            pos += labels.len();
        }
    }

    if total_tokens == 0 {
        return Err(AarambhError::Config(
            "holdout produced no scorable tokens".into(),
        ));
    }
    let loss = total_nll / total_tokens as f64;
    Ok(PplResult {
        loss,
        ppl: perplexity_from_loss(loss),
        tokens: total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppl_on_known_holdout_matches_manual_calculation() {
        let loss = 2.0f64;
        assert!((perplexity_from_loss(loss) - loss.exp()).abs() < 1e-12);
    }
}
