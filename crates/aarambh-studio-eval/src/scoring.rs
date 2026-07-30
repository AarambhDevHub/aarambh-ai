use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use candle_core::{DType, Tensor};

use crate::harness::EvalContext;

/// Log-probability summary for one candidate continuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuationScore {
    /// Sum of token log-probabilities.
    pub total_logprob: f64,
    /// Mean token log-probability.
    pub mean_logprob: f64,
    /// Number of continuation tokens scored.
    pub token_count: usize,
}

/// Scores text continuations against a prompt.
pub trait ContinuationScorer {
    /// Return model log-probability for `continuation` after `prompt`.
    fn score_continuation(&self, prompt: &str, continuation: &str) -> Result<ContinuationScore>;
}

/// Model-backed continuation scorer using full forward passes.
pub struct ModelLogProbScorer<'a> {
    context: &'a EvalContext,
}

impl<'a> ModelLogProbScorer<'a> {
    /// Create a scorer over an evaluation context.
    pub fn new(context: &'a EvalContext) -> Self {
        Self { context }
    }
}

impl ContinuationScorer for ModelLogProbScorer<'_> {
    fn score_continuation(&self, prompt: &str, continuation: &str) -> Result<ContinuationScore> {
        let mut prompt_ids = self.context.tokenizer().encode(prompt)?;
        if prompt_ids.is_empty()
            && let Some(bos) = self.context.tokenizer().bos_token_id()
        {
            prompt_ids.push(bos);
        }
        let continuation_ids = self.context.tokenizer().encode(continuation)?;
        if continuation_ids.is_empty() {
            return Err(AarambhError::Tokenizer(
                "continuation produced no tokens".into(),
            ));
        }
        if prompt_ids.is_empty() {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }

        let mut input_ids = prompt_ids.clone();
        input_ids.extend_from_slice(&continuation_ids);
        if input_ids.len() > self.context.max_seq_len() {
            return Err(AarambhError::Shape(format!(
                "eval sequence length {} exceeds max_seq_len {}",
                input_ids.len(),
                self.context.max_seq_len()
            )));
        }
        self.context.record_context_len(input_ids.len());

        let input = Tensor::from_vec(
            input_ids,
            (1, prompt_ids.len() + continuation_ids.len()),
            self.context.device(),
        )?;
        let logits = self.context.model().forward(&input)?;
        let mut total = 0.0f64;

        for (idx, target) in continuation_ids.iter().enumerate() {
            let pos = prompt_ids.len() + idx - 1;
            let row = logits.narrow(1, pos, 1)?.squeeze(1)?.squeeze(0)?;
            let log_probs = candle_nn::ops::log_softmax(&row.to_dtype(DType::F32)?, 0)?;
            let values = log_probs.to_vec1::<f32>()?;
            let value = values.get(*target as usize).copied().ok_or_else(|| {
                AarambhError::Shape(format!("target token id {target} is outside vocabulary"))
            })?;
            total += value as f64;
        }

        Ok(ContinuationScore {
            total_logprob: total,
            mean_logprob: total / continuation_ids.len() as f64,
            token_count: continuation_ids.len(),
        })
    }
}

/// Return the index of the highest scoring continuation.
pub fn best_continuation(scores: &[ContinuationScore], use_mean: bool) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let lhs = if use_mean {
                a.mean_logprob
            } else {
                a.total_logprob
            };
            let rhs = if use_mean {
                b.mean_logprob
            } else {
                b.total_logprob
            };
            lhs.total_cmp(&rhs)
        })
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_continuation_uses_requested_score() {
        let scores = [
            ContinuationScore {
                total_logprob: -2.0,
                mean_logprob: -1.0,
                token_count: 2,
            },
            ContinuationScore {
                total_logprob: -1.5,
                mean_logprob: -1.5,
                token_count: 1,
            },
        ];
        assert_eq!(best_continuation(&scores, false), Some(1));
        assert_eq!(best_continuation(&scores, true), Some(0));
    }
}
