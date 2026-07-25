use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_inference::{ThinkingController, ThinkingMode};
use aarambh_ai_tokenizer::{THINK_END_ID, THINK_START_ID};
use candle_core::Tensor;

use crate::harness::EvalContext;

/// Generate text greedily from a prompt.
pub fn greedy_generate(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<String> {
    let mut prompt_ids = context.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = context.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    if prompt_ids.len() >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "prompt length {} leaves no room in max_seq_len {}",
            prompt_ids.len(),
            context.max_seq_len()
        )));
    }

    let budget = max_new_tokens.min(context.max_seq_len() - prompt_ids.len());
    let mut caches = context.model().empty_kv_cache();
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let logits = context.model().forward_with_cache(&input, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);

    for step in 0..budget {
        let logits_vec = next_logits.to_vec1::<f32>()?;
        let token_id = argmax(&logits_vec) as u32;
        if token_id == context.tokenizer().eos_token_id() {
            break;
        }
        generated.push(token_id);
        context.record_context_len(prompt_ids.len() + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_ids.len() + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    context.tokenizer().decode(&generated)
}

/// Token accounting for a thinking-aware greedy generation.
///
/// `thinking_tokens` counts content tokens emitted inside the `<think>` block
/// (excluding the markers themselves), `completion_tokens` counts answer
/// tokens emitted after the block closes, and `total_tokens` is their sum.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingGenerationResult {
    /// Decoded completion text (markers stripped from the visible answer).
    pub text: String,
    /// Content tokens spent inside the thinking block.
    pub thinking_tokens: usize,
    /// Answer tokens emitted after the thinking block closed.
    pub completion_tokens: usize,
}

impl ThinkingGenerationResult {
    /// Total content tokens generated (thinking + completion, excluding markers).
    pub fn total_tokens(&self) -> usize {
        self.thinking_tokens + self.completion_tokens
    }
}

/// Generate text greedily while reusing the inference crate's
/// [`ThinkingController`] for forced `<think>`/`</think>` markers and budget
/// enforcement. Generation is deterministic (greedy argmax) regardless of the
/// thinking mode, matching the eval harness's deterministic-defaults policy.
///
/// The effective thinking budget is clamped by the controller to
/// `min(mode.budget(), max_new_tokens - reserve)` exactly as it is during
/// normal inference, so Max mode never exceeds the configured generation
/// budget.
pub fn greedy_generate_with_thinking(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
    thinking_mode: ThinkingMode,
) -> Result<ThinkingGenerationResult> {
    let mut prompt_ids = context.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = context.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    if prompt_ids.len() >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "prompt length {} leaves no room in max_seq_len {}",
            prompt_ids.len(),
            context.max_seq_len()
        )));
    }

    let budget = max_new_tokens.min(context.max_seq_len() - prompt_ids.len());
    let mut thinking = ThinkingController::for_generation(thinking_mode, max_new_tokens);
    let mut caches = context.model().empty_kv_cache();
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let logits = context.model().forward_with_cache(&input, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);
    let mut completion_tokens = 0usize;
    let eos = context.tokenizer().eos_token_id();

    for step in 0..budget {
        let (mut token_id, forced) = match thinking.take_forced_token() {
            Some(force) => (force.token_id(), true),
            None => {
                let logits_vec = next_logits.to_vec1::<f32>()?;
                (argmax(&logits_vec) as u32, false)
            }
        };

        // An EOS sampled inside an open thinking block forces the closing
        // marker instead of ending generation, exactly as the inference engine
        // does. A forced or naturally-sampled EOS outside thinking ends the
        // turn.
        if token_id == eos && !forced {
            if thinking.in_thinking_block() {
                token_id = THINK_END_ID;
            } else {
                break;
            }
        }

        let is_marker = token_id == THINK_START_ID || token_id == THINK_END_ID;
        let in_block_before = thinking.in_thinking_block();
        // on_token advances controller state (opens/closes the block, counts
        // thinking content, and may queue the next forced close marker).
        let _ = thinking.on_token(token_id);
        if !is_marker && !in_block_before {
            completion_tokens += 1;
        }

        generated.push(token_id);
        context.record_context_len(prompt_ids.len() + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_ids.len() + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    let thinking_tokens = thinking.tokens_used();
    let text = context.tokenizer().decode(&generated)?;
    Ok(ThinkingGenerationResult {
        text,
        thinking_tokens,
        completion_tokens,
    })
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[1] == 0 {
        return Err(AarambhError::Shape(format!(
            "expected logits [batch, seq, vocab], got {dims:?}"
        )));
    }
    Ok(logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use aarambh_ai_inference::ForceToken;

    use super::*;

    #[test]
    fn thinking_generation_result_total_sums_components() {
        let result = ThinkingGenerationResult {
            text: "answer".into(),
            thinking_tokens: 12,
            completion_tokens: 7,
        };
        assert_eq!(result.total_tokens(), 19);
    }

    #[test]
    fn force_token_marker_ids_match_tokenizer_constants() {
        assert_eq!(ForceToken::ThinkStart.token_id(), THINK_START_ID);
        assert_eq!(ForceToken::ThinkEnd.token_id(), THINK_END_ID);
    }
}
