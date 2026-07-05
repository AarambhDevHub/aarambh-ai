use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
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
