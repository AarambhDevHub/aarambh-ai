use aarambh_ai_core::Result;
use aarambh_ai_finetune::{MathVerifier, Verifier};
use serde::Deserialize;

use crate::generation::greedy_generate;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct Gsm8kExample {
    #[serde(alias = "prompt")]
    question: String,
    #[serde(alias = "ground_truth")]
    answer: String,
}

/// GSM8K exact numeric answer task.
pub struct Gsm8kSubsetTask;

impl EvalTask for Gsm8kSubsetTask {
    fn name(&self) -> &'static str {
        "gsm8k"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("gsm8k_subset").join("data.jsonl");
        let examples = read_jsonl::<Gsm8kExample>(&path, config.max_examples)?;
        let verifier = MathVerifier::default();
        let mut correct = 0usize;

        for example in &examples {
            let prompt = format!("{}\nAnswer:", example.question);
            let completion = greedy_generate(context, &prompt, config.max_new_tokens)?;
            if verifier.score(&completion, &example.answer) >= 1.0 {
                correct += 1;
            }
        }

        Ok(TaskScore::accuracy("gsm8k", correct, examples.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gsm8k_subset_reuses_math_verifier_exact_match() {
        let verifier = MathVerifier::default();
        assert_eq!(verifier.score("work\n#### 4", "#### 4"), 1.0);
        assert_eq!(verifier.score("work\n#### 5", "#### 4"), 0.0);
    }
}
