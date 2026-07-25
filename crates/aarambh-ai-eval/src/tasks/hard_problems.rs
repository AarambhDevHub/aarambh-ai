use aarambh_ai_core::Result;
use aarambh_ai_finetune::{MathVerifier, Verifier};
use serde::Deserialize;

use crate::generation::greedy_generate_with_thinking;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct HardProblemExample {
    #[serde(alias = "prompt", alias = "question")]
    question: String,
    #[serde(alias = "ground_truth", alias = "answer")]
    answer: String,
}

/// Hard-problems task for Phase 39 Max-mode validation.
///
/// The fixture (`data/eval/hard_problems/data.jsonl`) holds deterministic
/// problems selected to exercise a larger thinking budget than High mode's
/// 4,096-token ceiling. Each problem is scored with the shared `MathVerifier`,
/// and the task reports accuracy alongside average thinking, completion, and
/// total token counts so a High-vs-Max comparison can be read directly off the
/// scorecard's `details` map.
pub struct HardProblemsTask;

impl EvalTask for HardProblemsTask {
    fn name(&self) -> &'static str {
        "hard-problems"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("hard_problems").join("data.jsonl");
        let examples = read_jsonl::<HardProblemExample>(&path, config.max_examples)?;
        let verifier = MathVerifier::default();
        let mut correct = 0usize;
        let mut thinking_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut total_tokens = 0usize;

        for example in &examples {
            let prompt = format!("{}\nAnswer:", example.question);
            let result = greedy_generate_with_thinking(
                context,
                &prompt,
                config.max_new_tokens,
                config.thinking_mode,
            )?;
            if verifier.score(&result.text, &example.answer) >= 1.0 {
                correct += 1;
            }
            thinking_tokens += result.thinking_tokens;
            completion_tokens += result.completion_tokens;
            total_tokens += result.total_tokens();
        }

        let n = examples.len();
        let avg = |sum: usize| if n == 0 { 0.0 } else { sum as f64 / n as f64 };
        Ok(TaskScore::accuracy("hard-problems", correct, n)
            .with_detail("thinking_tokens", avg(thinking_tokens))
            .with_detail("completion_tokens", avg(completion_tokens))
            .with_detail("total_tokens", avg(total_tokens)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_problem_example_accepts_question_and_prompt_aliases() {
        let a: HardProblemExample =
            serde_json::from_str(r#"{"question":"q","answer":"42"}"#).unwrap();
        assert_eq!(a.question, "q");
        assert_eq!(a.answer, "42");
        let b: HardProblemExample =
            serde_json::from_str(r#"{"prompt":"p","ground_truth":"7"}"#).unwrap();
        assert_eq!(b.question, "p");
        assert_eq!(b.answer, "7");
    }

    #[test]
    fn hard_problems_task_name_is_stable() {
        assert_eq!(HardProblemsTask.name(), "hard-problems");
    }
}
