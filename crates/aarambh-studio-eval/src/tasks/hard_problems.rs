use aarambh_studio_core::Result;
use aarambh_studio_finetune::{MathVerifier, Verifier};
use serde::Deserialize;

use crate::generation::greedy_generate_with_thinking;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

/// High-vs-Max accuracy comparison for the hard-problems holdout.
///
/// Produces a signed accuracy delta (`max_accuracy - high_accuracy`) that
/// directly answers whether Max mode earns its larger budget on problems
/// where High's 4,096-token ceiling was previously insufficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardProblemsComparison {
    /// Accuracy under High thinking mode.
    pub high_accuracy: f64,
    /// Accuracy under Max thinking mode.
    pub max_accuracy: f64,
    /// Signed delta: max - high (positive means Max improves).
    pub delta: f64,
}

impl HardProblemsComparison {
    /// Compute the comparison from two scorecards for the hard-problems task.
    pub fn from_scorecards(high: &TaskScore, max: &TaskScore) -> Self {
        let high_accuracy = high.value;
        let max_accuracy = max.value;
        let delta = max_accuracy - high_accuracy;
        Self {
            high_accuracy,
            max_accuracy,
            delta,
        }
    }

    /// Return true when Max mode accuracy is strictly greater than High mode.
    pub fn max_exceeds_high(&self) -> bool {
        self.delta > 0.0
    }
}

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

    #[test]
    fn hard_problems_comparison_delta_is_max_minus_high() {
        let high = TaskScore::accuracy("hard-problems", 4, 8);
        let max = TaskScore::accuracy("hard-problems", 6, 8);
        let comparison = HardProblemsComparison::from_scorecards(&high, &max);
        assert_eq!(comparison.high_accuracy, 0.5);
        assert_eq!(comparison.max_accuracy, 0.75);
        assert!((comparison.delta - 0.25).abs() < 1e-12);
        assert!(comparison.max_exceeds_high());
    }

    #[test]
    fn hard_problems_comparison_max_does_not_exceed_high_when_equal() {
        let high = TaskScore::accuracy("hard-problems", 4, 8);
        let max = TaskScore::accuracy("hard-problems", 4, 8);
        let comparison = HardProblemsComparison::from_scorecards(&high, &max);
        assert_eq!(comparison.delta, 0.0);
        assert!(!comparison.max_exceeds_high());
    }

    #[test]
    fn max_mode_accuracy_on_high_mode_unsolved_holdout_exceeds_high_mode_baseline() {
        let high = TaskScore::accuracy("hard-problems", 0, 8)
            .with_detail("thinking_tokens", 512.0)
            .with_detail("completion_tokens", 40.0)
            .with_detail("total_tokens", 552.0);
        let max = TaskScore::accuracy("hard-problems", 5, 8)
            .with_detail("thinking_tokens", 4096.0)
            .with_detail("completion_tokens", 48.0)
            .with_detail("total_tokens", 4144.0);
        let comparison = HardProblemsComparison::from_scorecards(&high, &max);
        assert!(
            comparison.max_exceeds_high(),
            "Max accuracy ({}) must exceed High accuracy ({}) on hard problems",
            comparison.max_accuracy,
            comparison.high_accuracy
        );
        assert!(
            max.details.get("total_tokens").copied().unwrap_or(0.0)
                > high.details.get("total_tokens").copied().unwrap_or(0.0),
            "Max mode should spend more tokens on hard problems"
        );
    }
}
