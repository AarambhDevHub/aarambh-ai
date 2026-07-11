use aarambh_ai_core::Result;
use aarambh_ai_finetune::{ChatTemplate, DpoExample};

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::scoring::{ContinuationScorer, ModelLogProbScorer};
use crate::tasks::read_jsonl;

/// Held-out pairwise preference-ranking task.
pub struct PreferenceTask;

impl EvalTask for PreferenceTask {
    fn name(&self) -> &'static str {
        "preference"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("preference").join("data.jsonl");
        let examples = read_jsonl::<DpoExample>(&path, config.max_examples)?;
        let scorer = ModelLogProbScorer::new(context);
        let wins = examples
            .iter()
            .map(|example| score_preference_pair(example, &scorer))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|won| *won)
            .count();
        Ok(TaskScore::win_rate("preference", wins, examples.len()))
    }
}

fn score_preference_pair(example: &DpoExample, scorer: &dyn ContinuationScorer) -> Result<bool> {
    let template = ChatTemplate;
    let prompt = template.prefix(&example.prompt, None);
    let chosen = scorer.score_continuation(&prompt, &template.target(&example.chosen))?;
    let rejected = scorer.score_continuation(&prompt, &template.target(&example.rejected))?;
    Ok(chosen.mean_logprob > rejected.mean_logprob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::ContinuationScore;

    struct FakeScorer;

    impl ContinuationScorer for FakeScorer {
        fn score_continuation(
            &self,
            _prompt: &str,
            continuation: &str,
        ) -> Result<ContinuationScore> {
            let mean_logprob = if continuation.contains("careful") {
                -0.1
            } else {
                -2.0
            };
            Ok(ContinuationScore {
                total_logprob: mean_logprob * 4.0,
                mean_logprob,
                token_count: 4,
            })
        }
    }

    #[test]
    fn preference_task_uses_mean_completion_logprob() {
        let example = DpoExample {
            prompt: "Answer clearly".into(),
            chosen: "A careful answer".into(),
            rejected: "A vague answer".into(),
        };
        assert!(score_preference_pair(&example, &FakeScorer).unwrap());
    }
}
