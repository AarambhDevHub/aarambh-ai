use aarambh_ai_core::{AarambhError, Result};
use serde::Deserialize;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::scoring::{ContinuationScorer, ModelLogProbScorer, best_continuation};
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct HellaSwagExample {
    #[serde(alias = "ctx")]
    context: String,
    endings: Vec<String>,
    label: serde_json::Value,
}

/// HellaSwag multiple-choice completion task.
pub struct HellaSwagTask;

impl EvalTask for HellaSwagTask {
    fn name(&self) -> &'static str {
        "hellaswag"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("hellaswag").join("data.jsonl");
        let examples = read_jsonl::<HellaSwagExample>(&path, config.max_examples)?;
        let scorer = ModelLogProbScorer::new(context);
        let mut correct = 0usize;

        for example in &examples {
            let prediction = score_hellaswag_example(example, &scorer)?;
            if prediction == label_index(&example.label)? {
                correct += 1;
            }
        }

        Ok(TaskScore::accuracy("hellaswag", correct, examples.len()))
    }
}

fn score_hellaswag_example(
    example: &HellaSwagExample,
    scorer: &dyn ContinuationScorer,
) -> Result<usize> {
    if example.endings.is_empty() {
        return Err(AarambhError::Config(
            "HellaSwag examples must contain at least one ending".into(),
        ));
    }
    let scores = example
        .endings
        .iter()
        .map(|ending| {
            let continuation = if ending.starts_with(char::is_whitespace) {
                ending.clone()
            } else {
                format!(" {ending}")
            };
            scorer.score_continuation(&example.context, &continuation)
        })
        .collect::<Result<Vec<_>>>()?;
    best_continuation(&scores, true).ok_or_else(|| {
        AarambhError::Config("HellaSwag example produced no candidate scores".into())
    })
}

fn label_index(value: &serde_json::Value) -> Result<usize> {
    if let Some(index) = value.as_u64() {
        return Ok(index as usize);
    }
    let Some(text) = value.as_str() else {
        return Err(AarambhError::Config(
            "HellaSwag label must be a string or index".into(),
        ));
    };
    text.trim()
        .parse::<usize>()
        .map_err(|err| AarambhError::Config(format!("invalid HellaSwag label: {err}")))
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
            let mean = if continuation.contains("right") {
                -0.1
            } else {
                -2.0
            };
            Ok(ContinuationScore {
                total_logprob: mean * 4.0,
                mean_logprob: mean,
                token_count: 4,
            })
        }
    }

    #[test]
    fn hellaswag_scoring_uses_mean_logprob() {
        let example = HellaSwagExample {
            context: "A person starts".into(),
            endings: vec![" wrong".into(), " right".into()],
            label: serde_json::Value::from(1),
        };
        assert_eq!(score_hellaswag_example(&example, &FakeScorer).unwrap(), 1);
    }
}
