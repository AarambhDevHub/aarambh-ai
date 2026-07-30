use aarambh_studio_core::{AarambhError, Result};
use serde::Deserialize;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::scoring::{ContinuationScorer, ModelLogProbScorer, best_continuation};
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct MmluExample {
    question: String,
    choices: Vec<String>,
    answer: serde_json::Value,
}

/// MMLU-lite multiple-choice task.
pub struct MmluLiteTask;

impl EvalTask for MmluLiteTask {
    fn name(&self) -> &'static str {
        "mmlu"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("mmlu_lite").join("data.jsonl");
        let examples = read_jsonl::<MmluExample>(&path, config.max_examples)?;
        let scorer = ModelLogProbScorer::new(context);
        let mut correct = 0usize;

        for example in &examples {
            let prediction = score_mmlu_example(example, &scorer)?;
            if prediction == answer_index(&example.answer)? {
                correct += 1;
            }
        }

        Ok(TaskScore::accuracy("mmlu", correct, examples.len()))
    }
}

fn score_mmlu_example(example: &MmluExample, scorer: &dyn ContinuationScorer) -> Result<usize> {
    if example.choices.len() != 4 {
        return Err(AarambhError::Config(
            "MMLU-lite examples must contain exactly four choices".into(),
        ));
    }
    let prompt = format!(
        "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:",
        example.question,
        example.choices[0],
        example.choices[1],
        example.choices[2],
        example.choices[3]
    );
    let scores = [" A", " B", " C", " D"]
        .iter()
        .map(|label| scorer.score_continuation(&prompt, label))
        .collect::<Result<Vec<_>>>()?;
    best_continuation(&scores, false).ok_or_else(|| {
        AarambhError::Config("MMLU-lite example produced no candidate scores".into())
    })
}

fn answer_index(value: &serde_json::Value) -> Result<usize> {
    if let Some(index) = value.as_u64() {
        return Ok(index as usize);
    }
    let Some(text) = value.as_str() else {
        return Err(AarambhError::Config(
            "MMLU-lite answer must be a label or index".into(),
        ));
    };
    match text.trim().to_ascii_uppercase().as_str() {
        "A" => Ok(0),
        "B" => Ok(1),
        "C" => Ok(2),
        "D" => Ok(3),
        other => other.parse::<usize>().map_err(|err| {
            AarambhError::Config(format!("invalid MMLU-lite answer {other:?}: {err}"))
        }),
    }
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
            let value = if continuation.trim() == "C" {
                0.0
            } else {
                -10.0
            };
            Ok(ContinuationScore {
                total_logprob: value,
                mean_logprob: value,
                token_count: 1,
            })
        }
    }

    #[test]
    fn mmlu_lite_scoring_picks_highest_logprob_option() {
        let example = MmluExample {
            question: "2 + 2?".into(),
            choices: vec!["1".into(), "2".into(), "4".into(), "5".into()],
            answer: serde_json::Value::from("C"),
        };
        assert_eq!(score_mmlu_example(&example, &FakeScorer).unwrap(), 2);
        assert_eq!(answer_index(&example.answer).unwrap(), 2);
    }
}
