use aarambh_studio_core::Result;
use aarambh_studio_finetune::{CodeVerifier, Verifier};
use serde::Deserialize;

use crate::generation::greedy_generate;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct HumanEvalExample {
    prompt: String,
    #[serde(alias = "ground_truth")]
    test: String,
}

/// HumanEval-lite pass@1 task.
pub struct HumanEvalLiteTask;

impl EvalTask for HumanEvalLiteTask {
    fn name(&self) -> &'static str {
        "humaneval"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("humaneval_lite").join("data.jsonl");
        let examples = read_jsonl::<HumanEvalExample>(&path, config.max_examples)?;
        let verifier = CodeVerifier::default();
        let mut passed = 0usize;

        for example in &examples {
            let completion = greedy_generate(context, &example.prompt, config.max_new_tokens)?;
            let candidate = format!("{}{}", example.prompt, completion);
            if verifier.score(&candidate, &example.test) >= 1.0 {
                passed += 1;
            }
        }

        Ok(TaskScore::pass_at_1("humaneval", passed, examples.len()))
    }
}
