use aarambh_ai_core::{Result, TokenizerLike};
use aarambh_ai_inference::{
    GenerationConfig, InferenceEngine, ToolCall, ToolCallingConfig, ToolChoice, ToolDefinition,
};
use serde::Deserialize;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct ToolEvalExample {
    instruction: String,
    tools: Vec<ToolDefinition>,
    #[serde(default)]
    tool_call: Option<ToolCall>,
}

/// Single-turn tool selection and argument generation evaluation.
pub struct ToolCallingTask;

impl EvalTask for ToolCallingTask {
    fn name(&self) -> &'static str {
        "tool-calling"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("tool_calling").join("data.jsonl");
        let examples = read_jsonl::<ToolEvalExample>(&path, config.max_examples)?;
        let mut exact = 0usize;
        let mut selected = 0usize;
        let mut arguments_exact = 0usize;
        let mut valid = 0usize;
        let mut generated_calls = 0usize;
        let mut expected_calls = 0usize;
        let mut no_tool_correct = 0usize;
        let mut no_tool_total = 0usize;

        for example in &examples {
            let tools = ToolCallingConfig::new(example.tools.clone(), ToolChoice::Auto)?;
            let mut engine = InferenceEngine::new(
                context.model().clone(),
                context.tokenizer().clone(),
                context.device().clone(),
            )?;
            let mut generation = GenerationConfig::greedy(config.max_new_tokens);
            generation.tool_calling = Some(tools.clone());
            let output = engine.generate(&example.instruction, generation)?;
            let prompt = tools.render_prompt(&example.instruction)?;
            context.record_context_len(
                context.tokenizer().encode(&prompt)?.len() + output.token_ids.len(),
            );
            if let Some(call) = &output.tool_call {
                generated_calls += 1;
                if tools.validate_call(call).is_ok() {
                    valid += 1;
                }
                if example
                    .tool_call
                    .as_ref()
                    .is_some_and(|expected| expected.name == call.name)
                {
                    selected += 1;
                }
                if example
                    .tool_call
                    .as_ref()
                    .is_some_and(|expected| expected.arguments == call.arguments)
                {
                    arguments_exact += 1;
                }
            } else if example.tool_call.is_none() {
                no_tool_correct += 1;
            }
            if example.tool_call.is_none() {
                no_tool_total += 1;
            } else {
                expected_calls += 1;
            }
            if output.tool_call == example.tool_call {
                exact += 1;
            }
        }

        let count = examples.len();
        Ok(TaskScore::accuracy("tool-calling", exact, count)
            .with_detail("schema_valid_rate", ratio(valid, generated_calls))
            .with_detail("tool_name_accuracy", ratio(selected, expected_calls))
            .with_detail(
                "argument_exact_match",
                ratio(arguments_exact, expected_calls),
            )
            .with_detail("no_tool_accuracy", ratio(no_tool_correct, no_tool_total)))
    }
}

fn ratio(value: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_handles_empty_categories() {
        assert_eq!(ratio(0, 0), 0.0);
        assert_eq!(ratio(1, 2), 0.5);
    }
}
