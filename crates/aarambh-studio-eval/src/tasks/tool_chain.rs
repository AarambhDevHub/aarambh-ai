use std::cell::{Cell, RefCell};
use std::rc::Rc;

use aarambh_studio_agent::{
    AgentError, AgentResult, ChainDecoder, ChainEvent, EvictionPolicy, ReplayEntry,
    ReplayResultProvider, ToolChain, ToolChainConfig, ToolExchange, ToolResult,
};
use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_inference::{
    GenerationConfig, GenerationOutput, InferenceEngine, ToolCall, ToolCallingConfig, ToolChoice,
    ToolDefinition,
};
use aarambh_studio_safety::{SafetyInspector, SafetyPolicy, SafetyVerdict};
use aarambh_studio_tokenizer::{ASSISTANT, USER};
use serde::Deserialize;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct ToolChainEvalStep {
    call: ToolCall,
    result: ToolResult,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolChainEvalExample {
    instruction: String,
    tools: Vec<ToolDefinition>,
    steps: Vec<ToolChainEvalStep>,
    final_answer: String,
}

struct EvalChainDecoder {
    engine: InferenceEngine,
    tools: ToolCallingConfig,
    context_used: Rc<Cell<usize>>,
    safety_checks: Rc<Cell<usize>>,
    safety_violations: Rc<Cell<usize>>,
    safety: SafetyInspector,
}

impl ChainDecoder for EvalChainDecoder {
    fn context_limit(&self) -> usize {
        aarambh_studio_core::Configurable::config(self.engine.model()).max_seq_len
    }

    fn encode_prefix(
        &mut self,
        prompt: &str,
        _tools: &[ToolDefinition],
        summary: Option<&str>,
    ) -> AgentResult<Vec<u32>> {
        let prompt = match summary {
            Some(summary) => format!("Prior tool-chain summary:\n{summary}\n\n{prompt}"),
            None => prompt.to_string(),
        };
        let rendered = self.tools.render_prompt(&prompt)?;
        Ok(self.engine.encode_prompt(&rendered)?)
    }

    fn encode_result(&mut self, result: &ToolResult) -> AgentResult<Vec<u32>> {
        self.encode_result_metadata(result)
    }

    fn encode_result_metadata(&mut self, result: &ToolResult) -> AgentResult<Vec<u32>> {
        Ok(self.engine.tokenizer().encode(&format!(
            "{USER}\nTool result {}: {}\n{ASSISTANT}\n",
            result.call_id,
            result.transcript_text()
        ))?)
    }

    fn generate(
        &mut self,
        transcript_ids: &[u32],
        _pending_media: Option<&aarambh_studio_agent::ToolResultContent>,
        max_new_tokens: usize,
    ) -> AgentResult<GenerationOutput> {
        self.context_used
            .set(self.context_used.get().max(transcript_ids.len()));
        let mut config = GenerationConfig::greedy(max_new_tokens);
        config.tool_calling = Some(self.tools.clone());
        config.capture_steps = false;
        let output = self
            .engine
            .generate_from_token_ids(transcript_ids.to_vec(), config)?;
        let checked = self
            .safety
            .inspect_output("tool-chain eval", output.clone())?;
        self.safety_checks.set(self.safety_checks.get() + 1);
        if matches!(
            checked.verdict,
            SafetyVerdict::Block(_) | SafetyVerdict::Regenerate(_)
        ) {
            self.safety_violations.set(self.safety_violations.get() + 1);
        }
        Ok(output)
    }

    fn summarise(
        &mut self,
        previous_summary: Option<&str>,
        evicted: &[ToolExchange],
        _max_tokens: usize,
    ) -> AgentResult<String> {
        let mut summary = previous_summary.unwrap_or_default().to_string();
        for exchange in evicted {
            summary.push_str(&format!(
                "\n{} {} -> {}",
                exchange.request.call_id,
                exchange.request.call.name,
                exchange.result.transcript_text()
            ));
        }
        Ok(summary)
    }
}

/// Response-path evaluation for multi-step function-calling traces.
pub struct ToolChainTask;

impl EvalTask for ToolChainTask {
    fn name(&self) -> &'static str {
        "tool-chain"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("tool_chain").join("data.jsonl");
        let examples = read_jsonl::<ToolChainEvalExample>(&path, config.max_examples)?;
        let mut successful = 0usize;
        let mut call_name_matches = 0usize;
        let mut argument_matches = 0usize;
        let mut schema_valid = 0usize;
        let mut expected_calls = 0usize;
        let mut generated_calls = 0usize;
        let mut final_matches = 0usize;
        let mut three_plus_success = 0usize;
        let mut three_plus_total = 0usize;
        let mut max_step_failures = 0usize;
        let safety_checks = Rc::new(Cell::new(0usize));
        let safety_violations = Rc::new(Cell::new(0usize));

        for example in &examples {
            if example.steps.is_empty() {
                return Err(AarambhError::Config(
                    "tool-chain eval examples require at least one scripted step".into(),
                ));
            }
            expected_calls += example.steps.len();
            if example.steps.len() >= 3 {
                three_plus_total += 1;
            }
            let tools = ToolCallingConfig::new(example.tools.clone(), ToolChoice::Auto)?;
            let replay = example
                .steps
                .iter()
                .map(|step| ReplayEntry {
                    expected_call: Some(step.call.clone()),
                    result: step.result.clone(),
                })
                .collect::<Vec<_>>();
            let provider = ReplayResultProvider::new(replay).map_err(agent_error)?;
            let context_used = Rc::new(Cell::new(0usize));
            let mut policy = SafetyPolicy::strict();
            policy.audit_enabled = false;
            let decoder = EvalChainDecoder {
                engine: InferenceEngine::new(
                    context.model().clone(),
                    context.tokenizer().clone(),
                    context.device().clone(),
                )?,
                tools: tools.clone(),
                context_used: context_used.clone(),
                safety_checks: safety_checks.clone(),
                safety_violations: safety_violations.clone(),
                safety: SafetyInspector::new(policy),
            };
            let chain_config = ToolChainConfig {
                max_steps: config.agent_max_steps,
                max_tokens_per_step: config.max_new_tokens,
                context_reserve: 32.min(context.max_seq_len().saturating_sub(1)),
                keep_recent: 4,
                summary_tokens: 128,
                eviction_policy: EvictionPolicy::DropOldest,
            };
            let mut chain = ToolChain::new(decoder, provider, chain_config).map_err(agent_error)?;
            let observed = Rc::new(RefCell::new(Vec::<ToolCall>::new()));
            let event_calls = observed.clone();
            let outcome = chain.run_with_callback(
                example.instruction.clone(),
                example.tools.clone(),
                move |event| {
                    if let ChainEvent::ToolCall { request } = event {
                        event_calls.borrow_mut().push(request.call.clone());
                    }
                    Ok(())
                },
            );
            context.record_context_len(context_used.get());

            let observed = observed.borrow();
            generated_calls += observed.len();
            for (index, call) in observed.iter().enumerate() {
                if tools.validate_call(call).is_ok() {
                    schema_valid += 1;
                }
                if let Some(expected) = example.steps.get(index).map(|step| &step.call) {
                    if call.name == expected.name {
                        call_name_matches += 1;
                    }
                    if call.arguments == expected.arguments {
                        argument_matches += 1;
                    }
                }
            }
            match outcome {
                Ok(output) => {
                    let calls_match = observed.len() == example.steps.len()
                        && observed
                            .iter()
                            .zip(&example.steps)
                            .all(|(actual, expected)| actual == &expected.call);
                    let final_match =
                        normalize(&output.final_output.text) == normalize(&example.final_answer);
                    final_matches += usize::from(final_match);
                    if calls_match && final_match {
                        successful += 1;
                        if example.steps.len() >= 3 {
                            three_plus_success += 1;
                        }
                    }
                }
                Err(AgentError::MaxSteps { .. }) => max_step_failures += 1,
                Err(AgentError::ReplayMismatch { .. })
                | Err(AgentError::ResultProtocol(_))
                | Err(AgentError::Config(_))
                | Err(AgentError::Runtime(_)) => {}
            }
        }

        let count = examples.len();
        Ok(TaskScore::accuracy("tool-chain", successful, count)
            .with_detail("schema_valid_rate", ratio(schema_valid, generated_calls))
            .with_detail(
                "tool_name_accuracy",
                ratio(call_name_matches, expected_calls),
            )
            .with_detail(
                "argument_exact_match",
                ratio(argument_matches, expected_calls),
            )
            .with_detail("final_answer_match", ratio(final_matches, count))
            .with_detail(
                "three_plus_call_success",
                ratio(three_plus_success, three_plus_total),
            )
            .with_detail(
                "average_generated_calls",
                generated_calls as f64 / count as f64,
            )
            .with_detail("max_step_failure_rate", ratio(max_step_failures, count))
            .with_detail(
                "safety_pass_rate",
                1.0 - ratio(safety_violations.get(), safety_checks.get()),
            ))
    }
}

fn normalize(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn ratio(value: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64
    }
}

fn agent_error(error: AgentError) -> AarambhError {
    AarambhError::Config(format!("tool-chain evaluation: {error}"))
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn final_answer_normalization_is_whitespace_insensitive() {
        assert_eq!(normalize("  Result:\nFour "), "result: four");
    }
}
