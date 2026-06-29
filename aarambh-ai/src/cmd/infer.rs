use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_finetune::{Verifier, VerifierKind};
use aarambh_ai_inference::{
    GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep, InferenceEngine, Sampler,
    ThinkingMode,
};
use aarambh_ai_safety::{
    SafeResponse, SafetyGenerator, SafetyGuard, SafetyMode, SafetyPolicy, SafetyVerdict,
};
use aarambh_ai_selflearn::{SelfLearnBuildConfig, SelfLearnConfig, SelfLearnLoop, SelfLearnMode};
use aarambh_ai_tokenizer::{ASSISTANT, BpeTokenizer, THINK_END_ID, THINK_START_ID, USER};
use aarambh_ai_train::TrainingRunConfig;
use clap::Args;
use serde::Deserialize;

use crate::ui::predict_view;

const ANSI_DIM: &str = "\x1b[2m";
const ANSI_RESET: &str = "\x1b[0m";

#[derive(Debug, Args)]
pub struct InferArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub prompt: String,
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value = "none")]
    pub thinking: String,
    #[arg(long)]
    pub predict_view: bool,
    #[arg(long)]
    pub stream: bool,
    #[arg(long)]
    pub greedy: bool,
    #[arg(long, default_value = "strict")]
    pub safety: String,
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    #[arg(long, default_value = "disabled")]
    pub self_learn: String,
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    #[arg(long)]
    pub self_learn_reference: Option<PathBuf>,
    #[arg(long, default_value = "none")]
    pub self_learn_verifier: String,
    #[arg(long)]
    pub self_learn_ground_truth: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

pub fn run(args: InferArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    let tokenizer_path = tokenizer_path(&args, &run_config);
    let model_path = match args.model.clone() {
        Some(path) => path,
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let sampler = if args.greedy {
        Sampler::greedy()
    } else {
        Sampler::top_k_top_p(
            args.temperature,
            Some(args.top_k),
            Some(args.top_p),
            args.seed,
        )?
    };
    let thinking_mode = parse_thinking_mode(&args.thinking)?;
    let safety_mode = parse_safety_mode(&args.safety)?;
    let self_learn_mode = parse_self_learn_mode(&args.self_learn)?;

    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 5,
    };

    let prompt = prompt_for_mode(&args.prompt, thinking_mode);
    if self_learn_mode.is_enabled() {
        return run_self_learn_infer(
            &args,
            run_config,
            model_path,
            tokenizer_path,
            device,
            config,
            prompt,
            safety_mode,
            self_learn_mode,
            thinking_mode,
        );
    }

    let mut engine =
        InferenceEngine::from_paths(model_path, &run_config.model, tokenizer_path, device)?;
    let tokenizer_for_view = engine.tokenizer().clone();
    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(engine, policy);
        let mut stream_state = StreamState::default();
        let response = guard.generate_with_callback(&prompt, config, |step| {
            if args.predict_view {
                print!(
                    "{}",
                    predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
                );
            }
            if args.stream {
                stream_step(step, thinking_mode, &mut stream_state)?;
            }
            if args.predict_view || args.stream {
                io::stdout().flush()?;
            }
            Ok(())
        })?;
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let output = engine.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;

    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

fn tokenizer_path(args: &InferArgs, run_config: &TrainingRunConfig) -> PathBuf {
    args.tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

fn default_model_path(checkpoint_dir: &Path) -> anyhow::Result<PathBuf> {
    for pointer_name in ["latest.json", "best.json"] {
        let pointer_path = checkpoint_dir.join(pointer_name);
        if pointer_path.exists() {
            let file = fs::File::open(&pointer_path)?;
            let pointer: CheckpointPointer = serde_json::from_reader(file)?;
            return Ok(pointer.path.join("model.safetensors"));
        }
    }
    Err(anyhow::anyhow!(
        "no model provided and no latest.json or best.json found in {}",
        checkpoint_dir.display()
    ))
}

fn parse_thinking_mode(value: &str) -> anyhow::Result<ThinkingMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(ThinkingMode::None),
        "low" => Ok(ThinkingMode::Low),
        "medium" => Ok(ThinkingMode::Medium),
        "high" => Ok(ThinkingMode::High),
        other => Err(anyhow::anyhow!(
            "invalid thinking mode '{other}', expected none|low|medium|high"
        )),
    }
}

fn parse_safety_mode(value: &str) -> anyhow::Result<SafetyMode> {
    value.parse::<SafetyMode>().map_err(anyhow::Error::msg)
}

fn parse_self_learn_mode(value: &str) -> anyhow::Result<SelfLearnMode> {
    value.parse::<SelfLearnMode>().map_err(anyhow::Error::msg)
}

fn parse_self_learn_verifier(value: &str) -> anyhow::Result<Option<VerifierKind>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" | "disabled" | "off" => Ok(None),
        other => other
            .parse::<VerifierKind>()
            .map(Some)
            .map_err(anyhow::Error::msg),
    }
}

fn prompt_for_mode(prompt: &str, thinking_mode: ThinkingMode) -> String {
    if thinking_mode.is_enabled() {
        format!("{USER}\n{prompt}\n{ASSISTANT}\n")
    } else {
        prompt.to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_self_learn_infer(
    args: &InferArgs,
    mut run_config: TrainingRunConfig,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    device: candle_core::Device,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    self_learn_mode: SelfLearnMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    let mut self_config = SelfLearnConfig::for_mode(self_learn_mode)
        .with_replay_path(args.replay_path.clone())
        .with_state_dir(args.self_learn_state_dir.clone());
    self_config.grpo.max_new_tokens = args.max_tokens;
    self_config.critique.rewrite_max_tokens = self_config
        .critique
        .rewrite_max_tokens
        .min(args.max_tokens)
        .max(1);
    let reference_path = args
        .self_learn_reference
        .clone()
        .unwrap_or_else(|| model_path.clone());
    let verifier = parse_self_learn_verifier(&args.self_learn_verifier)?.map(VerifierKind::build);
    if verifier.is_some() && args.self_learn_ground_truth.is_none() {
        eprintln!(
            "[self-learn] deterministic verifier requested without --self-learn-ground-truth; online GRPO will be skipped"
        );
    }

    let tokenizer_for_view = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    let loop_ = SelfLearnLoop::from_paths(SelfLearnBuildConfig {
        model_config: {
            run_config.model.vocab_size = tokenizer_for_view.vocab_size();
            run_config.model.clone()
        },
        base_model_path: model_path,
        reference_model_path: reference_path,
        tokenizer_path,
        config: self_config,
        device,
        seed: run_config.train.seed,
    })?;
    let mut adapter = SelfLearnSafetyAdapter {
        loop_,
        verifier,
        ground_truth: args.self_learn_ground_truth.clone(),
    };

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = guard.generate_with_callback(&prompt, config, |step| {
            if args.predict_view {
                print!(
                    "{}",
                    predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
                );
            }
            if args.stream {
                stream_step(step, thinking_mode, &mut stream_state)?;
            }
            if args.predict_view || args.stream {
                io::stdout().flush()?;
            }
            Ok(())
        })?;
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        let mut adapter = guard.into_inner();
        if response.is_blocked() {
            adapter.loop_.discard_last_draft();
        } else {
            let learned = adapter
                .loop_
                .commit_last_draft(Some(response.text.clone()))?;
            print_self_learn_summary(&learned);
        }
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let output = adapter.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    let learned = adapter.loop_.commit_last_draft(Some(output.text.clone()))?;
    print_self_learn_summary(&learned);
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

struct SelfLearnSafetyAdapter {
    loop_: SelfLearnLoop,
    verifier: Option<Box<dyn Verifier>>,
    ground_truth: Option<String>,
}

impl SafetyGenerator for SelfLearnSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_ai_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_ai_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_ai_core::Result<()>,
    {
        self.loop_.generate_draft_with_callback(
            prompt,
            config,
            self.verifier.as_deref(),
            self.ground_truth.as_deref(),
            on_step,
        )
    }
}

fn print_self_learn_summary(response: &aarambh_ai_selflearn::SelfLearnResponse) {
    eprintln!(
        "[self-learn] critique_score={:.2} stored={} rewritten={} grpo={} metrics={}",
        response.critique_score,
        response.stored_in_replay,
        response.was_rewritten,
        response.used_grpo,
        response.metrics_summary
    );
}

#[derive(Default)]
struct StreamState {
    dim_active: bool,
    header_printed: bool,
    thinking_tokens: usize,
}

fn stream_step(
    step: &GenerationStep,
    thinking_mode: ThinkingMode,
    state: &mut StreamState,
) -> io::Result<()> {
    if !thinking_mode.is_enabled() {
        print!("{}", step.token_text);
        return Ok(());
    }

    match step.phase {
        GenerationPhase::Thinking => {
            if !state.header_printed {
                print!("[thinking]\n{ANSI_DIM}");
                state.header_printed = true;
                state.dim_active = true;
            }
            if !is_thinking_marker(step.token_id) {
                state.thinking_tokens += 1;
                print!("{}", step.token_text);
            }
        }
        GenerationPhase::Answer => {
            if state.dim_active {
                println!("{ANSI_RESET}");
                println!("[thinking: {} tokens]", state.thinking_tokens);
                state.dim_active = false;
            }
            print!("{}", step.token_text);
        }
    }
    Ok(())
}

fn finish_stream(state: &mut StreamState) {
    if state.dim_active {
        println!("{ANSI_RESET}");
        println!("[thinking: {} tokens]", state.thinking_tokens);
        state.dim_active = false;
    }
    println!();
}

fn print_safe_response(
    response: &SafeResponse,
    thinking_mode: ThinkingMode,
    stream: bool,
    stream_state: &mut StreamState,
) -> io::Result<()> {
    if let SafetyVerdict::Block(reason) = &response.verdict {
        println!("blocked by safety: {reason}");
        return Ok(());
    }

    let Some(output) = &response.output else {
        println!("blocked by safety");
        return Ok(());
    };

    if stream && !response.output_redacted {
        finish_stream(stream_state);
    } else {
        print_generation_output(output, thinking_mode)?;
    }
    Ok(())
}

fn print_generation_output(
    output: &GenerationOutput,
    thinking_mode: ThinkingMode,
) -> io::Result<()> {
    if !thinking_mode.is_enabled() {
        println!("{}", output.text);
        return Ok(());
    }

    println!("[thinking: {} tokens]", output.thinking_tokens);
    if !output.thinking_text.is_empty() {
        println!("{ANSI_DIM}{}{ANSI_RESET}", output.thinking_text);
    }
    println!("{}", output.text);
    Ok(())
}

fn is_thinking_marker(token_id: u32) -> bool {
    token_id == THINK_START_ID || token_id == THINK_END_ID
}
