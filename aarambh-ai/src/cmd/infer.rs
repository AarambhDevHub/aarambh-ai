use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use aarambh_ai_core::{AarambhError, TokenizerLike};
use aarambh_ai_finetune::{Verifier, VerifierKind};
use aarambh_ai_inference::{
    GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep, InferenceEngine, Sampler,
    SpeculativeConfig, SpeculativeEngine, ThinkingMode, ToolCallingConfig, ToolChoice,
    ToolDefinition,
};
use aarambh_ai_safety::{
    SafeResponse, SafeStreamEvent, SafetyGenerator, SafetyGuard, SafetyMode, SafetyPolicy,
    SafetyVerdict,
};
use aarambh_ai_selflearn::{
    SelfLearnBuildConfig, SelfLearnConfig, SelfLearnLoop, SelfLearnMode, VisionCache,
    VisionVerifierKind, require_vision_hardware,
};
use aarambh_ai_tokenizer::{
    ASSISTANT, BpeTokenizer, IMAGE, IMAGE_END, IMAGE_ID, THINK_END_ID, THINK_START_ID, USER,
};
use aarambh_ai_train::TrainingRunConfig;
use aarambh_ai_vision::{
    ClipVisionEncoder, ImagePreprocessor, ProjectorConfig, VisionEncoderConfig, VisionModel,
    VisionPreprocessConfig, VisionProjector, interleave_image_tokens,
};
use candle_core::Tensor;
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
    pub image: Option<PathBuf>,
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
    #[arg(long)]
    pub speculative: bool,
    #[arg(long)]
    pub draft_model: Option<PathBuf>,
    #[arg(long)]
    pub draft_config: Option<PathBuf>,
    #[arg(long)]
    pub draft_tokenizer: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub draft_tokens: usize,
    #[arg(long)]
    pub stats: bool,
    #[arg(long)]
    pub tools: Option<PathBuf>,
    #[arg(long, default_value = "auto")]
    pub tool_choice: String,
    #[arg(long, default_value = "strict")]
    pub safety: String,
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    #[arg(long, default_value = "disabled")]
    pub self_learn: String,
    #[arg(long)]
    pub replay_path: Option<PathBuf>,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    #[arg(long)]
    pub self_learn_reference: Option<PathBuf>,
    #[arg(long, default_value = "none")]
    pub self_learn_verifier: String,
    #[arg(long, default_value = "none")]
    pub self_learn_vision_verifier: String,
    #[arg(long)]
    pub self_learn_ground_truth: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

pub fn run(args: InferArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
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
    validate_speculative_args(&args, self_learn_mode)?;
    let tool_calling = load_tool_calling_config(&args, self_learn_mode)?;
    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 5,
        tool_calling,
        stop_sequences: Vec::new(),
        capture_steps: true,
    };

    let prompt = if config.tool_calling.is_some() {
        args.prompt.clone()
    } else {
        prompt_for_mode(&args.prompt, thinking_mode)
    };
    if args.speculative {
        return run_speculative_infer(
            &args,
            &run_config,
            model_path,
            tokenizer_path,
            device,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
        );
    }
    if self_learn_mode.is_enabled() {
        if let Some(image_path) = args.image.clone() {
            return run_vision_self_learn_infer(
                &args,
                run_config,
                run_device,
                model_path,
                tokenizer_path,
                image_path,
                device,
                dtype,
                config,
                prompt,
                safety_mode,
                self_learn_mode,
                thinking_mode,
            );
        }
        return run_self_learn_infer(
            &args,
            run_config,
            model_path,
            tokenizer_path,
            device,
            dtype,
            config,
            prompt,
            safety_mode,
            self_learn_mode,
            thinking_mode,
        );
    }

    let mut engine = InferenceEngine::from_paths_with_dtype(
        model_path,
        &run_config.model,
        tokenizer_path,
        device,
        dtype,
    )?;
    let tokenizer_for_view = engine.tokenizer().clone();
    if let Some(image_path) = args.image.clone() {
        return run_vision_infer(
            &args,
            &run_config,
            engine,
            image_path,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
            tokenizer_for_view,
        );
    }
    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(engine, policy);
        let mut stream_state = StreamState::default();
        let started = Instant::now();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        let elapsed = started.elapsed();
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
            if args.stats {
                print_generation_stats("target", output, elapsed, &run_config);
            }
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let started = Instant::now();
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
    let elapsed = started.elapsed();

    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    if args.stats {
        print_generation_stats("target", &output, elapsed, &run_config);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_infer(
    args: &InferArgs,
    target_config: &TrainingRunConfig,
    target_model: PathBuf,
    target_tokenizer: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    generation_config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    let draft_model = args
        .draft_model
        .as_ref()
        .expect("validated draft model")
        .clone();
    let draft_config_path = args.draft_config.as_ref().expect("validated draft config");
    let draft_config = TrainingRunConfig::from_toml(draft_config_path)?;
    let draft_tokenizer = args
        .draft_tokenizer
        .clone()
        .unwrap_or_else(|| target_tokenizer.clone());
    let speculative_config = SpeculativeConfig::new(args.draft_tokens)?;
    let mut engine = SpeculativeEngine::from_paths_with_dtype(
        target_model,
        &target_config.model,
        &target_tokenizer,
        draft_model,
        &draft_config.model,
        draft_tokenizer,
        device,
        dtype,
        speculative_config,
    )?;
    let tokenizer_for_view = engine.tokenizer().clone();

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(engine, policy);
        let mut stream_state = StreamState::default();
        let started = Instant::now();
        let response = if args.stream {
            guard.generate_streaming_with_callback(
                &prompt,
                generation_config,
                print_safe_stream_event,
            )?
        } else {
            guard.generate_with_callback(&prompt, generation_config, |step| {
                render_text_step(
                    args,
                    step,
                    thinking_mode,
                    &tokenizer_for_view,
                    &mut stream_state,
                )
            })?
        };
        let elapsed = started.elapsed();
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
            if args.stats {
                print_generation_stats("speculative", output, elapsed, target_config);
            }
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let started = Instant::now();
    let output = engine.generate_with_callback(&prompt, generation_config, |step| {
        render_text_step(
            args,
            step,
            thinking_mode,
            &tokenizer_for_view,
            &mut stream_state,
        )
    })?;
    let elapsed = started.elapsed();
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    if args.stats {
        print_generation_stats("speculative", &output, elapsed, target_config);
    }
    Ok(())
}

fn render_text_step(
    args: &InferArgs,
    step: &GenerationStep,
    thinking_mode: ThinkingMode,
    tokenizer: &BpeTokenizer,
    stream_state: &mut StreamState,
) -> aarambh_ai_core::Result<()> {
    if args.predict_view {
        print!(
            "{}",
            predict_view::render(step, tokenizer, args.temperature, args.top_p)
        );
    }
    if args.stream {
        stream_step(step, thinking_mode, stream_state)?;
    }
    if args.predict_view || args.stream {
        io::stdout().flush()?;
    }
    Ok(())
}

fn validate_speculative_args(
    args: &InferArgs,
    self_learn_mode: SelfLearnMode,
) -> anyhow::Result<()> {
    if !args.speculative {
        if args.draft_model.is_some()
            || args.draft_config.is_some()
            || args.draft_tokenizer.is_some()
        {
            return Err(
                AarambhError::Config("draft model options require --speculative".into()).into(),
            );
        }
        return Ok(());
    }
    if args.draft_model.is_none() {
        return Err(
            AarambhError::Config("--draft-model is required with --speculative".into()).into(),
        );
    }
    if args.draft_config.is_none() {
        return Err(
            AarambhError::Config("--draft-config is required with --speculative".into()).into(),
        );
    }
    if args.image.is_some() {
        return Err(AarambhError::Unsupported(
            "Phase 25 speculative decoding supports text inference only; --image is not supported"
                .into(),
        )
        .into());
    }
    if self_learn_mode.is_enabled() {
        return Err(AarambhError::Unsupported(
            "Phase 25 speculative decoding cannot be combined with --self-learn".into(),
        )
        .into());
    }
    SpeculativeConfig::new(args.draft_tokens)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolFile {
    Array(Vec<ToolEntry>),
    Object { tools: Vec<ToolEntry> },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolEntry {
    Native(ToolDefinition),
    OpenAi {
        r#type: String,
        function: ToolDefinition,
    },
}

fn load_tool_calling_config(
    args: &InferArgs,
    self_learn_mode: SelfLearnMode,
) -> anyhow::Result<Option<ToolCallingConfig>> {
    let Some(path) = &args.tools else {
        if !args.tool_choice.eq_ignore_ascii_case("auto") {
            return Err(AarambhError::Config("--tool-choice requires --tools".into()).into());
        }
        return Ok(None);
    };
    if args.image.is_some() {
        return Err(AarambhError::Unsupported(
            "Phase 26 tool calling supports text inference only; --image is not supported".into(),
        )
        .into());
    }
    if self_learn_mode.is_enabled() {
        return Err(AarambhError::Unsupported(
            "Phase 26 tool calling cannot be combined with --self-learn".into(),
        )
        .into());
    }
    let metadata = fs::metadata(path)?;
    if metadata.len() > 1024 * 1024 {
        return Err(AarambhError::Config(
            "tool definition file exceeds the 1 MiB request limit".into(),
        )
        .into());
    }
    let file = fs::File::open(path)?;
    let parsed: ToolFile = serde_json::from_reader(file)?;
    let entries = match parsed {
        ToolFile::Array(entries) | ToolFile::Object { tools: entries } => entries,
    };
    let definitions = entries
        .into_iter()
        .map(|entry| match entry {
            ToolEntry::Native(definition) => Ok(definition),
            ToolEntry::OpenAi { r#type, function } if r#type == "function" => Ok(function),
            ToolEntry::OpenAi {
                r#type: tool_type, ..
            } => Err(AarambhError::Config(format!(
                "unsupported OpenAI tool type {tool_type:?}; expected \"function\""
            ))),
        })
        .collect::<aarambh_ai_core::Result<Vec<_>>>()?;
    let choice = match args.tool_choice.trim() {
        value if value.eq_ignore_ascii_case("auto") => ToolChoice::Auto,
        value if value.eq_ignore_ascii_case("none") => ToolChoice::None,
        value if value.eq_ignore_ascii_case("required") => ToolChoice::Required,
        value => ToolChoice::Named(value.to_string()),
    };
    Ok(Some(ToolCallingConfig::new(definitions, choice)?))
}

fn print_generation_stats(
    mode: &str,
    output: &GenerationOutput,
    elapsed: Duration,
    run_config: &TrainingRunConfig,
) {
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let tokens_per_second = if elapsed.is_zero() {
        0.0
    } else {
        output.token_ids.len() as f64 / elapsed.as_secs_f64()
    };
    if let Some(stats) = &output.speculative_stats {
        eprintln!(
            "generation_stats mode={mode} tokens={} elapsed_ms={elapsed_ms:.3} tok_s={tokens_per_second:.3} target_decode_forwards={} draft_decode_forwards={} proposed={} accepted={} rejected={} acceptance_rate={:.4} accepted_per_target_forward={:.3}",
            output.token_ids.len(),
            stats.target_decode_forwards,
            stats.draft_decode_forwards,
            stats.draft_tokens_proposed,
            stats.draft_tokens_accepted,
            stats.draft_tokens_rejected,
            stats.acceptance_rate(),
            stats.accepted_tokens_per_target_forward(),
        );
    } else {
        eprintln!(
            "generation_stats mode={mode} tokens={} elapsed_ms={elapsed_ms:.3} tok_s={tokens_per_second:.3}",
            output.token_ids.len(),
        );
    }
    if let Some(dsa) = &run_config.model.dsa_config {
        let sparse_layers = (0..run_config.model.n_layers)
            .filter(|layer| {
                run_config.model.attention_kind_for_layer(*layer)
                    == aarambh_ai_core::AttentionKind::Sparse
            })
            .count();
        let seq_len = output.usage.total_tokens;
        let dtype_bytes = match run_config.dtype.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "bf16" => 2usize,
            _ => 4usize,
        };
        let kv_row_elements = 2 * run_config.model.n_kv_heads * run_config.model.head_dim();
        let stored_kv_bytes = sparse_layers * seq_len * kv_row_elements * dtype_bytes;
        let index_bytes = sparse_layers
            * seq_len.div_ceil(dsa.block_size)
            * run_config.model.head_dim()
            * std::mem::size_of::<f32>();
        let selected_tokens = seq_len.min(dsa.top_k_blocks * dsa.block_size);
        let selected_working_set_bytes =
            sparse_layers * selected_tokens * kv_row_elements * dtype_bytes;
        eprintln!(
            "dsa_cache_stats sparse_layers={sparse_layers} stored_cache_bytes={} selected_working_set_bytes={selected_working_set_bytes} selected_token_limit={selected_tokens}",
            stored_kv_bytes + index_bytes,
        );
    }
    if let Some(moe) = &run_config.model.moe
        && let (Ok(routed_experts), Ok(fine_dim), Ok(active_width)) = (
            moe.routed_expert_count(),
            moe.fine_grained_expert_dim(),
            moe.active_routed_width(),
        )
    {
        let moe_layers = (0..run_config.model.n_layers)
            .filter(|layer| moe.applies_to_layer(*layer))
            .count();
        let expert_params = 3u128 * run_config.model.hidden_dim as u128 * fine_dim as u128;
        let router_params = run_config.model.hidden_dim as u128 * routed_experts as u128;
        let total_params_per_layer =
            (routed_experts + moe.num_shared_experts) as u128 * expert_params + router_params;
        let active_params_per_token = (moe.top_k + moe.num_shared_experts) as u128 * expert_params;
        eprintln!(
            "moe_stats layers={moe_layers} routed_experts={routed_experts} active_routed={} shared_experts={} fine_dim={fine_dim} active_width={active_width} params_per_moe_layer={total_params_per_layer} active_expert_params_per_token={active_params_per_token}",
            moe.top_k, moe.num_shared_experts,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_vision_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    mut engine: InferenceEngine,
    image_path: PathBuf,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
    tokenizer_for_view: BpeTokenizer,
) -> anyhow::Result<()> {
    let runtime = load_vision_runtime(run_config, engine.device(), dtype)?;
    let prompt = ensure_image_prompt(&prompt);

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let adapter = VisionSafetyAdapter {
            engine,
            runtime,
            image_path,
        };
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let embeddings = build_vision_prompt_embeddings(&engine, &runtime, &image_path, &prompt)?;
    let mut stream_state = StreamState::default();
    let output = engine.generate_with_embeddings_callback(&embeddings, config, |step| {
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

struct VisionRuntime {
    model: VisionModel,
    preprocess: ImagePreprocessor,
    cache_salt: String,
}

fn load_vision_runtime(
    run_config: &TrainingRunConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<VisionRuntime> {
    let vision = run_config
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--image requires a [vision] config block"))?;
    let encoder_config = VisionEncoderConfig::from_json(&vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &vision.clip_weights_path,
        encoder_config.clone(),
        device,
        dtype,
    )?;
    let projector_path = match &vision.projector_path {
        Some(path) => path.clone(),
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let projector_config = ProjectorConfig {
        vit_d_model: encoder_config.vit_d_model,
        llm_d_model: run_config.model.hidden_dim,
        hidden_mult: vision.projector_hidden_mult,
    };
    let projector =
        VisionProjector::load_safetensors(&projector_path, projector_config, device, dtype)?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let cache_salt = format!(
        "clip_config={};clip_weights={};projector={};hidden_mult={};llm_hidden={}",
        vision.clip_config_path.display(),
        vision.clip_weights_path.display(),
        projector_path.display(),
        vision.projector_hidden_mult,
        run_config.model.hidden_dim
    );
    Ok(VisionRuntime {
        model: VisionModel::new(encoder, projector),
        preprocess,
        cache_salt,
    })
}

fn ensure_image_prompt(prompt: &str) -> String {
    if prompt.contains(IMAGE) {
        prompt.to_string()
    } else {
        format!("{IMAGE}{IMAGE_END}\n{prompt}")
    }
}

fn build_vision_prompt_embeddings(
    engine: &InferenceEngine,
    runtime: &VisionRuntime,
    image_path: &Path,
    prompt: &str,
) -> aarambh_ai_core::Result<Tensor> {
    engine.tokenizer().validate_vision_special_tokens()?;
    let mut prompt_ids = engine.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = engine.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Config(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), engine.device())?;
    let text_embeddings = engine.model().embed_tokens(&text)?;
    let image_tokens = project_image_tokens(runtime, image_path, engine.device())?;
    interleave_image_tokens(&prompt_ids, &text_embeddings, &image_tokens, IMAGE_ID)
}

fn project_image_tokens(
    runtime: &VisionRuntime,
    image_path: &Path,
    device: &candle_core::Device,
) -> aarambh_ai_core::Result<Tensor> {
    let image = runtime
        .preprocess
        .preprocess_path(image_path, device)?
        .unsqueeze(0)?;
    runtime.model.forward(&image)
}

struct VisionSafetyAdapter {
    engine: InferenceEngine,
    runtime: VisionRuntime,
    image_path: PathBuf,
}

impl SafetyGenerator for VisionSafetyAdapter {
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
        let embeddings =
            build_vision_prompt_embeddings(&self.engine, &self.runtime, &self.image_path, prompt)?;
        self.engine
            .generate_with_embeddings_callback(&embeddings, config, on_step)
    }
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

fn parse_vision_verifier(value: &str) -> anyhow::Result<VisionVerifierKind> {
    value
        .parse::<VisionVerifierKind>()
        .map_err(anyhow::Error::msg)
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
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    self_learn_mode: SelfLearnMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    let replay_path = args
        .replay_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/replay.jsonl"));
    let mut self_config = SelfLearnConfig::for_mode(self_learn_mode)
        .with_replay_path(replay_path)
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
        dtype,
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
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
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

#[allow(clippy::too_many_arguments)]
fn run_vision_self_learn_infer(
    args: &InferArgs,
    mut run_config: TrainingRunConfig,
    run_device: aarambh_ai_core::Device,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    image_path: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    self_learn_mode: SelfLearnMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    require_vision_hardware(&run_device)?;
    if self_learn_mode != SelfLearnMode::Gpu {
        return Err(anyhow::anyhow!(
            "vision self-learning requires --self-learn gpu"
        ));
    }
    let replay_path = args
        .replay_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/replay_buffer_v2.jsonl"));
    let mut self_config = SelfLearnConfig::for_mode(self_learn_mode)
        .with_replay_path(replay_path)
        .with_state_dir(args.self_learn_state_dir.clone());
    self_config.grpo.max_new_tokens = args.max_tokens;
    self_config.critique.rewrite_max_tokens = self_config
        .critique
        .rewrite_max_tokens
        .min(args.max_tokens)
        .max(1);

    let runtime = load_vision_runtime(&run_config, &device, dtype)?;
    let prompt = ensure_image_prompt(&prompt);
    let cache = VisionCache::new(&args.self_learn_state_dir);
    let image_ref = cache.image_ref(&image_path, &runtime.cache_salt)?;
    let image_tokens = match cache.load_projected_tokens(&image_ref, &device)? {
        Some(tokens) => tokens,
        None => {
            let tokens = project_image_tokens(&runtime, &image_path, &device)?;
            cache.save_projected_tokens(&image_ref, &tokens)?;
            tokens
        }
    };

    let reference_path = args
        .self_learn_reference
        .clone()
        .unwrap_or_else(|| model_path.clone());
    let vision_verifier_kind =
        parse_vision_verifier(&args.self_learn_vision_verifier)?.resolve_for_prompt(&prompt);
    let verifier = if vision_verifier_kind == VisionVerifierKind::None {
        None
    } else {
        if args.self_learn_ground_truth.is_none() {
            eprintln!(
                "[self-learn] vision verifier requested without --self-learn-ground-truth; grounded vision GRPO will be skipped"
            );
        }
        vision_verifier_kind
            .build()
            .map(|verifier| Box::new(verifier) as Box<dyn Verifier>)
    };

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
        dtype,
        seed: run_config.train.seed,
    })?;
    let mut adapter = VisionSelfLearnSafetyAdapter {
        loop_,
        image_tokens,
        image_ref,
        verifier,
        ground_truth: args.self_learn_ground_truth.clone(),
    };

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
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

struct VisionSelfLearnSafetyAdapter {
    loop_: SelfLearnLoop,
    image_tokens: Tensor,
    image_ref: PathBuf,
    verifier: Option<Box<dyn Verifier>>,
    ground_truth: Option<String>,
}

impl SafetyGenerator for VisionSelfLearnSafetyAdapter {
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
        self.loop_.generate_vision_draft_with_callback(
            prompt,
            &self.image_tokens,
            self.image_ref.clone(),
            config,
            self.verifier.as_deref(),
            self.ground_truth.as_deref(),
            on_step,
        )
    }
}

fn print_self_learn_summary(response: &aarambh_ai_selflearn::SelfLearnResponse) {
    eprintln!(
        "[self-learn] critique_score={:.2} stored={} rewritten={} grpo={} image_ref={} metrics={}",
        response.critique_score,
        response.stored_in_replay,
        response.was_rewritten,
        response.used_grpo,
        response
            .image_ref
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".into()),
        response.metrics_summary
    );
}

#[derive(Default)]
struct StreamState {
    dim_active: bool,
    header_printed: bool,
    thinking_tokens: usize,
    tool_buffer: String,
}

fn stream_step(
    step: &GenerationStep,
    _thinking_mode: ThinkingMode,
    state: &mut StreamState,
) -> io::Result<()> {
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
        GenerationPhase::ToolCall => state.tool_buffer.push_str(&step.token_text),
        GenerationPhase::Control => {}
    }
    Ok(())
}

fn finish_stream(state: &mut StreamState) {
    if state.dim_active {
        println!("{ANSI_RESET}");
        println!("[thinking: {} tokens]", state.thinking_tokens);
        state.dim_active = false;
    }
    if !state.tool_buffer.is_empty() {
        print!("{}", state.tool_buffer);
        state.tool_buffer.clear();
    }
    println!();
}

fn print_safe_stream_event(event: SafeStreamEvent) -> aarambh_ai_core::Result<()> {
    if let SafeStreamEvent::Text(text) = event {
        print!("{text}");
        io::stdout().flush()?;
    }
    Ok(())
}

fn print_safe_response(
    response: &SafeResponse,
    thinking_mode: ThinkingMode,
    stream: bool,
    stream_state: &mut StreamState,
) -> io::Result<()> {
    if stream {
        finish_stream(stream_state);
        if let SafetyVerdict::Block(reason) = &response.verdict {
            println!("blocked by safety: {reason}");
        }
        return Ok(());
    }
    if let SafetyVerdict::Block(reason) = &response.verdict {
        println!("blocked by safety: {reason}");
        return Ok(());
    }

    let Some(output) = &response.output else {
        println!("blocked by safety");
        return Ok(());
    };

    print_generation_output(output, thinking_mode)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> InferArgs {
        InferArgs {
            config: "target.toml".into(),
            model: Some("target.safetensors".into()),
            tokenizer: Some("tokenizer.json".into()),
            image: None,
            prompt: "test".into(),
            max_tokens: 8,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: Some(42),
            thinking: "none".into(),
            predict_view: false,
            stream: false,
            greedy: true,
            speculative: true,
            draft_model: Some("draft.safetensors".into()),
            draft_config: Some("draft.toml".into()),
            draft_tokenizer: None,
            draft_tokens: 4,
            stats: false,
            tools: None,
            tool_choice: "auto".into(),
            safety: "none".into(),
            safety_audit_log: "safety.jsonl".into(),
            self_learn: "disabled".into(),
            replay_path: None,
            self_learn_state_dir: "adapters/selflearn".into(),
            self_learn_reference: None,
            self_learn_verifier: "none".into(),
            self_learn_vision_verifier: "none".into(),
            self_learn_ground_truth: None,
        }
    }

    #[test]
    fn speculative_cli_requires_draft_model_and_config() {
        let mut args = args();
        args.draft_model = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled).is_err());
        args.draft_model = Some("draft.safetensors".into());
        args.draft_config = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled).is_err());
    }

    #[test]
    fn speculative_cli_rejects_unsupported_modes() {
        let mut args = args();
        args.image = Some("image.png".into());
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled).is_err());
        args.image = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Cpu).is_err());
    }

    #[test]
    fn draft_options_require_speculative_flag() {
        let mut args = args();
        args.speculative = false;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled).is_err());
        args.draft_model = None;
        args.draft_config = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled).is_ok());
    }
}
