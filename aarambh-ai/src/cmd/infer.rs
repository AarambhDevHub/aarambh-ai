use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use aarambh_ai_inference::{
    GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep, InferenceEngine, Sampler,
    ThinkingMode,
};
use aarambh_ai_tokenizer::{ASSISTANT, THINK_END_ID, THINK_START_ID, USER};
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

    let mut engine =
        InferenceEngine::from_paths(model_path, &run_config.model, tokenizer_path, device)?;
    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 5,
    };

    let prompt = prompt_for_mode(&args.prompt, thinking_mode);
    let tokenizer_for_view = engine.tokenizer().clone();
    let mut stream_dim_active = false;
    let mut stream_header_printed = false;
    let mut stream_thinking_tokens = 0usize;
    let output = engine.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(
                step,
                thinking_mode,
                &mut stream_dim_active,
                &mut stream_header_printed,
                &mut stream_thinking_tokens,
            )?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;

    if args.stream {
        if stream_dim_active {
            println!("{ANSI_RESET}");
            println!("[thinking: {stream_thinking_tokens} tokens]");
        }
        println!();
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

fn prompt_for_mode(prompt: &str, thinking_mode: ThinkingMode) -> String {
    if thinking_mode.is_enabled() {
        format!("{USER}\n{prompt}\n{ASSISTANT}\n")
    } else {
        prompt.to_string()
    }
}

fn stream_step(
    step: &GenerationStep,
    thinking_mode: ThinkingMode,
    dim_active: &mut bool,
    header_printed: &mut bool,
    thinking_tokens: &mut usize,
) -> io::Result<()> {
    if !thinking_mode.is_enabled() {
        print!("{}", step.token_text);
        return Ok(());
    }

    match step.phase {
        GenerationPhase::Thinking => {
            if !*header_printed {
                print!("[thinking]\n{ANSI_DIM}");
                *header_printed = true;
                *dim_active = true;
            }
            if !is_thinking_marker(step.token_id) {
                *thinking_tokens += 1;
                print!("{}", step.token_text);
            }
        }
        GenerationPhase::Answer => {
            if *dim_active {
                println!("{ANSI_RESET}");
                println!("[thinking: {thinking_tokens} tokens]");
                *dim_active = false;
            }
            print!("{}", step.token_text);
        }
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
