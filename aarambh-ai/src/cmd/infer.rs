use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use aarambh_ai_inference::{GenerationConfig, InferenceEngine, Sampler, ThinkingMode};
use aarambh_ai_train::TrainingRunConfig;
use clap::Args;
use serde::Deserialize;

use crate::ui::predict_view;

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
    if thinking_mode.is_enabled() {
        eprintln!(
            "thinking mode is accepted in Phase 6, but token forcing is implemented in Phase 7"
        );
    }

    let mut engine =
        InferenceEngine::from_paths(model_path, &run_config.model, tokenizer_path, device)?;
    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 5,
    };

    let tokenizer_for_view = engine.tokenizer().clone();
    let output = engine.generate_with_callback(&args.prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            print!("{}", step.token_text);
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;

    if args.stream {
        println!();
    } else {
        println!("{}", output.text);
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
