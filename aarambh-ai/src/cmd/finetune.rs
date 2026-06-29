use std::path::PathBuf;
use std::str::FromStr;

use aarambh_ai_finetune::{
    GrpoConfig, GrpoRunConfig, GrpoThinkingMode, LoraConfig, SftRunConfig, VerifierKind,
    merge_lora_from_paths, run_grpo_from_config, run_sft_from_config,
};
use aarambh_ai_train::TrainingRunConfig;
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct FinetuneArgs {
    #[command(subcommand)]
    pub command: FinetuneCommand,
}

#[derive(Debug, Subcommand)]
pub enum FinetuneCommand {
    Sft(FinetuneRunArgs),
    Qlora(FinetuneRunArgs),
    Grpo(GrpoArgs),
    Merge(MergeArgs),
}

#[derive(Debug, Args)]
pub struct FinetuneRunArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,
    #[arg(long, default_value = "attn.wq,attn.wk,attn.wv,attn.wo")]
    pub target_modules: String,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long)]
    pub max_steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
pub struct MergeArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub adapter: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Debug, Args)]
pub struct GrpoArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub reference: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value = "math-format")]
    pub verifier: String,
    #[arg(long, default_value_t = 8)]
    pub group_size: usize,
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    #[arg(long, default_value = "low")]
    pub thinking: String,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,
    #[arg(long, default_value = "attn.wq,attn.wk,attn.wv,attn.wo")]
    pub target_modules: String,
    #[arg(long, alias = "max-steps")]
    pub steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long, default_value_t = 0.01)]
    pub kl_coeff: f64,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

pub fn run(args: FinetuneArgs) -> anyhow::Result<()> {
    match args.command {
        FinetuneCommand::Sft(args) => run_finetune(args, false),
        FinetuneCommand::Qlora(args) => run_finetune(args, true),
        FinetuneCommand::Grpo(args) => run_grpo(args),
        FinetuneCommand::Merge(args) => run_merge(args),
    }
}

fn run_finetune(args: FinetuneRunArgs, qlora: bool) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    apply_train_overrides(&mut train_config, &args);
    train_config.checkpoint_dir = args.output.clone();

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };

    let config = SftRunConfig {
        model_config: run_config.model.clone(),
        train_config,
        base_model_path: args.base,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        device,
        qlora,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_sft_from_config(config)?;
    Ok(())
}

fn run_grpo(args: GrpoArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let verifier = VerifierKind::from_str(&args.verifier).map_err(anyhow::Error::msg)?;
    let thinking = GrpoThinkingMode::from_str(&args.thinking).map_err(anyhow::Error::msg)?;

    let mut train_config = run_config.train.clone();
    train_config.checkpoint_dir = args.output.clone();
    train_config.batch_size = 1;
    train_config.lr = args.lr.unwrap_or(1e-5);
    train_config.max_epochs = args.max_epochs.unwrap_or(usize::MAX);
    if let Some(value) = args.steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };
    let grpo_config = GrpoConfig {
        group_size: args.group_size,
        kl_coeff: args.kl_coeff,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: (args.top_p > 0.0 && args.top_p < 1.0).then_some(args.top_p),
        top_k: (args.top_k > 0).then_some(args.top_k),
        thinking,
    };

    let config = GrpoRunConfig {
        model_config: run_config.model,
        train_config,
        grpo_config,
        base_model_path: args.base,
        reference_model_path: args.reference,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        verifier,
        device,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_grpo_from_config(config)?;
    Ok(())
}

fn run_merge(args: MergeArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    let output = merge_lora_from_paths(
        &run_config.model,
        args.base,
        args.adapter,
        args.output,
        &device,
    )?;
    eprintln!("merged LoRA adapter written to {}", output.display());
    Ok(())
}

fn tokenizer_path(tokenizer: Option<&PathBuf>, run_config: &TrainingRunConfig) -> PathBuf {
    tokenizer
        .cloned()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

fn apply_train_overrides(train_config: &mut aarambh_ai_core::TrainConfig, args: &FinetuneRunArgs) {
    if let Some(value) = args.batch_size {
        train_config.batch_size = value;
    }
    if let Some(value) = args.max_steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.max_epochs {
        train_config.max_epochs = value;
    }
    if let Some(value) = args.lr {
        train_config.lr = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }
}
