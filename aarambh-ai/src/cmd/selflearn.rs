use std::path::PathBuf;

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_selflearn::{
    ReplayBuffer, SelfLearnBuildConfig, SelfLearnConfig, SelfLearnLoop, SelfLearnMode,
};
use aarambh_ai_tokenizer::BpeTokenizer;
use aarambh_ai_train::TrainingRunConfig;
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct SelflearnArgs {
    #[command(subcommand)]
    pub command: SelflearnCommand,
}

#[derive(Debug, Subcommand)]
pub enum SelflearnCommand {
    FlushGradients(SelflearnRunArgs),
    Replay(SelflearnRunArgs),
    Stats(StatsArgs),
    Reset(ResetArgs),
}

#[derive(Debug, Args)]
pub struct SelflearnRunArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub reference: Option<PathBuf>,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "cpu")]
    pub mode: String,
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
}

#[derive(Debug, Args)]
pub struct StatsArgs {
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
}

#[derive(Debug, Args)]
pub struct ResetArgs {
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    #[arg(long)]
    pub yes: bool,
}

pub fn run(args: SelflearnArgs) -> anyhow::Result<()> {
    match args.command {
        SelflearnCommand::FlushGradients(args) => run_flush(args),
        SelflearnCommand::Replay(args) => run_replay(args),
        SelflearnCommand::Stats(args) => run_stats(args),
        SelflearnCommand::Reset(args) => run_reset(args),
    }
}

fn run_flush(args: SelflearnRunArgs) -> anyhow::Result<()> {
    let mut loop_ = build_loop(args)?;
    match loop_.flush_pending_gradients()? {
        Some(norm) => println!("flushed pending self-learning gradients grad_norm={norm:.4}"),
        None => println!("no pending self-learning gradients to flush"),
    }
    Ok(())
}

fn run_replay(args: SelflearnRunArgs) -> anyhow::Result<()> {
    let mut loop_ = build_loop(args)?;
    match loop_.replay_finetune()? {
        Some(norm) => println!("self-learning replay fine-tune completed grad_norm={norm:.4}"),
        None => println!("no replay entries available for self-learning replay"),
    }
    Ok(())
}

fn run_stats(args: StatsArgs) -> anyhow::Result<()> {
    let config = SelfLearnConfig::for_cpu().with_replay_path(args.replay_path.clone());
    let replay = ReplayBuffer::load_jsonl(&args.replay_path, config.replay)?;
    let stats = replay.stats();
    println!(
        "Replay buffer: {} / {} entries  avg score: {:.2}",
        stats.len, stats.capacity, stats.avg_score
    );
    let mut topics = stats.topics.into_iter().collect::<Vec<_>>();
    topics.sort_by(|a, b| a.0.cmp(&b.0));
    for (topic, count) in topics {
        println!("{topic}: {count}");
    }
    let metrics = aarambh_ai_selflearn::LearningMetrics::load_jsonl(
        args.self_learn_state_dir.join("metrics.jsonl"),
    )?;
    println!("{}", metrics.summary());
    Ok(())
}

fn run_reset(args: ResetArgs) -> anyhow::Result<()> {
    if !args.yes {
        return Err(anyhow::anyhow!(
            "reset requires --yes because it deletes replay and self-learning state"
        ));
    }
    if args.replay_path.exists() {
        std::fs::remove_file(&args.replay_path)?;
    }
    if args.self_learn_state_dir.exists() {
        std::fs::remove_dir_all(&args.self_learn_state_dir)?;
    }
    println!("self-learning state reset");
    Ok(())
}

fn build_loop(args: SelflearnRunArgs) -> anyhow::Result<SelfLearnLoop> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let mode = args
        .mode
        .parse::<SelfLearnMode>()
        .map_err(anyhow::Error::msg)?;
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    let config = SelfLearnConfig::for_mode(mode)
        .with_replay_path(args.replay_path)
        .with_state_dir(args.self_learn_state_dir);
    SelfLearnLoop::from_paths(SelfLearnBuildConfig {
        model_config,
        base_model_path: args.base.clone(),
        reference_model_path: args.reference.unwrap_or(args.base),
        tokenizer_path,
        config,
        device,
        dtype,
        seed: run_config.train.seed,
    })
    .map_err(anyhow::Error::from)
}
