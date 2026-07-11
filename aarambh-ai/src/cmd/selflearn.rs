use std::path::PathBuf;

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_selflearn::{
    ReplayBuffer, SelfLearnBuildConfig, SelfLearnConfig, SelfLearnLoop, SelfLearnMode, VisionCache,
};
use aarambh_ai_tokenizer::BpeTokenizer;
use aarambh_ai_train::TrainingRunConfig;
use clap::{Args, Subcommand};

use crate::cmd::infer::{self, InferArgs};

#[derive(Debug, Args)]
pub struct SelflearnArgs {
    #[command(subcommand)]
    pub command: SelflearnCommand,
}

#[derive(Debug, Subcommand)]
pub enum SelflearnCommand {
    Start(Box<StartArgs>),
    FlushGradients(SelflearnRunArgs),
    Replay(SelflearnRunArgs),
    Stats(StatsArgs),
    Reset(ResetArgs),
}

#[derive(Debug, Args)]
pub struct StartArgs {
    #[arg(long, default_value = "text")]
    pub mode: String,
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
    #[arg(long)]
    pub stream: bool,
    #[arg(long)]
    pub greedy: bool,
    #[arg(long, default_value = "strict")]
    pub safety: String,
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    #[arg(long)]
    pub replay_path: Option<PathBuf>,
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    #[arg(long)]
    pub self_learn_reference: Option<PathBuf>,
    #[arg(long, default_value = "none")]
    pub self_learn_verifier: String,
    #[arg(long, default_value = "auto")]
    pub self_learn_vision_verifier: String,
    #[arg(long)]
    pub self_learn_ground_truth: Option<String>,
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
    #[arg(long, default_value = "text")]
    pub mode: String,
    #[arg(long)]
    pub replay_path: Option<PathBuf>,
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
        SelflearnCommand::Start(args) => run_start(*args),
        SelflearnCommand::FlushGradients(args) => run_flush(args),
        SelflearnCommand::Replay(args) => run_replay(args),
        SelflearnCommand::Stats(args) => run_stats(args),
        SelflearnCommand::Reset(args) => run_reset(args),
    }
}

fn run_start(args: StartArgs) -> anyhow::Result<()> {
    let mode = args.mode.trim().to_ascii_lowercase();
    let self_learn = match mode.as_str() {
        "text" => "cpu".to_string(),
        "vision" => {
            if args.image.is_none() {
                return Err(anyhow::anyhow!(
                    "selflearn start --mode vision requires --image"
                ));
            }
            "gpu".to_string()
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported selflearn start mode '{other}', expected text|vision"
            ));
        }
    };
    let replay_path = args
        .replay_path
        .or_else(|| (mode == "vision").then(|| PathBuf::from("data/replay_buffer_v2.jsonl")));
    infer::run(InferArgs {
        config: args.config,
        model: args.model,
        tokenizer: args.tokenizer,
        image: args.image,
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        thinking: "none".into(),
        predict_view: false,
        stream: args.stream,
        greedy: args.greedy,
        speculative: false,
        draft_model: None,
        draft_config: None,
        draft_tokenizer: None,
        draft_tokens: 4,
        stats: false,
        safety: args.safety,
        safety_audit_log: args.safety_audit_log,
        self_learn,
        replay_path,
        self_learn_state_dir: args.self_learn_state_dir,
        self_learn_reference: args.self_learn_reference,
        self_learn_verifier: args.self_learn_verifier,
        self_learn_vision_verifier: args.self_learn_vision_verifier,
        self_learn_ground_truth: args.self_learn_ground_truth,
    })
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
    let vision_mode = matches!(args.mode.trim().to_ascii_lowercase().as_str(), "vision");
    let replay_path = args.replay_path.unwrap_or_else(|| {
        if vision_mode {
            PathBuf::from("data/replay_buffer_v2.jsonl")
        } else {
            PathBuf::from("data/replay.jsonl")
        }
    });
    let config = if vision_mode {
        SelfLearnConfig::for_gpu()
    } else {
        SelfLearnConfig::for_cpu()
    }
    .with_replay_path(replay_path.clone());
    let replay = ReplayBuffer::load_jsonl(&replay_path, config.replay)?;
    let stats = replay.stats();
    let vision_entries = replay
        .entries()
        .iter()
        .filter(|entry| entry.is_vision())
        .count();
    println!(
        "Replay buffer: {} / {} entries  avg score: {:.2}",
        stats.len, stats.capacity, stats.avg_score
    );
    if vision_mode {
        let cache = VisionCache::new(&args.self_learn_state_dir);
        println!(
            "Vision entries: {}  cached image-token files: {}",
            vision_entries,
            cache.cached_entry_count()
        );
    }
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
