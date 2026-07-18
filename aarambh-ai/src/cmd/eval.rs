use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_eval::{
    EvalConfig, EvalContext, QatRobustnessReport, Scorecard, ScorecardComparison, run_all,
};
use aarambh_ai_quant::GgufFormat;
use aarambh_ai_tokenizer::BpeTokenizer;
use aarambh_ai_train::TrainingRunConfig;
use clap::Args;
use serde::Deserialize;

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "ppl")]
    pub tasks: String,
    #[arg(long, default_value = "data/eval")]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub max_examples: Option<usize>,
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    #[arg(long)]
    pub allow_code_exec: bool,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub markdown: Option<PathBuf>,
    #[arg(long, num_args = 2)]
    pub compare: Vec<PathBuf>,
    #[arg(long)]
    pub qat_compare: bool,
    #[arg(long, requires = "qat_compare")]
    pub baseline_model: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

pub fn run(args: EvalArgs) -> anyhow::Result<()> {
    if !args.compare.is_empty() {
        return run_compare(&args);
    }
    if args.qat_compare {
        return run_qat_compare(&args);
    }

    let config_path = args
        .config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--config is required unless --compare is used"))?;
    let run_config = TrainingRunConfig::from_toml(config_path)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let tokenizer_path = tokenizer_path(&args, &run_config);
    let model_path = match args.model.clone() {
        Some(path) => path,
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };

    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    let model =
        aarambh_ai_weights::load_any_model_with_dtype(&model_path, &model_config, &device, dtype)?;
    let context = EvalContext::new(model, tokenizer, device, dtype);
    let eval_config = EvalConfig {
        tasks: parse_tasks(&args.tasks),
        data_dir: args.data_dir.clone(),
        max_examples: args.max_examples,
        max_new_tokens: args.max_new_tokens,
        allow_code_exec: args.allow_code_exec,
        model_path: Some(model_path.display().to_string()),
        tokenizer_path: Some(tokenizer_path.display().to_string()),
        config_path: Some(config_path.display().to_string()),
    };

    let scorecard = run_all(&context, &eval_config)?;
    write_outputs(
        &scorecard.to_json()?,
        &scorecard.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

fn run_qat_compare(args: &EvalArgs) -> anyhow::Result<()> {
    let config_path = args
        .config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--config is required for --qat-compare"))?;
    let qat_path = args
        .model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--model is required for --qat-compare"))?;
    let baseline_path = args
        .baseline_model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--baseline-model is required for --qat-compare"))?;
    let run_config = TrainingRunConfig::from_toml(config_path)?;
    let qat = run_config.model.qat.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--qat-compare requires [model.qat] in the training config")
    })?;
    let format = match qat.bits.bits() {
        4 => GgufFormat::Q4KM,
        8 => GgufFormat::Q80,
        bits => return Err(anyhow::anyhow!("unsupported QAT comparison width {bits}")),
    };
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let tokenizer_path = tokenizer_path(args, &run_config);
    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();

    let temporary = TemporaryQatExports::new(format);
    export_for_comparison(
        baseline_path,
        &temporary.baseline,
        &model_config,
        &device,
        dtype,
        format,
    )?;
    export_for_comparison(
        qat_path,
        &temporary.qat,
        &model_config,
        &device,
        dtype,
        format,
    )?;

    let baseline_fp = evaluate_checkpoint(
        baseline_path,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let baseline_quantized = evaluate_checkpoint(
        &temporary.baseline,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let qat_fp = evaluate_checkpoint(
        qat_path,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let qat_quantized = evaluate_checkpoint(
        &temporary.qat,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let report =
        QatRobustnessReport::compare(baseline_fp, baseline_quantized, qat_fp, qat_quantized);
    write_outputs(
        &report.to_json()?,
        &report.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_checkpoint(
    model_path: &Path,
    model_config: &aarambh_ai_core::ModelConfig,
    tokenizer: &BpeTokenizer,
    tokenizer_path: &Path,
    config_path: &Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    args: &EvalArgs,
) -> anyhow::Result<Scorecard> {
    let model =
        aarambh_ai_weights::load_any_model_with_dtype(model_path, model_config, device, dtype)?;
    let context = EvalContext::new(model, tokenizer.clone(), device.clone(), dtype);
    run_all(
        &context,
        &EvalConfig {
            tasks: parse_tasks(&args.tasks),
            data_dir: args.data_dir.clone(),
            max_examples: args.max_examples,
            max_new_tokens: args.max_new_tokens,
            allow_code_exec: args.allow_code_exec,
            model_path: Some(model_path.display().to_string()),
            tokenizer_path: Some(tokenizer_path.display().to_string()),
            config_path: Some(config_path.display().to_string()),
        },
    )
    .map_err(Into::into)
}

fn export_for_comparison(
    source: &Path,
    output: &Path,
    model_config: &aarambh_ai_core::ModelConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    format: GgufFormat,
) -> anyhow::Result<()> {
    let model = aarambh_ai_weights::load_any_model_with_dtype(source, model_config, device, dtype)?;
    aarambh_ai_weights::save_gguf(&model, format, output)?;
    Ok(())
}

struct TemporaryQatExports {
    baseline: PathBuf,
    qat: PathBuf,
}

impl TemporaryQatExports {
    fn new(format: GgufFormat) -> Self {
        let suffix = match format {
            GgufFormat::Q80 => "q8",
            _ => "q4",
        };
        let nonce = format!("{}_{}", std::process::id(), suffix);
        Self {
            baseline: std::env::temp_dir().join(format!("aarambh_qat_baseline_{nonce}.gguf")),
            qat: std::env::temp_dir().join(format!("aarambh_qat_model_{nonce}.gguf")),
        }
    }
}

impl Drop for TemporaryQatExports {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.baseline);
        let _ = fs::remove_file(&self.qat);
    }
}

fn run_compare(args: &EvalArgs) -> anyhow::Result<()> {
    if args.compare.len() != 2 {
        return Err(anyhow::anyhow!(
            "--compare expects exactly two scorecard paths"
        ));
    }
    let before = read_scorecard(&args.compare[0])?;
    let after = read_scorecard(&args.compare[1])?;
    let comparison = ScorecardComparison::compare(&before, &after);
    write_outputs(
        &comparison.to_json()?,
        &comparison.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

fn tokenizer_path(args: &EvalArgs, run_config: &TrainingRunConfig) -> PathBuf {
    args.tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

fn default_model_path(checkpoint_dir: &Path) -> anyhow::Result<PathBuf> {
    for pointer_name in ["best.json", "latest.json"] {
        let pointer_path = checkpoint_dir.join(pointer_name);
        if pointer_path.exists() {
            let file = fs::File::open(&pointer_path)?;
            let pointer: CheckpointPointer = serde_json::from_reader(file)?;
            return Ok(pointer.path.join("model.safetensors"));
        }
    }
    Err(anyhow::anyhow!(
        "no model provided and no best.json or latest.json found in {}",
        checkpoint_dir.display()
    ))
}

fn parse_tasks(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn read_scorecard(path: &Path) -> anyhow::Result<Scorecard> {
    let file = fs::File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn write_outputs(
    json: &str,
    markdown: &str,
    out: Option<&Path>,
    markdown_out: Option<&Path>,
) -> anyhow::Result<()> {
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    if let Some(path) = markdown_out {
        fs::write(path, markdown)?;
    }
    if out.is_none() && markdown_out.is_none() {
        println!("{markdown}");
    }
    Ok(())
}
