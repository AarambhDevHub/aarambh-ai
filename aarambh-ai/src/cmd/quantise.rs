use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_data::dataset::PlaintextDataset;
use aarambh_ai_quant::{GgufFormat, QuantMethod, run_calibration};
use aarambh_ai_tokenizer::BpeTokenizer;
use aarambh_ai_train::TrainingRunConfig;
use aarambh_ai_weights::{load_any_model, save_gguf};
use clap::Args;

#[derive(Debug, Args)]
pub struct QuantiseArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "int8")]
    pub method: String,
    #[arg(long, default_value_t = 8)]
    pub bits: u8,
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    pub samples: usize,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(args: QuantiseArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    let method = QuantMethod::from_name(&args.method)?;
    let format = format_for(method, args.bits)?;

    let mut model_config = run_config.model.clone();
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    if tokenizer_path.exists() {
        let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
        tokenizer.validate_special_tokens()?;
        model_config.vocab_size = tokenizer.vocab_size();

        if matches!(method, QuantMethod::AwqInt4 | QuantMethod::GptqInt4) {
            let calibration_path = args.calibration_data.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--calibration-data is required for {} quantisation",
                    args.method
                )
            })?;
            let dataset = PlaintextDataset::from_file(calibration_path)?;
            let model = load_any_model(&args.model, &model_config, &device)?;
            let stats = run_calibration(
                &model,
                &tokenizer,
                &dataset,
                args.samples,
                model_config.max_seq_len,
                &device,
                matches!(method, QuantMethod::GptqInt4),
            )?;
            eprintln!(
                "calibration: {} layers from up to {} samples",
                stats.layer_names().len(),
                args.samples
            );
            write_parent_dir(&args.output)?;
            save_gguf(&model, format, &args.output)?;
            eprintln!(
                "quantised {:?} checkpoint written to {}",
                format,
                args.output.display()
            );
            return Ok(());
        }
    } else if matches!(method, QuantMethod::AwqInt4 | QuantMethod::GptqInt4) {
        return Err(anyhow::anyhow!(
            "tokenizer {} is required for calibration",
            tokenizer_path.display()
        ));
    }

    let model = load_any_model(&args.model, &model_config, &device)?;
    write_parent_dir(&args.output)?;
    save_gguf(&model, format, &args.output)?;
    eprintln!(
        "quantised {:?} checkpoint written to {}",
        format,
        args.output.display()
    );
    Ok(())
}

fn format_for(method: QuantMethod, bits: u8) -> anyhow::Result<GgufFormat> {
    match (method, bits) {
        (QuantMethod::Int8Absmax | QuantMethod::Q80, 8) => Ok(GgufFormat::Q80),
        (QuantMethod::AwqInt4 | QuantMethod::GptqInt4 | QuantMethod::Q4KM, 4) => {
            Ok(GgufFormat::Q4KM)
        }
        (QuantMethod::Q5KM, 5) => Ok(GgufFormat::Q5KM),
        _ => Err(anyhow::anyhow!(
            "invalid method/bits combination: method={method:?}, bits={bits}; use int8/8 or awq|gptq/4"
        )),
    }
}

fn write_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
