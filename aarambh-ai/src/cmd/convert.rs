use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_train::TrainingRunConfig;
use aarambh_ai_weights::{
    GgufFormat, HfArch, convert_hf_with_arch, load_any_model, save_gguf, save_model,
};
use clap::Args;

#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value = "llama3")]
    pub arch: String,
    #[arg(long)]
    pub gguf: bool,
    #[arg(long, default_value = "q4_k_m")]
    pub format: String,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    write_parent_dir(&args.output)?;

    if args.gguf {
        let format = GgufFormat::from_name(&args.format)?;
        let model = load_any_model(&args.input, &run_config.model, &device)?;
        save_gguf(&model, format, &args.output)?;
        eprintln!(
            "converted {} to {:?} GGUF at {}",
            args.input.display(),
            format,
            args.output.display()
        );
        return Ok(());
    }

    let arch = HfArch::from_name(&args.arch)?;
    let model = convert_hf_with_arch(&args.input, &run_config.model, arch, &device)?;
    save_model(&model, &args.output)?;
    eprintln!(
        "converted HF checkpoint {} ({arch:?}) to {}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn write_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
