use std::path::PathBuf;

use clap::Args;

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: PathBuf,
}

pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    aarambh_ai_train::run_training_from_config(args.config)?;
    Ok(())
}
