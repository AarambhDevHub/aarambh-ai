mod cmd;
mod ui;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "aarambh-ai")]
#[command(about = "Aarambh AI command line tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train(cmd::train::TrainArgs),
    Infer(cmd::infer::InferArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => cmd::train::run(args),
        Command::Infer(args) => cmd::infer::run(args),
    }
}
