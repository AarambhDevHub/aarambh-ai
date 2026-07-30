use aarambh_studio_core::Result;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::ppl::compute_ppl;
use crate::report::TaskScore;
use crate::tasks::first_existing;

/// Perplexity-on-holdout task.
pub struct PplTask;

impl EvalTask for PplTask {
    fn name(&self) -> &'static str {
        "ppl"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = first_existing(&[
            config.data_dir.join("ppl").join("holdout.txt"),
            config.data_dir.join("ppl.txt"),
            config.data_dir.join("holdout.txt"),
        ])?;
        compute_ppl(context, path, config.max_examples)
    }
}
