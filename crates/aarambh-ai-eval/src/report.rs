use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::{AarambhError, Result};
use serde::{Deserialize, Serialize};

/// Result for one evaluation task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskScore {
    /// Task name.
    pub name: String,
    /// Metric name, such as `accuracy`, `ppl`, or `pass@1`.
    pub metric: String,
    /// Primary metric value.
    pub value: f64,
    /// Whether larger values are better for this metric.
    pub higher_is_better: bool,
    /// Number of examples evaluated.
    pub examples: usize,
    /// Optional count of correct examples.
    pub correct: Option<usize>,
    /// Optional average language-model loss.
    pub loss: Option<f64>,
    /// Optional perplexity.
    pub ppl: Option<f64>,
    /// Optional secondary metrics for richer tasks.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, f64>,
}

impl TaskScore {
    /// Create an accuracy score from correct and total counts.
    pub fn accuracy(name: impl Into<String>, correct: usize, examples: usize) -> Self {
        let value = if examples == 0 {
            0.0
        } else {
            correct as f64 / examples as f64
        };
        Self {
            name: name.into(),
            metric: "accuracy".into(),
            value,
            higher_is_better: true,
            examples,
            correct: Some(correct),
            loss: None,
            ppl: None,
            details: BTreeMap::new(),
        }
    }

    /// Create a pass@1 score from pass and total counts.
    pub fn pass_at_1(name: impl Into<String>, passed: usize, examples: usize) -> Self {
        let mut score = Self::accuracy(name, passed, examples);
        score.metric = "pass@1".into();
        score
    }

    /// Create a pairwise preference win-rate score.
    pub fn win_rate(name: impl Into<String>, wins: usize, examples: usize) -> Self {
        let mut score = Self::accuracy(name, wins, examples);
        score.metric = "win_rate".into();
        score
    }

    /// Create a perplexity score from loss and token count.
    pub fn perplexity(loss: f64, ppl: f64, examples: usize) -> Self {
        Self {
            name: "ppl".into(),
            metric: "ppl".into(),
            value: ppl,
            higher_is_better: false,
            examples,
            correct: None,
            loss: Some(loss),
            ppl: Some(ppl),
            details: BTreeMap::new(),
        }
    }

    /// Attach one secondary metric.
    pub fn with_detail(mut self, name: impl Into<String>, value: f64) -> Self {
        self.details.insert(name.into(), value);
        self
    }
}

/// Full evaluation scorecard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Scorecard {
    /// Scorecard schema version.
    pub schema_version: u32,
    /// Optional model checkpoint path used for evaluation.
    pub model_path: Option<String>,
    /// Optional tokenizer path used for evaluation.
    pub tokenizer_path: Option<String>,
    /// Optional training/eval config path used for evaluation.
    pub config_path: Option<String>,
    /// Task scores in execution order.
    pub tasks: Vec<TaskScore>,
    /// Largest context length used by the run.
    pub context_len_used: usize,
    /// Generation token limit used by generative tasks.
    pub max_new_tokens: usize,
    /// UNIX timestamp in seconds.
    pub timestamp_unix: u64,
}

impl Scorecard {
    /// Build a scorecard from task scores and run metadata.
    pub fn new(
        tasks: Vec<TaskScore>,
        context_len_used: usize,
        max_new_tokens: usize,
        model_path: Option<String>,
        tokenizer_path: Option<String>,
        config_path: Option<String>,
    ) -> Self {
        Self {
            schema_version: 1,
            model_path,
            tokenizer_path,
            config_path,
            tasks,
            context_len_used,
            max_new_tokens,
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0),
        }
    }

    /// Serialize this scorecard to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(AarambhError::Json)
    }

    /// Render this scorecard as a Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| Task | Metric | Value | Examples |\n");
        out.push_str("|---|---:|---:|---:|\n");
        for task in &self.tasks {
            out.push_str(&format!(
                "| {} | {} | {:.4} | {} |\n",
                task.name, task.metric, task.value, task.examples
            ));
            for (name, value) in &task.details {
                out.push_str(&format!(
                    "| {}/{} | detail | {:.4} | {} |\n",
                    task.name, name, value, task.examples
                ));
            }
        }
        out.push_str(&format!(
            "\nContext length used: `{}`  \nMax new tokens: `{}`\n",
            self.context_len_used, self.max_new_tokens
        ));
        out
    }
}

/// Per-task score delta between two scorecards.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreDelta {
    /// Task name.
    pub name: String,
    /// Previous metric value.
    pub before: f64,
    /// New metric value.
    pub after: f64,
    /// Signed `after - before` delta.
    pub delta: f64,
    /// Human-readable direction using each task's metric preference.
    pub status: String,
}

/// Comparison between two scorecards.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScorecardComparison {
    /// Per-task deltas.
    pub deltas: Vec<ScoreDelta>,
}

impl ScorecardComparison {
    /// Compare two scorecards by matching task names.
    pub fn compare(before: &Scorecard, after: &Scorecard) -> Self {
        let before_map = before
            .tasks
            .iter()
            .map(|task| (task.name.clone(), task))
            .collect::<BTreeMap<_, _>>();
        let mut deltas = Vec::new();
        for after_task in &after.tasks {
            let Some(before_task) = before_map.get(&after_task.name) else {
                continue;
            };
            let delta = after_task.value - before_task.value;
            let improved = if after_task.higher_is_better {
                delta > 0.0
            } else {
                delta < 0.0
            };
            let worsened = if after_task.higher_is_better {
                delta < 0.0
            } else {
                delta > 0.0
            };
            let status = if improved {
                "better"
            } else if worsened {
                "worse"
            } else {
                "unchanged"
            };
            deltas.push(ScoreDelta {
                name: after_task.name.clone(),
                before: before_task.value,
                after: after_task.value,
                delta,
                status: status.into(),
            });
        }
        Self { deltas }
    }

    /// Serialize this comparison to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(AarambhError::Json)
    }

    /// Render this comparison as a Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("| Task | Before | After | Delta | Status |\n");
        out.push_str("|---|---:|---:|---:|---|\n");
        for delta in &self.deltas {
            out.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:+.4} | {} |\n",
                delta.name, delta.before, delta.after, delta.delta, delta.status
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scorecard_json_roundtrips_and_compare_reports_deltas() {
        let before = Scorecard::new(
            vec![TaskScore::accuracy("mmlu", 1, 2)],
            4,
            8,
            None,
            None,
            None,
        );
        let after = Scorecard::new(
            vec![TaskScore::accuracy("mmlu", 2, 2)],
            4,
            8,
            None,
            None,
            None,
        );
        let json = before.to_json().unwrap();
        let decoded: Scorecard = serde_json::from_str(&json).unwrap();
        assert_eq!(before.tasks, decoded.tasks);
        let comparison = ScorecardComparison::compare(&before, &after);
        assert_eq!(comparison.deltas[0].status, "better");
        assert_eq!(comparison.deltas[0].delta, 0.5);
    }
}
