use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::{AarambhError, Result};
use serde::{Deserialize, Serialize};

use crate::forgetting::{ForgettingDelta, ProbeSkip, RoutingDrift};

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
    /// Optional Phase 38 forgetting diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forgetting: Option<ForgettingReport>,
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
            schema_version: 2,
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
            forgetting: None,
        }
    }

    /// Attach Phase 38 forgetting diagnostics.
    pub fn with_forgetting(mut self, forgetting: ForgettingReport) -> Self {
        self.forgetting = Some(forgetting);
        self
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
        if let Some(forgetting) = &self.forgetting {
            out.push('\n');
            out.push_str(&forgetting.to_markdown());
        }
        out
    }
}

/// Forgetting diagnostics attached to one evaluation scorecard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForgettingReport {
    /// Optional baseline checkpoint or session identifier.
    pub baseline_checkpoint_or_session: Option<String>,
    /// Current checkpoint or session identifier.
    pub current_checkpoint_or_session: String,
    /// Absolute score-change threshold used for significance.
    pub significance_threshold: f64,
    /// Per-capability score comparisons available for the selected baseline.
    pub deltas: Vec<ForgettingDelta>,
    /// Optional MoE routing-drift comparisons.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub routing_drift: Vec<RoutingDrift>,
    /// Capabilities skipped by the current run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<ProbeSkip>,
}

impl ForgettingReport {
    /// Return the number of significantly regressed capabilities.
    pub fn forgotten_count(&self) -> usize {
        self.deltas
            .iter()
            .filter(|delta| delta.delta <= -self.significance_threshold)
            .count()
    }

    /// Return the number of significantly improved capabilities.
    pub fn improved_count(&self) -> usize {
        self.deltas
            .iter()
            .filter(|delta| delta.delta >= self.significance_threshold)
            .count()
    }

    /// Render a Markdown forgetting table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("## Forgetting Diagnostics\n\n");
        out.push_str(&format!(
            "Current: `{}`  \nBaseline: `{}`  \nSignificance threshold: `{:.4}`\n\n",
            self.current_checkpoint_or_session,
            self.baseline_checkpoint_or_session
                .as_deref()
                .unwrap_or("not selected"),
            self.significance_threshold
        ));
        out.push_str("| Capability | Before | After | Delta | Status |\n");
        out.push_str("|---|---:|---:|---:|---|\n");
        if self.deltas.is_empty() {
            out.push_str("| none | - | - | - | baseline recorded |\n");
        } else {
            for delta in &self.deltas {
                let status = if delta.delta <= -self.significance_threshold {
                    "forgotten"
                } else if delta.delta >= self.significance_threshold {
                    "improved"
                } else {
                    "stable"
                };
                out.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {:+.4} | {} |\n",
                    delta.capability_or_concept,
                    delta.score_before,
                    delta.score_after,
                    delta.delta,
                    status
                ));
            }
        }
        if !self.routing_drift.is_empty() {
            out.push_str("\n| Capability | Routing drift | Changed | Compared |\n");
            out.push_str("|---|---:|---:|---:|\n");
            for drift in &self.routing_drift {
                out.push_str(&format!(
                    "| {} | {:.4} | {} | {} |\n",
                    drift.capability,
                    drift.drift_rate,
                    drift.changed_examples,
                    drift.compared_examples
                ));
            }
        }
        if !self.skipped.is_empty() {
            out.push_str("\nSkipped probes:\n");
            for skipped in &self.skipped {
                out.push_str(&format!("- `{}`: {}\n", skipped.capability, skipped.reason));
            }
        }
        out.push_str(&format!(
            "\nForgotten: `{}` | Improved: `{}` | Skipped: `{}`\n",
            self.forgotten_count(),
            self.improved_count(),
            self.skipped.len()
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

/// Per-task robustness measurements from a four-way QAT comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QatTaskRobustness {
    /// Task name.
    pub name: String,
    /// Metric name.
    pub metric: String,
    /// Full-precision baseline value.
    pub baseline_fp: f64,
    /// Quantized baseline value.
    pub baseline_quantized: f64,
    /// Full-precision QAT-master value.
    pub qat_fp: f64,
    /// Quantized QAT value.
    pub qat_quantized: f64,
    /// Direction-normalized degradation caused by baseline quantization.
    pub baseline_quantization_drop: f64,
    /// Direction-normalized degradation caused by quantizing the QAT model.
    pub qat_quantization_drop: f64,
    /// Reduction in quantization degradation achieved by QAT.
    pub robustness_recovery: f64,
}

/// Four scorecards and normalized robustness deltas for QAT validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QatRobustnessReport {
    /// Scorecard for the full-precision baseline checkpoint.
    pub baseline_fp: Scorecard,
    /// Scorecard for the exported quantized baseline checkpoint.
    pub baseline_quantized: Scorecard,
    /// Scorecard for the full-precision QAT master checkpoint.
    pub qat_fp: Scorecard,
    /// Scorecard for the exported quantized QAT checkpoint.
    pub qat_quantized: Scorecard,
    /// Per-task robustness summary.
    pub tasks: Vec<QatTaskRobustness>,
}

impl QatRobustnessReport {
    /// Compare four scorecards by task name and metric direction.
    pub fn compare(
        baseline_fp: Scorecard,
        baseline_quantized: Scorecard,
        qat_fp: Scorecard,
        qat_quantized: Scorecard,
    ) -> Self {
        let baseline_quantized_by_name = score_map(&baseline_quantized);
        let qat_fp_by_name = score_map(&qat_fp);
        let qat_quantized_by_name = score_map(&qat_quantized);
        let tasks = baseline_fp
            .tasks
            .iter()
            .filter_map(|baseline| {
                let baseline_quantized = baseline_quantized_by_name.get(&baseline.name)?;
                let qat_fp = qat_fp_by_name.get(&baseline.name)?;
                let qat_quantized = qat_quantized_by_name.get(&baseline.name)?;
                let direction = if baseline.higher_is_better { 1.0 } else { -1.0 };
                let baseline_drop = (baseline.value - baseline_quantized.value) * direction;
                let qat_drop = (qat_fp.value - qat_quantized.value) * direction;
                Some(QatTaskRobustness {
                    name: baseline.name.clone(),
                    metric: baseline.metric.clone(),
                    baseline_fp: baseline.value,
                    baseline_quantized: baseline_quantized.value,
                    qat_fp: qat_fp.value,
                    qat_quantized: qat_quantized.value,
                    baseline_quantization_drop: baseline_drop,
                    qat_quantization_drop: qat_drop,
                    robustness_recovery: baseline_drop - qat_drop,
                })
            })
            .collect();
        Self {
            baseline_fp,
            baseline_quantized,
            qat_fp,
            qat_quantized,
            tasks,
        }
    }

    /// Serialize the complete robustness report to pretty JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(AarambhError::Json)
    }

    /// Render all four scorecards and robustness deltas as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();
        for (name, scorecard) in [
            ("Baseline FP", &self.baseline_fp),
            ("Baseline Quantized", &self.baseline_quantized),
            ("QAT FP Master", &self.qat_fp),
            ("QAT Quantized", &self.qat_quantized),
        ] {
            output.push_str(&format!("## {name}\n\n"));
            output.push_str(&scorecard.to_markdown());
            output.push_str("\n\n");
        }
        output.push_str("## Quantization Robustness\n\n");
        output.push_str(
            "| Task | Metric | Base FP | Base Quant | QAT FP | QAT Quant | Base Drop | QAT Drop | Recovery |\n",
        );
        output.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for task in &self.tasks {
            output.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:+.4} |\n",
                task.name,
                task.metric,
                task.baseline_fp,
                task.baseline_quantized,
                task.qat_fp,
                task.qat_quantized,
                task.baseline_quantization_drop,
                task.qat_quantization_drop,
                task.robustness_recovery,
            ));
        }
        output
    }
}

fn score_map(scorecard: &Scorecard) -> BTreeMap<String, &TaskScore> {
    scorecard
        .tasks
        .iter()
        .map(|task| (task.name.clone(), task))
        .collect()
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

    #[test]
    fn schema_v1_scorecard_without_forgetting_still_loads() {
        let json = r#"{
            "schema_version": 1,
            "model_path": null,
            "tokenizer_path": null,
            "config_path": null,
            "tasks": [],
            "context_len_used": 0,
            "max_new_tokens": 0,
            "timestamp_unix": 0
        }"#;
        let scorecard: Scorecard = serde_json::from_str(json).unwrap();
        assert!(scorecard.forgetting.is_none());
    }

    #[test]
    fn qat_report_normalizes_lower_is_better_metrics() {
        let scorecard = |ppl: f64| {
            Scorecard::new(
                vec![TaskScore::perplexity(ppl.ln(), ppl, 10)],
                4,
                0,
                None,
                None,
                None,
            )
        };
        let report = QatRobustnessReport::compare(
            scorecard(10.0),
            scorecard(14.0),
            scorecard(9.5),
            scorecard(11.0),
        );
        assert_eq!(report.tasks[0].baseline_quantization_drop, 4.0);
        assert_eq!(report.tasks[0].qat_quantization_drop, 1.5);
        assert_eq!(report.tasks[0].robustness_recovery, 2.5);
    }
}
