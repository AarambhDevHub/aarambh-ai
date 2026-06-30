use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use aarambh_ai_core::Result;
use serde::{Deserialize, Serialize};

use crate::replay::infer_topic;

const HISTORY_LIMIT: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Direction of a topic score trend.
pub enum TrendDirection {
    /// Scores are improving.
    Up,
    /// Scores are approximately unchanged.
    Flat,
    /// Scores are declining.
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// One persisted learning metric event.
pub struct MetricsEvent {
    /// Sequential step.
    pub step: usize,
    /// Inferred topic label.
    pub topic: String,
    /// Score in `[0, 1]`.
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
/// Tracks rolling self-learning scores by topic.
pub struct LearningMetrics {
    per_topic_scores: HashMap<String, VecDeque<f32>>,
    total_steps: usize,
    replay_count: usize,
}

impl LearningMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a score and infer topic from the prompt.
    pub fn record(&mut self, score: f32, prompt: &str) {
        let topic = infer_topic(prompt);
        self.record_topic(score, &topic);
    }

    /// Record a score under a specific topic.
    pub fn record_topic(&mut self, score: f32, topic: &str) {
        self.total_steps += 1;
        let scores = self.per_topic_scores.entry(topic.to_string()).or_default();
        scores.push_back(score.clamp(0.0, 1.0));
        while scores.len() > HISTORY_LIMIT {
            scores.pop_front();
        }
    }

    /// Increment replay fine-tuning count.
    pub fn record_replay(&mut self) {
        self.replay_count += 1;
    }

    /// Return total recorded response steps.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Return replay fine-tuning count.
    pub fn replay_count(&self) -> usize {
        self.replay_count
    }

    /// Return late-minus-early trend for a topic.
    pub fn topic_trend(&self, topic: &str) -> Option<f32> {
        let scores = self.per_topic_scores.get(topic)?;
        if scores.len() < 4 {
            return Some(0.0);
        }
        let mid = scores.len() / 2;
        let early = scores.iter().take(mid).sum::<f32>() / mid as f32;
        let late_count = scores.len() - mid;
        let late = scores.iter().skip(mid).sum::<f32>() / late_count as f32;
        Some(late - early)
    }

    /// Return a coarse trend direction for a topic.
    pub fn trend_direction(&self, topic: &str) -> Option<TrendDirection> {
        let trend = self.topic_trend(topic)?;
        if trend > 0.02 {
            Some(TrendDirection::Up)
        } else if trend < -0.02 {
            Some(TrendDirection::Down)
        } else {
            Some(TrendDirection::Flat)
        }
    }

    /// Return a compact text summary.
    pub fn summary(&self) -> String {
        let mut topics = self.per_topic_scores.keys().cloned().collect::<Vec<_>>();
        topics.sort();
        if topics.is_empty() {
            return "No self-learning metrics yet".into();
        }
        topics
            .iter()
            .filter_map(|topic| {
                let trend = self.topic_trend(topic)?;
                let arrow = match self.trend_direction(topic)? {
                    TrendDirection::Up => "↑",
                    TrendDirection::Flat => "→",
                    TrendDirection::Down => "↓",
                };
                Some(format!("{topic}: {arrow} {trend:+.2}"))
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Save all metrics as JSONL.
    pub fn save_jsonl(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(path)?;
        let mut step = 0usize;
        let mut topics = self.per_topic_scores.keys().cloned().collect::<Vec<_>>();
        topics.sort();
        for topic in topics {
            if let Some(scores) = self.per_topic_scores.get(&topic) {
                for score in scores {
                    step += 1;
                    serde_json::to_writer(
                        &mut file,
                        &MetricsEvent {
                            step,
                            topic: topic.clone(),
                            score: *score,
                        },
                    )?;
                    writeln!(file)?;
                }
            }
        }
        Ok(())
    }

    /// Append one metrics event to JSONL.
    pub fn append_event(path: impl AsRef<Path>, event: &MetricsEvent) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, event)?;
        writeln!(file)?;
        Ok(())
    }

    /// Load metrics from JSONL.
    pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut metrics = Self::new();
        if !path.exists() {
            return Ok(metrics);
        }
        let content = fs::read_to_string(path)?;
        for line in content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let event: MetricsEvent = serde_json::from_str(line)?;
            metrics.record_topic(event.score, &event.topic);
        }
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_trend_detects_improvement() {
        let mut metrics = LearningMetrics::new();
        for _ in 0..5 {
            metrics.record_topic(0.4, "math");
        }
        for _ in 0..5 {
            metrics.record_topic(0.8, "math");
        }
        assert!(metrics.topic_trend("math").unwrap() > 0.2);
        assert_eq!(metrics.trend_direction("math"), Some(TrendDirection::Up));
    }

    #[test]
    fn metrics_summary_handles_empty() {
        assert_eq!(
            LearningMetrics::new().summary(),
            "No self-learning metrics yet"
        );
    }
}
