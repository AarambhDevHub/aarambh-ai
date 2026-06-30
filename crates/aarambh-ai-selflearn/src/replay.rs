use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::config::ReplayConfig;

const HIGH_QUALITY_LOCK_SCORE: f32 = 0.90;
const TOPIC_BATCH_CAP: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// High-quality response stored for replay fine-tuning.
pub struct ReplayEntry {
    /// Original prompt.
    pub prompt: String,
    /// Response text.
    pub response: String,
    /// Quality score in `[0, 1]`.
    pub score: f32,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Inferred topic label.
    pub topic: String,
}

impl ReplayEntry {
    /// Create a replay entry and infer its topic.
    pub fn new(prompt: impl Into<String>, response: impl Into<String>, score: f32) -> Self {
        let prompt = prompt.into();
        Self {
            topic: infer_topic(&prompt),
            prompt,
            response: response.into(),
            score: score.clamp(0.0, 1.0),
            timestamp: now_unix_seconds(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Replay buffer summary statistics.
pub struct ReplayStats {
    /// Current entry count.
    pub len: usize,
    /// Maximum entry count.
    pub capacity: usize,
    /// Average stored score.
    pub avg_score: f32,
    /// Entry count by topic.
    pub topics: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
/// Bounded replay buffer with quality and topic-aware sampling.
pub struct ReplayBuffer {
    entries: Vec<ReplayEntry>,
    config: ReplayConfig,
    rng: StdRng,
}

impl ReplayBuffer {
    /// Create an empty replay buffer.
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Return replay configuration.
    pub fn config(&self) -> &ReplayConfig {
        &self.config
    }

    /// Return stored entries.
    pub fn entries(&self) -> &[ReplayEntry] {
        &self.entries
    }

    /// Return current entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true when no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Push an entry if it meets quality and capacity rules.
    pub fn push(&mut self, mut entry: ReplayEntry) -> bool {
        entry.score = entry.score.clamp(0.0, 1.0);
        if entry.score < self.config.min_score {
            return false;
        }
        if entry.topic.trim().is_empty() {
            entry.topic = infer_topic(&entry.prompt);
        }
        if self.entries.len() < self.config.capacity {
            self.entries.push(entry);
            return true;
        }
        let Some((idx, lowest)) = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        else {
            return false;
        };
        if lowest.score >= HIGH_QUALITY_LOCK_SCORE || entry.score <= lowest.score {
            return false;
        }
        self.entries[idx] = entry;
        true
    }

    /// Sample a quality-weighted replay batch.
    pub fn sample_batch(&mut self, n: usize) -> Vec<ReplayEntry> {
        if n == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let weights = self
            .entries
            .iter()
            .map(|entry| (entry.score.max(0.01) as f64).powi(2))
            .collect::<Vec<_>>();
        let Ok(distribution) = WeightedIndex::new(&weights) else {
            return Vec::new();
        };
        let mut selected = Vec::with_capacity(n.min(self.entries.len()));
        let mut topic_counts = HashMap::<String, usize>::new();
        let mut attempts = 0usize;
        while selected.len() < n.min(self.entries.len()) && attempts < self.entries.len() * 16 {
            attempts += 1;
            let idx = distribution.sample(&mut self.rng);
            let entry = &self.entries[idx];
            if selected
                .iter()
                .any(|selected: &ReplayEntry| selected.prompt == entry.prompt)
            {
                continue;
            }
            let count = topic_counts.entry(entry.topic.clone()).or_default();
            if *count >= TOPIC_BATCH_CAP {
                continue;
            }
            *count += 1;
            selected.push(entry.clone());
        }
        selected
    }

    /// Return true when replay should run at `step_count`.
    pub fn should_replay(&self, step_count: usize) -> bool {
        step_count > 0
            && step_count.is_multiple_of(self.config.replay_every_n)
            && self.entries.len() >= self.config.batch_size
    }

    /// Save the full buffer as JSONL.
    pub fn save_jsonl(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(path)?;
        for entry in &self.entries {
            serde_json::to_writer(&mut file, entry)?;
            writeln!(file)?;
        }
        Ok(())
    }

    /// Append one replay entry to JSONL.
    pub fn append_jsonl(path: impl AsRef<Path>, entry: &ReplayEntry) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, entry)?;
        writeln!(file)?;
        Ok(())
    }

    /// Load replay entries from JSONL.
    pub fn load_jsonl(path: impl AsRef<Path>, config: ReplayConfig) -> Result<Self> {
        let path = path.as_ref();
        let mut buffer = Self::new(config);
        if !path.exists() {
            return Ok(buffer);
        }
        let file = fs::File::open(path)?;
        for (idx, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: ReplayEntry = serde_json::from_str(line).map_err(|err| {
                AarambhError::Config(format!(
                    "invalid replay JSONL at line {} in {}: {err}",
                    idx + 1,
                    path.display()
                ))
            })?;
            let _ = buffer.push(entry);
        }
        Ok(buffer)
    }

    /// Return summary statistics.
    pub fn stats(&self) -> ReplayStats {
        let mut topics = HashMap::new();
        let mut total = 0.0f32;
        for entry in &self.entries {
            total += entry.score;
            *topics.entry(entry.topic.clone()).or_insert(0) += 1;
        }
        ReplayStats {
            len: self.entries.len(),
            capacity: self.config.capacity,
            avg_score: if self.entries.is_empty() {
                0.0
            } else {
                total / self.entries.len() as f32
            },
            topics,
        }
    }
}

/// Infer a broad topic label from a prompt.
pub fn infer_topic(prompt: &str) -> String {
    let text = prompt.to_ascii_lowercase();
    if contains_any(
        &text,
        &["solve", "calculate", "equation", "math", "+", "-", "*", "/"],
    ) {
        "math".into()
    } else if contains_any(
        &text,
        &["rust", "code", "function", "compile", "bug", "program"],
    ) {
        "code".into()
    } else if contains_any(&text, &["why", "reason", "prove", "explain step", "logic"]) {
        "reasoning".into()
    } else if contains_any(
        &text,
        &["who", "what is", "when", "where", "fact", "define"],
    ) {
        "factual".into()
    } else if contains_any(&text, &["story", "poem", "write", "creative", "song"]) {
        "creative".into()
    } else {
        "general".into()
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

/// Return current Unix time in seconds.
pub fn now_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(capacity: usize, min_score: f32) -> ReplayConfig {
        ReplayConfig {
            capacity,
            min_score,
            replay_every_n: 2,
            batch_size: 2,
            path: std::env::temp_dir().join("replay.jsonl"),
        }
    }

    fn entry(score: f32, topic: &str) -> ReplayEntry {
        ReplayEntry {
            prompt: format!("{topic} prompt {score}"),
            response: "answer".into(),
            score,
            timestamp: 1,
            topic: topic.into(),
        }
    }

    #[test]
    fn rejects_low_score_entries() {
        let mut buffer = ReplayBuffer::new(config(3, 0.7));
        assert!(!buffer.push(entry(0.5, "math")));
        assert!(buffer.is_empty());
    }

    #[test]
    fn replay_buffer_respects_capacity() {
        let mut buffer = ReplayBuffer::new(config(3, 0.0));
        for idx in 0..10 {
            let _ = buffer.push(entry(idx as f32 / 10.0, "math"));
        }
        assert_eq!(buffer.len(), 3);
        assert!(buffer.entries().iter().all(|entry| entry.score >= 0.7));
    }

    #[test]
    fn replay_buffer_never_evicts_high_quality() {
        let mut buffer = ReplayBuffer::new(config(2, 0.0));
        assert!(buffer.push(entry(0.95, "math")));
        assert!(buffer.push(entry(0.91, "code")));
        assert!(!buffer.push(entry(0.99, "reasoning")));
        assert_eq!(buffer.len(), 2);
        assert!(buffer.entries().iter().any(|entry| entry.score == 0.95));
        assert!(buffer.entries().iter().any(|entry| entry.score == 0.91));
    }

    #[test]
    fn replay_batch_has_topic_diversity() {
        let mut buffer = ReplayBuffer::new(config(20, 0.0));
        for idx in 0..10 {
            assert!(buffer.push(entry(0.8 + idx as f32 * 0.01, "math")));
            assert!(buffer.push(entry(0.8 + idx as f32 * 0.01, "code")));
        }
        let batch = buffer.sample_batch(8);
        let mut counts = HashMap::<String, usize>::new();
        for entry in batch {
            *counts.entry(entry.topic).or_default() += 1;
        }
        assert!(counts.values().all(|count| *count <= 2));
    }

    #[test]
    fn replay_persists_and_loads() {
        let path =
            std::env::temp_dir().join(format!("aarambh_replay_{}.jsonl", std::process::id()));
        let mut cfg = config(3, 0.0);
        cfg.path = path.clone();
        let mut buffer = ReplayBuffer::new(cfg.clone());
        assert!(buffer.push(ReplayEntry::new("Hello", "World", 0.9)));
        buffer.save_jsonl(&path).unwrap();
        let loaded = ReplayBuffer::load_jsonl(&path, cfg).unwrap();
        let _ = fs::remove_file(path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries()[0].prompt, "Hello");
    }

    #[test]
    fn topic_inference_covers_expected_topics() {
        assert_eq!(infer_topic("Solve 2 + 2"), "math");
        assert_eq!(infer_topic("Write Rust code"), "code");
        assert_eq!(infer_topic("Why is the sky blue?"), "reasoning");
        assert_eq!(infer_topic("Write a poem"), "creative");
    }
}
