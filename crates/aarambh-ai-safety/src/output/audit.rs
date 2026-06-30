use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Stage where a safety event occurred.
pub enum SafetyStage {
    /// Prompt/input stage.
    Input,
    /// Generated output stage.
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Privacy-preserving safety audit event.
pub struct SafetyEvent {
    /// Event creation time in Unix milliseconds.
    pub timestamp_unix_ms: u128,
    /// SHA-256 hash of the prompt.
    pub prompt_hash: String,
    /// Safety stage.
    pub stage: SafetyStage,
    /// Verdict label.
    pub verdict: String,
    /// Rule identifiers that fired.
    pub triggered_rules: Vec<String>,
    /// Check latency in milliseconds.
    pub latency_ms: u128,
}

impl SafetyEvent {
    /// Create a safety audit event.
    pub fn new(
        prompt_hash: String,
        stage: SafetyStage,
        verdict: impl Into<String>,
        triggered_rules: Vec<String>,
        latency_ms: u128,
    ) -> Self {
        Self {
            timestamp_unix_ms: now_unix_ms(),
            prompt_hash,
            stage,
            verdict: verdict.into(),
            triggered_rules,
            latency_ms,
        }
    }
}

/// Hash a prompt for audit logging without storing plaintext.
pub fn hash_prompt(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Append a safety event as JSONL.
pub fn log_event(event: &SafetyEvent, path: &Path) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, event)?;
    writeln!(file)?;
    Ok(())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_hash_is_sha256_hex() {
        let hash = hash_prompt("hello");
        assert_eq!(hash.len(), 64);
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn audit_log_excludes_prompt_text() {
        let path = std::env::temp_dir().join(format!(
            "aarambh-ai-safety-audit-{}.jsonl",
            std::process::id()
        ));
        let prompt = "secret prompt dev@example.com";
        let event = SafetyEvent::new(
            hash_prompt(prompt),
            SafetyStage::Input,
            "redact",
            vec!["input.pii.email".to_string()],
            1,
        );
        log_event(&event, &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(!content.contains(prompt));
        assert!(!content.contains("dev@example.com"));
        assert!(content.contains("prompt_hash"));
    }
}
