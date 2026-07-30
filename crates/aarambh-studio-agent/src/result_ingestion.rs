use std::collections::VecDeque;
use std::fs;
use std::io::BufRead;
use std::path::Path;

use aarambh_studio_inference::ToolCall;
use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentResult, ToolResult, ToolResultRequest};

/// Caller-owned source of externally executed tool results.
pub trait ToolResultProvider {
    /// Return the result corresponding to the supplied request.
    fn next_result(&mut self, request: &ToolResultRequest) -> AgentResult<ToolResult>;

    /// Validate provider completion after the model emits a final response.
    fn finish(&self) -> AgentResult<()> {
        Ok(())
    }
}

/// JSONL stdin result provider for interactive caller-executed chains.
pub struct StdinResultProvider<R> {
    reader: R,
    buffer: String,
}

impl<R: BufRead> StdinResultProvider<R> {
    /// Wrap a buffered input stream containing one [`ToolResult`] JSON object per line.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: String::new(),
        }
    }
}

impl<R: BufRead> ToolResultProvider for StdinResultProvider<R> {
    fn next_result(&mut self, request: &ToolResultRequest) -> AgentResult<ToolResult> {
        self.buffer.clear();
        let bytes = self.reader.read_line(&mut self.buffer)?;
        if bytes == 0 {
            return Err(AgentError::ResultProtocol(format!(
                "stdin ended while waiting for result {}",
                request.call_id
            )));
        }
        let result: ToolResult = serde_json::from_str(self.buffer.trim()).map_err(|error| {
            AgentError::ResultProtocol(format!(
                "invalid result JSON for {}: {error}",
                request.call_id
            ))
        })?;
        result.validate_for(&request.call_id)?;
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// One deterministic replay entry with optional expected-call validation.
pub struct ReplayEntry {
    /// Expected model call; omitted to validate only the call id and result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_call: Option<ToolCall>,
    /// Caller result returned for this replay step.
    pub result: ToolResult,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReplayLine {
    Entry(ReplayEntry),
    Direct(ToolResult),
}

/// Deterministic JSONL replay source for tests and evaluations.
pub struct ReplayResultProvider {
    entries: VecDeque<ReplayEntry>,
}

impl ReplayResultProvider {
    /// Load replay entries from a JSONL file.
    pub fn from_jsonl(path: impl AsRef<Path>) -> AgentResult<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        let mut entries = VecDeque::new();
        for (index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parsed: ReplayLine = serde_json::from_str(line).map_err(|error| {
                AgentError::ResultProtocol(format!(
                    "invalid replay JSONL at {} line {}: {error}",
                    path.display(),
                    index + 1
                ))
            })?;
            entries.push_back(match parsed {
                ReplayLine::Entry(entry) => entry,
                ReplayLine::Direct(result) => ReplayEntry {
                    expected_call: None,
                    result,
                },
            });
        }
        if entries.is_empty() {
            return Err(AgentError::ResultProtocol(format!(
                "replay file {} contains no results",
                path.display()
            )));
        }
        if entries.len() > 64 {
            return Err(AgentError::ResultProtocol(format!(
                "replay file {} exceeds the 64-result chain limit",
                path.display()
            )));
        }
        Ok(Self { entries })
    }

    /// Construct a replay source from already parsed entries.
    pub fn new(entries: Vec<ReplayEntry>) -> AgentResult<Self> {
        if entries.is_empty() || entries.len() > 64 {
            return Err(AgentError::ResultProtocol(
                "replay requires 1..=64 entries".into(),
            ));
        }
        Ok(Self {
            entries: entries.into(),
        })
    }

    /// Number of replay results not yet consumed.
    pub fn remaining(&self) -> usize {
        self.entries.len()
    }
}

impl ToolResultProvider for ReplayResultProvider {
    fn next_result(&mut self, request: &ToolResultRequest) -> AgentResult<ToolResult> {
        let entry = self.entries.pop_front().ok_or_else(|| {
            AgentError::ResultProtocol(format!(
                "replay exhausted before result {}",
                request.call_id
            ))
        })?;
        if let Some(expected) = entry.expected_call
            && expected != request.call
        {
            return Err(AgentError::ReplayMismatch {
                call_id: request.call_id.clone(),
                expected: Box::new(expected),
                actual: Box::new(request.call.clone()),
            });
        }
        entry.result.validate_for(&request.call_id)?;
        Ok(entry.result)
    }

    fn finish(&self) -> AgentResult<()> {
        if self.entries.is_empty() {
            Ok(())
        } else {
            Err(AgentError::ResultProtocol(format!(
                "model emitted a final response with {} replay results unconsumed",
                self.entries.len()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use aarambh_studio_inference::ToolCall;
    use serde_json::json;

    use super::{ReplayEntry, ReplayResultProvider, ToolResultProvider};
    use crate::{ToolResult, ToolResultRequest, ToolResultStatus};

    #[test]
    fn replay_rejects_wrong_call() {
        let expected = ToolCall {
            name: "lookup".into(),
            arguments: json!({"q": "one"}),
        };
        let mut replay = ReplayResultProvider::new(vec![ReplayEntry {
            expected_call: Some(expected),
            result: ToolResult {
                call_id: "call_0001".into(),
                status: ToolResultStatus::Error,
                content: None,
                error: Some("offline".into()),
            },
        }])
        .unwrap();
        let request = ToolResultRequest {
            call_id: "call_0001".into(),
            call: ToolCall {
                name: "lookup".into(),
                arguments: json!({"q": "two"}),
            },
        };
        assert!(replay.next_result(&request).is_err());
    }

    #[test]
    fn replay_rejects_unconsumed_results() {
        let replay = ReplayResultProvider::new(vec![ReplayEntry {
            expected_call: None,
            result: ToolResult {
                call_id: "call_0001".into(),
                status: ToolResultStatus::Error,
                content: None,
                error: Some("offline".into()),
            },
        }])
        .unwrap();
        assert!(replay.finish().is_err());
    }
}
