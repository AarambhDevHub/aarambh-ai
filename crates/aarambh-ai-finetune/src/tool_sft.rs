use std::fs;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_tokenizer::{
    ASSISTANT, ASSISTANT_ID, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD_ID, THINK_END, THINK_END_ID,
    THINK_START, THINK_START_ID, USER, USER_ID, VIRTUAL_ASCII_END, encode_virtual_json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sft::SftDataset;

const FINAL_MARKER: &str = "<final>";
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Tool definition embedded in a tool-SFT training example.
pub struct ToolSftDefinition {
    /// Stable function name.
    pub name: String,
    /// Human-readable function behavior.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the argument object.
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Expected function call in a tool-SFT example.
pub struct ToolSftCall {
    /// Selected function name.
    pub name: String,
    /// Function arguments.
    pub arguments: Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Outcome of an externally executed tool in a multi-step SFT trace.
pub enum ToolSftResultStatus {
    /// Tool execution succeeded.
    Ok,
    /// Tool execution failed.
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
/// Supported text-only result content for tool-chain supervision.
pub enum ToolSftResultContent {
    /// Bounded UTF-8 tool output.
    Text {
        /// Tool-produced text.
        text: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// One tool result in a multi-step SFT trace.
pub struct ToolSftResult {
    /// Successful or failed execution status.
    pub status: ToolSftResultStatus,
    /// Text content required for successful results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolSftResultContent>,
    /// Error text required for failed results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolSftResult {
    fn validate(&self) -> Result<()> {
        match (&self.status, &self.content, &self.error) {
            (
                ToolSftResultStatus::Ok,
                Some(ToolSftResultContent::Text { text }),
                None,
            ) if text.len() <= 64 * 1024 => Ok(()),
            (ToolSftResultStatus::Error, None, Some(error))
                if !error.trim().is_empty() && error.len() <= 64 * 1024 =>
            {
                Ok(())
            }
            _ => Err(AarambhError::Config(
                "multi-step tool result requires bounded text content for ok or a bounded non-empty error for error status"
                    .into(),
            )),
        }
    }

    fn text(&self) -> String {
        match (&self.status, &self.content, &self.error) {
            (ToolSftResultStatus::Ok, Some(ToolSftResultContent::Text { text }), None) => {
                text.clone()
            }
            (ToolSftResultStatus::Error, None, Some(error)) => format!("error: {error}"),
            _ => "invalid tool result".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One assistant call and caller result in a multi-step SFT trace.
pub struct ToolSftTurn {
    /// Expected schema-valid assistant call.
    pub tool_call: ToolSftCall,
    /// Caller-provided text or error result.
    pub tool_result: ToolSftResult,
    /// Optional hidden reasoning context, excluded from loss.
    #[serde(default)]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Long-horizon function-calling supervised example.
pub struct MultiStepToolSftExample {
    /// Initial user request.
    pub instruction: String,
    /// Tools available throughout the trace.
    pub tools: Vec<ToolSftDefinition>,
    /// Ordered call/result interactions.
    pub turns: Vec<ToolSftTurn>,
    /// Final assistant answer after all tool results.
    pub response: String,
    /// Optional hidden final reasoning context, excluded from loss.
    #[serde(default)]
    pub final_thinking: Option<String>,
}

impl MultiStepToolSftExample {
    /// Validate call schemas, result protocol, and bounded chain length.
    pub fn validate(&self) -> Result<()> {
        if self.instruction.trim().is_empty()
            || self.response.trim().is_empty()
            || self.tools.is_empty()
        {
            return Err(AarambhError::Config(
                "multi-step tool SFT requires instruction, tools, and final response".into(),
            ));
        }
        if self.turns.is_empty() || self.turns.len() > 64 {
            return Err(AarambhError::Config(
                "multi-step tool SFT requires 1..=64 turns".into(),
            ));
        }
        for turn in &self.turns {
            let definition = self
                .tools
                .iter()
                .find(|tool| tool.name == turn.tool_call.name)
                .ok_or_else(|| {
                    AarambhError::Config(format!(
                        "multi-step tool SFT call references unknown tool {:?}",
                        turn.tool_call.name
                    ))
                })?;
            validate_schema_value(
                &definition.parameters,
                &turn.tool_call.arguments,
                "$.arguments",
            )?;
            turn.tool_result.validate()?;
        }
        Ok(())
    }

    fn masked_ids(&self, tokenizer: &dyn TokenizerLike) -> Result<(Vec<u32>, Vec<u32>)> {
        require_virtual_vocabulary(tokenizer)?;
        let legacy = ToolSftExample {
            instruction: self.instruction.clone(),
            tools: self.tools.clone(),
            tool_call: None,
            response: "prefix".into(),
            thinking: None,
        };
        let mut ids = tokenizer.encode(&legacy.prompt()?)?;
        let mut mask = vec![0; ids.len()];
        for (index, turn) in self.turns.iter().enumerate() {
            if let Some(thinking) = &turn.thinking {
                append_masked(
                    &mut ids,
                    &mut mask,
                    tokenizer.encode(&format!("{THINK_START}\n{thinking}\n{THINK_END}\n"))?,
                    0,
                );
            }
            let call = serde_json::to_string(&turn.tool_call)?;
            append_masked(&mut ids, &mut mask, vec![USER_ID, USER_ID], 1);
            append_masked(&mut ids, &mut mask, encode_virtual_json(&call), 1);
            append_masked(&mut ids, &mut mask, vec![BOS_ID, PAD_ID], 1);
            let result = format!(
                "{USER}\nTool result call_{:04}: {}\n{ASSISTANT}\n",
                index + 1,
                turn.tool_result.text()
            );
            append_masked(&mut ids, &mut mask, tokenizer.encode(&result)?, 0);
        }
        if let Some(thinking) = &self.final_thinking {
            append_masked(
                &mut ids,
                &mut mask,
                tokenizer.encode(&format!("{THINK_START}\n{thinking}\n{THINK_END}\n"))?,
                0,
            );
        }
        append_masked(&mut ids, &mut mask, vec![ASSISTANT_ID, ASSISTANT_ID], 1);
        append_masked(&mut ids, &mut mask, tokenizer.encode(&self.response)?, 1);
        append_masked(&mut ids, &mut mask, vec![ENDOFTEXT_ID], 1);
        Ok((ids, mask))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One normalized function-calling supervised example.
pub struct ToolSftExample {
    /// User request.
    pub instruction: String,
    /// Tools available for this request.
    pub tools: Vec<ToolSftDefinition>,
    /// Expected call, or null for a direct answer.
    #[serde(default)]
    pub tool_call: Option<ToolSftCall>,
    /// Direct response when no tool is selected.
    #[serde(default)]
    pub response: String,
    /// Optional hidden reasoning target.
    #[serde(default)]
    pub thinking: Option<String>,
}

impl ToolSftExample {
    /// Validate tool names, outcome exclusivity, and call arguments.
    pub fn validate(&self) -> Result<()> {
        if self.instruction.trim().is_empty() {
            return Err(AarambhError::Config(
                "tool SFT instruction must not be empty".into(),
            ));
        }
        if self.tools.is_empty() {
            return Err(AarambhError::Config(
                "tool SFT example must provide at least one tool".into(),
            ));
        }
        match (&self.tool_call, self.response.trim().is_empty()) {
            (Some(call), true) => {
                let definition = self
                    .tools
                    .iter()
                    .find(|tool| tool.name == call.name)
                    .ok_or_else(|| {
                        AarambhError::Config(format!(
                            "tool SFT call references unknown tool {:?}",
                            call.name
                        ))
                    })?;
                validate_schema_value(&definition.parameters, &call.arguments, "$.arguments")
            }
            (None, false) => Ok(()),
            _ => Err(AarambhError::Config(
                "tool SFT example requires exactly one of tool_call or response".into(),
            )),
        }
    }

    /// Render the inference-compatible prompt prefix.
    pub fn prompt(&self) -> Result<String> {
        let tools = self
            .tools
            .iter()
            .map(|tool| {
                format!(
                    "{}: {}. Parameters: {}",
                    tool.name,
                    tool.description,
                    schema_summary(&tool.parameters)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "{USER}\nAvailable tools:\n{tools}\nChoose one tool when needed, otherwise answer directly.\nRequest: {}\n{ASSISTANT}\n",
            self.instruction
        ))
    }

    /// Render the assistant target including protocol and EOS markers.
    pub fn target(&self) -> Result<String> {
        let thinking = self
            .thinking
            .as_ref()
            .map(|thinking| format!("{THINK_START}\n{thinking}\n{THINK_END}\n"))
            .unwrap_or_default();
        if let Some(call) = &self.tool_call {
            Ok(format!(
                "{thinking}{TOOL_CALL_START}{}{TOOL_CALL_END}{ENDOFTEXT}",
                serde_json::to_string(call)?
            ))
        } else {
            Ok(format!(
                "{thinking}{FINAL_MARKER}{}{ENDOFTEXT}",
                self.response
            ))
        }
    }
}

#[derive(Debug, Clone)]
/// Tokenized tool-calling SFT dataset.
pub struct ToolSftDataset {
    inner: SftDataset,
}

impl ToolSftDataset {
    /// Load and strictly tokenize function-calling examples from JSONL.
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        tokenizer: &dyn TokenizerLike,
        max_seq_len: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        let mut legacy_pairs = Vec::new();
        let mut masked = Vec::new();
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid tool SFT JSONL at {} line {}: {error}",
                    path.display(),
                    line_index + 1
                ))
            })?;
            if value.get("turns").is_some() {
                let example: MultiStepToolSftExample =
                    serde_json::from_value(value).map_err(|error| {
                        AarambhError::Config(format!(
                            "invalid multi-step tool SFT example at {} line {}: {error}",
                            path.display(),
                            line_index + 1
                        ))
                    })?;
                example.validate()?;
                masked.push(example.masked_ids(tokenizer)?);
            } else {
                let example: ToolSftExample = serde_json::from_value(value).map_err(|error| {
                    AarambhError::Config(format!(
                        "invalid tool SFT example at {} line {}: {error}",
                        path.display(),
                        line_index + 1
                    ))
                })?;
                example.validate()?;
                legacy_pairs.push((
                    tokenizer.encode(&example.prompt()?)?,
                    example.target_ids(tokenizer)?,
                ));
            }
        }
        for (prefix, target) in legacy_pairs {
            let mut ids = prefix;
            let prefix_len = ids.len();
            ids.extend(target);
            let mut mask = vec![0; prefix_len];
            mask.resize(ids.len(), 1);
            masked.push((ids, mask));
        }
        Ok(Self {
            inner: SftDataset::from_masked_id_sequences(&masked, max_seq_len)?,
        })
    }

    pub(crate) fn into_inner(self) -> SftDataset {
        self.inner
    }
}

impl ToolSftExample {
    fn target_ids(&self, tokenizer: &dyn TokenizerLike) -> Result<Vec<u32>> {
        require_virtual_vocabulary(tokenizer)?;
        let mut ids = Vec::new();
        if let Some(thinking) = &self.thinking {
            ids.push(THINK_START_ID);
            ids.extend(tokenizer.encode(&format!("\n{thinking}\n"))?);
            ids.push(THINK_END_ID);
            ids.extend(tokenizer.encode("\n")?);
        }
        if let Some(call) = &self.tool_call {
            ids.extend([USER_ID, USER_ID]);
            ids.extend(encode_virtual_json(&serde_json::to_string(call)?));
            ids.extend([BOS_ID, PAD_ID]);
        } else {
            ids.extend([ASSISTANT_ID, ASSISTANT_ID]);
            ids.extend(tokenizer.encode(&self.response)?);
        }
        ids.push(ENDOFTEXT_ID);
        Ok(ids)
    }
}

fn require_virtual_vocabulary(tokenizer: &dyn TokenizerLike) -> Result<()> {
    if tokenizer.vocab_size() <= VIRTUAL_ASCII_END as usize {
        return Err(AarambhError::Tokenizer(format!(
            "tool SFT requires vocabulary size greater than {VIRTUAL_ASCII_END}"
        )));
    }
    Ok(())
}

fn append_masked(ids: &mut Vec<u32>, mask: &mut Vec<u32>, tokens: Vec<u32>, value: u32) {
    mask.extend(std::iter::repeat_n(value, tokens.len()));
    ids.extend(tokens);
}

fn validate_schema_value(schema: &Value, value: &Value, path: &str) -> Result<()> {
    let object = schema
        .as_object()
        .ok_or_else(|| AarambhError::Config(format!("schema at {path} must be an object")))?;
    if let Some(constant) = object.get("const") {
        return (constant == value)
            .then_some(())
            .ok_or_else(|| AarambhError::Config(format!("{path} does not match const")));
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        return values
            .contains(value)
            .then_some(())
            .ok_or_else(|| AarambhError::Config(format!("{path} is not in enum")));
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| object.contains_key("properties").then_some("object"))
        .unwrap_or("string");
    match kind {
        "object" => {
            let actual = value
                .as_object()
                .ok_or_else(|| schema_type(path, "object"))?;
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(required) = object.get("required").and_then(Value::as_array) {
                for name in required.iter().filter_map(Value::as_str) {
                    if !actual.contains_key(name) {
                        return Err(AarambhError::Config(format!(
                            "{path} is missing required property {name:?}"
                        )));
                    }
                }
            }
            for (name, value) in actual {
                let child = properties.get(name).ok_or_else(|| {
                    AarambhError::Config(format!("{path} has unknown property {name:?}"))
                })?;
                validate_schema_value(child, value, &format!("{path}.{name}"))?;
            }
            Ok(())
        }
        "array" => {
            let actual = value.as_array().ok_or_else(|| schema_type(path, "array"))?;
            let items = object.get("items").ok_or_else(|| {
                AarambhError::Config(format!("array schema at {path} requires items"))
            })?;
            for (index, value) in actual.iter().enumerate() {
                validate_schema_value(items, value, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        "string" => value
            .is_string()
            .then_some(())
            .ok_or_else(|| schema_type(path, "string")),
        "integer" => (value.as_i64().is_some() || value.as_u64().is_some())
            .then_some(())
            .ok_or_else(|| schema_type(path, "integer")),
        "number" => value
            .is_number()
            .then_some(())
            .ok_or_else(|| schema_type(path, "number")),
        "boolean" => value
            .is_boolean()
            .then_some(())
            .ok_or_else(|| schema_type(path, "boolean")),
        "null" => value
            .is_null()
            .then_some(())
            .ok_or_else(|| schema_type(path, "null")),
        other => Err(AarambhError::Unsupported(format!(
            "unsupported tool SFT schema type {other:?} at {path}"
        ))),
    }
}

fn schema_type(path: &str, expected: &str) -> AarambhError {
    AarambhError::Config(format!("{path} must be {expected}"))
}

fn schema_summary(schema: &Value) -> String {
    let Some(object) = schema.as_object() else {
        return "object".into();
    };
    let required = object
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(name, schema)| {
                    let kind = schema
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("value");
                    let suffix = if required.contains(name.as_str()) {
                        " required"
                    } else {
                        " optional"
                    };
                    format!("{name} {kind}{suffix}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|summary| !summary.is_empty())
        .unwrap_or_else(|| "none".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ByteTokenizer;

    impl TokenizerLike for ByteTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|byte| byte as u32 + 100).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids
                .iter()
                .filter_map(|id| u8::try_from(id.saturating_sub(100)).ok())
                .map(char::from)
                .collect())
        }

        fn vocab_size(&self) -> usize {
            512
        }

        fn eos_token_id(&self) -> u32 {
            ENDOFTEXT_ID
        }

        fn bos_token_id(&self) -> Option<u32> {
            Some(BOS_ID)
        }
    }

    #[test]
    fn formats_tool_and_direct_targets() {
        let tool = ToolSftDefinition {
            name: "weather".into(),
            description: "Weather".into(),
            parameters: serde_json::json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
            }),
        };
        let call = ToolSftExample {
            instruction: "Weather in Delhi?".into(),
            tools: vec![tool.clone()],
            tool_call: Some(ToolSftCall {
                name: "weather".into(),
                arguments: serde_json::json!({"city":"Delhi"}),
            }),
            response: String::new(),
            thinking: None,
        };
        call.validate().unwrap();
        assert!(call.target().unwrap().starts_with("<tool_call>{"));

        let direct = ToolSftExample {
            instruction: "Hello".into(),
            tools: vec![tool],
            tool_call: None,
            response: "Hi".into(),
            thinking: None,
        };
        assert_eq!(direct.target().unwrap(), "<final>Hi<|endoftext|>");
    }

    #[test]
    fn multi_step_mask_supervises_each_call_and_final_only() {
        let tool = ToolSftDefinition {
            name: "lookup".into(),
            description: "Lookup".into(),
            parameters: serde_json::json!({
                "type":"object",
                "properties":{"key":{"type":"string"}},
                "required":["key"]
            }),
        };
        let result = ToolSftResult {
            status: ToolSftResultStatus::Ok,
            content: Some(ToolSftResultContent::Text {
                text: "value".into(),
            }),
            error: None,
        };
        let example = MultiStepToolSftExample {
            instruction: "Find two values".into(),
            tools: vec![tool],
            turns: vec![
                ToolSftTurn {
                    tool_call: ToolSftCall {
                        name: "lookup".into(),
                        arguments: serde_json::json!({"key":"one"}),
                    },
                    tool_result: result.clone(),
                    thinking: Some("first".into()),
                },
                ToolSftTurn {
                    tool_call: ToolSftCall {
                        name: "lookup".into(),
                        arguments: serde_json::json!({"key":"two"}),
                    },
                    tool_result: result,
                    thinking: None,
                },
            ],
            response: "Both found".into(),
            final_thinking: Some("combine".into()),
        };
        example.validate().unwrap();
        let (ids, mask) = example.masked_ids(&ByteTokenizer).unwrap();
        assert_eq!(ids.len(), mask.len());
        let transitions = mask.windows(2).filter(|pair| pair[0] != pair[1]).count();
        assert_eq!(transitions, 5);
        assert_eq!(mask.first(), Some(&0));
        assert_eq!(mask.last(), Some(&1));
    }
}
