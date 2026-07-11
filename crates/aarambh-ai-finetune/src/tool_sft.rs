use std::fs;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};
use aarambh_ai_tokenizer::{
    ASSISTANT, ASSISTANT_ID, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD_ID, THINK_END, THINK_END_ID,
    THINK_START, THINK_START_ID, USER, USER_ID,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sft::SftDataset;

const FINAL_MARKER: &str = "<final>";
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const VIRTUAL_ASCII_BASE: u32 = 9;
const VIRTUAL_ASCII_FIRST: u8 = 0x20;
const VIRTUAL_ASCII_END: u32 = VIRTUAL_ASCII_BASE + (0x7e - VIRTUAL_ASCII_FIRST) as u32;

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
        let mut pairs = Vec::new();
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let example: ToolSftExample = serde_json::from_str(line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid tool SFT JSONL at {} line {}: {error}",
                    path.display(),
                    line_index + 1
                ))
            })?;
            example.validate().map_err(|error| {
                AarambhError::Config(format!(
                    "invalid tool SFT example at {} line {}: {error}",
                    path.display(),
                    line_index + 1
                ))
            })?;
            pairs.push((
                tokenizer.encode(&example.prompt()?)?,
                example.target_ids(tokenizer)?,
            ));
        }
        Ok(Self {
            inner: SftDataset::from_id_sequences(&pairs, max_seq_len, true)?,
        })
    }

    pub(crate) fn into_inner(self) -> SftDataset {
        self.inner
    }
}

impl ToolSftExample {
    fn target_ids(&self, tokenizer: &dyn TokenizerLike) -> Result<Vec<u32>> {
        if tokenizer.vocab_size() <= VIRTUAL_ASCII_END as usize {
            return Err(AarambhError::Tokenizer(format!(
                "tool SFT requires vocabulary size greater than {VIRTUAL_ASCII_END}"
            )));
        }
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

fn encode_virtual_json(text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    for character in text.chars() {
        if character.is_ascii() && (' '..='~').contains(&character) {
            ids.push(virtual_ascii_id(character));
        } else {
            for escaped in json_unicode_escape(character).chars() {
                ids.push(virtual_ascii_id(escaped));
            }
        }
    }
    ids
}

fn virtual_ascii_id(character: char) -> u32 {
    VIRTUAL_ASCII_BASE + (character as u8 - VIRTUAL_ASCII_FIRST) as u32
}

fn json_unicode_escape(character: char) -> String {
    let code = character as u32;
    if code <= 0xffff {
        format!("\\u{code:04x}")
    } else {
        let adjusted = code - 0x1_0000;
        let high = 0xd800 + (adjusted >> 10);
        let low = 0xdc00 + (adjusted & 0x3ff);
        format!("\\u{high:04x}\\u{low:04x}")
    }
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
}
