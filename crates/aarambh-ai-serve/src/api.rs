use aarambh_ai_inference::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// Supported text-only chat message content.
pub enum ChatContent {
    /// Plain text content.
    Text(String),
    /// OpenAI-style text content parts.
    Parts(Vec<ContentPart>),
}

impl ChatContent {
    /// Flatten supported content into a single text string.
    pub fn into_text(self) -> Result<String, &'static str> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Parts(parts) => parts
                .into_iter()
                .map(|part| match part {
                    ContentPart::Text { text } => Ok(text),
                    ContentPart::Unsupported => Err("only text content parts are supported"),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|parts| parts.join("")),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// One request message content part.
pub enum ContentPart {
    /// A text content part.
    Text {
        /// Text payload.
        text: String,
    },
    /// A non-text content part rejected by Phase 27.
    #[serde(other)]
    Unsupported,
}

#[derive(Debug, Clone, Deserialize)]
/// One input chat message.
pub struct ChatMessage {
    /// Message author role.
    pub role: String,
    /// Message content.
    pub content: ChatContent,
}

#[derive(Debug, Clone, Deserialize)]
/// OpenAI function-tool wrapper.
pub struct FunctionTool {
    /// Tool type; Phase 27 accepts only `function`.
    pub r#type: String,
    /// Function definition.
    pub function: ToolDefinition,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// OpenAI tool-choice request value.
pub enum ApiToolChoice {
    /// `auto`, `none`, or `required`.
    Mode(String),
    /// One explicitly selected function.
    Named {
        /// Choice type; must be `function`.
        r#type: String,
        /// Selected function payload.
        function: NamedFunction,
    },
}

#[derive(Debug, Clone, Deserialize)]
/// Function name inside a named tool choice.
pub struct NamedFunction {
    /// Selected function name.
    pub name: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
/// Additional options for an SSE response.
pub struct StreamOptions {
    /// Emit one usage-only chunk before `[DONE]`.
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
/// Request body for `/v1/chat/completions`.
pub struct ChatCompletionRequest {
    /// Served model identifier.
    pub model: String,
    /// Ordered conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Preferred modern completion-token limit.
    pub max_completion_tokens: Option<usize>,
    /// Compatibility completion-token limit.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling mass.
    pub top_p: Option<f32>,
    /// Reproducible sampler seed.
    pub seed: Option<u64>,
    /// String or string-array stop sequences.
    pub stop: Option<StringOrStrings>,
    /// Return SSE chunks.
    #[serde(default)]
    pub stream: bool,
    /// SSE usage controls.
    #[serde(default)]
    pub stream_options: StreamOptions,
    /// Number of choices; only one is supported.
    pub n: Option<usize>,
    /// Optional function definitions.
    pub tools: Option<Vec<FunctionTool>>,
    /// Tool selection policy.
    pub tool_choice: Option<ApiToolChoice>,
    /// Must be false because Phase 26 emits one call.
    pub parallel_tool_calls: Option<bool>,
    /// Thinking-budget selector.
    pub reasoning_effort: Option<String>,
    /// Frequency penalty; non-zero values are unsupported.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty; non-zero values are unsupported.
    pub presence_penalty: Option<f32>,
    /// Log probability output; true is unsupported.
    pub logprobs: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
/// One string or a list of strings.
pub enum StringOrStrings {
    /// One string.
    One(String),
    /// Multiple strings.
    Many(Vec<String>),
}

impl StringOrStrings {
    /// Convert into a vector without changing order.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Request body for legacy `/v1/completions`.
pub struct CompletionRequest {
    /// Served model identifier.
    pub model: String,
    /// Text prompt.
    pub prompt: Value,
    /// Completion-token limit.
    pub max_tokens: Option<usize>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling mass.
    pub top_p: Option<f32>,
    /// Reproducible sampler seed.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Option<StringOrStrings>,
    /// Return SSE chunks.
    #[serde(default)]
    pub stream: bool,
    /// Number of choices; only one is supported.
    pub n: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize)]
/// OpenAI-compatible token usage.
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: usize,
    /// Completion tokens.
    pub completion_tokens: usize,
    /// Total tokens.
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
/// One generated function call.
pub struct ToolCallResponse {
    /// Stable call identifier.
    pub id: String,
    /// Tool type, always `function`.
    pub r#type: &'static str,
    /// Called function.
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Serialize)]
/// Function name and serialized JSON arguments.
pub struct FunctionCallResponse {
    /// Function name.
    pub name: String,
    /// JSON arguments serialized as a string.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
/// Assistant message returned by chat completions.
pub struct AssistantMessage {
    /// Message role, always `assistant`.
    pub role: &'static str,
    /// Text response, or `None` for tool calls and content filtering.
    pub content: Option<String>,
    /// Optional function calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallResponse>>,
}

#[derive(Debug, Clone, Serialize)]
/// One chat completion choice.
pub struct ChatChoice {
    /// Choice index.
    pub index: usize,
    /// Assistant output.
    pub message: AssistantMessage,
    /// OpenAI finish-reason string.
    pub finish_reason: String,
    /// Log probabilities are not produced.
    pub logprobs: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
/// Non-streaming chat completion response.
pub struct ChatCompletionResponse {
    /// Completion identifier.
    pub id: String,
    /// Object type, always `chat.completion`.
    pub object: &'static str,
    /// Unix creation timestamp.
    pub created: u64,
    /// Served model identifier.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Token usage.
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
/// One legacy completion choice.
pub struct CompletionChoice {
    /// Generated text.
    pub text: String,
    /// Choice index.
    pub index: usize,
    /// Log probabilities are not produced.
    pub logprobs: Option<Value>,
    /// OpenAI finish-reason string.
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
/// Non-streaming legacy completion response.
pub struct CompletionResponse {
    /// Completion identifier.
    pub id: String,
    /// Object type, always `text_completion`.
    pub object: &'static str,
    /// Unix creation timestamp.
    pub created: u64,
    /// Served model identifier.
    pub model: String,
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    /// Token usage.
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
/// One locally loaded model descriptor.
pub struct ModelObject {
    /// Public model identifier.
    pub id: String,
    /// Object type, always `model`.
    pub object: &'static str,
    /// Unix load timestamp.
    pub created: u64,
    /// Model owner label.
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Serialize)]
/// Response body for `/v1/models`.
pub struct ModelList {
    /// Object type, always `list`.
    pub object: &'static str,
    /// Loaded model descriptors.
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize)]
/// OpenAI-compatible error envelope.
pub struct ErrorResponse {
    /// Error payload.
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize)]
/// Structured API error details.
pub struct ErrorBody {
    /// Human-readable message.
    pub message: String,
    /// Stable error category.
    pub r#type: String,
    /// Related request parameter.
    pub param: Option<String>,
    /// Stable machine-readable code.
    pub code: String,
}
