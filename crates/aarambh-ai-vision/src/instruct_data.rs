use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Result};
use serde::Deserialize;

/// Normalized vision-language instruction example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VqaExample {
    /// Image path, relative to the configured image root when not absolute.
    pub image_path: PathBuf,
    /// User question or visual instruction.
    pub question: String,
    /// Assistant target answer.
    pub answer: String,
    /// Optional hidden thinking target placed before the final answer.
    pub thinking: Option<String>,
}

impl VqaExample {
    /// Build a normalized VQA example.
    pub fn new(
        image_path: impl Into<PathBuf>,
        question: impl Into<String>,
        answer: impl Into<String>,
        thinking: Option<String>,
    ) -> Result<Self> {
        let question = strip_image_marker(&question.into());
        let answer = answer.into().trim().to_string();
        if question.is_empty() {
            return Err(AarambhError::Config(
                "VQA example question must not be empty".into(),
            ));
        }
        if answer.is_empty() {
            return Err(AarambhError::Config(
                "VQA example answer must not be empty".into(),
            ));
        }
        Ok(Self {
            image_path: image_path.into(),
            question,
            answer,
            thinking,
        })
    }
}

/// Load VQA instruction examples from JSONL.
pub fn load_vqa_jsonl(
    path: impl AsRef<Path>,
    max_samples: Option<usize>,
) -> Result<Vec<VqaExample>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path).map_err(|err| {
        AarambhError::Io(std::io::Error::new(
            err.kind(),
            format!("failed to read {}: {err}", path.display()),
        ))
    })?;
    let mut examples = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw: RawVqaRecord = serde_json::from_str(line).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {err}",
                path.display(),
                line_idx + 1
            ))
        })?;
        examples.push(raw.into_example().map_err(|err| {
            AarambhError::Config(format!(
                "invalid VQA example in {} line {}: {err}",
                path.display(),
                line_idx + 1
            ))
        })?);
        if max_samples.is_some_and(|max| examples.len() >= max) {
            break;
        }
    }
    if examples.is_empty() {
        return Err(AarambhError::Config(format!(
            "{} contains no VQA examples",
            path.display()
        )));
    }
    Ok(examples)
}

#[derive(Debug, Deserialize)]
struct RawVqaRecord {
    #[serde(default)]
    image: Option<PathBuf>,
    #[serde(default)]
    image_path: Option<PathBuf>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    conversations: Vec<ConversationTurn>,
}

impl RawVqaRecord {
    fn into_example(self) -> Result<VqaExample> {
        let image_path = self.image_path.or(self.image).ok_or_else(|| {
            AarambhError::Config("VQA example is missing image or image_path".into())
        })?;
        if let (Some(question), Some(answer)) = (self.question, self.answer) {
            return VqaExample::new(image_path, question, answer, self.thinking);
        }
        let question = self
            .conversations
            .iter()
            .find(|turn| turn.is_human())
            .map(|turn| turn.value.clone())
            .ok_or_else(|| AarambhError::Config("LLaVA record has no human turn".into()))?;
        let answer = self
            .conversations
            .iter()
            .skip_while(|turn| !turn.is_human())
            .find(|turn| turn.is_assistant())
            .map(|turn| turn.value.clone())
            .ok_or_else(|| {
                AarambhError::Config("LLaVA record has no assistant turn after human turn".into())
            })?;
        VqaExample::new(image_path, question, answer, self.thinking)
    }
}

#[derive(Debug, Deserialize)]
struct ConversationTurn {
    #[serde(default, alias = "role")]
    from: String,
    value: String,
}

impl ConversationTurn {
    fn is_human(&self) -> bool {
        matches!(
            self.from.trim().to_ascii_lowercase().as_str(),
            "human" | "user"
        )
    }

    fn is_assistant(&self) -> bool {
        matches!(
            self.from.trim().to_ascii_lowercase().as_str(),
            "gpt" | "assistant"
        )
    }
}

fn strip_image_marker(value: &str) -> String {
    value
        .replace("<image>", "")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_vqa_record() {
        let raw: RawVqaRecord = serde_json::from_str(
            r#"{"image":"red.png","question":"What color?","answer":"red","thinking":"look"}"#,
        )
        .unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.image_path, PathBuf::from("red.png"));
        assert_eq!(example.question, "What color?");
        assert_eq!(example.answer, "red");
        assert_eq!(example.thinking.as_deref(), Some("look"));
    }

    #[test]
    fn parses_llava_conversation_record() {
        let raw: RawVqaRecord = serde_json::from_str(
            r#"{"image":"cat.jpg","conversations":[{"from":"human","value":"<image>\nWhat animal is shown?"},{"from":"gpt","value":"A cat."}]}"#,
        )
        .unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.question, "What animal is shown?");
        assert_eq!(example.answer, "A cat.");
    }
}
