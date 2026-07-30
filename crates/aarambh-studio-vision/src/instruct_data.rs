use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result};
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

/// Normalized document question-answering instruction example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocQaExample {
    /// PDF or single raster document path when `page_paths` is empty.
    pub document_path: Option<PathBuf>,
    /// Ordered raster page paths for a multi-page scanned document.
    pub page_paths: Vec<PathBuf>,
    /// User question with any explicit document marker removed.
    pub question: String,
    /// Accepted answers in deterministic preference order.
    pub answers: Vec<String>,
    /// Optional 1-based source page selection.
    pub pages: Option<Vec<usize>>,
    /// Optional hidden thinking target.
    pub thinking: Option<String>,
    /// Optional evaluation categories such as `table`.
    pub tags: Vec<String>,
}

impl DocQaExample {
    /// Return the first accepted answer used for instruction tuning.
    pub fn primary_answer(&self) -> &str {
        &self.answers[0]
    }

    /// Convert the configured source paths into a document source.
    pub fn source(&self) -> crate::DocumentSource {
        match &self.document_path {
            Some(path) => crate::DocumentSource::File(path.clone()),
            None => crate::DocumentSource::PageImages(self.page_paths.clone()),
        }
    }
}

/// Load canonical document-QA records from JSONL.
pub fn load_document_qa_jsonl(
    path: impl AsRef<Path>,
    max_samples: Option<usize>,
) -> Result<Vec<DocQaExample>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)?;
    let mut examples = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw = serde_json::from_str::<RawDocQaRecord>(line).map_err(|error| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {error}",
                path.display(),
                line_idx + 1
            ))
        })?;
        examples.push(raw.into_example().map_err(|error| {
            AarambhError::Config(format!(
                "invalid document QA example in {} line {}: {error}",
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
            "{} contains no document QA examples",
            path.display()
        )));
    }
    Ok(examples)
}

#[derive(Debug, Deserialize)]
struct RawDocQaRecord {
    #[serde(default, alias = "document")]
    document_path: Option<PathBuf>,
    #[serde(default)]
    page_paths: Vec<PathBuf>,
    question: String,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    answers: Vec<String>,
    #[serde(default)]
    pages: Option<Vec<usize>>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

impl RawDocQaRecord {
    fn into_example(self) -> Result<DocQaExample> {
        if self.document_path.is_some() != self.page_paths.is_empty() {
            return Err(AarambhError::Config(
                "document QA example requires exactly one of document_path or page_paths".into(),
            ));
        }
        let question = self
            .question
            .replace("<document>", "")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if question.is_empty() {
            return Err(AarambhError::Config(
                "document QA question must not be empty".into(),
            ));
        }
        let mut answers = self.answers;
        if let Some(answer) = self.answer {
            answers.insert(0, answer);
        }
        for answer in &mut answers {
            *answer = answer.trim().to_string();
        }
        answers.retain(|answer| !answer.is_empty());
        answers.dedup();
        if answers.is_empty() {
            return Err(AarambhError::Config(
                "document QA example requires at least one answer".into(),
            ));
        }
        if self
            .pages
            .as_ref()
            .is_some_and(|pages| pages.is_empty() || pages.contains(&0))
        {
            return Err(AarambhError::Config(
                "document QA pages must contain non-zero 1-based page numbers".into(),
            ));
        }
        Ok(DocQaExample {
            document_path: self.document_path,
            page_paths: self.page_paths,
            question,
            answers,
            pages: self.pages,
            thinking: self.thinking,
            tags: self.tags,
        })
    }
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

    #[test]
    fn parses_document_record_with_answer_aliases() {
        let raw: RawDocQaRecord = serde_json::from_str(
            r#"{"document_path":"invoice.pdf","question":"<document>\nTotal?","answer":"$4","answers":["4 USD"],"pages":[2],"tags":["table"]}"#,
        )
        .unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.question, "Total?");
        assert_eq!(example.answers, vec!["$4", "4 USD"]);
        assert_eq!(example.pages, Some(vec![2]));
        assert_eq!(example.tags, vec!["table"]);
    }

    #[test]
    fn document_record_requires_exactly_one_source_kind() {
        let raw: RawDocQaRecord = serde_json::from_str(
            r#"{"document_path":"a.pdf","page_paths":["a.png"],"question":"Q","answer":"A"}"#,
        )
        .unwrap();
        assert!(raw.into_example().is_err());
    }
}
