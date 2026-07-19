use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Result};
use serde::Deserialize;

/// Normalized video question-answer instruction example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoQaExample {
    /// Video path, relative to the configured video root when not absolute.
    pub video_path: PathBuf,
    /// User question, including normalized multiple-choice options when present.
    pub question: String,
    /// Assistant target answer.
    pub answer: String,
    /// Optional hidden thinking target.
    pub thinking: Option<String>,
}

impl VideoQaExample {
    /// Create and validate a normalized video QA example.
    pub fn new(
        video_path: impl Into<PathBuf>,
        question: impl Into<String>,
        answer: impl Into<String>,
        thinking: Option<String>,
    ) -> Result<Self> {
        let question = question.into().replace("<video>", "").trim().to_string();
        let answer = answer.into().trim().to_string();
        let video_path = video_path.into();
        if video_path.as_os_str().is_empty() || question.is_empty() || answer.is_empty() {
            return Err(AarambhError::Config(
                "video QA path, question, and answer must not be empty".into(),
            ));
        }
        Ok(Self {
            video_path,
            question,
            answer,
            thinking,
        })
    }
}

/// Load normalized JSONL video instructions or official NExT-QA CSV records.
pub fn load_video_qa(
    path: impl AsRef<Path>,
    max_samples: Option<usize>,
) -> Result<Vec<VideoQaExample>> {
    let path = path.as_ref();
    let examples = if path.extension().and_then(|ext| ext.to_str()) == Some("csv") {
        load_nextqa_csv(path, max_samples)?
    } else {
        load_video_jsonl(path, max_samples)?
    };
    if examples.is_empty() {
        return Err(AarambhError::Config(format!(
            "{} contains no video QA examples",
            path.display()
        )));
    }
    Ok(examples)
}

fn load_video_jsonl(path: &Path, max_samples: Option<usize>) -> Result<Vec<VideoQaExample>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();
    for (line_index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let raw = serde_json::from_str::<RawVideoRecord>(line).map_err(|error| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {error}",
                path.display(),
                line_index + 1
            ))
        })?;
        let video = raw.video_path.or(raw.video).ok_or_else(|| {
            AarambhError::Config(format!(
                "{} line {} is missing video or video_path",
                path.display(),
                line_index + 1
            ))
        })?;
        examples.push(VideoQaExample::new(
            video,
            raw.question,
            raw.answer,
            raw.thinking,
        )?);
        if max_samples.is_some_and(|max| examples.len() >= max) {
            break;
        }
    }
    Ok(examples)
}

fn load_nextqa_csv(path: &Path, max_samples: Option<usize>) -> Result<Vec<VideoQaExample>> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|error| AarambhError::Config(format!("failed to read NExT-QA CSV: {error}")))?;
    let mut examples = Vec::new();
    for record in reader.deserialize::<NextQaRecord>() {
        let record = record.map_err(|error| {
            AarambhError::Config(format!(
                "invalid NExT-QA row in {}: {error}",
                path.display()
            ))
        })?;
        let options = [&record.a0, &record.a1, &record.a2, &record.a3, &record.a4];
        if record.answer >= options.len() {
            return Err(AarambhError::Config(format!(
                "NExT-QA answer index {} is outside 0..5",
                record.answer
            )));
        }
        let question = format!(
            "{}\nA. {}\nB. {}\nC. {}\nD. {}\nE. {}\nAnswer with only the option letter.",
            record.question, record.a0, record.a1, record.a2, record.a3, record.a4
        );
        let answer = ((b'A' + record.answer as u8) as char).to_string();
        let mut video = PathBuf::from(record.video);
        if video.extension().is_none() {
            video.set_extension("mp4");
        }
        examples.push(VideoQaExample::new(video, question, answer, None)?);
        if max_samples.is_some_and(|max| examples.len() >= max) {
            break;
        }
    }
    Ok(examples)
}

#[derive(Debug, Deserialize)]
struct RawVideoRecord {
    #[serde(default)]
    video: Option<PathBuf>,
    #[serde(default)]
    video_path: Option<PathBuf>,
    question: String,
    answer: String,
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NextQaRecord {
    video: String,
    question: String,
    answer: usize,
    a0: String,
    a1: String,
    a2: String,
    a3: String,
    a4: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_explicit_video_marker() {
        let example =
            VideoQaExample::new("clip.mp4", "<video> What happens?", "running", None).unwrap();
        assert_eq!(example.question, "What happens?");
    }
}
