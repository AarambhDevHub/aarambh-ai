use std::fs;
use std::path::Path;

use aarambh_ai_core::Result;

pub trait TextDataset {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &str;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct PlaintextDataset {
    lines: Vec<String>,
}

impl PlaintextDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        Ok(Self { lines })
    }

    pub fn from_lines(lines: Vec<String>) -> Self {
        Self { lines }
    }
}

impl TextDataset for PlaintextDataset {
    fn len(&self) -> usize {
        self.lines.len()
    }

    fn get(&self, i: usize) -> &str {
        &self.lines[i]
    }
}

pub struct JsonlDataset {
    texts: Vec<String>,
}

impl JsonlDataset {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let texts: Vec<String> = content
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                if l.is_empty() {
                    return None;
                }
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                let text = v.get("text")?.as_str()?.to_string();
                (!text.is_empty()).then_some(text)
            })
            .collect();
        Ok(Self { texts })
    }

    pub fn from_texts(texts: Vec<String>) -> Self {
        Self { texts }
    }
}

impl TextDataset for JsonlDataset {
    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get(&self, i: usize) -> &str {
        &self.texts[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_from_file() {
        let content = "hello\nworld\nfoo";
        let dir = std::env::temp_dir();
        let path = dir.join("test_plaintext.txt");
        std::fs::write(&path, content).unwrap();
        let ds = PlaintextDataset::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.get(0), "hello");
        assert_eq!(ds.get(1), "world");
        assert_eq!(ds.get(2), "foo");
    }

    #[test]
    fn jsonl_from_file() {
        let content = r#"{"text": "hello world"}
{"text": "foo bar"}
{"text": "baz"}"#;
        let dir = std::env::temp_dir();
        let path = dir.join("test_jsonl.jsonl");
        std::fs::write(&path, content).unwrap();
        let ds = JsonlDataset::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.get(0), "hello world");
        assert_eq!(ds.get(1), "foo bar");
    }

    #[test]
    fn jsonl_skips_bad_lines() {
        let content = r#"{"text": "valid"}
not json
{"text": "also valid"}"#;
        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_jsonl.jsonl");
        std::fs::write(&path, content).unwrap();
        let ds = JsonlDataset::from_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn dataset_is_empty() {
        let ds = PlaintextDataset::from_lines(vec![]);
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }
}
