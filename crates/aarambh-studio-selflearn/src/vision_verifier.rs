use std::str::FromStr;

use aarambh_studio_finetune::Verifier;
use serde::{Deserialize, Serialize};

/// Built-in grounded verifier selector for checkable vision questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VisionVerifierKind {
    /// Disable grounded vision verification.
    None,
    /// Infer count, color, or presence verification from the prompt.
    Auto,
    /// Count verifier for questions such as "How many objects?".
    Count,
    /// Color verifier for questions such as "What color is it?".
    Color,
    /// Yes/no presence verifier.
    Presence,
    /// Exact normalized answer verifier.
    Exact,
}

impl VisionVerifierKind {
    /// Return a concrete verifier kind for a prompt when this kind is `Auto`.
    pub fn resolve_for_prompt(self, prompt: &str) -> Self {
        if self != Self::Auto {
            return self;
        }
        let text = normalize(prompt);
        if text.contains("how many") || text.contains("count") || text.contains("number of") {
            Self::Count
        } else if text.contains("what color")
            || text.contains("which color")
            || text.contains("colour")
        {
            Self::Color
        } else if text.starts_with("is ")
            || text.starts_with("are ")
            || text.starts_with("do ")
            || text.starts_with("does ")
            || text.contains(" is there ")
            || text.contains(" are there ")
        {
            Self::Presence
        } else {
            Self::Exact
        }
    }

    /// Build a verifier for this concrete kind.
    pub fn build(self) -> Option<VisionVerifier> {
        match self {
            Self::None => None,
            Self::Auto => Some(VisionVerifier::new(Self::Exact)),
            kind => Some(VisionVerifier::new(kind)),
        }
    }
}

impl FromStr for VisionVerifierKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "disabled" | "off" => Ok(Self::None),
            "auto" => Ok(Self::Auto),
            "count" | "counting" => Ok(Self::Count),
            "color" | "colour" => Ok(Self::Color),
            "presence" | "yes-no" | "yes_no" | "yesno" => Ok(Self::Presence),
            "exact" | "vqa" => Ok(Self::Exact),
            other => Err(format!(
                "unsupported vision verifier '{other}', expected none|auto|count|color|presence|exact"
            )),
        }
    }
}

/// Grounded verifier for checkable vision-language answers.
#[derive(Debug, Clone, Copy)]
pub struct VisionVerifier {
    kind: VisionVerifierKind,
}

impl VisionVerifier {
    /// Create a verifier for a concrete checkable vision task.
    pub fn new(kind: VisionVerifierKind) -> Self {
        Self { kind }
    }

    /// Return the verifier kind.
    pub fn kind(&self) -> VisionVerifierKind {
        self.kind
    }
}

impl Verifier for VisionVerifier {
    fn score(&self, completion: &str, ground_truth: &str) -> f32 {
        match self.kind {
            VisionVerifierKind::None => 0.0,
            VisionVerifierKind::Auto | VisionVerifierKind::Exact => {
                score_exact(completion, ground_truth)
            }
            VisionVerifierKind::Count => score_count(completion, ground_truth),
            VisionVerifierKind::Color => score_color(completion, ground_truth),
            VisionVerifierKind::Presence => score_presence(completion, ground_truth),
        }
    }
}

fn score_exact(completion: &str, ground_truth: &str) -> f32 {
    let completion = normalize(completion);
    let expected = normalize(ground_truth);
    if expected.is_empty() {
        return 0.0;
    }
    if completion == expected || contains_word(&completion, &expected) {
        1.0
    } else {
        0.0
    }
}

fn score_count(completion: &str, ground_truth: &str) -> f32 {
    match (extract_count(completion), extract_count(ground_truth)) {
        (Some(predicted), Some(expected)) if predicted == expected => 1.0,
        _ => 0.0,
    }
}

fn score_color(completion: &str, ground_truth: &str) -> f32 {
    let expected = normalize_color(ground_truth);
    if expected.is_empty() {
        return 0.0;
    }
    let completion = normalize(completion);
    if !contains_word(&completion, &expected) {
        return 0.0;
    }
    let contradictory = [
        "red", "green", "blue", "yellow", "black", "white", "orange", "purple", "pink", "brown",
        "gray", "grey",
    ]
    .into_iter()
    .any(|color| color != expected && contains_word(&completion, color));
    if contradictory { 0.5 } else { 1.0 }
}

fn score_presence(completion: &str, ground_truth: &str) -> f32 {
    match (extract_bool(completion), extract_bool(ground_truth)) {
        (Some(predicted), Some(expected)) if predicted == expected => 1.0,
        _ => 0.0,
    }
}

fn extract_count(text: &str) -> Option<i64> {
    let normalized = normalize(text);
    for token in normalized.split_whitespace().rev() {
        if let Ok(value) = token.parse::<i64>() {
            return Some(value);
        }
        if let Some(value) = number_word(token) {
            return Some(value);
        }
    }
    None
}

fn extract_bool(text: &str) -> Option<bool> {
    let normalized = normalize(text);
    if contains_word(&normalized, "no")
        || contains_word(&normalized, "false")
        || normalized.starts_with("there is no")
        || normalized.starts_with("there are no")
    {
        Some(false)
    } else if contains_word(&normalized, "yes")
        || contains_word(&normalized, "true")
        || normalized.starts_with("there is")
        || normalized.starts_with("there are")
    {
        Some(true)
    } else {
        None
    }
}

fn normalize_color(text: &str) -> String {
    normalize(text)
        .split_whitespace()
        .find(|token| {
            matches!(
                *token,
                "red"
                    | "green"
                    | "blue"
                    | "yellow"
                    | "black"
                    | "white"
                    | "orange"
                    | "purple"
                    | "pink"
                    | "brown"
                    | "gray"
                    | "grey"
            )
        })
        .unwrap_or("")
        .to_string()
}

fn number_word(token: &str) -> Option<i64> {
    match token {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        _ => None,
    }
}

fn contains_word(text: &str, needle: &str) -> bool {
    text.split_whitespace().any(|token| token == needle)
}

fn normalize(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_verifier_scores_counting_questions_exactly() {
        let verifier = VisionVerifier::new(VisionVerifierKind::Count);
        assert_eq!(verifier.score("There are three squares.", "3"), 1.0);
        assert_eq!(verifier.score("There are two squares.", "3"), 0.0);
    }

    #[test]
    fn vision_verifier_scores_color_questions() {
        let verifier = VisionVerifier::new(VisionVerifierKind::Color);
        assert_eq!(verifier.score("The square is red.", "red"), 1.0);
        assert_eq!(verifier.score("The square is blue.", "red"), 0.0);
    }

    #[test]
    fn vision_verifier_scores_presence_questions() {
        let verifier = VisionVerifier::new(VisionVerifierKind::Presence);
        assert_eq!(verifier.score("Yes, there is a car.", "yes"), 1.0);
        assert_eq!(verifier.score("No, there is not.", "yes"), 0.0);
    }

    #[test]
    fn auto_verifier_resolves_from_prompt() {
        assert_eq!(
            VisionVerifierKind::Auto.resolve_for_prompt("How many cats are visible?"),
            VisionVerifierKind::Count
        );
        assert_eq!(
            VisionVerifierKind::Auto.resolve_for_prompt("What color is the square?"),
            VisionVerifierKind::Color
        );
    }
}
