use std::str::FromStr;

use aarambh_ai_tokenizer::{THINK_END, THINK_START};
use serde::{Deserialize, Serialize};

/// Scores generated completions against ground-truth answers.
pub trait Verifier: Send + Sync {
    /// Return a reward score in the range expected by the verifier.
    fn score(&self, completion: &str, ground_truth: &str) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// Built-in verifier selector.
pub enum VerifierKind {
    /// Numeric final-answer verifier.
    Math,
    /// Thinking-format verifier.
    Format,
    /// Weighted math plus format verifier.
    MathFormat,
}

impl VerifierKind {
    /// Build the selected verifier.
    pub fn build(self) -> Box<dyn Verifier> {
        match self {
            Self::Math => Box::new(MathVerifier::default()),
            Self::Format => Box::new(FormatVerifier),
            Self::MathFormat => Box::new(CompositeVerifier::new(vec![
                (Box::new(MathVerifier::default()), 0.8),
                (Box::new(FormatVerifier), 0.2),
            ])),
        }
    }
}

impl FromStr for VerifierKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "math" => Ok(Self::Math),
            "format" => Ok(Self::Format),
            "math-format" | "math_format" | "composite" => Ok(Self::MathFormat),
            other => Err(format!(
                "unsupported verifier '{other}', expected math, format, or math-format"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Verifies numeric final answers with tolerance.
pub struct MathVerifier {
    tolerance: f64,
}

impl Default for MathVerifier {
    fn default() -> Self {
        Self { tolerance: 1e-4 }
    }
}

impl MathVerifier {
    /// Create a math verifier with absolute/relative tolerance.
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl Verifier for MathVerifier {
    fn score(&self, completion: &str, ground_truth: &str) -> f32 {
        let Some(predicted) = extract_final_number(completion) else {
            return 0.0;
        };
        let Some(expected) = extract_final_number(ground_truth) else {
            return 0.0;
        };
        let abs_err = (predicted - expected).abs();
        let rel_tol = self.tolerance * expected.abs().max(1.0);
        if abs_err <= self.tolerance.max(rel_tol) {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// Verifies that thinking markers contain non-empty thinking text.
pub struct FormatVerifier;

impl Verifier for FormatVerifier {
    fn score(&self, completion: &str, _ground_truth: &str) -> f32 {
        let Some(start) = completion.find(THINK_START) else {
            return 0.0;
        };
        let Some(end) = completion.find(THINK_END) else {
            return 0.5;
        };
        if end <= start {
            return 0.5;
        }
        let thinking = &completion[start + THINK_START.len()..end];
        if thinking.trim().is_empty() { 0.5 } else { 1.0 }
    }
}

/// Weighted composition of multiple verifiers.
pub struct CompositeVerifier {
    verifiers: Vec<(Box<dyn Verifier>, f32)>,
    weight_sum: f32,
}

impl CompositeVerifier {
    /// Create a composite verifier from verifier/weight pairs.
    pub fn new(verifiers: Vec<(Box<dyn Verifier>, f32)>) -> Self {
        let weight_sum = verifiers
            .iter()
            .map(|(_, weight)| weight.max(0.0))
            .sum::<f32>()
            .max(f32::EPSILON);
        Self {
            verifiers,
            weight_sum,
        }
    }
}

impl Verifier for CompositeVerifier {
    fn score(&self, completion: &str, ground_truth: &str) -> f32 {
        self.verifiers
            .iter()
            .map(|(verifier, weight)| weight.max(0.0) * verifier.score(completion, ground_truth))
            .sum::<f32>()
            / self.weight_sum
    }
}

/// Extract the final numeric answer from text.
pub fn extract_final_number(text: &str) -> Option<f64> {
    let source = text
        .rsplit_once("####")
        .map(|(_, answer)| answer)
        .unwrap_or(text);
    let mut last = None;
    let mut current = String::new();
    let mut has_digit = false;

    for ch in source.chars() {
        let sign_at_start = matches!(ch, '-' | '+') && current.is_empty();
        let numeric_char = ch.is_ascii_digit() || ch == '.' || ch == ',';
        if sign_at_start || numeric_char {
            if ch.is_ascii_digit() {
                has_digit = true;
            }
            current.push(ch);
            continue;
        }
        if has_digit && let Some(value) = parse_number(&current) {
            last = Some(value);
        }
        current.clear();
        has_digit = false;
    }

    if has_digit && let Some(value) = parse_number(&current) {
        last = Some(value);
    }
    last
}

fn parse_number(value: &str) -> Option<f64> {
    let normalized = value
        .trim_matches(|ch: char| !ch.is_ascii_digit() && !matches!(ch, '-' | '+' | '.'))
        .replace(',', "");
    if normalized.is_empty() || matches!(normalized.as_str(), "+" | "-" | ".") {
        return None;
    }
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_verifier_accepts_gsm8k_final_answer() {
        let verifier = MathVerifier::default();
        assert_eq!(
            verifier.score("work here\n#### 1,081", "The final answer is 1081."),
            1.0
        );
    }

    #[test]
    fn math_verifier_handles_negative_and_decimal_answers() {
        let verifier = MathVerifier::default();
        assert_eq!(verifier.score("answer: -4.50", "#### -4.5"), 1.0);
        assert_eq!(verifier.score("answer: -4.25", "#### -4.5"), 0.0);
    }

    #[test]
    fn math_verifier_missing_number_scores_zero() {
        let verifier = MathVerifier::default();
        assert_eq!(verifier.score("I do not know", "#### 42"), 0.0);
    }

    #[test]
    fn format_verifier_rewards_complete_think_block() {
        let verifier = FormatVerifier;
        assert_eq!(
            verifier.score("<think>\n2 + 2 = 4\n</think>\nAnswer: 4", ""),
            1.0
        );
        assert_eq!(verifier.score("<think>\nunfinished", ""), 0.5);
        assert_eq!(verifier.score("Answer only", ""), 0.0);
    }

    #[test]
    fn composite_verifier_uses_weighted_average() {
        let verifier = CompositeVerifier::new(vec![
            (Box::new(MathVerifier::default()), 0.75),
            (Box::new(FormatVerifier), 0.25),
        ]);
        let score = verifier.score("<think>calc</think>\nAnswer: 4", "#### 4");
        assert!((score - 1.0).abs() < 1e-6);
        let score = verifier.score("Answer: 4", "#### 4");
        assert!((score - 0.75).abs() < 1e-6);
    }
}
