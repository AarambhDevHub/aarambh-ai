use aarambh_ai_core::Result;
use aarambh_ai_inference::{GenerationConfig, InferenceEngine, Sampler, ThinkingMode};
use serde::Deserialize;

use crate::config::CritiqueConfig;

#[derive(Debug, Clone, PartialEq)]
/// Parsed critique result and optional rewrite.
pub struct CritiqueResult {
    /// Response selected by critique.
    pub response: String,
    /// Critique score in `[0, 1]`.
    pub score: f32,
    /// Critique reason string.
    pub reason: String,
    /// Whether the response came from a rewrite attempt.
    pub was_rewritten: bool,
}

/// Generation interface used by critique helpers.
pub trait CritiqueGenerator {
    /// Generate text from a prompt and generation config.
    fn generate_text(&mut self, prompt: &str, config: GenerationConfig) -> Result<String>;
}

impl CritiqueGenerator for InferenceEngine {
    fn generate_text(&mut self, prompt: &str, config: GenerationConfig) -> Result<String> {
        Ok(self.generate(prompt, config)?.text)
    }
}

/// Score a response and optionally rewrite it when below threshold.
pub fn critique_response<G: CritiqueGenerator>(
    generator: &mut G,
    prompt: &str,
    response: &str,
    config: &CritiqueConfig,
) -> Result<CritiqueResult> {
    if !config.enabled {
        return Ok(CritiqueResult {
            response: response.to_string(),
            score: 0.5,
            reason: "critique disabled".into(),
            was_rewritten: false,
        });
    }

    let mut best = CritiqueResult {
        response: response.to_string(),
        score: 0.0,
        reason: String::new(),
        was_rewritten: false,
    };
    let mut current = response.to_string();
    for rewrite_idx in 0..=config.max_rewrites {
        let critique_prompt = fill_template(&config.prompt_template, prompt, &current);
        let critique_text =
            generator.generate_text(&critique_prompt, critique_generation_config(config))?;
        let parsed = parse_critique_response(&critique_text);
        let candidate = CritiqueResult {
            response: current.clone(),
            score: parsed.score,
            reason: parsed.reason,
            was_rewritten: rewrite_idx > 0,
        };
        if candidate.score >= best.score {
            best = candidate;
        }
        if best.score >= config.rewrite_threshold || rewrite_idx == config.max_rewrites {
            break;
        }
        let rewrite_prompt = rewrite_prompt(prompt, &current);
        current = generator.generate_text(&rewrite_prompt, rewrite_generation_config(config))?;
    }
    Ok(best)
}

#[derive(Debug, Deserialize)]
struct RawCritique {
    score: f32,
    #[serde(default)]
    reason: String,
}

/// Parse critique JSON into a normalized result.
pub fn parse_critique_response(text: &str) -> CritiqueResult {
    let json = extract_json_object(text).unwrap_or(text);
    let parsed = serde_json::from_str::<RawCritique>(json).ok();
    match parsed {
        Some(raw) if raw.score.is_finite() => CritiqueResult {
            response: String::new(),
            score: raw.score.clamp(0.0, 1.0),
            reason: raw.reason,
            was_rewritten: false,
        },
        _ => CritiqueResult {
            response: String::new(),
            score: 0.5,
            reason: "malformed critique JSON".into(),
            was_rewritten: false,
        },
    }
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then_some(&text[start..=end])
}

fn fill_template(template: &str, prompt: &str, response: &str) -> String {
    template
        .replace("{prompt}", prompt)
        .replace("{original_prompt}", prompt)
        .replace("{response}", response)
        .replace("{model_response}", response)
}

fn critique_generation_config(config: &CritiqueConfig) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: config.max_tokens,
        sampler: Sampler::greedy(),
        thinking_mode: ThinkingMode::None,
        top_candidates: 5,
    }
}

fn rewrite_generation_config(config: &CritiqueConfig) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: config.rewrite_max_tokens,
        sampler: Sampler::top_k_top_p(0.5, Some(40), Some(0.9), Some(42))
            .unwrap_or_else(|_| Sampler::greedy()),
        thinking_mode: ThinkingMode::None,
        top_candidates: 5,
    }
}

fn rewrite_prompt(prompt: &str, response: &str) -> String {
    format!(
        "<|user|>\nImprove the response. Keep it accurate, clear, and concise.\n\nQuestion: {prompt}\nCurrent response: {response}\n<|assistant|>\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingGenerator {
        calls: Vec<GenerationConfig>,
    }

    impl CritiqueGenerator for RecordingGenerator {
        fn generate_text(&mut self, prompt: &str, config: GenerationConfig) -> Result<String> {
            self.calls.push(config);
            if prompt.contains("Improve the response") {
                Ok("better".into())
            } else {
                Ok(r#"{"score": 0.1, "reason": "needs work"}"#.into())
            }
        }
    }

    #[test]
    fn critique_parses_valid_json_score() {
        let result = parse_critique_response(r#"{"score": 0.85, "reason": "clear and correct"}"#);
        assert!((result.score - 0.85).abs() < 1e-4);
        assert_eq!(result.reason, "clear and correct");
    }

    #[test]
    fn critique_handles_malformed_json_gracefully() {
        let result = parse_critique_response("The answer looks good to me.");
        assert_eq!(result.score, 0.5);
    }

    #[test]
    fn critique_clamps_scores() {
        assert_eq!(parse_critique_response(r#"{"score": 2.0}"#).score, 1.0);
        assert_eq!(parse_critique_response(r#"{"score": -1.0}"#).score, 0.0);
    }

    #[test]
    fn rewrite_uses_configured_token_budget() {
        let mut generator = RecordingGenerator { calls: Vec::new() };
        let mut config = CritiqueConfig {
            enabled: true,
            rewrite_threshold: 0.7,
            max_rewrites: 1,
            max_tokens: 9,
            rewrite_max_tokens: 7,
            prompt_template: "{prompt}\n{response}".into(),
        };
        let _ = critique_response(&mut generator, "question", "bad", &config).unwrap();
        assert_eq!(generator.calls[0].max_new_tokens, 9);
        assert_eq!(generator.calls[1].max_new_tokens, 7);

        config.rewrite_max_tokens = 3;
        generator.calls.clear();
        let _ = critique_response(&mut generator, "question", "bad", &config).unwrap();
        assert_eq!(generator.calls[1].max_new_tokens, 3);
    }
}
