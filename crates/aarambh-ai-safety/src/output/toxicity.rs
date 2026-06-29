use std::collections::HashMap;

use crate::input::normalize_signal;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToxicityCategory {
    HateSpeech,
    Violence,
    SexualContent,
    SelfHarm,
    IllegalActivity,
}

impl ToxicityCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::HateSpeech => "hate_speech",
            Self::Violence => "violence",
            Self::SexualContent => "sexual_content",
            Self::SelfHarm => "self_harm",
            Self::IllegalActivity => "illegal_activity",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToxicityScore {
    pub overall: f32,
    pub categories: HashMap<ToxicityCategory, f32>,
    pub matched_rules: Vec<String>,
}

impl ToxicityScore {
    pub fn is_triggered(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }
}

pub fn score_toxicity(text: &str) -> ToxicityScore {
    let text = normalize_signal(text);
    let mut categories = HashMap::new();
    let mut matched_rules = Vec::new();

    for (category, phrase, weight, rule_id) in TOXICITY_RULES {
        if contains_rule_phrase(&text, phrase) {
            let entry = categories.entry(*category).or_insert(0.0f32);
            *entry = (*entry).max(*weight);
            matched_rules.push((*rule_id).to_string());
        }
    }

    let overall = categories.values().copied().fold(0.0f32, f32::max).min(1.0);
    ToxicityScore {
        overall,
        categories,
        matched_rules,
    }
}

fn contains_rule_phrase(text: &str, phrase: &str) -> bool {
    let mut offset = 0usize;
    while let Some(relative_start) = text[offset..].find(phrase) {
        let start = offset + relative_start;
        let end = start + phrase.len();
        if has_phrase_boundaries(text, start, end) {
            return true;
        }
        offset = end;
    }
    false
}

fn has_phrase_boundaries(text: &str, start: usize, end: usize) -> bool {
    let before_ok = text[..start]
        .chars()
        .next_back()
        .is_none_or(|ch| !is_word_char(ch));
    let after_ok = text[end..]
        .chars()
        .next()
        .is_none_or(|ch| !is_word_char(ch));
    before_ok && after_ok
}

fn is_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

const TOXICITY_RULES: &[(ToxicityCategory, &str, f32, &str)] = &[
    (
        ToxicityCategory::Violence,
        "i will kill you",
        0.95,
        "output.violence.direct_threat",
    ),
    (
        ToxicityCategory::Violence,
        "i will hurt you",
        0.90,
        "output.violence.direct_threat_hurt",
    ),
    (
        ToxicityCategory::Violence,
        "i will hurt everyone",
        0.90,
        "output.violence.mass_threat",
    ),
    (
        ToxicityCategory::Violence,
        "how to make a bomb",
        0.90,
        "output.violence.weapon_instruction",
    ),
    (
        ToxicityCategory::Violence,
        "murder",
        0.45,
        "output.violence.murder",
    ),
    (
        ToxicityCategory::SelfHarm,
        "kill myself",
        0.95,
        "output.self_harm.direct",
    ),
    (
        ToxicityCategory::SelfHarm,
        "suicide",
        0.85,
        "output.self_harm.suicide",
    ),
    (
        ToxicityCategory::SelfHarm,
        "self harm",
        0.80,
        "output.self_harm.keyword",
    ),
    (
        ToxicityCategory::IllegalActivity,
        "steal credit card",
        0.90,
        "output.illegal.credit_card_theft",
    ),
    (
        ToxicityCategory::IllegalActivity,
        "hack into",
        0.80,
        "output.illegal.intrusion",
    ),
    (
        ToxicityCategory::IllegalActivity,
        "malware",
        0.80,
        "output.illegal.malware",
    ),
    (
        ToxicityCategory::HateSpeech,
        "inferior race",
        0.90,
        "output.hate.inferior_race",
    ),
    (
        ToxicityCategory::HateSpeech,
        "exterminate",
        0.80,
        "output.hate.exterminate",
    ),
    (
        ToxicityCategory::SexualContent,
        "explicit sexual",
        0.80,
        "output.sexual.explicit",
    ),
    (
        ToxicityCategory::SexualContent,
        "porn",
        0.75,
        "output.sexual.porn",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violence_scores_high() {
        let score = score_toxicity("I will kill you.");
        assert!(score.is_triggered(0.7));
        assert_eq!(score.categories[&ToxicityCategory::Violence], 0.95);
    }

    #[test]
    fn neutral_text_scores_low() {
        let score = score_toxicity("The model wrote a short poem.");
        assert!(!score.is_triggered(0.7));
    }

    #[test]
    fn literary_murderer_word_does_not_block() {
        let score = score_toxicity("approach murderer, grave; sirrah");
        assert!(!score.is_triggered(0.7));
        assert!(score.matched_rules.is_empty());
    }

    #[test]
    fn standalone_generic_murder_is_logged_but_not_blocked() {
        let score = score_toxicity("The play mentions murder.");
        assert!(!score.is_triggered(0.7));
        assert_eq!(score.categories[&ToxicityCategory::Violence], 0.45);
    }
}
