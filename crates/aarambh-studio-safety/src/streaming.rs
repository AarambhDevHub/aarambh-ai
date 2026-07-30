use aarambh_studio_inference::{GenerationPhase, GenerationStep};

use crate::input::{PiiKind, detect_pii};
use crate::output::toxicity::score_toxicity;
use crate::policy::{PiiPolicy, SafetyPolicy, ViolationAction};

const TOXICITY_LOOKBEHIND_CHARS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
/// One safety-approved streaming action.
pub enum SafeStreamEvent {
    /// Text that may be released to the client.
    Text(String),
    /// Terminal safety block; no later text may be emitted.
    Blocked(String),
}

/// Incremental output filter that prevents split-token safety bypasses.
pub struct StreamingSafetyFilter {
    policy: SafetyPolicy,
    visible_pending: String,
    toxicity_tail: String,
    redacting: Option<PiiKind>,
    blocked: Option<String>,
    output_redacted: bool,
}

impl StreamingSafetyFilter {
    /// Create a filter for one generation request.
    pub fn new(policy: SafetyPolicy) -> Self {
        Self {
            policy,
            visible_pending: String::new(),
            toxicity_tail: String::new(),
            redacting: None,
            blocked: None,
            output_redacted: false,
        }
    }

    /// Inspect one generated step and return newly approved stream events.
    pub fn push_step(&mut self, step: &GenerationStep) -> Vec<SafeStreamEvent> {
        if self.blocked.is_some() {
            return Vec::new();
        }
        if self.scan_toxicity(&step.token_text) {
            return vec![SafeStreamEvent::Blocked(
                self.blocked.clone().unwrap_or_default(),
            )];
        }
        if step.phase != GenerationPhase::Answer {
            return Vec::new();
        }
        self.push_visible(&step.token_text, false)
    }

    /// Flush the final safe suffix after generation completes.
    pub fn finish(&mut self) -> Vec<SafeStreamEvent> {
        if let Some(reason) = &self.blocked {
            return vec![SafeStreamEvent::Blocked(reason.clone())];
        }
        self.push_visible("", true)
    }

    /// Return true when streaming was terminally blocked.
    pub fn is_blocked(&self) -> bool {
        self.blocked.is_some()
    }

    /// Return whether at least one PII span was replaced.
    pub fn output_redacted(&self) -> bool {
        self.output_redacted
    }

    fn scan_toxicity(&mut self, fragment: &str) -> bool {
        if !self.policy.check_toxicity {
            return false;
        }
        self.toxicity_tail.push_str(fragment);
        let score = score_toxicity(&self.toxicity_tail);
        if score.is_triggered(self.policy.toxicity_threshold) {
            match self.policy.on_output_violation {
                ViolationAction::Allow | ViolationAction::Warn => {}
                ViolationAction::Redact | ViolationAction::Block | ViolationAction::Regenerate => {
                    self.blocked = Some("toxic output detected".to_string());
                    return true;
                }
            }
        }
        trim_to_last_chars(&mut self.toxicity_tail, TOXICITY_LOOKBEHIND_CHARS);
        false
    }

    fn push_visible(&mut self, fragment: &str, final_flush: bool) -> Vec<SafeStreamEvent> {
        let mut events = Vec::new();
        self.consume_redacted_continuation(fragment, &mut events);
        if self.blocked.is_some() {
            return events;
        }
        self.process_pii(final_flush, &mut events);
        if self.blocked.is_some() {
            return events;
        }

        let boundary = if final_flush {
            self.visible_pending.len()
        } else {
            stable_prefix_len(&self.visible_pending)
        };
        if boundary > 0 {
            let suffix = self.visible_pending.split_off(boundary);
            let safe = std::mem::replace(&mut self.visible_pending, suffix);
            if !safe.is_empty() {
                events.push(SafeStreamEvent::Text(safe));
            }
        }
        events
    }

    fn consume_redacted_continuation(&mut self, fragment: &str, events: &mut Vec<SafeStreamEvent>) {
        let Some(kind) = self.redacting else {
            self.visible_pending.push_str(fragment);
            return;
        };
        let split = fragment
            .char_indices()
            .find_map(|(idx, ch)| (!continues_pii(kind, ch)).then_some(idx));
        match split {
            Some(idx) => {
                self.redacting = None;
                self.visible_pending.push_str(&fragment[idx..]);
            }
            None => {
                if fragment.is_empty() {
                    self.redacting = None;
                }
            }
        }
        if let Some(reason) = &self.blocked {
            events.push(SafeStreamEvent::Blocked(reason.clone()));
        }
    }

    fn process_pii(&mut self, final_flush: bool, events: &mut Vec<SafeStreamEvent>) {
        if self.policy.output_pii == PiiPolicy::Off || self.visible_pending.is_empty() {
            return;
        }
        loop {
            let findings = detect_pii(&self.visible_pending);
            let Some(finding) = findings.items.iter().min_by_key(|finding| finding.start) else {
                return;
            };
            match self.policy.output_pii {
                PiiPolicy::Off | PiiPolicy::Warn => return,
                PiiPolicy::Block => {
                    self.blocked = Some("output PII detected".to_string());
                    events.push(SafeStreamEvent::Blocked(
                        self.blocked.clone().unwrap_or_default(),
                    ));
                    return;
                }
                PiiPolicy::Redact => {
                    let prefix = self.visible_pending[..finding.start].to_string();
                    let suffix = self.visible_pending[finding.end..].to_string();
                    let finding_ends_pending = finding.end == self.visible_pending.len();
                    if !prefix.is_empty() {
                        events.push(SafeStreamEvent::Text(prefix));
                    }
                    events.push(SafeStreamEvent::Text(
                        finding.kind.replacement().to_string(),
                    ));
                    self.output_redacted = true;
                    self.visible_pending = suffix;
                    if finding_ends_pending && !final_flush {
                        self.redacting = Some(finding.kind);
                        return;
                    }
                }
            }
        }
    }
}

fn stable_prefix_len(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let lookbehind_start = char_boundary_before_last(text, TOXICITY_LOOKBEHIND_CHARS);
    let sensitive_start = text
        .char_indices()
        .rev()
        .find_map(|(idx, ch)| (!is_sensitive_tail_char(ch)).then_some(idx + ch.len_utf8()))
        .unwrap_or(0);
    lookbehind_start.min(sensitive_start)
}

fn char_boundary_before_last(text: &str, chars: usize) -> usize {
    text.char_indices()
        .rev()
        .nth(chars.saturating_sub(1))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn trim_to_last_chars(text: &mut String, chars: usize) {
    let start = char_boundary_before_last(text, chars);
    if start > 0 {
        *text = text[start..].to_string();
    }
}

fn is_sensitive_tail_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || matches!(
            ch,
            '@' | '.' | '_' | '%' | '+' | '-' | '(' | ')' | ':' | '/' | '='
        )
        || ch.is_ascii_whitespace()
}

fn continues_pii(kind: PiiKind, ch: char) -> bool {
    match kind {
        PiiKind::Email => ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-'),
        PiiKind::Phone | PiiKind::NationalId | PiiKind::CreditCard => {
            ch.is_ascii_digit() || matches!(ch, ' ' | '-' | '.' | '(' | ')' | '+')
        }
        PiiKind::ApiKey => ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'),
    }
}

#[cfg(test)]
mod tests {
    use aarambh_studio_inference::{GenerationPhase, GenerationStep};

    use super::*;

    fn step(text: &str) -> GenerationStep {
        GenerationStep {
            step: 1,
            token_id: 1,
            token_text: text.to_string(),
            candidates: Vec::new(),
            phase: GenerationPhase::Answer,
            forced: false,
        }
    }

    fn text(events: Vec<SafeStreamEvent>) -> String {
        events
            .into_iter()
            .filter_map(|event| match event {
                SafeStreamEvent::Text(text) => Some(text),
                SafeStreamEvent::Blocked(_) => None,
            })
            .collect()
    }

    #[test]
    fn split_email_is_redacted_before_release() {
        let mut filter = StreamingSafetyFilter::new(SafetyPolicy::strict());
        let mut output = text(filter.push_step(&step("Contact dev")));
        output.push_str(&text(filter.push_step(&step("@example"))));
        output.push_str(&text(filter.push_step(&step(".com now."))));
        output.push_str(&text(filter.finish()));
        assert!(!output.contains("dev@example.com"));
        assert!(output.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn split_toxic_phrase_blocks_before_phrase_release() {
        let mut filter = StreamingSafetyFilter::new(SafetyPolicy::strict());
        let mut events = filter.push_step(&step("I will "));
        events.extend(filter.push_step(&step("kill you")));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SafeStreamEvent::Blocked(_)))
        );
        assert!(!text(events).to_ascii_lowercase().contains("kill you"));
    }
}
