#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Supported personally identifiable information kind.
pub enum PiiKind {
    /// Email address.
    Email,
    /// Phone number.
    Phone,
    /// National identifier such as SSN-style patterns.
    NationalId,
    /// Credit-card number.
    CreditCard,
    /// API key or secret-like token.
    ApiKey,
}

impl PiiKind {
    /// Return the redaction replacement string.
    pub fn replacement(self) -> &'static str {
        match self {
            Self::Email => "[REDACTED_EMAIL]",
            Self::Phone => "[REDACTED_PHONE]",
            Self::NationalId => "[REDACTED_ID]",
            Self::CreditCard => "[REDACTED_CARD]",
            Self::ApiKey => "[REDACTED_SECRET]",
        }
    }

    /// Return a stable label for rule identifiers.
    pub fn label(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::NationalId => "national_id",
            Self::CreditCard => "credit_card",
            Self::ApiKey => "api_key",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// One detected PII span.
pub struct PiiFinding {
    /// PII kind.
    pub kind: PiiKind,
    /// Byte start offset.
    pub start: usize,
    /// Byte end offset.
    pub end: usize,
    /// Detector confidence.
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Collection of PII findings.
pub struct PiiFindings {
    /// Detected PII spans.
    pub items: Vec<PiiFinding>,
}

impl PiiFindings {
    /// Return true when there are no findings.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Return rule ids for each finding with a prefix.
    pub fn rules(&self, prefix: &str) -> Vec<String> {
        self.items
            .iter()
            .map(|item| format!("{prefix}.pii.{}", item.kind.label()))
            .collect()
    }
}

/// Detect PII spans in text.
pub fn detect_pii(text: &str) -> PiiFindings {
    let mut items = Vec::new();
    detect_emails(text, &mut items);
    detect_credit_cards(text, &mut items);
    detect_national_ids(text, &mut items);
    detect_phones(text, &mut items);
    detect_api_keys(text, &mut items);
    PiiFindings {
        items: dedupe_findings(items),
    }
}

/// Redact detected PII spans from text.
pub fn redact_pii(text: &str, findings: &PiiFindings) -> String {
    if findings.items.is_empty() {
        return text.to_string();
    }
    let mut items = findings.items.clone();
    items.sort_by_key(|item| item.start);

    let mut redacted = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for item in items {
        if item.start < cursor || item.end > text.len() || !text.is_char_boundary(item.start) {
            continue;
        }
        if !text.is_char_boundary(item.end) {
            continue;
        }
        redacted.push_str(&text[cursor..item.start]);
        redacted.push_str(item.kind.replacement());
        cursor = item.end;
    }
    redacted.push_str(&text[cursor..]);
    redacted
}

fn detect_emails(text: &str, items: &mut Vec<PiiFinding>) {
    let bytes = text.as_bytes();
    for at in bytes
        .iter()
        .enumerate()
        .filter_map(|(idx, b)| (*b == b'@').then_some(idx))
    {
        let mut start = at;
        while start > 0 && is_email_local(bytes[start - 1]) {
            start -= 1;
        }
        let mut end = at + 1;
        while end < bytes.len() && is_email_domain(bytes[end]) {
            end += 1;
        }
        while end > at + 1 && matches!(bytes[end - 1], b'.' | b'-') {
            end -= 1;
        }
        if start < at && is_valid_email_domain(&text[at + 1..end]) {
            items.push(PiiFinding {
                kind: PiiKind::Email,
                start,
                end,
                confidence: 0.98,
            });
        }
    }
}

fn detect_credit_cards(text: &str, items: &mut Vec<PiiFinding>) {
    for (start, end, digits) in digitish_spans(text, 13, 32) {
        if (13..=19).contains(&digits.len()) && luhn_valid(&digits) {
            items.push(PiiFinding {
                kind: PiiKind::CreditCard,
                start,
                end,
                confidence: 0.99,
            });
        }
    }
}

fn detect_national_ids(text: &str, items: &mut Vec<PiiFinding>) {
    let bytes = text.as_bytes();
    if bytes.len() >= 11 {
        for idx in 0..=bytes.len() - 11 {
            if idx > 0 && bytes[idx - 1].is_ascii_digit() {
                continue;
            }
            if idx + 11 < bytes.len() && bytes[idx + 11].is_ascii_digit() {
                continue;
            }
            if bytes[idx..idx + 3].iter().all(u8::is_ascii_digit)
                && bytes[idx + 3] == b'-'
                && bytes[idx + 4..idx + 6].iter().all(u8::is_ascii_digit)
                && bytes[idx + 6] == b'-'
                && bytes[idx + 7..idx + 11].iter().all(u8::is_ascii_digit)
                && &text[idx..idx + 3] != "000"
                && &text[idx + 4..idx + 6] != "00"
                && &text[idx + 7..idx + 11] != "0000"
            {
                items.push(PiiFinding {
                    kind: PiiKind::NationalId,
                    start: idx,
                    end: idx + 11,
                    confidence: 0.98,
                });
            }
        }
    }

    for (start, end, digits) in digitish_spans(text, 12, 16) {
        if digits.len() == 12
            && text[start..end].chars().any(|ch| ch.is_ascii_whitespace())
            && !luhn_valid(&digits)
        {
            items.push(PiiFinding {
                kind: PiiKind::NationalId,
                start,
                end,
                confidence: 0.75,
            });
        }
    }
}

fn detect_phones(text: &str, items: &mut Vec<PiiFinding>) {
    for (start, end, digits) in digitish_spans(text, 7, 24) {
        if !(7..=15).contains(&digits.len()) || digits.len() >= 13 {
            continue;
        }
        let candidate = &text[start..end];
        let has_phone_shape = candidate.starts_with('+')
            || candidate.contains('(')
            || candidate.contains('-')
            || candidate.matches(' ').count() >= 2;
        if has_phone_shape {
            items.push(PiiFinding {
                kind: PiiKind::Phone,
                start,
                end,
                confidence: 0.82,
            });
        }
    }
}

fn detect_api_keys(text: &str, items: &mut Vec<PiiFinding>) {
    for (start, end, token) in token_spans(text) {
        if token.len() < 10 {
            continue;
        }
        let known_prefix = ["sk-", "ghp_", "github_pat_", "xoxb-", "AKIA"]
            .iter()
            .any(|prefix| token.starts_with(prefix));
        let high_entropy = token.len() >= 32 && looks_high_entropy(token);
        if known_prefix || high_entropy {
            items.push(PiiFinding {
                kind: PiiKind::ApiKey,
                start,
                end,
                confidence: if known_prefix { 0.98 } else { 0.72 },
            });
        }
    }
}

fn digitish_spans(text: &str, min_span: usize, max_span: usize) -> Vec<(usize, usize, String)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < bytes.len() {
        if !is_digitish(bytes[idx]) {
            idx += 1;
            continue;
        }
        let start = idx;
        let mut end = idx;
        let mut digits = String::new();
        while end < bytes.len() && is_digitish(bytes[end]) && end - start <= max_span {
            if bytes[end].is_ascii_digit() {
                digits.push(bytes[end] as char);
            }
            end += 1;
        }
        let (trimmed_start, trimmed_end) = trim_span(text, start, end);
        if trimmed_end > trimmed_start && trimmed_end - trimmed_start >= min_span {
            spans.push((trimmed_start, trimmed_end, digits));
        }
        idx = end.max(start + 1);
    }
    spans
}

fn token_spans(text: &str) -> Vec<(usize, usize, &str)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if is_secret_token_char(ch) {
            start.get_or_insert(idx);
        } else if let Some(span_start) = start.take()
            && idx > span_start
        {
            spans.push((span_start, idx, &text[span_start..idx]));
        }
    }
    if let Some(span_start) = start {
        spans.push((span_start, text.len(), &text[span_start..]));
    }
    spans
}

fn trim_span(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    let bytes = text.as_bytes();
    while start < end && !bytes[start].is_ascii_digit() && bytes[start] != b'+' {
        start += 1;
    }
    while end > start && !bytes[end - 1].is_ascii_digit() && bytes[end - 1] != b')' {
        end -= 1;
    }
    (start, end)
}

fn dedupe_findings(mut items: Vec<PiiFinding>) -> Vec<PiiFinding> {
    items.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then_with(|| b.confidence.total_cmp(&a.confidence))
            .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
    });
    let mut deduped: Vec<PiiFinding> = Vec::new();
    for item in items {
        if let Some(existing) = deduped
            .iter_mut()
            .find(|existing| overlaps(existing, &item))
        {
            if item.confidence > existing.confidence {
                *existing = item;
            }
        } else {
            deduped.push(item);
        }
    }
    deduped.sort_by_key(|item| item.start);
    deduped
}

fn overlaps(a: &PiiFinding, b: &PiiFinding) -> bool {
    a.start < b.end && b.start < a.end
}

fn is_email_local(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'%' | b'+' | b'-')
}

fn is_email_domain(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-')
}

fn is_valid_email_domain(domain: &str) -> bool {
    let Some((_, tld)) = domain.rsplit_once('.') else {
        return false;
    };
    tld.len() >= 2
        && tld.bytes().all(|b| b.is_ascii_alphabetic())
        && domain
            .split('.')
            .all(|part| !part.is_empty() && !part.starts_with('-') && !part.ends_with('-'))
}

fn is_digitish(byte: u8) -> bool {
    byte.is_ascii_digit() || matches!(byte, b' ' | b'-' | b'.' | b'(' | b')' | b'+')
}

fn is_secret_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')
}

fn luhn_valid(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut double = false;
    for digit in digits.bytes().rev() {
        let mut value = (digit - b'0') as u32;
        if double {
            value *= 2;
            if value > 9 {
                value -= 9;
            }
        }
        sum += value;
        double = !double;
    }
    sum != 0 && sum.is_multiple_of(10)
}

fn looks_high_entropy(token: &str) -> bool {
    let lower = token.bytes().any(|b| b.is_ascii_lowercase());
    let upper = token.bytes().any(|b| b.is_ascii_uppercase());
    let digit = token.bytes().any(|b| b.is_ascii_digit());
    let symbol = token.bytes().any(|b| matches!(b, b'-' | b'_' | b'.'));
    let diversity = [lower, upper, digit, symbol]
        .iter()
        .filter(|enabled| **enabled)
        .count();
    diversity >= 3 && shannon_entropy(token) >= 3.5
}

fn shannon_entropy(token: &str) -> f32 {
    let mut counts = [0usize; 128];
    let mut total = 0usize;
    for byte in token.bytes().filter(u8::is_ascii) {
        counts[byte as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|count| **count > 0)
        .map(|count| {
            let p = *count as f32 / total as f32;
            -p * p.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_redacts_email() {
        let findings = detect_pii("email me at dev@example.com please");
        assert!(
            findings
                .items
                .iter()
                .any(|item| item.kind == PiiKind::Email)
        );
        assert_eq!(
            redact_pii("email me at dev@example.com please", &findings),
            "email me at [REDACTED_EMAIL] please"
        );
    }

    #[test]
    fn detects_credit_card_with_luhn() {
        let findings = detect_pii("card 4111-1111-1111-1111");
        assert!(
            findings
                .items
                .iter()
                .any(|item| item.kind == PiiKind::CreditCard)
        );
        let invalid = detect_pii("card 4111-1111-1111-1112");
        assert!(
            !invalid
                .items
                .iter()
                .any(|item| item.kind == PiiKind::CreditCard)
        );
    }

    #[test]
    fn detects_api_key_prefix() {
        let findings = detect_pii("token sk-test_123456789abcdef");
        assert!(
            findings
                .items
                .iter()
                .any(|item| item.kind == PiiKind::ApiKey)
        );
    }

    #[test]
    fn detects_phone_number() {
        let findings = detect_pii("call +1 (415) 555-0100");
        assert!(
            findings
                .items
                .iter()
                .any(|item| item.kind == PiiKind::Phone)
        );
    }
}
