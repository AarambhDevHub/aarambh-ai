use crate::input::pii::{PiiFindings, detect_pii, redact_pii};

/// Detect PII in generated output text.
pub fn detect_output_pii(text: &str) -> PiiFindings {
    detect_pii(text)
}

/// Redact PII in generated output text.
pub fn redact_output_pii(text: &str, findings: &PiiFindings) -> String {
    redact_pii(text, findings)
}
