use crate::input::pii::{PiiFindings, detect_pii, redact_pii};

pub fn detect_output_pii(text: &str) -> PiiFindings {
    detect_pii(text)
}

pub fn redact_output_pii(text: &str, findings: &PiiFindings) -> String {
    redact_pii(text, findings)
}
