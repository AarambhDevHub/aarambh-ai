//! Input and output safety filters, policy controls, verdicts, and audit logging.
#![deny(missing_docs)]

/// Safety guard wrapper around generation engines.
pub mod guard;
/// Prompt-side safety detectors.
pub mod input;
/// Output-side safety detectors and audit records.
pub mod output;
/// Safety policy presets and actions.
pub mod policy;
/// Safety verdict types.
pub mod verdict;

pub use guard::{SafeResponse, SafetyGenerator, SafetyGuard};
pub use input::{
    InjectionScore, JailbreakScore, PiiFinding, PiiFindings, PiiKind, detect_injection,
    detect_jailbreak, detect_pii, redact_pii,
};
pub use output::{
    SafetyEvent, SafetyStage, ToxicityCategory, ToxicityScore, hash_prompt, log_event,
    score_toxicity,
};
pub use policy::{PiiPolicy, SafetyMode, SafetyPolicy, ViolationAction};
pub use verdict::SafetyVerdict;
