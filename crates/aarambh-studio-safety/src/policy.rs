use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Policy for handling detected PII.
pub enum PiiPolicy {
    /// Disable PII handling.
    Off,
    /// Record a warning only.
    Warn,
    /// Redact PII and continue.
    Redact,
    /// Block when PII is detected.
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Action taken after a safety violation.
pub enum ViolationAction {
    /// Allow the content.
    Allow,
    /// Warn and continue.
    Warn,
    /// Redact and continue.
    Redact,
    /// Block the response.
    Block,
    /// Try regenerating the response.
    Regenerate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Named safety policy preset.
pub enum SafetyMode {
    /// Strict production policy.
    Strict,
    /// Less aggressive production policy.
    Permissive,
    /// Research policy that audits but usually allows.
    Research,
    /// Disable safety guard.
    None,
}

impl FromStr for SafetyMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "strict" => Ok(Self::Strict),
            "permissive" => Ok(Self::Permissive),
            "research" => Ok(Self::Research),
            "none" | "off" | "disabled" => Ok(Self::None),
            other => Err(format!(
                "invalid safety mode '{other}', expected strict|permissive|research|none"
            )),
        }
    }
}

#[derive(Debug, Clone)]
/// Safety policy thresholds, actions, and audit configuration.
pub struct SafetyPolicy {
    /// Whether to check prompt-injection rules.
    pub check_prompt_injection: bool,
    /// Prompt-injection block threshold.
    pub injection_threshold: f32,
    /// Whether to check jailbreak rules.
    pub check_jailbreak: bool,
    /// Jailbreak block threshold.
    pub jailbreak_threshold: f32,
    /// Input PII policy.
    pub input_pii: PiiPolicy,
    /// Optional maximum prompt length in characters.
    pub max_prompt_chars: Option<usize>,
    /// Whether to check output toxicity.
    pub check_toxicity: bool,
    /// Output toxicity threshold.
    pub toxicity_threshold: f32,
    /// Output PII policy.
    pub output_pii: PiiPolicy,
    /// Action for input violations.
    pub on_input_violation: ViolationAction,
    /// Action for output violations.
    pub on_output_violation: ViolationAction,
    /// Maximum output regenerations after unsafe output.
    pub max_regenerations: usize,
    /// Whether to write safety audit events.
    pub audit_enabled: bool,
    /// Optional JSONL audit log path.
    pub audit_path: Option<PathBuf>,
}

impl SafetyPolicy {
    /// Return the strict production policy.
    pub fn strict() -> Self {
        Self {
            check_prompt_injection: true,
            injection_threshold: 0.65,
            check_jailbreak: true,
            jailbreak_threshold: 0.65,
            input_pii: PiiPolicy::Redact,
            max_prompt_chars: Some(16_384),
            check_toxicity: true,
            toxicity_threshold: 0.70,
            output_pii: PiiPolicy::Redact,
            on_input_violation: ViolationAction::Block,
            on_output_violation: ViolationAction::Regenerate,
            max_regenerations: 3,
            audit_enabled: true,
            audit_path: Some(PathBuf::from("safety_audit.jsonl")),
        }
    }

    /// Return the permissive production policy.
    pub fn permissive() -> Self {
        Self {
            check_prompt_injection: true,
            injection_threshold: 0.85,
            check_jailbreak: true,
            jailbreak_threshold: 0.85,
            input_pii: PiiPolicy::Warn,
            max_prompt_chars: Some(32_768),
            check_toxicity: true,
            toxicity_threshold: 0.90,
            output_pii: PiiPolicy::Redact,
            on_input_violation: ViolationAction::Block,
            on_output_violation: ViolationAction::Block,
            max_regenerations: 1,
            audit_enabled: true,
            audit_path: Some(PathBuf::from("safety_audit.jsonl")),
        }
    }

    /// Return the research policy.
    pub fn research() -> Self {
        Self {
            check_prompt_injection: true,
            injection_threshold: 0.65,
            check_jailbreak: true,
            jailbreak_threshold: 0.65,
            input_pii: PiiPolicy::Warn,
            max_prompt_chars: None,
            check_toxicity: true,
            toxicity_threshold: 0.70,
            output_pii: PiiPolicy::Warn,
            on_input_violation: ViolationAction::Allow,
            on_output_violation: ViolationAction::Allow,
            max_regenerations: 0,
            audit_enabled: true,
            audit_path: Some(PathBuf::from("safety_audit.jsonl")),
        }
    }

    /// Return a policy for a named mode, or `None` for disabled safety.
    pub fn for_mode(mode: SafetyMode) -> Option<Self> {
        match mode {
            SafetyMode::Strict => Some(Self::strict()),
            SafetyMode::Permissive => Some(Self::permissive()),
            SafetyMode::Research => Some(Self::research()),
            SafetyMode::None => None,
        }
    }

    /// Override the audit log path.
    pub fn with_audit_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.audit_path = Some(path.into());
        self
    }
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self::strict()
    }
}
