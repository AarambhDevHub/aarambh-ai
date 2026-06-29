use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiPolicy {
    Off,
    Warn,
    Redact,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationAction {
    Allow,
    Warn,
    Redact,
    Block,
    Regenerate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyMode {
    Strict,
    Permissive,
    Research,
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
pub struct SafetyPolicy {
    pub check_prompt_injection: bool,
    pub injection_threshold: f32,
    pub check_jailbreak: bool,
    pub jailbreak_threshold: f32,
    pub input_pii: PiiPolicy,
    pub max_prompt_chars: Option<usize>,
    pub check_toxicity: bool,
    pub toxicity_threshold: f32,
    pub output_pii: PiiPolicy,
    pub on_input_violation: ViolationAction,
    pub on_output_violation: ViolationAction,
    pub max_regenerations: usize,
    pub audit_enabled: bool,
    pub audit_path: Option<PathBuf>,
}

impl SafetyPolicy {
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

    pub fn for_mode(mode: SafetyMode) -> Option<Self> {
        match mode {
            SafetyMode::Strict => Some(Self::strict()),
            SafetyMode::Permissive => Some(Self::permissive()),
            SafetyMode::Research => Some(Self::research()),
            SafetyMode::None => None,
        }
    }

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
