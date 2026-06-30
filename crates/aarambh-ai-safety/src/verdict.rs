#[derive(Debug, Clone, PartialEq, Eq)]
/// Safety decision produced by input or output checks.
pub enum SafetyVerdict {
    /// Allow content unchanged.
    Allow,
    /// Block content with reason.
    Block(String),
    /// Redact content with reason.
    Redact(String),
    /// Regenerate content with reason.
    Regenerate(String),
}

impl SafetyVerdict {
    /// Return the stable verdict label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block(_) => "block",
            Self::Redact(_) => "redact",
            Self::Regenerate(_) => "regenerate",
        }
    }

    /// Return the verdict reason, when present.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::Block(reason) | Self::Redact(reason) | Self::Regenerate(reason) => Some(reason),
        }
    }
}
