#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyVerdict {
    Allow,
    Block(String),
    Redact(String),
    Regenerate(String),
}

impl SafetyVerdict {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block(_) => "block",
            Self::Redact(_) => "redact",
            Self::Regenerate(_) => "regenerate",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::Block(reason) | Self::Redact(reason) | Self::Regenerate(reason) => Some(reason),
        }
    }
}
