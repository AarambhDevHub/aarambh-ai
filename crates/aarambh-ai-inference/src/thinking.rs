use aarambh_ai_tokenizer::{THINK_END_ID, THINK_START_ID};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    None,
    Low,
    Medium,
    High,
}

impl ThinkingMode {
    pub fn budget(self) -> usize {
        match self {
            Self::None => 0,
            Self::Low => 256,
            Self::Medium => 1024,
            Self::High => 4096,
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceToken {
    ThinkEnd,
}

#[derive(Debug, Clone)]
pub struct ThinkingController {
    mode: ThinkingMode,
    in_thinking_block: bool,
    tokens_used: usize,
    started: bool,
}

impl ThinkingController {
    pub fn new(mode: ThinkingMode) -> Self {
        Self {
            mode,
            in_thinking_block: false,
            tokens_used: 0,
            started: false,
        }
    }

    pub fn mode(&self) -> ThinkingMode {
        self.mode
    }

    pub fn in_thinking_block(&self) -> bool {
        self.in_thinking_block
    }

    pub fn tokens_used(&self) -> usize {
        self.tokens_used
    }

    pub fn should_force_think_start(&self) -> bool {
        false
    }

    pub fn on_token(&mut self, token_id: u32) -> Option<ForceToken> {
        if self.mode == ThinkingMode::None {
            return None;
        }
        if token_id == THINK_START_ID {
            self.started = true;
            self.in_thinking_block = true;
            self.tokens_used = 0;
            return None;
        }
        if !self.in_thinking_block {
            return None;
        }
        if token_id == THINK_END_ID {
            self.in_thinking_block = false;
            return None;
        }
        self.tokens_used += 1;
        if self.tokens_used > self.mode.budget() {
            self.in_thinking_block = false;
            return Some(ForceToken::ThinkEnd);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_low_budget_is_recorded() {
        let ctrl = ThinkingController::new(ThinkingMode::Low);
        assert_eq!(ctrl.mode().budget(), 256);
        assert!(!ctrl.should_force_think_start());
    }

    #[test]
    fn think_end_token_closes_block() {
        let mut ctrl = ThinkingController::new(ThinkingMode::Medium);
        ctrl.on_token(THINK_START_ID);
        assert!(ctrl.in_thinking_block());
        ctrl.on_token(THINK_END_ID);
        assert!(!ctrl.in_thinking_block());
    }

    #[test]
    fn thinking_none_never_opens_block() {
        let mut ctrl = ThinkingController::new(ThinkingMode::None);
        ctrl.on_token(THINK_START_ID);
        assert!(!ctrl.in_thinking_block());
    }
}
