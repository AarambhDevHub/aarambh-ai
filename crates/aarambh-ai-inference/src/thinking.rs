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
    ThinkStart,
    ThinkEnd,
}

impl ForceToken {
    pub fn token_id(self) -> u32 {
        match self {
            Self::ThinkStart => THINK_START_ID,
            Self::ThinkEnd => THINK_END_ID,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThinkingController {
    mode: ThinkingMode,
    in_thinking_block: bool,
    tokens_used: usize,
    started: bool,
    closed: bool,
    budget: usize,
    pending_force: Option<ForceToken>,
}

impl ThinkingController {
    pub fn new(mode: ThinkingMode) -> Self {
        Self::with_budget(mode, mode.budget())
    }

    pub fn for_generation(mode: ThinkingMode, max_new_tokens: usize) -> Self {
        let budget = if mode.is_enabled() {
            mode.budget().min(max_new_tokens.saturating_sub(32))
        } else {
            0
        };
        Self::with_budget(mode, budget)
    }

    fn with_budget(mode: ThinkingMode, budget: usize) -> Self {
        Self {
            mode,
            in_thinking_block: false,
            tokens_used: 0,
            started: false,
            closed: false,
            budget,
            pending_force: None,
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

    pub fn effective_budget(&self) -> usize {
        self.budget
    }

    pub fn has_started(&self) -> bool {
        self.started
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn should_force_think_start(&self) -> bool {
        self.mode.is_enabled() && !self.started && !self.closed && self.pending_force.is_none()
    }

    pub fn take_forced_token(&mut self) -> Option<ForceToken> {
        self.pending_force.take().or_else(|| {
            self.should_force_think_start()
                .then_some(ForceToken::ThinkStart)
        })
    }

    pub fn on_token(&mut self, token_id: u32) -> Option<ForceToken> {
        if self.mode == ThinkingMode::None {
            return None;
        }
        if token_id == THINK_START_ID && !self.started {
            self.started = true;
            self.in_thinking_block = true;
            self.tokens_used = 0;
            if self.budget == 0 {
                self.pending_force = Some(ForceToken::ThinkEnd);
                return self.pending_force;
            }
            return None;
        }
        if !self.in_thinking_block {
            return None;
        }
        if token_id == THINK_END_ID {
            self.in_thinking_block = false;
            self.closed = true;
            return None;
        }
        self.tokens_used += 1;
        if self.tokens_used >= self.budget {
            self.pending_force = Some(ForceToken::ThinkEnd);
            return self.pending_force;
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
        assert_eq!(ctrl.effective_budget(), 256);
        assert!(ctrl.should_force_think_start());
    }

    #[test]
    fn think_end_token_closes_block() {
        let mut ctrl = ThinkingController::new(ThinkingMode::Medium);
        ctrl.on_token(THINK_START_ID);
        assert!(ctrl.in_thinking_block());
        ctrl.on_token(THINK_END_ID);
        assert!(!ctrl.in_thinking_block());
        assert!(ctrl.is_closed());
    }

    #[test]
    fn thinking_none_never_opens_block() {
        let mut ctrl = ThinkingController::new(ThinkingMode::None);
        ctrl.on_token(THINK_START_ID);
        assert!(!ctrl.in_thinking_block());
    }

    #[test]
    fn thinking_start_forced_only_once() {
        let mut ctrl = ThinkingController::new(ThinkingMode::Low);
        assert_eq!(ctrl.take_forced_token(), Some(ForceToken::ThinkStart));
        ctrl.on_token(THINK_START_ID);
        assert_eq!(ctrl.take_forced_token(), None);
    }

    #[test]
    fn thinking_low_budget_is_enforced() {
        let mut ctrl = ThinkingController::new(ThinkingMode::Low);
        ctrl.on_token(THINK_START_ID);
        for _ in 0..255 {
            assert_eq!(ctrl.on_token(42), None);
        }
        assert_eq!(ctrl.on_token(42), Some(ForceToken::ThinkEnd));
        assert_eq!(ctrl.take_forced_token(), Some(ForceToken::ThinkEnd));
        ctrl.on_token(THINK_END_ID);
        assert!(!ctrl.in_thinking_block());
        assert!(ctrl.is_closed());
        assert_eq!(ctrl.tokens_used(), 256);
    }

    #[test]
    fn forced_tokens_do_not_count_as_thinking_content() {
        let mut ctrl = ThinkingController::for_generation(ThinkingMode::Low, 4);
        assert_eq!(ctrl.effective_budget(), 0);
        ctrl.on_token(THINK_START_ID);
        assert_eq!(ctrl.tokens_used(), 0);
        assert_eq!(ctrl.take_forced_token(), Some(ForceToken::ThinkEnd));
        ctrl.on_token(THINK_END_ID);
        assert_eq!(ctrl.tokens_used(), 0);
    }

    #[test]
    fn thinking_budgets_increase_by_mode() {
        assert!(ThinkingMode::Medium.budget() > ThinkingMode::Low.budget());
        assert!(ThinkingMode::High.budget() > ThinkingMode::Medium.budget());
    }
}
