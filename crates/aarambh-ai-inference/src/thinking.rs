use std::fmt;
use std::str::FromStr;

use aarambh_ai_tokenizer::{THINK_END_ID, THINK_START_ID};

/// Canonical lower-case spelling of the `none` thinking mode.
pub const MODE_NONE: &str = "none";
/// Canonical lower-case spelling of the `low` thinking mode.
pub const MODE_LOW: &str = "low";
/// Canonical lower-case spelling of the `medium` thinking mode.
pub const MODE_MEDIUM: &str = "medium";
/// Canonical lower-case spelling of the `high` thinking mode.
pub const MODE_HIGH: &str = "high";
/// Canonical lower-case spelling of the `max` thinking mode.
pub const MODE_MAX: &str = "max";

/// Parse a thinking-mode token (case-insensitive) into a [`ThinkingMode`].
///
/// This is the single canonical parser shared by every CLI command and the
/// serving API so that the accepted vocabulary stays identical everywhere.
pub fn parse_thinking_mode(value: &str) -> Result<ThinkingMode, String> {
    ThinkingMode::from_str(value)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Thinking budget mode.
pub enum ThinkingMode {
    /// Disable thinking markers and budget.
    #[default]
    None,
    /// Low thinking budget.
    Low,
    /// Medium thinking budget.
    Medium,
    /// High thinking budget.
    High,
    /// Max thinking budget (Phase 39).
    Max,
}

impl ThinkingMode {
    /// Return the nominal token budget for this mode.
    pub fn budget(self) -> usize {
        match self {
            Self::None => 0,
            Self::Low => 256,
            Self::Medium => 1024,
            Self::High => 4096,
            Self::Max => 16384,
        }
    }

    /// Return true when thinking is enabled.
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Return the default `(temperature, top_p)` sampling pair for this mode.
    ///
    /// These defaults extend v1's per-mode table (`ARCHITECTURE_V3.md` §48.3).
    /// They are only applied when the caller does not supply explicit sampling
    /// parameters and never override user-provided values.
    pub fn default_sampler(self) -> (f32, f32) {
        match self {
            Self::None => (0.70, 0.90),
            Self::Low => (0.75, 0.92),
            Self::Medium => (0.80, 0.95),
            Self::High => (0.80, 0.95),
            Self::Max => (0.85, 0.97),
        }
    }
}

impl FromStr for ThinkingMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            MODE_NONE => Ok(Self::None),
            MODE_LOW => Ok(Self::Low),
            MODE_MEDIUM => Ok(Self::Medium),
            MODE_HIGH => Ok(Self::High),
            MODE_MAX => Ok(Self::Max),
            other => Err(format!(
                "invalid thinking mode '{other}', expected none|low|medium|high|max"
            )),
        }
    }
}

impl fmt::Display for ThinkingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spelling = match self {
            Self::None => MODE_NONE,
            Self::Low => MODE_LOW,
            Self::Medium => MODE_MEDIUM,
            Self::High => MODE_HIGH,
            Self::Max => MODE_MAX,
        };
        f.write_str(spelling)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Token that should be forced by the thinking controller.
pub enum ForceToken {
    /// Force the thinking start marker.
    ThinkStart,
    /// Force the thinking end marker.
    ThinkEnd,
}

impl ForceToken {
    /// Return the tokenizer id for this forced token.
    pub fn token_id(self) -> u32 {
        match self {
            Self::ThinkStart => THINK_START_ID,
            Self::ThinkEnd => THINK_END_ID,
        }
    }
}

#[derive(Debug, Clone)]
/// Tracks thinking marker state and budget enforcement.
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
    /// Create a controller with the mode's nominal budget.
    pub fn new(mode: ThinkingMode) -> Self {
        Self::with_budget(mode, mode.budget())
    }

    /// Create a controller clamped to a generation token budget.
    pub fn for_generation(mode: ThinkingMode, max_new_tokens: usize) -> Self {
        Self::for_generation_with_reserve(mode, max_new_tokens, 32)
    }

    /// Create a controller while reserving tokens for a post-thinking action.
    pub fn for_generation_with_reserve(
        mode: ThinkingMode,
        max_new_tokens: usize,
        reserve: usize,
    ) -> Self {
        let budget = if mode.is_enabled() {
            mode.budget().min(max_new_tokens.saturating_sub(reserve))
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

    /// Return the configured thinking mode.
    pub fn mode(&self) -> ThinkingMode {
        self.mode
    }

    /// Return true when currently inside a thinking block.
    pub fn in_thinking_block(&self) -> bool {
        self.in_thinking_block
    }

    /// Return thinking content tokens used so far.
    pub fn tokens_used(&self) -> usize {
        self.tokens_used
    }

    /// Return the effective generation-time thinking budget.
    pub fn effective_budget(&self) -> usize {
        self.budget
    }

    /// Return true after the thinking block has started.
    pub fn has_started(&self) -> bool {
        self.started
    }

    /// Return true after the thinking block has closed.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Return true when a thinking start marker should be forced.
    pub fn should_force_think_start(&self) -> bool {
        self.mode.is_enabled() && !self.started && !self.closed && self.pending_force.is_none()
    }

    /// Take the next forced token, if one is pending or required.
    pub fn take_forced_token(&mut self) -> Option<ForceToken> {
        self.pending_force.take().or_else(|| {
            self.should_force_think_start()
                .then_some(ForceToken::ThinkStart)
        })
    }

    /// Update controller state after a token and return any pending force.
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
        assert!(ThinkingMode::Max.budget() > ThinkingMode::High.budget());
    }

    #[test]
    fn thinking_mode_max_budget_is_16384_tokens() {
        assert_eq!(ThinkingMode::Max.budget(), 16384);
        assert!(ThinkingMode::Max.is_enabled());
    }

    #[test]
    fn thinking_mode_parser_accepts_all_five_modes_case_insensitively() {
        assert_eq!("none".parse::<ThinkingMode>().unwrap(), ThinkingMode::None);
        assert_eq!("LOW".parse::<ThinkingMode>().unwrap(), ThinkingMode::Low);
        assert_eq!(
            "Medium".parse::<ThinkingMode>().unwrap(),
            ThinkingMode::Medium
        );
        assert_eq!("high".parse::<ThinkingMode>().unwrap(), ThinkingMode::High);
        assert_eq!("max".parse::<ThinkingMode>().unwrap(), ThinkingMode::Max);
        assert_eq!(
            "  MaX  ".parse::<ThinkingMode>().unwrap(),
            ThinkingMode::Max
        );
    }

    #[test]
    fn thinking_mode_parser_rejects_unknown_values() {
        assert!("ultra".parse::<ThinkingMode>().is_err());
        assert!("higher".parse::<ThinkingMode>().is_err());
        assert!("".parse::<ThinkingMode>().is_err());
    }

    #[test]
    fn thinking_mode_display_outputs_canonical_lower_case() {
        assert_eq!(ThinkingMode::None.to_string(), "none");
        assert_eq!(ThinkingMode::Low.to_string(), "low");
        assert_eq!(ThinkingMode::Medium.to_string(), "medium");
        assert_eq!(ThinkingMode::High.to_string(), "high");
        assert_eq!(ThinkingMode::Max.to_string(), "max");
    }

    #[test]
    fn thinking_mode_display_round_trips_through_parser() {
        for mode in [
            ThinkingMode::None,
            ThinkingMode::Low,
            ThinkingMode::Medium,
            ThinkingMode::High,
            ThinkingMode::Max,
        ] {
            assert_eq!(mode.to_string().parse::<ThinkingMode>().unwrap(), mode);
        }
    }

    #[test]
    fn max_mode_sampling_defaults_are_more_exploratory_than_high_mode() {
        let (high_temp, high_top_p) = ThinkingMode::High.default_sampler();
        let (max_temp, max_top_p) = ThinkingMode::Max.default_sampler();
        assert!(max_temp > high_temp, "max temperature must exceed high");
        assert!(max_top_p > high_top_p, "max top_p must exceed high");
        // The full per-mode table from ARCHITECTURE_V3.md §48.3.
        assert_eq!(ThinkingMode::None.default_sampler(), (0.70, 0.90));
        assert_eq!(ThinkingMode::Low.default_sampler(), (0.75, 0.92));
        assert_eq!(ThinkingMode::Medium.default_sampler(), (0.80, 0.95));
        assert_eq!(ThinkingMode::High.default_sampler(), (0.80, 0.95));
        assert_eq!(ThinkingMode::Max.default_sampler(), (0.85, 0.97));
    }

    #[test]
    fn thinking_controller_force_closes_max_mode_at_budget_exactly_like_other_modes() {
        // No special-cased logic path for Max — same on_token()/
        // take_forced_token() mechanism as None/Low/Medium/High.
        let mut ctrl = ThinkingController::new(ThinkingMode::Max);
        assert_eq!(ctrl.effective_budget(), 16384);
        assert_eq!(ctrl.take_forced_token(), Some(ForceToken::ThinkStart));
        ctrl.on_token(THINK_START_ID);
        assert!(ctrl.in_thinking_block());
        // Feed up to (budget - 1) content tokens without force-closing.
        for _ in 0..(16384 - 1) {
            assert_eq!(ctrl.on_token(42), None);
        }
        // The budget-th content token forces the closing marker.
        assert_eq!(ctrl.on_token(42), Some(ForceToken::ThinkEnd));
        assert_eq!(ctrl.take_forced_token(), Some(ForceToken::ThinkEnd));
        ctrl.on_token(THINK_END_ID);
        assert!(!ctrl.in_thinking_block());
        assert!(ctrl.is_closed());
        assert_eq!(ctrl.tokens_used(), 16384);
    }
}
