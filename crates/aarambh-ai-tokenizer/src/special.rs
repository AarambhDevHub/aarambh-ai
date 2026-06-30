/// End-of-text token string.
pub const ENDOFTEXT: &str = "<|endoftext|>";
/// Padding token string.
pub const PAD: &str = "<|pad|>";
/// Beginning-of-sequence token string.
pub const BOS: &str = "<|bos|>";
/// Thinking-section start token string.
pub const THINK_START: &str = "<think>";
/// Thinking-section end token string.
pub const THINK_END: &str = "</think>";
/// User role token string.
pub const USER: &str = "<|user|>";
/// Assistant role token string.
pub const ASSISTANT: &str = "<|assistant|>";

/// End-of-text token id.
pub const ENDOFTEXT_ID: u32 = 0;
/// Padding token id.
pub const PAD_ID: u32 = 1;
/// Beginning-of-sequence token id.
pub const BOS_ID: u32 = 2;
/// Thinking-section start token id.
pub const THINK_START_ID: u32 = 3;
/// Thinking-section end token id.
pub const THINK_END_ID: u32 = 4;
/// User role token id.
pub const USER_ID: u32 = 5;
/// Assistant role token id.
pub const ASSISTANT_ID: u32 = 6;

/// Reserved special token table in required id order.
pub const SPECIAL_TOKENS: [(&str, u32); 7] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
];

/// Number of reserved special tokens.
pub const SPECIAL_TOKEN_COUNT: usize = SPECIAL_TOKENS.len();
