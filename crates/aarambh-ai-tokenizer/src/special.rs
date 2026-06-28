pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const PAD: &str = "<|pad|>";
pub const BOS: &str = "<|bos|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";

pub const ENDOFTEXT_ID: u32 = 0;
pub const PAD_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const THINK_START_ID: u32 = 3;
pub const THINK_END_ID: u32 = 4;
pub const USER_ID: u32 = 5;
pub const ASSISTANT_ID: u32 = 6;

pub const SPECIAL_TOKENS: [(&str, u32); 7] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
];

pub const SPECIAL_TOKEN_COUNT: usize = SPECIAL_TOKENS.len();
