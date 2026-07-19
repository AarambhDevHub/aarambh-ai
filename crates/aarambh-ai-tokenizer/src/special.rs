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
/// Image placeholder token string.
pub const IMAGE: &str = "<image>";
/// Image prefix boundary token string.
pub const IMAGE_END: &str = "<image_end>";
/// Video placeholder token string.
pub const VIDEO: &str = "<video>";
/// Video prefix boundary token string.
pub const VIDEO_END: &str = "<video_end>";
/// Separator token inserted between sampled video frames.
pub const FRAME_SEP: &str = "<frame_sep>";

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
/// Image placeholder token id.
pub const IMAGE_ID: u32 = 7;
/// Image prefix boundary token id.
pub const IMAGE_END_ID: u32 = 8;
/// Video placeholder token id.
pub const VIDEO_ID: u32 = 9;
/// Video prefix boundary token id.
pub const VIDEO_END_ID: u32 = 10;
/// Sampled-frame separator token id.
pub const FRAME_SEP_ID: u32 = 11;

/// Reserved special token table in required id order.
pub const SPECIAL_TOKENS: [(&str, u32); 12] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
    (VIDEO, VIDEO_ID),
    (VIDEO_END, VIDEO_END_ID),
    (FRAME_SEP, FRAME_SEP_ID),
];

/// Image-capable reserved token table accepted by v2 checkpoints.
pub const VISION_SPECIAL_TOKENS: [(&str, u32); 9] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
];

/// Text-only reserved special token table accepted by legacy checkpoints.
pub const TEXT_SPECIAL_TOKENS: [(&str, u32); 7] = [
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
