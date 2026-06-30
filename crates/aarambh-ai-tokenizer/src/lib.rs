//! BPE tokenizer wrapper, vocabulary utilities, and Aarambh special tokens.
#![deny(missing_docs)]

/// Byte-pair-encoding tokenizer implementation.
pub mod bpe;
/// Reserved special token definitions.
pub mod special;
/// Vocabulary lookup table.
pub mod vocab;

pub use bpe::BpeTokenizer;
pub use special::{
    ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID,
    SPECIAL_TOKEN_COUNT, SPECIAL_TOKENS, THINK_END, THINK_END_ID, THINK_START, THINK_START_ID,
    USER, USER_ID,
};
pub use vocab::Vocab;
