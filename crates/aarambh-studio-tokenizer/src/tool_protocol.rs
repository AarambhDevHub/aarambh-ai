use aarambh_studio_core::{Result, TokenizerLike};

use crate::{
    ASSISTANT_ID, BOS_ID, BpeTokenizer, ENDOFTEXT_ID, PAD_ID, THINK_END_ID, THINK_START_ID, USER_ID,
};

/// First token id reserved for printable virtual ASCII tool-protocol bytes.
pub const VIRTUAL_ASCII_BASE: u32 = 9;
/// First printable ASCII byte represented by the virtual tool-protocol range.
pub const VIRTUAL_ASCII_FIRST: u8 = 0x20;
/// Last printable ASCII byte represented by the virtual tool-protocol range.
pub const VIRTUAL_ASCII_LAST: u8 = 0x7e;
/// Last token id reserved for printable virtual ASCII tool-protocol bytes.
pub const VIRTUAL_ASCII_END: u32 =
    VIRTUAL_ASCII_BASE + (VIRTUAL_ASCII_LAST - VIRTUAL_ASCII_FIRST) as u32;

/// Encode JSON into the virtual printable-ASCII token range used by tool calls.
///
/// Non-ASCII scalar values are converted to JSON-compatible UTF-16 escape
/// sequences so every emitted token remains in the deterministic virtual range.
pub fn encode_virtual_json(text: &str) -> Vec<u32> {
    let mut ids = Vec::with_capacity(text.len());
    for character in text.chars() {
        if character.is_ascii() && (' '..='~').contains(&character) {
            ids.push(virtual_ascii_id(character));
        } else {
            for escaped in json_unicode_escape(character).chars() {
                ids.push(virtual_ascii_id(escaped));
            }
        }
    }
    ids
}

/// Decode one tokenizer or virtual tool-protocol token to its JSON fragment.
pub fn tool_json_token_text(token_id: u32, tokenizer: &BpeTokenizer) -> Result<String> {
    let structural = match token_id {
        BOS_ID => Some("{"),
        PAD_ID => Some("}"),
        THINK_START_ID => Some("["),
        THINK_END_ID => Some("]"),
        USER_ID => Some("\""),
        ASSISTANT_ID => Some(":"),
        ENDOFTEXT_ID => Some(","),
        _ => None,
    };
    match structural {
        Some(text) => Ok(text.to_string()),
        None if (VIRTUAL_ASCII_BASE..=VIRTUAL_ASCII_END).contains(&token_id) => {
            let byte = VIRTUAL_ASCII_FIRST + (token_id - VIRTUAL_ASCII_BASE) as u8;
            Ok(char::from(byte).to_string())
        }
        None => tokenizer.decode(&[token_id]),
    }
}

fn virtual_ascii_id(character: char) -> u32 {
    VIRTUAL_ASCII_BASE + (character as u8 - VIRTUAL_ASCII_FIRST) as u32
}

fn json_unicode_escape(character: char) -> String {
    let code = character as u32;
    if code <= 0xffff {
        format!("\\u{code:04x}")
    } else {
        let adjusted = code - 0x1_0000;
        let high = 0xd800 + (adjusted >> 10);
        let low = 0xdc00 + (adjusted & 0x3ff);
        format!("\\u{high:04x}\\u{low:04x}")
    }
}

#[cfg(test)]
mod tests {
    use super::{VIRTUAL_ASCII_END, encode_virtual_json};

    #[test]
    fn virtual_json_is_ascii_and_deterministic() {
        let ids = encode_virtual_json(r#"{"city":"Pune","mark":"✓"}"#);
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|id| *id <= VIRTUAL_ASCII_END));
        assert_eq!(ids, encode_virtual_json(r#"{"city":"Pune","mark":"✓"}"#));
    }
}
