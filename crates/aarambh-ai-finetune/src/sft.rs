use aarambh_ai_tokenizer::{ASSISTANT, ENDOFTEXT, THINK_END, THINK_START, USER};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingSftExample {
    pub instruction: String,
    pub thinking: String,
    pub response: String,
}

pub fn format_thinking_sft(example: &ThinkingSftExample) -> String {
    format!(
        "{USER}\n{}\n{ASSISTANT}\n{THINK_START}\n{}\n{THINK_END}\n{}{}",
        example.instruction, example.thinking, example.response, ENDOFTEXT
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_ai_core::TokenizerLike;
    use aarambh_ai_tokenizer::{
        ASSISTANT_ID, BpeTokenizer, ENDOFTEXT_ID, THINK_END_ID, THINK_START_ID, USER_ID, Vocab,
    };
    use std::collections::HashMap;

    #[test]
    fn thinking_sft_format_matches_phase_9_contract() {
        let example = ThinkingSftExample {
            instruction: "What is 23 x 47?".into(),
            thinking: "23 x 40 = 920; 23 x 7 = 161; total = 1081".into(),
            response: "The answer is 1081.".into(),
        };

        let formatted = format_thinking_sft(&example);
        assert_eq!(
            formatted,
            "<|user|>\nWhat is 23 x 47?\n<|assistant|>\n<think>\n23 x 40 = 920; 23 x 7 = 161; total = 1081\n</think>\nThe answer is 1081.<|endoftext|>"
        );
    }

    #[test]
    fn thinking_sft_format_uses_reserved_special_token_ids() {
        let tokenizer = test_tokenizer();
        let formatted = format_thinking_sft(&ThinkingSftExample {
            instruction: "Hi".into(),
            thinking: "Plan".into(),
            response: "Hello".into(),
        });
        let ids = tokenizer.encode(&formatted).unwrap();

        assert!(ids.contains(&USER_ID));
        assert!(ids.contains(&ASSISTANT_ID));
        assert!(ids.contains(&THINK_START_ID));
        assert!(ids.contains(&THINK_END_ID));
        assert!(ids.contains(&ENDOFTEXT_ID));
    }

    fn test_tokenizer() -> BpeTokenizer {
        let mut token_to_id = HashMap::from([
            (USER.to_string(), USER_ID),
            (ASSISTANT.to_string(), ASSISTANT_ID),
            (THINK_START.to_string(), THINK_START_ID),
            (THINK_END.to_string(), THINK_END_ID),
            (ENDOFTEXT.to_string(), ENDOFTEXT_ID),
        ]);
        let mut id_to_token = vec![String::new(); 12];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        for (token, id) in [("H", 7), ("i", 8), ("P", 9), ("l", 10), ("a", 11)] {
            token_to_id.insert(token.to_string(), id);
            id_to_token[id as usize] = token.to_string();
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: vec![],
            merge_rank: HashMap::new(),
        }
    }
}
