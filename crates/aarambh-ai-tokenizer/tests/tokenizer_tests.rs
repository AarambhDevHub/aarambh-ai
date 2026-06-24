use std::collections::HashMap;

use aarambh_ai_core::TokenizerLike;
use aarambh_ai_tokenizer::{
    BpeTokenizer, Vocab,
    special::{ASSISTANT_ID, BOS_ID, ENDOFTEXT_ID, PAD_ID, THINK_END_ID, THINK_START_ID, USER_ID},
};

#[test]
fn special_token_ids_are_correct() {
    assert_eq!(ENDOFTEXT_ID, 0);
    assert_eq!(PAD_ID, 1);
    assert_eq!(BOS_ID, 2);
    assert_eq!(THINK_START_ID, 3);
    assert_eq!(THINK_END_ID, 4);
    assert_eq!(USER_ID, 5);
    assert_eq!(ASSISTANT_ID, 6);
}

#[test]
fn vocab_get_id_and_get_token() {
    let token_to_id = HashMap::from([
        ("hello".into(), 0u32),
        ("world".into(), 1u32),
        (" ".into(), 2u32),
    ]);
    let id_to_token = vec!["hello".into(), "world".into(), " ".into()];
    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };

    assert_eq!(vocab.get_id("hello"), Some(0));
    assert_eq!(vocab.get_id("world"), Some(1));
    assert_eq!(vocab.get_id("unknown"), None);
    assert_eq!(vocab.get_token(0), Some("hello"));
    assert_eq!(vocab.get_token(1), Some("world"));
    assert_eq!(vocab.get_token(99), None);
}

#[test]
fn vocab_roundtrip_via_json() {
    let token_to_id = HashMap::from([("a".into(), 0u32), ("b".into(), 1u32)]);
    let id_to_token = vec!["a".into(), "b".into()];
    let original = Vocab {
        token_to_id,
        id_to_token,
    };

    let dir = std::env::temp_dir();
    let path = dir.join("test_vocab.json");
    original.save_json(&path).unwrap();
    let loaded = Vocab::from_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.get_id("a"), Some(0));
    assert_eq!(loaded.get_id("b"), Some(1));
    assert_eq!(loaded.get_token(0), Some("a"));
}

#[test]
fn bpe_tokenizer_encode_decode_roundtrip() {
    let token_to_id = HashMap::from([
        ("h".into(), 0u32),
        ("e".into(), 1u32),
        ("l".into(), 2u32),
        ("o".into(), 3u32),
        (" ".into(), 4u32),
        ("w".into(), 5u32),
        ("r".into(), 6u32),
        ("d".into(), 7u32),
        ("he".into(), 8u32),
        ("ll".into(), 9u32),
        ("hell".into(), 10u32),
        ("hello".into(), 11u32),
        ("wo".into(), 12u32),
        ("rl".into(), 13u32),
        ("worl".into(), 14u32),
        ("world".into(), 15u32),
    ]);
    let id_to_token = vec![
        "h".into(),
        "e".into(),
        "l".into(),
        "o".into(),
        " ".into(),
        "w".into(),
        "r".into(),
        "d".into(),
        "he".into(),
        "ll".into(),
        "hell".into(),
        "hello".into(),
        "wo".into(),
        "rl".into(),
        "worl".into(),
        "world".into(),
    ];
    let merges: Vec<(String, String)> = vec![
        ("h".into(), "e".into()),
        ("l".into(), "l".into()),
        ("he".into(), "ll".into()),
        ("hell".into(), "o".into()),
        ("w".into(), "o".into()),
        ("r".into(), "l".into()),
        ("wo".into(), "rl".into()),
        ("worl".into(), "d".into()),
    ];

    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };

    let merge_rank: HashMap<(String, String), usize> = merges
        .iter()
        .enumerate()
        .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
        .collect();

    let tokenizer = BpeTokenizer {
        vocab,
        merges,
        merge_rank,
    };

    let text = "hello world";
    let ids = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn bpe_tokenizer_implements_tokenizer_like() {
    let token_to_id = HashMap::from([("a".into(), 0u32), ("b".into(), 1u32)]);
    let id_to_token = vec!["a".into(), "b".into()];
    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };
    let merge_rank = HashMap::new();

    let tokenizer = BpeTokenizer {
        vocab,
        merges: vec![],
        merge_rank,
    };

    assert_eq!(tokenizer.vocab_size(), 2);
    assert_eq!(tokenizer.eos_token_id(), 0);
    assert_eq!(tokenizer.bos_token_id(), Some(2));
}
