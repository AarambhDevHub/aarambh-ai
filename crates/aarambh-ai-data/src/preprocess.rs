use aarambh_ai_core::TokenizerLike;

use crate::dataset::TextDataset;

pub fn chunk_and_tokenize(
    dataset: &dyn TextDataset,
    tokenizer: &dyn TokenizerLike,
    max_seq_len: usize,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut all_ids = Vec::new();
    for i in 0..dataset.len() {
        if let Ok(ids) = tokenizer.encode(dataset.get(i)) {
            all_ids.extend(ids);
        }
    }

    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos + max_seq_len < all_ids.len() {
        let input = all_ids[pos..pos + max_seq_len].to_vec();
        let label = all_ids[pos + 1..pos + max_seq_len + 1].to_vec();
        chunks.push((input, label));
        pos += max_seq_len;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::PlaintextDataset;
    use aarambh_ai_core::TokenizerLike;
    use std::collections::HashMap;

    struct DummyTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TokenizerLike for DummyTokenizer {
        fn encode(&self, text: &str) -> aarambh_ai_core::Result<Vec<u32>> {
            Ok(text
                .chars()
                .filter_map(|c| self.vocab.get(&c.to_string()).copied())
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> aarambh_ai_core::Result<String> {
            let rev: HashMap<u32, String> =
                self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            Ok(ids
                .iter()
                .filter_map(|id| rev.get(id).map(|s| s.as_str()))
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn labels_are_shifted_by_one() {
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([
                ("a".into(), 0),
                ("b".into(), 1),
                ("c".into(), 2),
                ("d".into(), 3),
            ]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcd".into()]);
        let chunks = chunk_and_tokenize(&dataset, &tokenizer, 2);
        assert!(!chunks.is_empty());
        let (input, label) = &chunks[0];
        assert_eq!(input, &[0, 1]);
        assert_eq!(label, &[1, 2]);
        assert_eq!(input[1], label[0]);
    }

    #[test]
    fn multiple_chunks() {
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([
                ("a".into(), 0),
                ("b".into(), 1),
                ("c".into(), 2),
                ("d".into(), 3),
                ("e".into(), 4),
                ("f".into(), 5),
            ]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdef".into()]);
        let chunks = chunk_and_tokenize(&dataset, &tokenizer, 2);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, &[0, 1]);
        assert_eq!(chunks[0].1, &[1, 2]);
        assert_eq!(chunks[1].0, &[2, 3]);
        assert_eq!(chunks[1].1, &[3, 4]);
    }
}
