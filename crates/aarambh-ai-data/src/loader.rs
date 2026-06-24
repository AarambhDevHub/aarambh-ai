use candle_core::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use aarambh_ai_core::{Device, Result, TokenizerLike};

use crate::dataset::TextDataset;
use crate::preprocess::chunk_and_tokenize;

pub struct Batch {
    pub input_ids: Tensor,
    pub labels: Tensor,
    pub attention_mask: Tensor,
}

pub struct DataLoader {
    chunks: Vec<(Vec<u32>, Vec<u32>)>,
    batch_size: usize,
    shuffle: bool,
    device: Device,
    rng: StdRng,
    pos: usize,
}

impl DataLoader {
    pub fn new(
        dataset: &dyn TextDataset,
        tokenizer: &dyn TokenizerLike,
        batch_size: usize,
        max_seq_len: usize,
        shuffle: bool,
        device: Device,
    ) -> Self {
        let chunks = chunk_and_tokenize(dataset, tokenizer, max_seq_len);
        let rng = StdRng::from_entropy();
        let pos = 0;
        DataLoader {
            chunks,
            batch_size,
            shuffle,
            device,
            rng,
            pos,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.chunks.shuffle(&mut self.rng);
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len() / self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for DataLoader {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.chunks.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.chunks.len());
        if end - self.pos < self.batch_size {
            self.pos = end;
            return None;
        }

        let batch_chunks = &self.chunks[self.pos..end];
        self.pos = end;

        let seq_len = batch_chunks[0].0.len();

        let mut input_ids = Vec::with_capacity(self.batch_size * seq_len);
        let mut labels = Vec::with_capacity(self.batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(self.batch_size * seq_len);

        for (input, label) in batch_chunks {
            input_ids.extend_from_slice(input);
            labels.extend_from_slice(label);
            attention_mask.extend(std::iter::repeat_n(1u32, seq_len));
        }

        let candle_device = match self.device.to_candle() {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        let input_ids =
            match Tensor::from_vec(input_ids, (self.batch_size, seq_len), &candle_device) {
                Ok(t) => t,
                Err(e) => return Some(Err(e.into())),
            };
        let labels = match Tensor::from_vec(labels, (self.batch_size, seq_len), &candle_device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e.into())),
        };
        let attention_mask =
            match Tensor::from_vec(attention_mask, (self.batch_size, seq_len), &candle_device) {
                Ok(t) => t,
                Err(e) => return Some(Err(e.into())),
            };

        Some(Ok(Batch {
            input_ids,
            labels,
            attention_mask,
        }))
    }
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
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .chars()
                .filter_map(|c| self.vocab.get(&c.to_string()).copied())
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let rev: HashMap<u32, String> =
                self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            Ok(ids
                .iter()
                .filter_map(|id| rev.get(id).map(|s| s.as_str()))
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len() as usize
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn dataloader_batch_shape() {
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([
                ("a".into(), 0),
                ("b".into(), 1),
                ("c".into(), 2),
                ("d".into(), 3),
                ("e".into(), 4),
                ("f".into(), 5),
                ("g".into(), 6),
                ("h".into(), 7),
            ]),
        };
        let dataset = PlaintextDataset::from_lines(vec![
            "abcdefgh".into(),
            "abcdefgh".into(),
            "abcdefgh".into(),
            "abcdefgh".into(),
            "abcdefgh".into(),
        ]);
        let device = Device::Cpu;
        let mut loader = DataLoader::new(&dataset, &tokenizer, 4, 4, false, device);
        let batch = loader.next().unwrap().unwrap();
        assert_eq!(batch.input_ids.shape().dims(), &[4, 4]);
        assert_eq!(batch.labels.shape().dims(), &[4, 4]);
        assert_eq!(batch.attention_mask.shape().dims(), &[4, 4]);
    }

    #[test]
    fn dataloader_exhaustion() {
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([("a".into(), 0), ("b".into(), 1)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["ab".into(), "ab".into()]);
        let device = Device::Cpu;
        let mut loader = DataLoader::new(&dataset, &tokenizer, 2, 1, false, device);
        assert!(loader.next().is_some());
        assert!(loader.next().is_none());
    }
}
