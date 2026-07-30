use candle_core::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use aarambh_studio_core::{Device, Result, TokenizerLike};

use crate::dataset::TextDataset;
use crate::preprocess::chunk_and_tokenize;

/// Tensor batch emitted by [`DataLoader`].
pub struct Batch {
    /// Input token ids with shape `[batch, seq]`.
    pub input_ids: Tensor,
    /// Next-token labels with shape `[batch, seq]`.
    pub labels: Tensor,
    /// Attention mask with shape `[batch, seq]`.
    pub attention_mask: Tensor,
}

/// Rank-local shard descriptor for distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataShard {
    /// Zero-based shard rank.
    pub rank: usize,
    /// Total number of shards.
    pub count: usize,
    /// Deterministic RNG seed for this shard.
    pub seed: u64,
}

/// Mini-batch iterator over fixed-length token chunks.
pub struct DataLoader {
    chunks: Vec<(Vec<u32>, Vec<u32>)>,
    batch_size: usize,
    shuffle: bool,
    device: Device,
    rng: StdRng,
    pos: usize,
}

impl DataLoader {
    /// Build a loader by tokenizing and chunking a dataset.
    pub fn new(
        dataset: &dyn TextDataset,
        tokenizer: &dyn TokenizerLike,
        batch_size: usize,
        max_seq_len: usize,
        shuffle: bool,
        device: Device,
    ) -> Self {
        let chunks = chunk_and_tokenize(dataset, tokenizer, max_seq_len);
        Self::from_chunks(chunks, batch_size, shuffle, device, None)
    }

    /// Build a loader with a deterministic RNG seed.
    pub fn new_with_seed(
        dataset: &dyn TextDataset,
        tokenizer: &dyn TokenizerLike,
        batch_size: usize,
        max_seq_len: usize,
        shuffle: bool,
        device: Device,
        seed: u64,
    ) -> Self {
        let chunks = chunk_and_tokenize(dataset, tokenizer, max_seq_len);
        Self::from_chunks(chunks, batch_size, shuffle, device, Some(seed))
    }

    /// Build a deterministic rank-local shard for data-parallel training.
    pub fn new_sharded(
        dataset: &dyn TextDataset,
        tokenizer: &dyn TokenizerLike,
        batch_size: usize,
        max_seq_len: usize,
        shuffle: bool,
        device: Device,
        shard: DataShard,
    ) -> Self {
        let chunks = chunk_and_tokenize(dataset, tokenizer, max_seq_len);
        let chunks = shard_chunks(chunks, batch_size, shard.rank, shard.count);
        Self::from_chunks(chunks, batch_size, shuffle, device, Some(shard.seed))
    }

    fn from_chunks(
        chunks: Vec<(Vec<u32>, Vec<u32>)>,
        batch_size: usize,
        shuffle: bool,
        device: Device,
        seed: Option<u64>,
    ) -> Self {
        let rng = StdRng::from_entropy();
        let rng = seed.map_or(rng, StdRng::seed_from_u64);
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

    /// Reset iteration and reshuffle chunks when enabled.
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.chunks.shuffle(&mut self.rng);
        }
    }

    /// Return the number of full batches.
    pub fn len(&self) -> usize {
        self.chunks.len() / self.batch_size
    }

    /// Return true when the loader has no full batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn shard_chunks(
    chunks: Vec<(Vec<u32>, Vec<u32>)>,
    batch_size: usize,
    shard_rank: usize,
    shard_count: usize,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    if shard_count <= 1 || batch_size == 0 {
        return chunks;
    }
    if shard_rank >= shard_count {
        return Vec::new();
    }

    let samples_per_sync_step = batch_size.saturating_mul(shard_count);
    if samples_per_sync_step == 0 {
        return Vec::new();
    }
    let full_sync_steps = chunks.len() / samples_per_sync_step;
    let mut sharded = Vec::with_capacity(full_sync_steps * batch_size);
    for step in 0..full_sync_steps {
        let start = step * samples_per_sync_step + shard_rank * batch_size;
        let end = start + batch_size;
        sharded.extend(chunks[start..end].iter().cloned());
    }
    sharded
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
    use aarambh_studio_core::TokenizerLike;
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

    #[test]
    fn sharded_dataloader_produces_equal_disjoint_batches() {
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([
                ("a".into(), 0),
                ("b".into(), 1),
                ("c".into(), 2),
                ("d".into(), 3),
            ]),
        };
        let dataset = PlaintextDataset::from_lines(vec![
            "abcd".into(),
            "abcd".into(),
            "abcd".into(),
            "abcd".into(),
            "abcd".into(),
        ]);
        let rank0 = DataLoader::new_sharded(
            &dataset,
            &tokenizer,
            2,
            1,
            false,
            Device::Cpu,
            DataShard {
                rank: 0,
                count: 2,
                seed: 42,
            },
        );
        let rank1 = DataLoader::new_sharded(
            &dataset,
            &tokenizer,
            2,
            1,
            false,
            Device::Cpu,
            DataShard {
                rank: 1,
                count: 2,
                seed: 42,
            },
        );
        assert_eq!(rank0.len(), rank1.len());
        assert!(!rank0.is_empty());
        assert_ne!(rank0.chunks, rank1.chunks);
    }
}
