use std::collections::HashMap;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result, TokenizerLike};

use crate::special;
use crate::vocab::Vocab;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub vocab: Vocab,
    pub merges: Vec<(String, String)>,
    pub merge_rank: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    pub fn train(corpus_path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        use tokenizers::models::bpe::{BPE, BpeTrainer};
        use tokenizers::tokenizer::{Tokenizer as HfTokenizer, Trainer};

        let content = std::fs::read_to_string(corpus_path.as_ref()).map_err(AarambhError::Io)?;

        let words: Vec<String> = content.split_whitespace().map(String::from).collect();

        let mut trainer = BpeTrainer::new(2, vocab_size);
        trainer
            .feed(words.iter(), |s| Ok(vec![s.to_owned()]))
            .map_err(|e| AarambhError::Tokenizer(e.to_string()))?;

        let mut model = BPE::default();
        trainer
            .train(&mut model)
            .map_err(|e| AarambhError::Tokenizer(e.to_string()))?;

        let hf = HfTokenizer::new(model);
        let tmp = std::env::temp_dir().join(format!("aarambh_bpe_{}.json", std::process::id()));
        hf.save(&tmp, false)
            .map_err(|e| AarambhError::Tokenizer(format!("Save failed: {e}")))?;

        let result = Self::from_pretrained(&tmp);
        let _ = std::fs::remove_file(&tmp);
        result
    }

    pub fn from_pretrained(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let json: serde_json::Value = serde_json::from_str(&content).map_err(AarambhError::Json)?;

        let vocab_obj = json["model"]["vocab"].as_object().ok_or_else(|| {
            AarambhError::Tokenizer("missing model.vocab in tokenizer.json".into())
        })?;

        let token_to_id: HashMap<String, u32> = vocab_obj
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
            .collect();

        let merges: Vec<(String, String)> = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| {
                AarambhError::Tokenizer("missing model.merges in tokenizer.json".into())
            })?
            .iter()
            .filter_map(parse_merge)
            .collect();

        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let mut id_to_token: Vec<String> = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }

        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        let vocab = Vocab {
            token_to_id,
            id_to_token,
        };

        Ok(Self {
            vocab,
            merges,
            merge_rank,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.vocab.save_json(path)
    }

    pub fn save_pretrained(&self, path: impl AsRef<Path>) -> Result<()> {
        let vocab = self
            .vocab
            .token_to_id
            .iter()
            .map(|(token, id)| (token.clone(), serde_json::Value::from(*id)))
            .collect::<serde_json::Map<_, _>>();
        let merges = self
            .merges
            .iter()
            .map(|(left, right)| {
                serde_json::Value::Array(vec![
                    serde_json::Value::from(left.clone()),
                    serde_json::Value::from(right.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
            }
        });
        let content = serde_json::to_string_pretty(&json).map_err(AarambhError::Json)?;
        std::fs::write(path.as_ref(), content).map_err(AarambhError::Io)?;
        Ok(())
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            if symbols.len() <= 1 {
                break;
            }

            let mut best_idx = None;
            let mut best_rank = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(rank) = self.merge_rank.get(&pair)
                    && *rank < best_rank
                {
                    best_rank = *rank;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                    symbols[idx] = merged;
                    symbols.remove(idx + 1);
                }
                None => break,
            }
        }

        symbols
            .into_iter()
            .filter_map(|s| self.vocab.token_to_id.get(&s).copied())
            .collect()
    }
}

fn parse_merge(value: &serde_json::Value) -> Option<(String, String)> {
    if let Some(s) = value.as_str() {
        let parts = s.split_once(' ')?;
        return Some((parts.0.to_string(), parts.1.to_string()));
    }

    let array = value.as_array()?;
    if array.len() != 2 {
        return None;
    }
    Some((
        array[0].as_str()?.to_string(),
        array[1].as_str()?.to_string(),
    ))
}

impl TokenizerLike for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for word in text.split_inclusive(|c: char| c.is_whitespace()) {
            ids.extend(self.encode_word(word));
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &id in ids {
            match self.vocab.get_token(id) {
                Some(t) => text.push_str(t),
                None => {
                    return Err(AarambhError::Tokenizer(format!("unknown token id: {id}")));
                }
            }
        }
        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.id_to_token.len()
    }

    fn eos_token_id(&self) -> u32 {
        special::ENDOFTEXT_ID
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(special::BOS_ID)
    }
}
