use std::path::Path;

use aarambh_ai_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_ai_model::AarambhModel;
use aarambh_ai_tokenizer::BpeTokenizer;
use candle_core::Tensor;

use crate::kvcache::KvCache;
use crate::sampler::{Sampler, TokenCandidate};
use crate::thinking::{ThinkingController, ThinkingMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    MaxTokens,
    EosToken,
    ContextLimit,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub sampler: Sampler,
    pub thinking_mode: ThinkingMode,
    pub top_candidates: usize,
}

impl GenerationConfig {
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            sampler: Sampler::greedy(),
            thinking_mode: ThinkingMode::None,
            top_candidates: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStep {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub candidates: Vec<TokenCandidate>,
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub steps: Vec<GenerationStep>,
}

pub struct InferenceEngine {
    model: AarambhModel,
    tokenizer: BpeTokenizer,
    device: candle_core::Device,
}

impl InferenceEngine {
    pub fn new(
        model: AarambhModel,
        tokenizer: BpeTokenizer,
        device: candle_core::Device,
    ) -> Result<Self> {
        tokenizer.validate_special_tokens()?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn from_paths(
        model_path: impl AsRef<Path>,
        model_config: &aarambh_ai_core::ModelConfig,
        tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
    ) -> Result<Self> {
        let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
        tokenizer.validate_special_tokens()?;
        let mut model_config = model_config.clone();
        model_config.vocab_size = tokenizer.vocab_size();
        let model = aarambh_ai_weights::load_model(model_path, &model_config, &device)?;
        Self::new(model, tokenizer, device)
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    pub fn model(&self) -> &AarambhModel {
        &self.model
    }

    pub fn generate(&mut self, prompt: &str, config: GenerationConfig) -> Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        mut config: GenerationConfig,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            if let Some(bos) = self.tokenizer.bos_token_id() {
                prompt_ids.push(bos);
            } else {
                return Err(AarambhError::Config(
                    "prompt produced no tokens and tokenizer has no BOS token".into(),
                ));
            }
        }

        let max_seq_len = self.model.config().max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {} tokens but model max_seq_len is {max_seq_len}",
                prompt_ids.len()
            )));
        }
        let available = max_seq_len - prompt_ids.len();
        let max_new_tokens = config.max_new_tokens.min(available);
        if max_new_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                token_ids: Vec::new(),
                finish_reason: FinishReason::ContextLimit,
                steps: Vec::new(),
            });
        }

        let mut cache = KvCache::for_model(&self.model);
        let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)?;
        let logits = self
            .model
            .forward_with_cache(&input, 0, cache.layers_mut())?;
        let mut next_logits = last_logits(&logits)?;

        let mut thinking = ThinkingController::new(config.thinking_mode);
        let mut generated_ids = Vec::with_capacity(max_new_tokens);
        let mut text = String::new();
        let mut steps = Vec::with_capacity(max_new_tokens);
        let mut finish_reason = FinishReason::MaxTokens;

        for step in 0..max_new_tokens {
            let logits_vec = next_logits.to_vec1::<f32>()?;
            let candidates = config
                .sampler
                .top_candidates(&logits_vec, config.top_candidates)?;
            let token_id = config.sampler.sample(&logits_vec)?;

            if token_id == self.tokenizer.eos_token_id() {
                finish_reason = FinishReason::EosToken;
                break;
            }

            let token_text = self.tokenizer.decode(&[token_id])?;
            let generation_step = GenerationStep {
                step: step + 1,
                token_id,
                token_text: token_text.clone(),
                candidates,
            };
            on_step(&generation_step)?;
            thinking.on_token(token_id);

            generated_ids.push(token_id);
            text.push_str(&token_text);
            steps.push(generation_step);

            if step + 1 == max_new_tokens {
                if generated_ids.len() == available {
                    finish_reason = FinishReason::ContextLimit;
                }
                break;
            }

            let offset = prompt_ids.len() + generated_ids.len() - 1;
            let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
            let logits = self
                .model
                .forward_with_cache(&input, offset, cache.layers_mut())?;
            next_logits = last_logits(&logits)?;
        }

        Ok(GenerationOutput {
            text,
            token_ids: generated_ids,
            finish_reason,
            steps,
        })
    }
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "expected logits shape [1, seq, vocab], got {dims:?}"
        )));
    }
    let seq_len = dims[1];
    let vocab = dims[2];
    Ok(logits.narrow(1, seq_len - 1, 1)?.reshape((vocab,))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_ai_core::ModelConfig;
    use aarambh_ai_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn test_tokenizer() -> BpeTokenizer {
        let pairs: [(&str, u32); 12] = [
            (ENDOFTEXT, ENDOFTEXT_ID),
            (PAD, PAD_ID),
            (BOS, BOS_ID),
            (THINK_START, THINK_START_ID),
            (THINK_END, THINK_END_ID),
            (USER, USER_ID),
            (ASSISTANT, ASSISTANT_ID),
            ("H", 7),
            ("e", 8),
            ("l", 9),
            ("o", 10),
            (" ", 11),
        ];
        let token_to_id = pairs
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect::<HashMap<_, _>>();
        let mut id_to_token = vec![String::new(); 12];
        for (token, id) in pairs {
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

    fn test_engine() -> InferenceEngine {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AarambhModel::new(&config, vb).unwrap();
        InferenceEngine::new(model, test_tokenizer(), device).unwrap()
    }

    #[test]
    fn greedy_generation_is_deterministic() {
        let mut engine1 = test_engine();
        let mut engine2 = test_engine();
        let out1 = engine1
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let out2 = engine2
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(out1.text, out2.text);
        assert_eq!(out1.token_ids, out2.token_ids);
    }

    #[test]
    fn generate_respects_max_tokens() {
        let mut engine = test_engine();
        let out = engine
            .generate("Hello", GenerationConfig::greedy(5))
            .unwrap();
        assert!(out.token_ids.len() <= 5);
    }

    #[test]
    fn invalid_tokenizer_special_ids_are_rejected() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AarambhModel::new(&config, vb).unwrap();
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id: HashMap::from([("!".to_string(), 0)]),
                id_to_token: vec!["!".to_string()],
            },
            merges: vec![],
            merge_rank: HashMap::new(),
        };
        assert!(InferenceEngine::new(model, tokenizer, device).is_err());
    }
}
