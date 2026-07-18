use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use aarambh_ai_core::{AarambhError, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::rollout::StudentRollout;

/// Stable prompt identifier and exact text supplied to generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptExample {
    /// Identifier used to group rollouts and join teacher data.
    pub id: String,
    /// Exact prompt string encoded by the inference engine.
    pub prompt: String,
}

#[derive(Debug, Deserialize)]
struct RawPromptExample {
    id: Option<String>,
    prompt: String,
}

/// Validated in-memory prompt dataset.
#[derive(Debug, Clone)]
pub struct PromptDataset {
    examples: Vec<PromptExample>,
}

impl PromptDataset {
    /// Load `{id?, prompt}` records from JSONL.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut examples = Vec::new();
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let raw: RawPromptExample = serde_json::from_str(line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid distillation prompt JSONL at line {}: {error}",
                    line_index + 1
                ))
            })?;
            examples.push(PromptExample {
                id: raw.id.unwrap_or_else(|| format!("line-{}", line_index + 1)),
                prompt: raw.prompt,
            });
        }
        Self::from_examples(examples).map_err(|error| {
            AarambhError::Config(format!(
                "distillation prompt dataset {} is invalid: {error}",
                path.as_ref().display()
            ))
        })
    }

    /// Validate and store already loaded prompt examples.
    pub fn from_examples(examples: Vec<PromptExample>) -> Result<Self> {
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "distillation prompt dataset is empty".into(),
            ));
        }
        let mut ids = HashSet::with_capacity(examples.len());
        for example in &examples {
            if example.id.trim().is_empty() || example.prompt.trim().is_empty() {
                return Err(AarambhError::Config(
                    "distillation prompt ids and text must be non-empty".into(),
                ));
            }
            if !ids.insert(example.id.clone()) {
                return Err(AarambhError::Config(format!(
                    "duplicate distillation prompt id '{}'",
                    example.id
                )));
            }
        }
        Ok(Self { examples })
    }

    /// Return the number of prompts.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return true when no prompts are present.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Return all prompts in source order.
    pub fn examples(&self) -> &[PromptExample] {
        &self.examples
    }
}

/// One teacher-approved reference response and its quality weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceAnswer {
    /// Reference completion text.
    pub completion: String,
    /// Quality weight in `[0, 1]`.
    pub score: f32,
}

/// Scored references associated with one prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredReferenceRecord {
    /// Prompt identifier matching the rollout dataset.
    pub id: String,
    /// Exact prompt text, retained for validation and standalone use.
    pub prompt: String,
    /// One or more acceptable teacher references.
    pub references: Vec<ReferenceAnswer>,
}

/// Validated scored-reference teacher dataset.
#[derive(Debug, Clone)]
pub struct ScoredReferenceDataset {
    records: Vec<ScoredReferenceRecord>,
    by_id: HashMap<String, usize>,
}

impl ScoredReferenceDataset {
    /// Load scored-reference records from JSONL.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut records = Vec::new();
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record = serde_json::from_str(line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid scored-reference JSONL at line {}: {error}",
                    line_index + 1
                ))
            })?;
            records.push(record);
        }
        Self::from_records(records).map_err(|error| {
            AarambhError::Config(format!(
                "scored-reference dataset {} is invalid: {error}",
                path.as_ref().display()
            ))
        })
    }

    /// Validate and store scored-reference records.
    pub fn from_records(records: Vec<ScoredReferenceRecord>) -> Result<Self> {
        if records.is_empty() {
            return Err(AarambhError::Config(
                "scored-reference teacher dataset is empty".into(),
            ));
        }
        let mut by_id = HashMap::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            if record.id.trim().is_empty() || record.prompt.trim().is_empty() {
                return Err(AarambhError::Config(
                    "scored-reference ids and prompts must be non-empty".into(),
                ));
            }
            if record.references.is_empty() {
                return Err(AarambhError::Config(format!(
                    "scored-reference prompt '{}' has no references",
                    record.id
                )));
            }
            for reference in &record.references {
                if reference.completion.trim().is_empty()
                    || !reference.score.is_finite()
                    || !(0.0..=1.0).contains(&reference.score)
                {
                    return Err(AarambhError::Config(format!(
                        "scored-reference prompt '{}' has an invalid completion or score",
                        record.id
                    )));
                }
            }
            if by_id.insert(record.id.clone(), index).is_some() {
                return Err(AarambhError::Config(format!(
                    "duplicate scored-reference id '{}'",
                    record.id
                )));
            }
        }
        Ok(Self { records, by_id })
    }

    /// Find references by prompt identifier.
    pub fn get(&self, id: &str) -> Option<&ScoredReferenceRecord> {
        self.by_id.get(id).map(|index| &self.records[*index])
    }

    /// Return all records in source order.
    pub fn records(&self) -> &[ScoredReferenceRecord] {
        &self.records
    }

    /// Build a prompt dataset directly from the scored records.
    pub fn prompts(&self) -> Result<PromptDataset> {
        PromptDataset::from_examples(
            self.records
                .iter()
                .map(|record| PromptExample {
                    id: record.id.clone(),
                    prompt: record.prompt.clone(),
                })
                .collect(),
        )
    }

    /// Ensure every prompt has an exact matching scored-reference record.
    pub fn validate_prompt_coverage(&self, prompts: &PromptDataset) -> Result<()> {
        for prompt in prompts.examples() {
            let record = self.get(&prompt.id).ok_or_else(|| {
                AarambhError::Config(format!(
                    "scored-reference teacher is missing prompt id '{}'",
                    prompt.id
                ))
            })?;
            if record.prompt != prompt.prompt {
                return Err(AarambhError::Config(format!(
                    "prompt text mismatch for scored-reference id '{}'",
                    prompt.id
                )));
            }
        }
        Ok(())
    }
}

/// Static teacher-generated completion used by the offline baseline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineExample {
    /// Prompt identifier.
    pub id: String,
    /// Exact prompt text.
    pub prompt: String,
    /// Frozen teacher-generated completion.
    pub completion: String,
}

/// Validated static completion dataset for matched offline distillation.
#[derive(Debug, Clone)]
pub struct OfflineDataset {
    examples: Vec<OfflineExample>,
}

impl OfflineDataset {
    /// Load offline completion records from JSONL.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut examples = Vec::new();
        for (line_index, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            examples.push(serde_json::from_str(line).map_err(|error| {
                AarambhError::Config(format!(
                    "invalid offline distillation JSONL at line {}: {error}",
                    line_index + 1
                ))
            })?);
        }
        Self::from_examples(examples)
    }

    /// Validate and store offline examples.
    pub fn from_examples(examples: Vec<OfflineExample>) -> Result<Self> {
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "offline distillation dataset is empty".into(),
            ));
        }
        let mut ids = HashSet::with_capacity(examples.len());
        for example in &examples {
            if example.id.trim().is_empty()
                || example.prompt.trim().is_empty()
                || example.completion.trim().is_empty()
            {
                return Err(AarambhError::Config(
                    "offline distillation fields must be non-empty".into(),
                ));
            }
            if !ids.insert(example.id.clone()) {
                return Err(AarambhError::Config(format!(
                    "duplicate offline distillation id '{}'",
                    example.id
                )));
            }
        }
        Ok(Self { examples })
    }

    /// Return offline examples in source order.
    pub fn examples(&self) -> &[OfflineExample] {
        &self.examples
    }
}

/// Padded prompt-plus-completion replay tensors and packed completion rows.
#[derive(Debug)]
pub struct ReplayBatch {
    /// Next-token model inputs shaped `[batch, sequence]`.
    pub input_ids: Tensor,
    /// Next-token labels shaped `[batch, sequence]`.
    pub labels: Tensor,
    /// Completion-only policy mask shaped `[batch, sequence]`.
    pub completion_mask: Tensor,
    /// Flattened row indexes for all unforced completion targets.
    pub packed_row_indices: Tensor,
    /// Number of selected completion rows belonging to each rollout.
    pub completion_counts: Vec<usize>,
}

impl ReplayBatch {
    /// Build a right-padded replay batch from student rollouts.
    pub fn from_rollouts(
        rollouts: &[StudentRollout],
        pad_id: u32,
        device: &Device,
    ) -> Result<Self> {
        if rollouts.is_empty() {
            return Err(AarambhError::Config(
                "distillation replay batch is empty".into(),
            ));
        }
        let mut sequences = Vec::with_capacity(rollouts.len());
        let mut masks = Vec::with_capacity(rollouts.len());
        let mut max_len = 0usize;
        for rollout in rollouts {
            if rollout.prompt_token_ids.is_empty()
                || rollout.completion_token_ids.is_empty()
                || rollout.completion_token_ids.len() != rollout.loss_mask.len()
            {
                return Err(AarambhError::Config(format!(
                    "rollout '{}' has empty prompt/completion or inconsistent completion data",
                    rollout.prompt_id
                )));
            }
            let mut full = rollout.prompt_token_ids.clone();
            full.extend_from_slice(&rollout.completion_token_ids);
            if full.len() < 2 {
                return Err(AarambhError::Config(
                    "distillation replay sequence requires at least two tokens".into(),
                ));
            }
            let seq_len = full.len() - 1;
            let mut target_mask = vec![0u32; seq_len];
            for (completion_index, &enabled) in rollout.loss_mask.iter().enumerate() {
                let target_position = rollout.prompt_token_ids.len() + completion_index;
                let row = target_position - 1;
                target_mask[row] = u32::from(enabled);
            }
            max_len = max_len.max(seq_len);
            sequences.push(full);
            masks.push(target_mask);
        }

        let batch = rollouts.len();
        let mut inputs = Vec::with_capacity(batch * max_len);
        let mut labels = Vec::with_capacity(batch * max_len);
        let mut completion_mask = Vec::with_capacity(batch * max_len);
        let mut packed_rows = Vec::new();
        let mut completion_counts = Vec::with_capacity(batch);
        for (batch_index, (full, mask)) in sequences.iter().zip(&masks).enumerate() {
            let seq_len = full.len() - 1;
            inputs.extend_from_slice(&full[..seq_len]);
            inputs.resize(inputs.len() + (max_len - seq_len), pad_id);
            labels.extend_from_slice(&full[1..]);
            labels.resize(labels.len() + (max_len - seq_len), pad_id);
            completion_mask.extend_from_slice(mask);
            completion_mask.resize(completion_mask.len() + (max_len - seq_len), 0);
            let mut count = 0usize;
            for (row, enabled) in mask.iter().enumerate() {
                if *enabled != 0 {
                    packed_rows.push((batch_index * max_len + row) as u32);
                    count += 1;
                }
            }
            if count == 0 {
                return Err(AarambhError::Config(format!(
                    "rollout '{}' has no trainable completion tokens",
                    rollouts[batch_index].prompt_id
                )));
            }
            completion_counts.push(count);
        }

        Ok(Self {
            input_ids: Tensor::from_vec(inputs, (batch, max_len), device)?,
            labels: Tensor::from_vec(labels, (batch, max_len), device)?,
            completion_mask: Tensor::from_vec(completion_mask, (batch, max_len), device)?,
            packed_row_indices: Tensor::from_vec(packed_rows.clone(), packed_rows.len(), device)?,
            completion_counts,
        })
    }

    /// Select completion-position rows from `[batch, sequence, width]` logits.
    pub fn pack_logits(&self, logits: &Tensor) -> Result<Tensor> {
        let (batch, sequence, width) = logits.dims3()?;
        if self.input_ids.dims() != [batch, sequence] {
            return Err(AarambhError::Shape(format!(
                "replay logits shape {:?} does not match input shape {:?}",
                logits.dims(),
                self.input_ids.dims()
            )));
        }
        Ok(logits
            .reshape((batch * sequence, width))?
            .index_select(&self.packed_row_indices, 0)?)
    }

    /// Select target token IDs corresponding to packed completion rows.
    pub fn packed_labels(&self) -> Result<Tensor> {
        let (batch, sequence) = self.labels.dims2()?;
        Ok(self
            .labels
            .reshape(batch * sequence)?
            .index_select(&self.packed_row_indices, 0)?)
    }

    /// Return the number of rollout sequences.
    pub fn batch_size(&self) -> usize {
        self.completion_counts.len()
    }

    /// Return the number of trainable completion tokens.
    pub fn completion_tokens(&self) -> usize {
        self.completion_counts.iter().sum()
    }

    /// Return a floating completion mask for existing loss helpers.
    pub fn completion_mask_f32(&self) -> Result<Tensor> {
        Ok(self.completion_mask.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollout::RolloutFinish;

    fn rollout(
        id: &str,
        prompt_tokens: Vec<u32>,
        completion_tokens: Vec<u32>,
        loss_mask: Vec<bool>,
    ) -> StudentRollout {
        StudentRollout {
            prompt_id: id.to_string(),
            prompt: "prompt".to_string(),
            prompt_token_ids: prompt_tokens,
            completion_token_ids: completion_tokens,
            completion_text: "completion".to_string(),
            loss_mask,
            rollout_index: 0,
            finish_reason: RolloutFinish::MaxTokens,
        }
    }

    #[test]
    fn prompt_dataset_rejects_duplicate_ids() {
        let examples = vec![
            PromptExample {
                id: "same".into(),
                prompt: "first".into(),
            },
            PromptExample {
                id: "same".into(),
                prompt: "second".into(),
            },
        ];
        assert!(PromptDataset::from_examples(examples).is_err());
    }

    #[test]
    fn replay_batch_masks_prompts_forced_tokens_and_padding() {
        let device = Device::Cpu;
        let batch = ReplayBatch::from_rollouts(
            &[
                rollout("a", vec![2, 7], vec![8, 9], vec![true, false]),
                rollout("b", vec![2], vec![7, 8], vec![true, true]),
            ],
            1,
            &device,
        )
        .unwrap();

        assert_eq!(
            batch.input_ids.to_vec2::<u32>().unwrap(),
            vec![vec![2, 7, 8], vec![2, 7, 1]]
        );
        assert_eq!(
            batch.labels.to_vec2::<u32>().unwrap(),
            vec![vec![7, 8, 9], vec![7, 8, 1]]
        );
        assert_eq!(
            batch.completion_mask.to_vec2::<u32>().unwrap(),
            vec![vec![0, 1, 0], vec![1, 1, 0]]
        );
        assert_eq!(batch.completion_counts, vec![1, 2]);
        assert_eq!(
            batch.packed_labels().unwrap().to_vec1::<u32>().unwrap(),
            vec![8, 7, 8]
        );

        let logits = Tensor::arange(0u32, 18, &device)
            .unwrap()
            .reshape((2, 3, 3))
            .unwrap();
        assert_eq!(
            batch
                .pack_logits(&logits)
                .unwrap()
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![3, 4, 5], vec![9, 10, 11], vec![12, 13, 14]]
        );
    }

    #[test]
    fn replay_batch_rejects_empty_prompt_tokens() {
        let result = ReplayBatch::from_rollouts(
            &[rollout("bad", Vec::new(), vec![7], vec![true])],
            1,
            &Device::Cpu,
        );
        assert!(result.is_err());
    }
}
