use std::fs;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{Device, Tensor};

/// Description of rows inserted into vocabulary-shaped checkpoint tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyExpansion {
    /// Row at which new vocabulary entries are inserted.
    pub insertion_id: usize,
    /// Existing row copied to initialize each inserted entry, in insertion order.
    pub source_ids: Vec<usize>,
}

impl VocabularyExpansion {
    /// Validate this expansion against an existing vocabulary size.
    pub fn validate(&self, old_vocab_size: usize) -> Result<()> {
        if self.source_ids.is_empty() {
            return Err(AarambhError::Config(
                "vocabulary expansion requires at least one source row".into(),
            ));
        }
        if self.insertion_id > old_vocab_size {
            return Err(AarambhError::Config(format!(
                "vocabulary insertion id {} exceeds old vocabulary size {old_vocab_size}",
                self.insertion_id
            )));
        }
        if let Some(source) = self
            .source_ids
            .iter()
            .find(|source| **source >= old_vocab_size)
        {
            return Err(AarambhError::Config(format!(
                "vocabulary expansion source id {source} exceeds old vocabulary size {old_vocab_size}"
            )));
        }
        Ok(())
    }

    /// Return the vocabulary size after applying this expansion.
    pub fn new_vocab_size(&self, old_vocab_size: usize) -> usize {
        old_vocab_size + self.source_ids.len()
    }
}

/// Summary of a SafeTensors vocabulary expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyExpansionReport {
    /// Source vocabulary size.
    pub old_vocab_size: usize,
    /// Expanded vocabulary size.
    pub new_vocab_size: usize,
    /// Number of vocabulary-shaped tensors expanded.
    pub expanded_tensors: usize,
    /// Total tensors copied to the output checkpoint.
    pub total_tensors: usize,
}

/// Expand embedding and untied LM-head rows in a SafeTensors checkpoint.
///
/// All non-vocabulary tensors are copied unchanged. The input and output paths
/// must differ so a failed write can never damage the source checkpoint.
pub fn expand_safetensors_vocabulary(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    old_vocab_size: usize,
    expansion: &VocabularyExpansion,
) -> Result<VocabularyExpansionReport> {
    expansion.validate(old_vocab_size)?;
    let input = input.as_ref();
    let output = output.as_ref();
    if input == output {
        return Err(AarambhError::Config(
            "vocabulary migration input and output paths must differ".into(),
        ));
    }
    if input.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        return Err(AarambhError::Config(
            "vocabulary migration requires a SafeTensors source checkpoint; migrate before GGUF quantisation"
                .into(),
        ));
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut tensors = candle_core::safetensors::load(input, &Device::Cpu)?;
    let mut expanded_tensors = 0usize;
    for name in ["embedding.weight", "lm_head.weight"] {
        let Some(tensor) = tensors.get(name) else {
            continue;
        };
        let dims = tensor.dims();
        if dims.len() != 2 || dims[0] != old_vocab_size {
            return Err(AarambhError::Checkpoint(format!(
                "checkpoint tensor {name} must have shape [{old_vocab_size}, hidden], got {dims:?}"
            )));
        }
        let expanded = expand_rows(tensor, expansion)?;
        tensors.insert(name.to_string(), expanded);
        expanded_tensors += 1;
    }
    if expanded_tensors == 0 {
        return Err(AarambhError::Checkpoint(
            "checkpoint is missing embedding.weight".into(),
        ));
    }
    candle_core::safetensors::save(&tensors, output)?;
    Ok(VocabularyExpansionReport {
        old_vocab_size,
        new_vocab_size: expansion.new_vocab_size(old_vocab_size),
        expanded_tensors,
        total_tensors: tensors.len(),
    })
}

fn expand_rows(tensor: &Tensor, expansion: &VocabularyExpansion) -> Result<Tensor> {
    let rows = tensor.dim(0)?;
    let mut parts = Vec::with_capacity(expansion.source_ids.len() + 2);
    if expansion.insertion_id > 0 {
        parts.push(tensor.narrow(0, 0, expansion.insertion_id)?);
    }
    for source in &expansion.source_ids {
        parts.push(tensor.narrow(0, *source, 1)?);
    }
    if expansion.insertion_id < rows {
        parts.push(tensor.narrow(0, expansion.insertion_id, rows - expansion.insertion_id)?);
    }
    let refs = parts.iter().collect::<Vec<_>>();
    Ok(Tensor::cat(&refs, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn vocabulary_expansion_inserts_cloned_rows_and_preserves_tail() {
        let nonce = format!(
            "{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        );
        let input = std::env::temp_dir().join(format!("aarambh_vocab_input_{nonce}.safetensors"));
        let output = std::env::temp_dir().join(format!("aarambh_vocab_output_{nonce}.safetensors"));
        let tensor = Tensor::from_vec(
            (0..20).map(|value| value as f32).collect::<Vec<_>>(),
            (10, 2),
            &Device::Cpu,
        )
        .unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("embedding.weight".to_string(), tensor)]),
            &input,
        )
        .unwrap();
        let report = expand_safetensors_vocabulary(
            &input,
            &output,
            10,
            &VocabularyExpansion {
                insertion_id: 9,
                source_ids: vec![7, 8, 8],
            },
        )
        .unwrap();
        assert_eq!(report.new_vocab_size, 13);
        let loaded = candle_core::safetensors::load(&output, &Device::Cpu).unwrap();
        let rows = loaded["embedding.weight"].to_vec2::<f32>().unwrap();
        assert_eq!(rows[9], rows[7]);
        assert_eq!(rows[10], rows[8]);
        assert_eq!(rows[11], rows[8]);
        assert_eq!(rows[12], vec![18.0, 19.0]);
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
