use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Supported HuggingFace source checkpoint architecture families.
pub enum HfArch {
    /// Llama 2 style tensor naming.
    Llama2,
    /// Llama 3 style tensor naming.
    Llama3,
    /// Mistral style tensor naming.
    Mistral,
    /// Qwen2 style tensor naming.
    Qwen2,
}

impl HfArch {
    /// Parse a HuggingFace architecture name.
    pub fn from_name(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "llama2" => Ok(Self::Llama2),
            "llama3" => Ok(Self::Llama3),
            "mistral" => Ok(Self::Mistral),
            "qwen2" => Ok(Self::Qwen2),
            other => Err(AarambhError::Config(format!(
                "unsupported HF architecture '{other}', expected llama2|llama3|mistral|qwen2"
            ))),
        }
    }
}

/// Convert a HuggingFace checkpoint with the default Llama 3 mapping on CPU.
pub fn convert_hf(hf_path: &Path, cfg: &ModelConfig) -> Result<AarambhModel> {
    convert_hf_with_arch(hf_path, cfg, HfArch::Llama3, &Device::Cpu)
}

/// Convert a HuggingFace checkpoint into an Aarambh model.
pub fn convert_hf_with_arch(
    hf_path: &Path,
    cfg: &ModelConfig,
    arch: HfArch,
    device: &Device,
) -> Result<AarambhModel> {
    let tensors = convert_hf_tensors(hf_path, cfg, arch, device)?;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    AarambhModel::new(cfg, vb)
}

/// Convert HuggingFace checkpoint tensors into Aarambh tensor names.
pub fn convert_hf_tensors(
    hf_path: &Path,
    cfg: &ModelConfig,
    arch: HfArch,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let source = load_hf_tensors(hf_path, device)?;
    let expected = expected_shapes(cfg);
    let mut converted = HashMap::new();

    for (hf_name, tensor) in source {
        let Some(name) = map_key(&hf_name, arch) else {
            continue;
        };
        if cfg.tie_embeddings && name == "lm_head.weight" {
            continue;
        }
        let Some(expected_shape) = expected.get(&name) else {
            continue;
        };
        let tensor = adapt_tensor(&name, tensor.to_dtype(DType::F32)?, expected_shape, cfg)?;
        converted.insert(name, tensor);
    }

    for (name, shape) in expected {
        if cfg.tie_embeddings && name == "lm_head.weight" {
            continue;
        }
        if !converted.contains_key(&name) {
            return Err(AarambhError::Checkpoint(format!(
                "converted HF checkpoint is missing required tensor {name} with shape {shape:?}"
            )));
        }
    }

    Ok(converted)
}

fn load_hf_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let files = safetensor_files(path)?;
    let mut tensors = HashMap::new();
    for file in files {
        let loaded = candle_core::safetensors::load(&file, device)?;
        tensors.extend(loaded);
    }
    Ok(tensors)
}

fn safetensor_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let value: serde_json::Value = serde_json::from_reader(fs::File::open(&index_path)?)?;
        let weight_map = value
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                AarambhError::Checkpoint(format!(
                    "{} does not contain a weight_map object",
                    index_path.display()
                ))
            })?;
        let mut files = BTreeSet::new();
        for file in weight_map.values().filter_map(|value| value.as_str()) {
            files.insert(path.join(file));
        }
        return Ok(files.into_iter().collect());
    }

    let mut files = fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        return Err(AarambhError::Checkpoint(format!(
            "no safetensors files found in {}",
            path.display()
        )));
    }
    Ok(files)
}

fn map_key(name: &str, _arch: HfArch) -> Option<String> {
    if name == "model.embed_tokens.weight" || name == "transformer.wte.weight" {
        return Some("embedding.weight".to_string());
    }
    if name == "model.norm.weight" || name == "transformer.ln_f.weight" {
        return Some("final_norm.weight".to_string());
    }
    if name == "lm_head.weight" {
        return Some("lm_head.weight".to_string());
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, suffix) = rest.split_once('.')?;
    let mapped_suffix = match suffix {
        "input_layernorm.weight" => "norm1.weight",
        "post_attention_layernorm.weight" => "norm2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "mlp.gate_proj.weight" => "ffn.w_gate.weight",
        "mlp.up_proj.weight" => "ffn.w_up.weight",
        "mlp.down_proj.weight" => "ffn.w_down.weight",
        _ => return None,
    };
    Some(format!("blocks.{layer}.{mapped_suffix}"))
}

fn expected_shapes(cfg: &ModelConfig) -> HashMap<String, Vec<usize>> {
    let mut shapes = HashMap::new();
    let head_dim = cfg.head_dim();
    shapes.insert(
        "embedding.weight".to_string(),
        vec![cfg.vocab_size, cfg.hidden_dim],
    );
    for layer in 0..cfg.n_layers {
        shapes.insert(format!("blocks.{layer}.norm1.weight"), vec![cfg.hidden_dim]);
        shapes.insert(format!("blocks.{layer}.norm2.weight"), vec![cfg.hidden_dim]);
        shapes.insert(
            format!("blocks.{layer}.attn.wq.weight"),
            vec![cfg.n_heads * head_dim, cfg.hidden_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.attn.wk.weight"),
            vec![cfg.n_kv_heads * head_dim, cfg.hidden_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.attn.wv.weight"),
            vec![cfg.n_kv_heads * head_dim, cfg.hidden_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.attn.wo.weight"),
            vec![cfg.hidden_dim, cfg.n_heads * head_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.ffn.w_gate.weight"),
            vec![cfg.ffn_dim, cfg.hidden_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.ffn.w_up.weight"),
            vec![cfg.ffn_dim, cfg.hidden_dim],
        );
        shapes.insert(
            format!("blocks.{layer}.ffn.w_down.weight"),
            vec![cfg.hidden_dim, cfg.ffn_dim],
        );
    }
    shapes.insert("final_norm.weight".to_string(), vec![cfg.hidden_dim]);
    if !cfg.tie_embeddings {
        shapes.insert(
            "lm_head.weight".to_string(),
            vec![cfg.vocab_size, cfg.hidden_dim],
        );
    }
    shapes
}

fn adapt_tensor(
    name: &str,
    tensor: Tensor,
    expected_shape: &[usize],
    cfg: &ModelConfig,
) -> Result<Tensor> {
    let dims = tensor.dims().to_vec();
    if dims == expected_shape {
        return Ok(tensor);
    }
    if (name.ends_with("attn.wk.weight") || name.ends_with("attn.wv.weight"))
        && dims.len() == 2
        && expected_shape.len() == 2
        && dims[1] == expected_shape[1]
    {
        if dims[0] < expected_shape[0] {
            return Err(AarambhError::Unsupported(format!(
                "{name} has fewer KV rows ({}) than required ({}) for n_kv_heads={}",
                dims[0], expected_shape[0], cfg.n_kv_heads
            )));
        }
        return Ok(tensor.narrow(0, 0, expected_shape[0])?);
    }
    Err(AarambhError::Shape(format!(
        "{name} has shape {dims:?}, expected {expected_shape:?}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_standard_llama_key() {
        assert_eq!(
            map_key("model.layers.3.self_attn.q_proj.weight", HfArch::Llama3).unwrap(),
            "blocks.3.attn.wq.weight"
        );
    }

    #[test]
    fn rejects_unknown_arch() {
        assert!(HfArch::from_name("bert").is_err());
    }
}
