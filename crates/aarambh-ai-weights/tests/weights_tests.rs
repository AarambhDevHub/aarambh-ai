use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::ModelConfig;
use aarambh_ai_model::AarambhModel;
use aarambh_ai_weights::{GgufFormat, load_gguf, load_model, save_gguf, save_model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn mini_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        hidden_dim: 64,
        ffn_dim: 128,
        n_layers: 2,
        n_heads: 1,
        n_kv_heads: 1,
        max_seq_len: 16,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        tie_embeddings: true,
    }
}

fn temp_safetensors_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "aarambh-ai-model-{}-{nanos}.safetensors",
        std::process::id()
    ))
}

fn temp_gguf_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "aarambh-ai-model-{}-{nanos}.gguf",
        std::process::id()
    ))
}

#[test]
fn safetensors_roundtrip_preserves_weights_and_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_safetensors_path();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path, &cfg, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    let w1 = model.get_weight("blocks.0.attn.wq.weight").unwrap();
    let w2 = loaded.get_weight("blocks.0.attn.wq.weight").unwrap();
    let weight_diff = (w1 - w2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(weight_diff < 1e-6, "weight diff: {weight_diff}");

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits1 = model.forward(&ids).unwrap();
    let logits2 = loaded.forward(&ids).unwrap();
    let logits_diff = (logits1 - logits2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(logits_diff < 1e-6, "logits diff: {logits_diff}");
}

#[test]
fn gguf_save_load_roundtrip_produces_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits = loaded.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
    let max = logits
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max.is_finite());
}
