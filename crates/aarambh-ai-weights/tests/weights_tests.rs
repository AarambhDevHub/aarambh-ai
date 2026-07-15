use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_ai_core::{GatedDeltaNetConfig, HybridAttentionSchedule, ModelConfig, MoeConfig};
use aarambh_ai_model::AarambhModel;
use aarambh_ai_weights::{
    GgufFormat, load_gguf, load_model, load_retrofit_into_varmap, save_gguf, save_model,
};
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
        rope_scaling: None,
        moe: None,
        attention_schedule: None,
        norm_eps: 1e-5,
        tie_embeddings: true,
    }
}

fn moe_mini_config() -> ModelConfig {
    ModelConfig {
        moe: Some(MoeConfig {
            num_experts: 4,
            top_k: 2,
            expert_ffn_dim: 64,
            aux_loss_weight: 0.01,
            every_n_layers: 2,
        }),
        ..mini_config()
    }
}

fn hybrid_mini_config() -> ModelConfig {
    ModelConfig {
        attention_schedule: Some(HybridAttentionSchedule {
            full_attention_every_n: 2,
            gated_deltanet: GatedDeltaNetConfig {
                n_heads: 1,
                key_head_dim: 16,
                value_head_dim: 32,
                conv_kernel_size: 4,
                chunk_size: 16,
            },
        }),
        ..mini_config()
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

#[test]
fn moe_safetensors_roundtrip_preserves_logits() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_safetensors_path();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path, &cfg, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        loaded
            .get_weight("blocks.1.ffn.experts.0.w_gate.weight")
            .is_some()
    );
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
fn moe_gguf_roundtrip_produces_logits() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
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
    assert!(
        loaded
            .get_weight("blocks.1.ffn.experts.0.w_gate.weight")
            .is_some()
    );
}

#[test]
fn retrofit_load_preserves_full_layers_and_initializes_deltanet() {
    let device = Device::Cpu;
    let dense_cfg = mini_config();
    let dense_vars = VarMap::new();
    let dense = AarambhModel::new(
        &dense_cfg,
        VarBuilder::from_varmap(&dense_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&dense, &path).unwrap();

    let hybrid_cfg = hybrid_mini_config();
    let mut hybrid_vars = VarMap::new();
    let hybrid = AarambhModel::new(
        &hybrid_cfg,
        VarBuilder::from_varmap(&hybrid_vars, DType::F32, &device),
    )
    .unwrap();
    let report =
        load_retrofit_into_varmap(&path, &hybrid_cfg, &mut hybrid_vars, &device, DType::F32)
            .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(report.loaded_tensors > 0);
    assert_eq!(report.initialized_deltanet_tensors, 13);
    let source = dense.get_weight("blocks.0.attn.wq.weight").unwrap();
    let loaded = hybrid.get_weight("blocks.0.attn.wq.weight").unwrap();
    let diff = (source - loaded)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-6, "retrofit full-layer weight diff: {diff}");
    assert!(
        hybrid
            .get_weight("blocks.1.deltanet.q_proj.weight")
            .is_some()
    );
    assert!(hybrid.get_weight("blocks.1.attn.wq.weight").is_none());
}

#[test]
fn hybrid_gguf_roundtrip_keeps_float_recurrent_parameters() {
    let device = Device::Cpu;
    let cfg = hybrid_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        loaded
            .get_weight("blocks.1.deltanet.A_log")
            .unwrap()
            .dtype(),
        DType::F32
    );
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    assert_eq!(loaded.forward(&ids).unwrap().dims(), [1, 3, 128]);
}
