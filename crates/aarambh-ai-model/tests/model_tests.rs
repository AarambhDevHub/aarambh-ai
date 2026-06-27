use aarambh_ai_core::ModelConfig;
use aarambh_ai_model::AarambhModel;
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

fn mini_model(device: &Device) -> AarambhModel {
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    AarambhModel::new(&cfg, vb).unwrap()
}

#[test]
fn all_four_model_configs_validate() {
    for cfg in [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::medium(),
        ModelConfig::large(),
    ] {
        AarambhModel::validate_config(&cfg).unwrap();
    }
}

#[test]
fn tiny_forward_produces_correct_shape() {
    let device = Device::Cpu;
    let cfg = ModelConfig::tiny();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let ids = Tensor::zeros((1, 16), DType::U32, &device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 16, 32000]);
}

#[test]
fn mini_forward_produces_correct_shape_and_finite_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let model = mini_model(&device);
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 6, cfg.vocab_size]);

    let max = logits
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max.is_finite());
    assert!(max < 10.0, "initial logits are too large: {max}");
}

#[test]
fn cached_forward_matches_full_forward_for_next_token() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let full_logits = model.forward(&ids).unwrap();
    let full_last = full_logits.narrow(1, 3, 1).unwrap();

    let mut caches = model.empty_kv_cache();
    let mut cached_last = None;
    for pos in 0..4 {
        let token = ids.narrow(1, pos, 1).unwrap();
        cached_last = Some(model.forward_with_cache(&token, pos, &mut caches).unwrap());
    }

    let cached_last = cached_last.unwrap();
    let max_diff = (full_last - cached_last)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "cached/full mismatch: {max_diff}");
}

#[test]
fn tied_embedding_reuses_lm_head_tensor() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    assert_eq!(
        model.get_weight("embedding.weight").unwrap().id(),
        model.get_weight("lm_head.weight").unwrap().id()
    );
    assert!(!model.named_tensors().contains_key("lm_head.weight"));
}

#[test]
fn untied_lm_head_is_saved_separately() {
    let device = Device::Cpu;
    let mut cfg = mini_config();
    cfg.tie_embeddings = false;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    assert_ne!(
        model.get_weight("embedding.weight").unwrap().id(),
        model.get_weight("lm_head.weight").unwrap().id()
    );
    assert!(model.named_tensors().contains_key("lm_head.weight"));
}

#[test]
fn invalid_config_is_rejected() {
    let mut cfg = mini_config();
    cfg.hidden_dim = 96;
    cfg.n_heads = 2;
    let err = AarambhModel::validate_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("head_dim must be 64"));
}

#[test]
fn readme_model_scale_table_matches_model_config() {
    let readme = include_str!("../../../README.md");
    assert!(readme.contains("| Tiny | 25M | 384 | 8 | 6 | 2 | 1,024 | 512 | 10,000 |"));
    assert!(readme.contains("| Small | 117M | 768 | 12 | 12 | 4 | 2,688 | 1,024 | 10,000 |"));
    assert!(readme.contains("| Medium | 360M | 1,024 | 24 | 16 | 8 | 3,392 | 2,048 | 500,000 |"));
    assert!(readme.contains("| Large | 1.3B | 2,048 | 24 | 32 | 8 | 6,656 | 4,096 | 500,000 |"));
}

#[test]
#[ignore = "Large model construction allocates multiple GB; run manually for release validation."]
fn all_four_full_scales_construct() {
    let device = Device::Cpu;
    for cfg in [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::medium(),
        ModelConfig::large(),
    ] {
        let vb = VarBuilder::zeros(DType::F32, &device);
        AarambhModel::new(&cfg, vb).unwrap();
    }
}
