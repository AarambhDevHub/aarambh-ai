use candle_core::{DType, Device, Tensor};

use aarambh_ai_core::ModelConfig;
use aarambh_ai_nn::{
    GroupedQueryAttention, RMSNorm, RopeCache, SwiGluFfn, TransformerBlock,
};

fn create_causal_mask(seq_len: usize, device: &Device) -> Tensor {
    let tril = Tensor::tril2(seq_len, DType::U32, device).unwrap();
    let zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device).unwrap();
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device).unwrap();
    let mask = tril.where_cond(&zeros, &neg_inf).unwrap();
    mask.unsqueeze(0).unwrap().unsqueeze(0).unwrap()
}

#[test]
fn rmsnorm_output_shape_unchanged() {
    let device = Device::Cpu;
    let size = 384;
    let weight = Tensor::ones(size, DType::F32, &device).unwrap();
    let norm = RMSNorm::new(weight, 1e-5);
    let x = Tensor::randn(0f32, 1f32, (2, 16, size), &device).unwrap();
    let out = norm.forward(&x).unwrap();
    assert_eq!(out.shape(), x.shape());
}

#[test]
fn rope_preserves_vector_magnitude() {
    let device = Device::Cpu;
    let rope = RopeCache::new(512, 64, 10000.0, &device).unwrap();
    let q = Tensor::randn(0f32, 1f32, (1, 4, 8, 64), &device).unwrap();
    let (q_rot, _) = rope.apply(&q, &q, 0).unwrap();
    let norm_before: f32 = q.sqr().unwrap().sum_all().unwrap().sqrt().unwrap()
        .to_scalar::<f32>().unwrap();
    let norm_after: f32 = q_rot.sqr().unwrap().sum_all().unwrap().sqrt().unwrap()
        .to_scalar::<f32>().unwrap();
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "RoPE changed magnitude: {norm_before} → {norm_after}",
    );
}

#[test]
fn swiglu_ffn_shape_unchanged() {
    let device = Device::Cpu;
    let cfg = ModelConfig::tiny();

    let w_gate = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.ffn_dim, cfg.hidden_dim), &device).unwrap(),
        None,
    );
    let w_up = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.ffn_dim, cfg.hidden_dim), &device).unwrap(),
        None,
    );
    let w_down = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.hidden_dim, cfg.ffn_dim), &device).unwrap(),
        None,
    );

    let ffn = SwiGluFfn::new(w_gate, w_up, w_down);
    let x = Tensor::randn(0f32, 1f32, (2, 16, cfg.hidden_dim), &device).unwrap();
    let out = ffn.forward(&x).unwrap();
    assert_eq!(out.shape(), x.shape());
}

#[test]
fn gqa_output_shape() {
    let device = Device::Cpu;
    let cfg = ModelConfig::tiny();
    let rope =
        RopeCache::new(cfg.max_seq_len, cfg.head_dim(), cfg.rope_theta, &device).unwrap();
    let mask = create_causal_mask(16, &device);

    let wq = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wk = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_kv_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wv = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_kv_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wo = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.hidden_dim, cfg.n_heads * cfg.head_dim()),
            &device,
        )
        .unwrap(),
        None,
    );

    let attn = GroupedQueryAttention::new(
        wq,
        wk,
        wv,
        wo,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim(),
    );
    let x = Tensor::randn(0f32, 1f32, (1, 16, cfg.hidden_dim), &device).unwrap();
    let out = attn.forward(&x, &rope, Some(&mask), None, 0).unwrap();
    assert_eq!(out.shape().dims(), &[1, 16, cfg.hidden_dim]);
}

#[test]
fn transformer_block_output_shape() {
    let device = Device::Cpu;
    let cfg = ModelConfig::tiny();
    let rope =
        RopeCache::new(cfg.max_seq_len, cfg.head_dim(), cfg.rope_theta, &device).unwrap();
    let mask = create_causal_mask(16, &device);

    let norm1 =
        RMSNorm::new(Tensor::ones(cfg.hidden_dim, DType::F32, &device).unwrap(), 1e-5);
    let norm2 =
        RMSNorm::new(Tensor::ones(cfg.hidden_dim, DType::F32, &device).unwrap(), 1e-5);

    let wq = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wk = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_kv_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wv = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.n_kv_heads * cfg.head_dim(), cfg.hidden_dim),
            &device,
        )
        .unwrap(),
        None,
    );
    let wo = candle_nn::Linear::new(
        Tensor::randn(
            0f32,
            1f32,
            (cfg.hidden_dim, cfg.n_heads * cfg.head_dim()),
            &device,
        )
        .unwrap(),
        None,
    );
    let attn = GroupedQueryAttention::new(
        wq,
        wk,
        wv,
        wo,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim(),
    );

    let w_gate = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.ffn_dim, cfg.hidden_dim), &device).unwrap(),
        None,
    );
    let w_up = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.ffn_dim, cfg.hidden_dim), &device).unwrap(),
        None,
    );
    let w_down = candle_nn::Linear::new(
        Tensor::randn(0f32, 1f32, (cfg.hidden_dim, cfg.ffn_dim), &device).unwrap(),
        None,
    );
    let ffn = SwiGluFfn::new(w_gate, w_up, w_down);

    let block = TransformerBlock::new(norm1, attn, norm2, ffn);
    let x = Tensor::randn(0f32, 1f32, (2, 16, cfg.hidden_dim), &device).unwrap();
    let out = block.forward(&x, &rope, Some(&mask), None, 0).unwrap();
    assert_eq!(out.shape().dims(), &[2, 16, cfg.hidden_dim]);
}
