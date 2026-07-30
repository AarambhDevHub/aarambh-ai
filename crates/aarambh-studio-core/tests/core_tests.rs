use aarambh_studio_core::*;

#[test]
fn tiny_config_head_dim_is_correct() {
    let cfg = ModelConfig::tiny();
    assert_eq!(cfg.head_dim(), 64);
}

#[test]
fn all_four_configs_construct() {
    let _ = ModelConfig::tiny();
    let _ = ModelConfig::small();
    let _ = ModelConfig::medium();
    let _ = ModelConfig::large();
}

#[test]
fn device_best_available_is_cpu_on_i3() {
    assert_eq!(Device::best_available(), Device::Cpu);
}

#[test]
fn dtype_size_bytes() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
}

#[test]
fn default_train_config_effective_batch() {
    let cfg = TrainConfig::default();
    assert_eq!(cfg.batch_size * cfg.grad_accum_steps, 32);
}

#[test]
fn default_train_config_beta2_is_correct() {
    let cfg = TrainConfig::default();
    assert!(
        (cfg.beta2 - 0.95).abs() < 1e-9,
        "beta2 should be 0.95, got {}",
        cfg.beta2
    );
}

#[test]
fn hybrid_schedule_resolves_phase29_dimensions() {
    let schedule = HybridAttentionSchedule::default();
    let resolved = schedule.validate(8, 384, 6).unwrap();
    assert_eq!(resolved.n_heads, 3);
    assert_eq!(resolved.key_head_dim, 96);
    assert_eq!(resolved.value_head_dim, 192);
    assert_eq!(schedule.kind_for_layer(0), AttentionKind::Full);
    assert_eq!(schedule.kind_for_layer(1), AttentionKind::GatedDeltaNet);
    assert_eq!(schedule.kind_for_layer(4), AttentionKind::Full);
}
