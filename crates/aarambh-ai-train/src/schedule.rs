use aarambh_ai_core::TrainConfig;

#[derive(Debug, Clone)]
pub struct CosineScheduleWithWarmup {
    max_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl CosineScheduleWithWarmup {
    pub fn new(max_lr: f64, warmup_steps: usize, total_steps: usize, min_lr_ratio: f64) -> Self {
        let total_steps = total_steps.max(1);
        let min_lr = max_lr * min_lr_ratio;
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    pub fn from_train_config(config: &TrainConfig) -> Self {
        Self::new(
            config.lr,
            config.warmup_steps,
            config.max_steps,
            config.min_lr_ratio,
        )
    }

    pub fn lr_at_step(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.max_lr * (step + 1) as f64 / self.warmup_steps as f64;
        }

        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let decay_step = step.saturating_sub(self.warmup_steps).min(decay_steps);
        let progress = decay_step as f64 / decay_steps as f64;
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.min_lr + (self.max_lr - self.min_lr) * cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_is_monotonic() {
        let schedule = CosineScheduleWithWarmup::new(1e-3, 4, 20, 0.1);
        let values = (0..4).map(|s| schedule.lr_at_step(s)).collect::<Vec<_>>();
        assert!(values.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn decay_is_monotonic() {
        let schedule = CosineScheduleWithWarmup::new(1e-3, 2, 20, 0.1);
        let values = (2..20).map(|s| schedule.lr_at_step(s)).collect::<Vec<_>>();
        assert!(values.windows(2).all(|w| w[1] <= w[0] + f64::EPSILON));
    }
}
