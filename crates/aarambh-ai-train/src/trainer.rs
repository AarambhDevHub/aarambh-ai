use std::time::Instant;

use aarambh_ai_core::{AarambhError, ModelConfig, Result, TrainConfig};
use aarambh_ai_data::{Batch, DataLoader};
use aarambh_ai_model::AarambhModel;
use candle_core::DType;
use candle_core::backprop::GradStore;
use candle_nn::{VarBuilder, VarMap};

use crate::checkpoint::{CheckpointManager, TrainState};
use crate::loss::cross_entropy_loss;
use crate::optim::{AdamW, AdamWConfig, GradMap, clip_gradients};
use crate::schedule::CosineScheduleWithWarmup;

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: usize,
    pub loss: f64,
    pub perplexity: f64,
    pub lr: f64,
    pub grad_norm: Option<f64>,
    pub did_optimizer_step: bool,
}

pub struct Trainer {
    model: AarambhModel,
    varmap: VarMap,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    checkpoint: CheckpointManager,
    train_loader: DataLoader,
    val_loader: Option<DataLoader>,
    train_config: TrainConfig,
    device: candle_core::Device,
    state: TrainState,
    pending_grads: GradMap,
    last_loss: Option<f64>,
    tokens_since_log: usize,
    last_log_at: Instant,
}

impl Trainer {
    pub fn new(
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Option<DataLoader>,
        device: candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        if train_config.grad_accum_steps == 0 {
            return Err(AarambhError::Config(
                "grad_accum_steps must be greater than zero".into(),
            ));
        }
        if train_config.max_steps == 0 {
            return Err(AarambhError::Config(
                "max_steps must be greater than zero".into(),
            ));
        }
        if train_loader.is_empty() {
            return Err(AarambhError::Config(
                "training dataloader has no full batches".into(),
            ));
        }

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = AarambhModel::new(&model_config, vb)?;
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        let schedule = CosineScheduleWithWarmup::from_train_config(&train_config);
        let checkpoint = CheckpointManager::new(train_config.checkpoint_dir.clone());

        Ok(Self {
            model,
            varmap,
            optimizer,
            schedule,
            checkpoint,
            train_loader,
            val_loader,
            train_config,
            device,
            state: TrainState::default(),
            pending_grads: GradMap::new(),
            last_loss: None,
            tokens_since_log: 0,
            last_log_at: Instant::now(),
        })
    }

    pub fn state(&self) -> &TrainState {
        &self.state
    }

    pub fn model(&self) -> &AarambhModel {
        &self.model
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    pub fn load_latest_checkpoint(&mut self) -> Result<bool> {
        match self
            .checkpoint
            .load_latest(&mut self.varmap, &mut self.optimizer, &self.device)?
        {
            Some(state) => {
                self.state = state;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    pub fn train_step(&mut self, batch: Batch) -> Result<TrainingMetrics> {
        let token_count = batch.input_ids.elem_count();
        let logits = self.model.forward_train(&batch.input_ids)?;
        let loss = cross_entropy_loss(&logits, &batch.labels, &batch.attention_mask)?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(AarambhError::Config(format!(
                "non-finite training loss: {loss_value}"
            )));
        }

        let scaled_loss = loss.affine(1.0 / self.train_config.grad_accum_steps as f64, 0.0)?;
        let grads = scaled_loss.backward()?;
        self.accumulate_gradients(&grads)?;
        self.state.micro_step += 1;
        self.state.train_loss = Some(loss_value);
        self.last_loss = Some(loss_value);
        self.tokens_since_log += token_count;

        let should_step = self
            .state
            .micro_step
            .is_multiple_of(self.train_config.grad_accum_steps);
        if should_step {
            let (lr, grad_norm) = self.optimizer_step()?;
            Ok(TrainingMetrics {
                step: self.state.step,
                loss: loss_value,
                perplexity: loss_value.exp(),
                lr,
                grad_norm: Some(grad_norm),
                did_optimizer_step: true,
            })
        } else {
            Ok(TrainingMetrics {
                step: self.state.step,
                loss: loss_value,
                perplexity: loss_value.exp(),
                lr: self.schedule.lr_at_step(self.state.step),
                grad_norm: None,
                did_optimizer_step: false,
            })
        }
    }

    pub fn train_epoch(&mut self) -> Result<()> {
        self.train_loader.reset();
        while self.state.step < self.train_config.max_steps {
            let Some(batch) = self.train_loader.next() else {
                break;
            };
            let metrics = self.train_step(batch?)?;
            if metrics.did_optimizer_step {
                self.after_optimizer_step(&metrics)?;
            }
        }
        self.flush_pending_step()?;
        self.state.epoch += 1;
        Ok(())
    }

    pub fn train(&mut self) -> Result<()> {
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            self.train_epoch()?;
        }
        self.checkpoint
            .save(&self.varmap, &self.optimizer, &self.state)?;
        Ok(())
    }

    pub fn validate(&mut self) -> Result<Option<f64>> {
        let Some(loader) = self.val_loader.as_mut() else {
            return Ok(None);
        };
        loader.reset();

        let mut total = 0f64;
        let mut batches = 0usize;
        for batch in loader.by_ref() {
            let batch = batch?;
            let logits = self.model.forward_train(&batch.input_ids)?;
            let loss = cross_entropy_loss(&logits, &batch.labels, &batch.attention_mask)?;
            total += loss.to_scalar::<f32>()? as f64;
            batches += 1;
        }

        if batches == 0 {
            return Ok(None);
        }
        let loss = total / batches as f64;
        self.state.val_loss = Some(loss);
        Ok(Some(loss))
    }

    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        let mut updates = Vec::new();
        for param in self.optimizer.parameters() {
            let Some(grad) = grads.get(param.tensor()) else {
                continue;
            };
            let grad = grad.detach();
            let next = match self.pending_grads.get(param.name()) {
                Some(existing) => ((existing + &grad)?).detach(),
                None => grad,
            };
            updates.push((param.name().to_string(), next));
        }

        if updates.is_empty() {
            return Err(AarambhError::Config(
                "backward produced no parameter gradients".into(),
            ));
        }

        for (name, grad) in updates {
            self.pending_grads.insert(name, grad);
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<(f64, f64)> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer.step(&self.pending_grads, lr)?;
        self.pending_grads.clear();
        self.state.step += 1;
        Ok((lr, grad_norm))
    }

    fn flush_pending_step(&mut self) -> Result<()> {
        if self.pending_grads.is_empty() || self.state.step >= self.train_config.max_steps {
            return Ok(());
        }
        let (lr, grad_norm) = self.optimizer_step()?;
        let loss = self.last_loss.unwrap_or(0.0);
        let metrics = TrainingMetrics {
            step: self.state.step,
            loss,
            perplexity: loss.exp(),
            lr,
            grad_norm: Some(grad_norm),
            did_optimizer_step: true,
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&mut self, metrics: &TrainingMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let grad_norm = metrics.grad_norm.unwrap_or(0.0);
            println!(
                "step={} loss={:.4} ppl={:.2} lr={:.6} grad_norm={:.4} tok/s={:.2}",
                metrics.step,
                metrics.loss,
                metrics.perplexity,
                metrics.lr,
                grad_norm,
                self.tokens_per_second_since_last_log()
            );
        }

        if self.train_config.eval_steps > 0
            && metrics.step.is_multiple_of(self.train_config.eval_steps)
            && let Some(val_loss) = self.validate()?
        {
            let improved = self.state.best_val_loss.is_none_or(|best| val_loss < best);
            if improved {
                self.state.best_val_loss = Some(val_loss);
                self.checkpoint
                    .save_best(&self.varmap, &self.optimizer, &self.state)?;
            }
            println!(
                "eval step={} val_loss={:.4} val_ppl={:.2}",
                metrics.step,
                val_loss,
                val_loss.exp()
            );
        }

        if self.train_config.save_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.save_every_n_steps)
        {
            self.checkpoint
                .save(&self.varmap, &self.optimizer, &self.state)?;
        }
        Ok(())
    }

    fn tokens_per_second_since_last_log(&mut self) -> f64 {
        let elapsed = self.last_log_at.elapsed().as_secs_f64();
        let tokens = self.tokens_since_log;
        self.tokens_since_log = 0;
        self.last_log_at = Instant::now();
        if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_ai_core::{Device as AarambhDevice, TokenizerLike};
    use aarambh_ai_data::dataset::PlaintextDataset;
    use std::collections::HashMap;

    struct CharTokenizer {
        ids: HashMap<char, u32>,
    }

    impl TokenizerLike for CharTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .chars()
                .filter_map(|c| self.ids.get(&c).copied())
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let rev = self
                .ids
                .iter()
                .map(|(k, v)| (*v, *k))
                .collect::<HashMap<_, _>>();
            Ok(ids.iter().filter_map(|id| rev.get(id)).collect())
        }

        fn vocab_size(&self) -> usize {
            self.ids.len()
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn tiny_training_loss_decreases() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdabcdabcdabcdabcdabcd".into()]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let candle_device = device.to_candle().unwrap();
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let train_config = TrainConfig {
            lr: 1e-2,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: 4,
            warmup_steps: 0,
            min_lr_ratio: 0.1,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: std::env::temp_dir().join("aarambh_train_loss_decreases"),
        };

        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            candle_device,
            DType::F32,
        )
        .unwrap();
        let mut eval_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device);
        let eval_batch = eval_loader.next().unwrap().unwrap();
        let first = cross_entropy_loss(
            &trainer.model.forward_train(&eval_batch.input_ids).unwrap(),
            &eval_batch.labels,
            &eval_batch.attention_mask,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
        trainer.train().unwrap();
        let last = cross_entropy_loss(
            &trainer.model.forward_train(&eval_batch.input_ids).unwrap(),
            &eval_batch.labels,
            &eval_batch.attention_mask,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
        assert!(
            last < first,
            "loss did not decrease: first={first}, last={last}"
        );
    }
}
