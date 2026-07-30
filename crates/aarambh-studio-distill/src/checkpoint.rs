use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_train::AdamW;
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

/// Exact resumable state for a distillation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillState {
    /// Completed optimizer updates.
    pub step: usize,
    /// Completed prompt-dataset epochs.
    pub epoch: usize,
    /// Completed micro-batches.
    pub micro_step: usize,
    /// Cursor into the current deterministic prompt order.
    pub prompt_position: usize,
    /// Current deterministic prompt ordering.
    pub prompt_order: Vec<usize>,
    /// Most recent total training loss.
    pub train_loss: Option<f64>,
    /// Total student completion tokens generated so far.
    pub rollout_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
    step: usize,
}

/// Full-model checkpoint manager for distillation state and optimizer moments.
#[derive(Debug, Clone)]
pub struct DistillCheckpointManager {
    root: PathBuf,
}

impl DistillCheckpointManager {
    /// Create a checkpoint manager rooted at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { root: path.into() }
    }

    /// Return the checkpoint root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Save a numbered checkpoint and update `latest.json`.
    pub fn save(
        &self,
        varmap: &VarMap,
        optimizer: &AdamW,
        state: &DistillState,
        manifest: &serde_json::Value,
    ) -> Result<PathBuf> {
        self.save_named(
            varmap,
            optimizer,
            state,
            manifest,
            format!("step_{:06}", state.step),
            true,
        )
    }

    /// Save the final checkpoint without changing the latest numbered pointer.
    pub fn save_final(
        &self,
        varmap: &VarMap,
        optimizer: &AdamW,
        state: &DistillState,
        manifest: &serde_json::Value,
    ) -> Result<PathBuf> {
        self.save_named(
            varmap,
            optimizer,
            state,
            manifest,
            "final".to_string(),
            false,
        )
    }

    /// Restore the checkpoint referenced by `latest.json` when present.
    pub fn load_latest(
        &self,
        varmap: &mut VarMap,
        optimizer: &mut AdamW,
        expected_manifest: &serde_json::Value,
        device: &candle_core::Device,
    ) -> Result<Option<DistillState>> {
        let pointer_path = self.root.join("latest.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: CheckpointPointer = read_json(&pointer_path)?;
        let actual_manifest: serde_json::Value = read_json(pointer.path.join("manifest.json"))?;
        if &actual_manifest != expected_manifest {
            return Err(AarambhError::Checkpoint(
                "distillation resume manifest does not match the requested run".into(),
            ));
        }
        varmap.load(pointer.path.join("model.safetensors"))?;
        optimizer.load_state(pointer.path.join("optimizer.safetensors"), device)?;
        let state: DistillState = read_json(pointer.path.join("distill_state.json"))?;
        if state.step != pointer.step {
            return Err(AarambhError::Checkpoint(format!(
                "distillation pointer step {} does not match state step {}",
                pointer.step, state.step
            )));
        }
        optimizer.set_step(state.step);
        Ok(Some(state))
    }

    fn save_named(
        &self,
        varmap: &VarMap,
        optimizer: &AdamW,
        state: &DistillState,
        manifest: &serde_json::Value,
        name: String,
        update_latest: bool,
    ) -> Result<PathBuf> {
        fs::create_dir_all(&self.root)?;
        let directory = self.root.join(name);
        fs::create_dir_all(&directory)?;
        varmap.save(directory.join("model.safetensors"))?;
        optimizer.save_state(directory.join("optimizer.safetensors"))?;
        write_json(directory.join("distill_state.json"), state)?;
        write_json(directory.join("manifest.json"), manifest)?;
        if update_latest {
            write_json(
                self.root.join("latest.json"),
                &CheckpointPointer {
                    path: directory.clone(),
                    step: state.step,
                },
            )?;
        }
        Ok(directory)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let file = fs::File::open(path.as_ref())?;
    serde_json::from_reader(file).map_err(AarambhError::Json)
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_train::{AdamWConfig, GradMap};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Init, VarMap};

    #[test]
    fn checkpoint_roundtrip_restores_exact_distill_state_and_manifest() {
        let device = Device::Cpu;
        let directory =
            std::env::temp_dir().join(format!("aarambh_distill_checkpoint_{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let mut varmap = VarMap::new();
        varmap
            .get((2,), "w", Init::Const(1.0), DType::F32, &device)
            .unwrap();
        let optimizer_config = AdamWConfig {
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut optimizer = AdamW::from_varmap(&varmap, optimizer_config.clone()).unwrap();
        optimizer
            .step(
                &GradMap::from([(
                    "w".into(),
                    Tensor::from_vec(vec![0.25f32, -0.25], 2, &device).unwrap(),
                )]),
                1e-3,
            )
            .unwrap();
        let state = DistillState {
            step: 1,
            epoch: 2,
            micro_step: 3,
            prompt_position: 1,
            prompt_order: vec![2, 0, 1],
            train_loss: Some(0.75),
            rollout_tokens: 64,
        };
        let manifest = serde_json::json!({"teacher": "local", "schema": 1});
        let manager = DistillCheckpointManager::new(&directory);
        manager
            .save(&varmap, &optimizer, &state, &manifest)
            .unwrap();

        varmap
            .set_one("w", Tensor::zeros(2, DType::F32, &device).unwrap())
            .unwrap();
        let mut loaded_optimizer = AdamW::from_varmap(&varmap, optimizer_config).unwrap();
        let loaded = manager
            .load_latest(&mut varmap, &mut loaded_optimizer, &manifest, &device)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.step, state.step);
        assert_eq!(loaded.prompt_position, state.prompt_position);
        assert_eq!(loaded.prompt_order, state.prompt_order);
        assert_eq!(loaded_optimizer.step_num(), state.step);
        assert_ne!(
            varmap
                .data()
                .lock()
                .unwrap()
                .get("w")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0, 0.0]
        );

        let wrong_manifest = serde_json::json!({"teacher": "dataset", "schema": 1});
        assert!(
            manager
                .load_latest(&mut varmap, &mut loaded_optimizer, &wrong_manifest, &device,)
                .is_err()
        );
        let _ = fs::remove_dir_all(directory);
    }
}
