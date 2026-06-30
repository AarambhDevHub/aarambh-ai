use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::optim::AdamW;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
/// Serializable training progress saved with each checkpoint.
pub struct TrainState {
    /// Completed optimizer steps.
    pub step: usize,
    /// Completed training epochs.
    pub epoch: usize,
    /// Completed micro-batches including gradient accumulation.
    pub micro_step: usize,
    /// Most recent training loss.
    pub train_loss: Option<f64>,
    /// Most recent validation loss.
    pub val_loss: Option<f64>,
    /// Best validation loss observed so far.
    pub best_val_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
    step: usize,
}

#[derive(Debug, Clone)]
/// Saves and restores model, optimizer, and training state checkpoints.
pub struct CheckpointManager {
    dir: PathBuf,
}

impl CheckpointManager {
    /// Create a checkpoint manager rooted at `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Return the checkpoint root directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Save a step checkpoint and update the latest pointer.
    pub fn save(&self, varmap: &VarMap, optimizer: &AdamW, state: &TrainState) -> Result<PathBuf> {
        self.save_named(
            varmap,
            optimizer,
            state,
            format!("step_{:06}", state.step),
            "latest.json",
        )
    }

    /// Save the best checkpoint and update the best pointer.
    pub fn save_best(
        &self,
        varmap: &VarMap,
        optimizer: &AdamW,
        state: &TrainState,
    ) -> Result<PathBuf> {
        self.save_named(varmap, optimizer, state, "best".to_string(), "best.json")
    }

    /// Load the checkpoint referenced by the latest pointer.
    pub fn load_latest(
        &self,
        varmap: &mut VarMap,
        optimizer: &mut AdamW,
        device: &candle_core::Device,
    ) -> Result<Option<TrainState>> {
        let pointer_path = self.dir.join("latest.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let file = fs::File::open(&pointer_path)?;
        let pointer: CheckpointPointer = serde_json::from_reader(file)?;
        let state = self.load_from_dir(&pointer.path, varmap, optimizer, device)?;
        Ok(Some(state))
    }

    /// Load model, optimizer, and state from a checkpoint directory.
    pub fn load_from_dir(
        &self,
        path: impl AsRef<Path>,
        varmap: &mut VarMap,
        optimizer: &mut AdamW,
        device: &candle_core::Device,
    ) -> Result<TrainState> {
        let path = path.as_ref();
        varmap.load(path.join("model.safetensors"))?;
        optimizer.load_state(path.join("optimizer.safetensors"), device)?;

        let file = fs::File::open(path.join("train_state.json"))?;
        let state: TrainState = serde_json::from_reader(file)?;
        optimizer.set_step(state.step);
        Ok(state)
    }

    fn save_named(
        &self,
        varmap: &VarMap,
        optimizer: &AdamW,
        state: &TrainState,
        name: String,
        pointer_name: &str,
    ) -> Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let checkpoint_dir = self.dir.join(name);
        fs::create_dir_all(&checkpoint_dir)?;

        varmap.save(checkpoint_dir.join("model.safetensors"))?;
        optimizer.save_state(checkpoint_dir.join("optimizer.safetensors"))?;
        write_json(checkpoint_dir.join("train_state.json"), state)?;

        let pointer = CheckpointPointer {
            path: checkpoint_dir.clone(),
            step: state.step,
        };
        write_json(self.dir.join(pointer_name), &pointer)?;
        Ok(checkpoint_dir)
    }
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::{AdamW, AdamWConfig, GradMap};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Init, VarMap};

    #[test]
    fn checkpoint_roundtrip_restores_weights_optimizer_and_state() {
        let device = Device::Cpu;
        let dir = std::env::temp_dir().join(format!(
            "aarambh_checkpoint_roundtrip_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);

        let mut varmap = VarMap::new();
        varmap
            .get((2,), "w", Init::Const(1.0), DType::F32, &device)
            .unwrap();
        let mut optimizer = AdamW::from_varmap(
            &varmap,
            AdamWConfig {
                beta1: 0.9,
                beta2: 0.95,
                epsilon: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();
        let grads = GradMap::from([(
            "w".to_string(),
            Tensor::from_vec(vec![0.25f32, -0.25], (2,), &device).unwrap(),
        )]);
        optimizer.step(&grads, 1e-3).unwrap();

        let state = TrainState {
            step: 1,
            epoch: 0,
            micro_step: 1,
            train_loss: Some(2.0),
            val_loss: Some(1.5),
            best_val_loss: Some(1.5),
        };
        let manager = CheckpointManager::new(&dir);
        manager.save(&varmap, &optimizer, &state).unwrap();

        let zeros = Tensor::zeros((2,), DType::F32, &device).unwrap();
        varmap.set_one("w", &zeros).unwrap();
        let mut loaded_optimizer = AdamW::from_varmap(
            &varmap,
            AdamWConfig {
                beta1: 0.9,
                beta2: 0.95,
                epsilon: 1e-8,
                weight_decay: 0.0,
            },
        )
        .unwrap();

        let loaded_state = manager
            .load_latest(&mut varmap, &mut loaded_optimizer, &device)
            .unwrap()
            .unwrap();
        let data = varmap.data().lock().unwrap();
        let restored = data.get("w").unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(loaded_state.step, 1);
        assert_eq!(loaded_optimizer.step_num(), 1);
        assert_ne!(restored, vec![0.0, 0.0]);

        let _ = fs::remove_dir_all(&dir);
    }
}
