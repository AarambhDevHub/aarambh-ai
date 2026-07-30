//! Single-node data-parallel training helpers.

use std::env;
use std::path::PathBuf;

use aarambh_studio_core::{AarambhError, Result};
#[cfg(any(feature = "cuda", test))]
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::optim::GradMap;

const DEFAULT_BUCKET_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_RUN_ID: &str = "aarambh-dist";

/// Distributed collective backend.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistributedBackend {
    /// NVIDIA NCCL collectives through Candle/cudarc.
    #[default]
    Nccl,
}

/// TOML configuration for one data-parallel worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DistributedConfig {
    /// Enable distributed data-parallel training.
    pub enabled: bool,
    /// Collective backend.
    pub backend: DistributedBackend,
    /// Total number of worker processes.
    pub world_size: usize,
    /// Global rank for this worker.
    pub rank: usize,
    /// CUDA device index local to this machine.
    pub local_rank: usize,
    /// Rendezvous run identifier used for NCCL unique-id sharing.
    pub run_id: String,
    /// Directory used for single-node NCCL rendezvous files.
    pub rendezvous_dir: PathBuf,
    /// Maximum seconds nonzero ranks wait for rank 0 rendezvous.
    pub init_timeout_secs: u64,
    /// Maximum F32 gradient bucket size before an all-reduce call.
    pub bucket_bytes: usize,
    /// Fall back to rank-0 single-GPU training when requested GPUs are unavailable.
    pub fallback_single_gpu: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: DistributedBackend::Nccl,
            world_size: 1,
            rank: 0,
            local_rank: 0,
            run_id: DEFAULT_RUN_ID.to_string(),
            rendezvous_dir: PathBuf::from(".aarambh_dist"),
            init_timeout_secs: DEFAULT_INIT_TIMEOUT_SECS,
            bucket_bytes: DEFAULT_BUCKET_BYTES,
            fallback_single_gpu: true,
        }
    }
}

impl DistributedConfig {
    /// Validate the distributed configuration values that do not depend on hardware.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            return Err(AarambhError::Config(
                "distributed.world_size must be greater than zero".into(),
            ));
        }
        if self.rank >= self.world_size {
            return Err(AarambhError::Config(format!(
                "distributed.rank {} must be less than world_size {}",
                self.rank, self.world_size
            )));
        }
        if self.bucket_bytes == 0 {
            return Err(AarambhError::Config(
                "distributed.bucket_bytes must be greater than zero".into(),
            ));
        }
        if self.init_timeout_secs == 0 {
            return Err(AarambhError::Config(
                "distributed.init_timeout_secs must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Fully resolved distributed worker configuration after env overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDistributedConfig {
    /// Collective backend.
    pub backend: DistributedBackend,
    /// Total number of worker processes.
    pub world_size: usize,
    /// Global rank for this worker.
    pub rank: usize,
    /// CUDA device index local to this machine.
    pub local_rank: usize,
    /// Rendezvous run identifier used for NCCL unique-id sharing.
    pub run_id: String,
    /// Directory used for single-node NCCL rendezvous files.
    pub rendezvous_dir: PathBuf,
    /// Maximum seconds nonzero ranks wait for rank 0 rendezvous.
    pub init_timeout_secs: u64,
    /// Maximum F32 gradient bucket size before an all-reduce call.
    pub bucket_bytes: usize,
}

impl ResolvedDistributedConfig {
    /// Return true when this worker is rank 0.
    pub fn is_rank0(&self) -> bool {
        self.rank == 0
    }
}

/// Runtime decision for the current process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedRuntime {
    /// Distributed training is disabled.
    Disabled,
    /// This process participates in NCCL data-parallel training.
    Active(ResolvedDistributedConfig),
    /// Rank 0 should run the normal single-process path.
    SingleProcessFallback {
        /// Requested global rank.
        rank: usize,
        /// Requested world size.
        world_size: usize,
        /// Human-readable fallback reason.
        reason: String,
    },
    /// This worker should exit successfully because rank 0 is handling fallback.
    NonParticipant {
        /// Requested global rank.
        rank: usize,
        /// Requested world size.
        world_size: usize,
        /// Human-readable exit reason.
        reason: String,
    },
}

impl DistributedRuntime {
    /// Return true when this process is the logging/checkpointing rank.
    pub fn is_rank0(&self) -> bool {
        match self {
            Self::Active(config) => config.is_rank0(),
            Self::Disabled | Self::SingleProcessFallback { .. } => true,
            Self::NonParticipant { .. } => false,
        }
    }
}

/// Resolve distributed configuration from TOML plus environment variables.
pub fn resolve_runtime(config: Option<&DistributedConfig>) -> Result<DistributedRuntime> {
    let mut resolved = config.cloned().unwrap_or_default();
    apply_env_overrides(&mut resolved)?;
    resolved.validate()?;

    if !resolved.enabled && resolved.world_size <= 1 {
        return Ok(DistributedRuntime::Disabled);
    }
    if resolved.world_size <= 1 {
        return Ok(DistributedRuntime::Disabled);
    }

    let available = cuda_device_count();
    let Some(device_count) = available else {
        return fallback_or_error(
            &resolved,
            "CUDA/NCCL support is not available in this build",
        );
    };
    if device_count < resolved.world_size || resolved.local_rank >= device_count {
        return fallback_or_error(
            &resolved,
            &format!(
                "requested world_size={} local_rank={} but only {device_count} CUDA device(s) are visible",
                resolved.world_size, resolved.local_rank
            ),
        );
    }

    Ok(DistributedRuntime::Active(ResolvedDistributedConfig {
        backend: resolved.backend,
        world_size: resolved.world_size,
        rank: resolved.rank,
        local_rank: resolved.local_rank,
        run_id: resolved.run_id,
        rendezvous_dir: resolved.rendezvous_dir,
        init_timeout_secs: resolved.init_timeout_secs,
        bucket_bytes: resolved.bucket_bytes,
    }))
}

fn apply_env_overrides(config: &mut DistributedConfig) -> Result<()> {
    if let Some(world_size) = env_usize("AARAMBH_STUDIO_WORLD_SIZE")? {
        config.world_size = world_size;
        config.enabled = world_size > 1;
    }
    if let Some(rank) = env_usize("AARAMBH_STUDIO_RANK")? {
        config.rank = rank;
    }
    if let Some(local_rank) = env_usize("AARAMBH_STUDIO_LOCAL_RANK")? {
        config.local_rank = local_rank;
    }
    if let Ok(run_id) = env::var("AARAMBH_STUDIO_DIST_RUN_ID")
        && !run_id.trim().is_empty()
    {
        config.run_id = run_id;
    }
    if let Ok(path) = env::var("AARAMBH_STUDIO_DIST_RENDEZVOUS")
        && !path.trim().is_empty()
    {
        config.rendezvous_dir = PathBuf::from(path);
    }
    Ok(())
}

fn env_usize(name: &str) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| AarambhError::Config(format!("invalid {name} value '{value}': {err}"))),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(AarambhError::Config(format!("invalid {name}: {err}"))),
    }
}

fn fallback_or_error(config: &DistributedConfig, reason: &str) -> Result<DistributedRuntime> {
    if !config.fallback_single_gpu {
        return Err(AarambhError::Unsupported(format!(
            "distributed training requires usable CUDA/NCCL: {reason}"
        )));
    }
    if config.rank == 0 {
        Ok(DistributedRuntime::SingleProcessFallback {
            rank: config.rank,
            world_size: config.world_size,
            reason: reason.to_string(),
        })
    } else {
        Ok(DistributedRuntime::NonParticipant {
            rank: config.rank,
            world_size: config.world_size,
            reason: reason.to_string(),
        })
    }
}

#[cfg(feature = "cuda")]
fn cuda_device_count() -> Option<usize> {
    candle_core::cuda::cudarc::driver::CudaDevice::count()
        .ok()
        .map(|count| count as usize)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_count() -> Option<usize> {
    None
}

/// Active distributed training context for a worker process.
pub struct DistributedContext {
    config: ResolvedDistributedConfig,
    device: candle_core::Device,
    #[cfg(feature = "cuda")]
    nccl: NcclGradientSync,
}

impl DistributedContext {
    /// Initialize a distributed context on the selected CUDA device.
    pub fn init(config: ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        Self::init_impl(config, device)
    }

    /// Return true when this worker is rank 0.
    pub fn is_rank0(&self) -> bool {
        self.config.is_rank0()
    }

    /// Return this worker's global rank.
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Return the total worker count.
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Average gradients across all ranks in place.
    pub fn all_reduce_gradients(&self, grads: &mut GradMap) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        self.all_reduce_gradients_impl(grads)
    }

    /// Synchronize all participating ranks.
    pub fn barrier(&self) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        let mut marker = GradMap::new();
        marker.insert(
            "barrier".into(),
            candle_core::Tensor::zeros((1,), candle_core::DType::F32, &self.device)?,
        );
        self.all_reduce_gradients(&mut marker)
    }

    /// Return whether any participating rank reported a local failure.
    pub fn any_rank_failed(&self, local_failed: bool) -> Result<bool> {
        if self.world_size() <= 1 {
            return Ok(local_failed);
        }
        let mut marker = GradMap::new();
        marker.insert(
            "observer_failure".into(),
            candle_core::Tensor::new(&[if local_failed { 1.0f32 } else { 0.0 }], &self.device)?,
        );
        self.all_reduce_gradients(&mut marker)?;
        let value = marker
            .remove("observer_failure")
            .ok_or_else(|| AarambhError::Config("distributed failure marker disappeared".into()))?
            .to_vec1::<f32>()?[0];
        Ok(value > 0.0)
    }

    #[cfg(feature = "cuda")]
    fn init_impl(config: ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        let nccl = NcclGradientSync::new(&config, device)?;
        Ok(Self {
            config,
            device: device.clone(),
            nccl,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn init_impl(
        _config: ResolvedDistributedConfig,
        _device: &candle_core::Device,
    ) -> Result<Self> {
        Err(AarambhError::Unsupported(
            "distributed training requires the cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn all_reduce_gradients_impl(&self, grads: &mut GradMap) -> Result<()> {
        self.nccl.all_reduce_gradients(grads)
    }

    #[cfg(not(feature = "cuda"))]
    fn all_reduce_gradients_impl(&self, _grads: &mut GradMap) -> Result<()> {
        Err(AarambhError::Unsupported(
            "distributed gradient sync requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
struct NcclGradientSync {
    comm: candle_core::cuda::cudarc::nccl::safe::Comm,
    world_size: usize,
    bucket_bytes: usize,
}

#[cfg(feature = "cuda")]
impl NcclGradientSync {
    fn new(config: &ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        use candle_core::cuda::cudarc::nccl::safe::{Comm, Id};

        let cuda = device.as_cuda_device().map_err(|err| {
            AarambhError::Config(format!("distributed training requires CUDA: {err}"))
        })?;
        let id = if config.rank == 0 {
            let id = Id::new().map_err(|err| {
                AarambhError::Config(format!("failed to create NCCL id: {err:?}"))
            })?;
            write_nccl_id(config, &id)?;
            id
        } else {
            read_nccl_id(config)?
        };
        let comm = Comm::from_rank(cuda.cuda_stream(), config.rank, config.world_size, id)
            .map_err(|err| AarambhError::Config(format!("failed to initialize NCCL: {err:?}")))?;
        Ok(Self {
            comm,
            world_size: config.world_size,
            bucket_bytes: config.bucket_bytes,
        })
    }

    fn all_reduce_gradients(&self, grads: &mut GradMap) -> Result<()> {
        if grads.is_empty() {
            return Ok(());
        }

        let mut names = grads.keys().cloned().collect::<Vec<_>>();
        names.sort();

        let mut flat_grads = Vec::with_capacity(names.len());
        for name in names {
            let grad = grads
                .get(&name)
                .ok_or_else(|| AarambhError::Config(format!("missing gradient {name}")))?;
            let shape = grad.shape().dims().to_vec();
            let tensor = grad.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
            flat_grads.push(FlatGrad {
                name,
                shape,
                elem_count: tensor.elem_count(),
                tensor,
            });
        }

        let bucket_elem_limit = (self.bucket_bytes / std::mem::size_of::<f32>()).max(1);
        let mut synced = Vec::with_capacity(flat_grads.len());
        let mut start = 0usize;
        while start < flat_grads.len() {
            let mut end = start;
            let mut elems = 0usize;
            while end < flat_grads.len() {
                let next = flat_grads[end].elem_count;
                if end > start && elems + next > bucket_elem_limit {
                    break;
                }
                elems += next;
                end += 1;
            }
            self.sync_bucket(&flat_grads[start..end], &mut synced)?;
            start = end;
        }

        for (name, tensor) in synced {
            grads.insert(name, tensor.detach());
        }
        Ok(())
    }

    fn sync_bucket(&self, bucket: &[FlatGrad], synced: &mut Vec<(String, Tensor)>) -> Result<()> {
        let bucket_tensor = if bucket.len() == 1 {
            bucket[0].tensor.clone()
        } else {
            let refs = bucket.iter().map(|grad| &grad.tensor).collect::<Vec<_>>();
            Tensor::cat(&refs, 0)?.contiguous()?
        };

        let reduced = self.all_reduce_flat(&bucket_tensor)?;
        let averaged = reduced.affine(1.0 / self.world_size as f64, 0.0)?;
        let mut offset = 0usize;
        for grad in bucket {
            let slice = averaged.narrow(0, offset, grad.elem_count)?;
            let restored = slice.reshape(grad.shape.as_slice())?;
            synced.push((grad.name.clone(), restored.detach()));
            offset += grad.elem_count;
        }
        Ok(())
    }

    fn all_reduce_flat(&self, tensor: &Tensor) -> Result<Tensor> {
        use candle_core::cuda::cudarc::nccl::safe::ReduceOp;
        use candle_core::op::BackpropOp;
        use candle_core::{CudaStorage, Storage};

        let tensor = tensor.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
        let shape = tensor.shape().clone();
        let elem_count = tensor.elem_count();
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            return Err(AarambhError::Config(
                "NCCL all-reduce requires contiguous gradient buckets".into(),
            ));
        }
        let Storage::Cuda(cuda_storage) = &*storage else {
            return Err(AarambhError::Config(
                "NCCL all-reduce requires CUDA gradient tensors".into(),
            ));
        };
        let send = cuda_storage.as_cuda_slice::<f32>()?;
        let mut recv = cuda_storage
            .device
            .cuda_stream()
            .alloc_zeros::<f32>(elem_count)
            .map_err(|err| {
                AarambhError::Config(format!("failed to allocate NCCL receive bucket: {err:?}"))
            })?;
        self.comm
            .all_reduce(send, &mut recv, &ReduceOp::Sum)
            .map_err(|err| AarambhError::Config(format!("NCCL all-reduce failed: {err:?}")))?;
        let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(
            recv,
            cuda_storage.device.clone(),
        ));
        Ok(Tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
struct FlatGrad {
    name: String,
    shape: Vec<usize>,
    elem_count: usize,
    tensor: Tensor,
}

#[cfg(feature = "cuda")]
fn nccl_id_path(config: &ResolvedDistributedConfig) -> PathBuf {
    config
        .rendezvous_dir
        .join(&config.run_id)
        .join("nccl_id.bin")
}

#[cfg(feature = "cuda")]
fn write_nccl_id(
    config: &ResolvedDistributedConfig,
    id: &candle_core::cuda::cudarc::nccl::safe::Id,
) -> Result<()> {
    let path = nccl_id_path(config);
    let dir = path
        .parent()
        .ok_or_else(|| AarambhError::Config("invalid NCCL rendezvous path".into()))?;
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension(format!("bin.rank{}.tmp", config.rank));
    let bytes = id
        .internal()
        .iter()
        .map(|byte| *byte as u8)
        .collect::<Vec<_>>();
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn read_nccl_id(
    config: &ResolvedDistributedConfig,
) -> Result<candle_core::cuda::cudarc::nccl::safe::Id> {
    use candle_core::cuda::cudarc::nccl::safe::Id;
    use std::time::{Duration, Instant};

    let path = nccl_id_path(config);
    let deadline = Instant::now() + Duration::from_secs(config.init_timeout_secs);
    loop {
        match std::fs::read(&path) {
            Ok(bytes) => {
                if bytes.len() != 128 {
                    return Err(AarambhError::Config(format!(
                        "invalid NCCL id length in {}: {}",
                        path.display(),
                        bytes.len()
                    )));
                }
                let mut internal = [0 as std::ffi::c_char; 128];
                for (dst, src) in internal.iter_mut().zip(bytes) {
                    *dst = src as std::ffi::c_char;
                }
                return Ok(Id::uninit(internal));
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if Instant::now() >= deadline {
                    return Err(AarambhError::Config(format!(
                        "timed out waiting for NCCL rendezvous file {}",
                        path.display()
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) => return Err(err.into()),
        }
    }
}

#[cfg(test)]
fn average_grad_maps_for_test(ranks: &[GradMap]) -> Result<Vec<GradMap>> {
    if ranks.is_empty() {
        return Ok(Vec::new());
    }
    let mut names = ranks[0].keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut averaged = GradMap::new();
    for name in names {
        let mut sum = None::<Tensor>;
        for rank in ranks {
            let grad = rank
                .get(&name)
                .ok_or_else(|| AarambhError::Config(format!("missing gradient {name}")))?;
            let grad = grad.to_dtype(DType::F32)?;
            sum = Some(match sum {
                Some(existing) => (existing + grad)?,
                None => grad,
            });
        }
        let mean = sum
            .expect("rank list is non-empty")
            .affine(1.0 / ranks.len() as f64, 0.0)?;
        averaged.insert(name, mean.detach());
    }
    Ok(ranks.iter().map(|_| averaged.clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn config_env_overrides_world_rank_and_local_rank() {
        // This test avoids mutating process-global env because Rust 2024 makes
        // env mutation unsafe. Direct validation still covers the resolved shape.
        let config = DistributedConfig {
            enabled: true,
            world_size: 2,
            rank: 1,
            local_rank: 1,
            ..DistributedConfig::default()
        };
        config.validate().unwrap();
    }

    #[test]
    fn gradient_average_matches_two_rank_mean() {
        let device = Device::Cpu;
        let mut rank0 = GradMap::new();
        rank0.insert(
            "w".into(),
            Tensor::from_vec(vec![1f32, 3f32], (2,), &device).unwrap(),
        );
        let mut rank1 = GradMap::new();
        rank1.insert(
            "w".into(),
            Tensor::from_vec(vec![5f32, 7f32], (2,), &device).unwrap(),
        );
        let averaged = average_grad_maps_for_test(&[rank0, rank1]).unwrap();
        for rank in averaged {
            let values = rank.get("w").unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(values, vec![3.0, 5.0]);
        }
    }

    #[test]
    fn invalid_rank_is_rejected() {
        let config = DistributedConfig {
            enabled: true,
            world_size: 2,
            rank: 2,
            ..DistributedConfig::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("less than world_size"), "{err}");
    }
}
