use candle_core::{DType, Result, Tensor};

/// Run inference-only DSA attention using selected causal block ids.
///
/// CPU F32 uses the Rayon online-softmax path with the additive sparse mask.
/// CUDA F32/F16/BF16 uses the compiled sparse PTX kernel when available. Other
/// combinations retain the exact Candle additive-mask fallback.
#[allow(clippy::too_many_arguments)]
pub fn dsa_sparse_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    selected_blocks: &[u32],
    selected_per_query: usize,
    block_size: usize,
    scale: f64,
) -> Result<Tensor> {
    if q.device().is_cpu() && q.dtype() == DType::F32 {
        return crate::dispatch::attention_forward_additive(q, k, v, mask, scale);
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    let _ = (selected_blocks, selected_per_query, block_size);
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    if q.device().is_cuda()
        && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && let Ok(output) = cuda::forward(
            q,
            k,
            v,
            selected_blocks,
            selected_per_query,
            block_size,
            scale,
        )
    {
        return Ok(output);
    }
    crate::dispatch::attention_forward_candle(q, k, v, Some(mask), scale)
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp3, DType, Error, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const PTX: &str = include_str!(env!("AARAMBH_CUDA_DSA_PTX"));
    const MODULE: &str = "aarambh_deepseek_sparse_attention";
    const THREADS: u32 = 256;

    #[derive(Debug, Clone)]
    struct SparseAttention {
        selected_blocks: Vec<u32>,
        selected_per_query: usize,
        block_size: usize,
        batch: usize,
        heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f64,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        selected_blocks: &[u32],
        selected_per_query: usize,
        block_size: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let (batch, heads, q_len, head_dim) = q.dims4()?;
        let (k_batch, k_heads, kv_len, k_dim) = k.dims4()?;
        if v.dims4()? != (k_batch, k_heads, kv_len, k_dim)
            || batch != k_batch
            || heads != k_heads
            || head_dim != k_dim
            || q.dtype() != k.dtype()
            || q.dtype() != v.dtype()
            || !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || selected_per_query == 0
            || block_size == 0
            || selected_blocks.len() != batch * q_len * selected_per_query
        {
            return Err(Error::msg(format!(
                "invalid DSA CUDA inputs q={:?} k={:?} v={:?} selected={}",
                q.dims(),
                k.dims(),
                v.dims(),
                selected_blocks.len()
            )));
        }
        q.apply_op3_no_bwd(
            k,
            v,
            &SparseAttention {
                selected_blocks: selected_blocks.to_vec(),
                selected_per_query,
                block_size,
                batch,
                heads,
                q_len,
                kv_len,
                head_dim,
                scale,
            },
        )
    }

    impl CustomOp3 for SparseAttention {
        fn name(&self) -> &'static str {
            "aarambh-dsa-sparse-attention"
        }

        fn cpu_fwd(
            &self,
            _q: &CpuStorage,
            _q_layout: &Layout,
            _k: &CpuStorage,
            _k_layout: &Layout,
            _v: &CpuStorage,
            _v_layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Err(Error::msg("DSA PTX custom op is CUDA-only"))
        }

        fn cuda_fwd(
            &self,
            q: &CudaStorage,
            q_layout: &Layout,
            k: &CudaStorage,
            k_layout: &Layout,
            v: &CudaStorage,
            v_layout: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
                return Err(Error::msg("DSA CUDA q/k/v tensors must be contiguous"));
            }
            match q.dtype() {
                DType::F32 => launch::<f32>(
                    "aarambh_dsa_sparse_f32",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    self,
                ),
                DType::F16 => launch::<f16>(
                    "aarambh_dsa_sparse_f16",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    self,
                ),
                DType::BF16 => launch::<bf16>(
                    "aarambh_dsa_sparse_bf16",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    self,
                ),
                dtype => Err(Error::msg(format!("DSA CUDA does not support {dtype:?}"))),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch<T>(
        function_name: &str,
        q: &CudaStorage,
        q_layout: &Layout,
        k: &CudaStorage,
        k_layout: &Layout,
        v: &CudaStorage,
        v_layout: &Layout,
        params: &SparseAttention,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
    {
        let dev = q.device.clone();
        let elements = params.batch * params.heads * params.q_len * params.head_dim;
        let mut output = dev.alloc_zeros::<T>(elements)?;
        let selected = dev.clone_htod(&params.selected_blocks)?;
        let q = q.as_cuda_slice::<T>()?.slice(q_layout.start_offset()..);
        let k = k.as_cuda_slice::<T>()?.slice(k_layout.start_offset()..);
        let v = v.as_cuda_slice::<T>()?.slice(v_layout.start_offset()..);
        let function = dev.get_or_load_custom_func(function_name, MODULE, PTX)?;
        let launch = LaunchConfig {
            grid_dim: ((elements as u32).div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch = params.batch as i32;
        let heads = params.heads as i32;
        let q_len = params.q_len as i32;
        let kv_len = params.kv_len as i32;
        let head_dim = params.head_dim as i32;
        let selected_per_query = params.selected_per_query as i32;
        let block_size = params.block_size as i32;
        let scale = params.scale as f32;
        let mut builder = function.builder();
        builder
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&selected)
            .arg(&mut output)
            .arg(&batch)
            .arg(&heads)
            .arg(&q_len)
            .arg(&kv_len)
            .arg(&head_dim)
            .arg(&selected_per_query)
            .arg(&block_size)
            .arg(&scale);
        // SAFETY: all layouts, lengths, dtypes, and launch dimensions are
        // validated before device pointers are passed to the PTX kernel.
        unsafe { builder.launch(launch).w()? };
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev),
            Shape::from((params.batch, params.heads, params.q_len, params.head_dim)),
        ))
    }
}
