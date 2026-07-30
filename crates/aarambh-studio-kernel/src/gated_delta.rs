use candle_core::{Result, Tensor};

/// Run one optimized FP32 Gated DeltaNet recurrent update.
pub fn gated_delta_recurrent(
    packed: &Tensor,
    state: &Tensor,
    key_dim: usize,
    value_dim: usize,
) -> Result<Tensor> {
    match crate::dispatch::gated_delta_path(packed, state) {
        crate::dispatch::KernelPath::CpuSimd => {
            return crate::cpu::gated_delta::cpu_gated_delta_recurrent(
                packed, state, key_dim, value_dim,
            );
        }
        #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
        crate::dispatch::KernelPath::CudaGatedDelta => {
            return cuda::gated_delta_recurrent(packed, state, key_dim, value_dim);
        }
        _ => {}
    }
    Err(candle_core::Error::msg(
        "optimized Gated DeltaNet recurrence is unavailable for this device",
    ))
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor};

    const PTX: &str = include_str!(env!("AARAMBH_STUDIO_CUDA_GATED_DELTA_PTX"));
    const MODULE: &str = "aarambh_gated_delta";

    #[derive(Debug, Clone, Copy)]
    struct CudaGatedDelta {
        batch: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    }

    pub fn gated_delta_recurrent(
        packed: &Tensor,
        state: &Tensor,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Tensor> {
        let (batch, heads, width) = packed.dims3()?;
        if packed.dtype() != DType::F32
            || state.dtype() != DType::F32
            || !packed.device().same_device(state.device())
            || width != key_dim * 2 + value_dim + 2
            || state.dims() != [batch, heads, key_dim, value_dim]
        {
            return Err(Error::msg(format!(
                "invalid CUDA gated delta tensors packed={:?}/{:?}, state={:?}/{:?}",
                packed.dims(),
                packed.dtype(),
                state.dims(),
                state.dtype()
            )));
        }
        packed.apply_op2_no_bwd(
            state,
            &CudaGatedDelta {
                batch,
                heads,
                key_dim,
                value_dim,
            },
        )
    }

    impl CustomOp2 for CudaGatedDelta {
        fn name(&self) -> &'static str {
            "aarambh-gated-delta-cuda"
        }

        fn cpu_fwd(
            &self,
            _s1: &CpuStorage,
            _l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Err(Error::msg("CUDA gated delta custom op is CUDA-only"))
        }

        fn cuda_fwd(
            &self,
            packed: &CudaStorage,
            packed_layout: &Layout,
            state: &CudaStorage,
            state_layout: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            if !packed_layout.is_contiguous() || !state_layout.is_contiguous() {
                return Err(Error::msg("CUDA gated delta inputs must be contiguous"));
            }
            let rows = self.batch * self.heads;
            let output_width = self.key_dim * self.value_dim + self.value_dim;
            let dev = packed.device.clone();
            let mut output = dev.alloc_zeros::<f32>(rows * output_width)?;
            let packed = packed
                .as_cuda_slice::<f32>()?
                .slice(packed_layout.start_offset()..);
            let state = state
                .as_cuda_slice::<f32>()?
                .slice(state_layout.start_offset()..);
            let func =
                dev.get_or_load_custom_func("aarambh_gated_delta_recurrent_f32", MODULE, PTX)?;
            let config = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let rows = rows as i32;
            let key_dim = self.key_dim as i32;
            let value_dim = self.value_dim as i32;
            let mut builder = func.builder();
            builder
                .arg(&mut output)
                .arg(&packed)
                .arg(&state)
                .arg(&rows)
                .arg(&key_dim)
                .arg(&value_dim);
            // SAFETY: Shapes and contiguous layouts are validated above; each
            // CUDA block owns one disjoint recurrent-state row in the output.
            unsafe { builder.launch(config).w()? };
            Ok((
                CudaStorage::wrap_cuda_slice(output, dev),
                Shape::from((self.batch, self.heads, output_width)),
            ))
        }
    }
}
