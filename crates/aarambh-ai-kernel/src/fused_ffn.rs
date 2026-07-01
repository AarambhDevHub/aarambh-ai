use candle_core::{Error, Result, Tensor};
use candle_core::backend::BackendStorage;

/// Return true when CUDA PTX kernels were compiled into this crate.
pub fn cuda_kernels_compiled() -> bool {
    cfg!(aarambh_cuda_kernels)
}

/// Run the CUDA fused SwiGLU kernel.
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    {
        cuda::fused_swiglu(gate, up)
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    {
        let _ = (gate, up);
        Err(Error::msg(
            "Fused CUDA SwiGLU is unavailable because aarambh-ai-kernel was built without CUDA PTX kernels",
        ))
    }
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const PTX: &str = include_str!(env!("AARAMBH_CUDA_SWIGLU_PTX"));
    const MODULE: &str = "aarambh_swiglu_fused";
    const BLOCK_SIZE: u32 = 256;

    #[derive(Clone, Copy, Debug)]
    struct FusedSwiGlu {
        total: usize,
    }

    pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        validate_tensors(gate, up)?;
        gate.apply_op2_no_bwd(
            up,
            &FusedSwiGlu {
                total: gate.elem_count(),
            },
        )
    }

    impl CustomOp2 for FusedSwiGlu {
        fn name(&self) -> &'static str {
            "aarambh-fused-swiglu"
        }

        fn cpu_fwd(
            &self,
            _s1: &CpuStorage,
            _l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Err(Error::msg("fused SwiGLU custom op is CUDA-only"))
        }

        fn cuda_fwd(
            &self,
            gate: &CudaStorage,
            gate_layout: &Layout,
            up: &CudaStorage,
            up_layout: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            validate_layout(gate_layout, "SwiGLU gate")?;
            validate_layout(up_layout, "SwiGLU up")?;
            let shape = gate_layout.shape().clone();
            match gate.dtype() {
                DType::F32 => launch::<f32>(
                    "aarambh_swiglu_f32",
                    gate,
                    gate_layout,
                    up,
                    up_layout,
                    self.total,
                    shape,
                ),
                DType::F16 => launch::<f16>(
                    "aarambh_swiglu_f16",
                    gate,
                    gate_layout,
                    up,
                    up_layout,
                    self.total,
                    shape,
                ),
                DType::BF16 => launch::<bf16>(
                    "aarambh_swiglu_bf16",
                    gate,
                    gate_layout,
                    up,
                    up_layout,
                    self.total,
                    shape,
                ),
                dtype => Err(Error::msg(format!(
                    "fused SwiGLU supports F32/F16/BF16 on CUDA, got {dtype:?}"
                ))),
            }
        }
    }

    fn validate_tensors(gate: &Tensor, up: &Tensor) -> Result<()> {
        if !gate.device().same_device(up.device()) || !gate.device().is_cuda() {
            return Err(Error::msg(
                "fused SwiGLU requires tensors on the same CUDA device",
            ));
        }
        if gate.shape() != up.shape() {
            return Err(Error::msg(format!(
                "fused SwiGLU requires matching shapes, got {:?} and {:?}",
                gate.dims(),
                up.dims()
            )));
        }
        if gate.dtype() != up.dtype() {
            return Err(Error::msg(format!(
                "fused SwiGLU requires matching dtypes, got {:?} and {:?}",
                gate.dtype(),
                up.dtype()
            )));
        }
        if !matches!(gate.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::msg(format!(
                "fused SwiGLU supports F32/F16/BF16 on CUDA, got {:?}",
                gate.dtype()
            )));
        }
        Ok(())
    }

    fn validate_layout(layout: &Layout, name: &str) -> Result<()> {
        if !layout.is_contiguous() {
            return Err(Error::msg(format!(
                "{name} must be contiguous for fused CUDA SwiGLU"
            )));
        }
        Ok(())
    }

    fn launch<T>(
        fn_name: &str,
        gate: &CudaStorage,
        gate_layout: &Layout,
        up: &CudaStorage,
        up_layout: &Layout,
        total: usize,
        shape: Shape,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
    {
        let dev = gate.device.clone();
        let mut out = dev.alloc_zeros::<T>(total)?;
        let gate_slice = gate
            .as_cuda_slice::<T>()?
            .slice(gate_layout.start_offset()..);
        let up_slice = up.as_cuda_slice::<T>()?.slice(up_layout.start_offset()..);
        let func = dev.get_or_load_custom_func(fn_name, MODULE, PTX)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let total = total as i32;
        let mut builder = func.builder();
        builder
            .arg(&mut out)
            .arg(&gate_slice)
            .arg(&up_slice)
            .arg(&total);
        unsafe { builder.launch(cfg).w()? };
        Ok((CudaStorage::wrap_cuda_slice(out, dev), shape))
    }
}
