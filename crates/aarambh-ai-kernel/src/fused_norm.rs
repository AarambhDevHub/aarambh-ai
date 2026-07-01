use candle_core::{Result, Tensor};

/// Return true when CUDA PTX kernels were compiled into this crate.
pub fn cuda_kernels_compiled() -> bool {
    cfg!(aarambh_cuda_kernels)
}

/// Run the CUDA fused RMSNorm kernel.
pub fn fused_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    {
        cuda::fused_rms_norm(x, weight, eps)
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    {
        let _ = (x, weight, eps);
        Err(candle_core::Error::msg(
            "Fused CUDA RMSNorm is unavailable because aarambh-ai-kernel was built without CUDA PTX kernels",
        ))
    }
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const PTX: &str = include_str!(env!("AARAMBH_CUDA_RMS_NORM_PTX"));
    const MODULE: &str = "aarambh_rms_norm_fused";
    const BLOCK_SIZE: u32 = 256;

    #[derive(Clone, Copy, Debug)]
    struct FusedRmsNorm {
        eps: f32,
        rows: usize,
        hidden: usize,
    }

    pub fn fused_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        validate_tensors(x, weight)?;
        let hidden = *x
            .dims()
            .last()
            .ok_or_else(|| Error::msg("rms_norm expects rank >= 1"))?;
        let rows = x.elem_count() / hidden;
        x.apply_op2_no_bwd(weight, &FusedRmsNorm { eps, rows, hidden })
    }

    impl CustomOp2 for FusedRmsNorm {
        fn name(&self) -> &'static str {
            "aarambh-fused-rms-norm"
        }

        fn cpu_fwd(
            &self,
            _s1: &CpuStorage,
            _l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Err(Error::msg("fused RMSNorm custom op is CUDA-only"))
        }

        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            x_layout: &Layout,
            weight: &CudaStorage,
            weight_layout: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            validate_layout(x_layout, "rms_norm input")?;
            validate_layout(weight_layout, "rms_norm weight")?;
            let shape = x_layout.shape().clone();
            match x.dtype() {
                DType::F32 => launch::<f32>(
                    "aarambh_rms_norm_f32",
                    x,
                    x_layout,
                    weight,
                    weight_layout,
                    self.rows,
                    self.hidden,
                    self.eps,
                    shape,
                ),
                DType::F16 => launch::<f16>(
                    "aarambh_rms_norm_f16",
                    x,
                    x_layout,
                    weight,
                    weight_layout,
                    self.rows,
                    self.hidden,
                    self.eps,
                    shape,
                ),
                DType::BF16 => launch::<bf16>(
                    "aarambh_rms_norm_bf16",
                    x,
                    x_layout,
                    weight,
                    weight_layout,
                    self.rows,
                    self.hidden,
                    self.eps,
                    shape,
                ),
                dtype => Err(Error::msg(format!(
                    "fused RMSNorm supports F32/F16/BF16 on CUDA, got {dtype:?}"
                ))),
            }
        }
    }

    fn validate_tensors(x: &Tensor, weight: &Tensor) -> Result<()> {
        if !x.device().same_device(weight.device()) || !x.device().is_cuda() {
            return Err(Error::msg(
                "fused RMSNorm requires tensors on the same CUDA device",
            ));
        }
        if x.dtype() != weight.dtype() {
            return Err(Error::msg(format!(
                "fused RMSNorm requires matching dtypes, got {:?} and {:?}",
                x.dtype(),
                weight.dtype()
            )));
        }
        let Some(&hidden) = x.dims().last() else {
            return Err(Error::msg("rms_norm expects rank >= 1"));
        };
        if hidden == 0 {
            return Err(Error::msg("rms_norm hidden dimension must be non-zero"));
        }
        if weight.dims() != [hidden] {
            return Err(Error::msg(format!(
                "rms_norm weight must have shape [{hidden}], got {:?}",
                weight.dims()
            )));
        }
        if !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::msg(format!(
                "fused RMSNorm supports F32/F16/BF16 on CUDA, got {:?}",
                x.dtype()
            )));
        }
        Ok(())
    }

    fn validate_layout(layout: &Layout, name: &str) -> Result<()> {
        if !layout.is_contiguous() {
            return Err(Error::msg(format!(
                "{name} must be contiguous for fused CUDA RMSNorm"
            )));
        }
        Ok(())
    }

    fn launch<T>(
        fn_name: &str,
        x: &CudaStorage,
        x_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        rows: usize,
        hidden: usize,
        eps: f32,
        shape: Shape,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
    {
        let dev = x.device.clone();
        let mut out = dev.alloc_zeros::<T>(rows * hidden)?;
        let x_slice = x.as_cuda_slice::<T>()?.slice(x_layout.start_offset()..);
        let weight_slice = weight
            .as_cuda_slice::<T>()?
            .slice(weight_layout.start_offset()..);
        let func = dev.get_or_load_custom_func(fn_name, MODULE, PTX)?;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
        };
        let rows = rows as i32;
        let hidden = hidden as i32;
        let mut builder = func.builder();
        builder
            .arg(&mut out)
            .arg(&x_slice)
            .arg(&weight_slice)
            .arg(&rows)
            .arg(&hidden)
            .arg(&eps);
        unsafe { builder.launch(cfg).w()? };
        Ok((CudaStorage::wrap_cuda_slice(out, dev), shape))
    }
}
