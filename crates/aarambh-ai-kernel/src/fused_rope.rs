use candle_core::{Error, Result, Tensor};

pub fn cuda_kernels_compiled() -> bool {
    cfg!(aarambh_cuda_kernels)
}

pub fn fused_rope_apply(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    seqlen_offset: usize,
) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    {
        cuda::fused_rope_apply(x, cos, sin, seqlen_offset)
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    {
        let _ = (x, cos, sin, seqlen_offset);
        Err(Error::msg(
            "Fused CUDA RoPE is unavailable because aarambh-ai-kernel was built without CUDA PTX kernels",
        ))
    }
}

pub fn fused_rope_stub() -> Result<()> {
    Err(Error::msg(
        "Fused CUDA RoPE stubs were replaced by Phase 14 PTX kernels",
    ))
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp3, DType, Error, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const PTX: &str = include_str!(env!("AARAMBH_CUDA_ROPE_PTX"));
    const MODULE: &str = "aarambh_rope_apply";

    #[derive(Clone, Copy, Debug)]
    struct FusedRope {
        total: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        seqlen_offset: usize,
    }

    pub fn fused_rope_apply(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (seq_len, heads, head_dim) = validate_tensors(x, cos, sin, seqlen_offset)?;
        x.apply_op3_no_bwd(
            cos,
            sin,
            &FusedRope {
                total: x.elem_count(),
                seq_len,
                heads,
                head_dim,
                seqlen_offset,
            },
        )
    }

    impl CustomOp3 for FusedRope {
        fn name(&self) -> &'static str {
            "aarambh-fused-rope"
        }

        fn cpu_fwd(
            &self,
            _s1: &CpuStorage,
            _l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
            _s3: &CpuStorage,
            _l3: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Err(Error::msg("fused RoPE custom op is CUDA-only"))
        }

        fn cuda_fwd(
            &self,
            x: &CudaStorage,
            x_layout: &Layout,
            cos: &CudaStorage,
            cos_layout: &Layout,
            sin: &CudaStorage,
            sin_layout: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            validate_layout(x_layout, "RoPE input")?;
            validate_layout(cos_layout, "RoPE cos")?;
            validate_layout(sin_layout, "RoPE sin")?;
            let shape = x_layout.shape().clone();
            match x.dtype() {
                DType::F32 => launch::<f32>(
                    "aarambh_rope_apply_f32",
                    x,
                    x_layout,
                    cos,
                    cos_layout,
                    sin,
                    sin_layout,
                    *self,
                    shape,
                ),
                DType::F16 => launch::<f16>(
                    "aarambh_rope_apply_f16",
                    x,
                    x_layout,
                    cos,
                    cos_layout,
                    sin,
                    sin_layout,
                    *self,
                    shape,
                ),
                DType::BF16 => launch::<bf16>(
                    "aarambh_rope_apply_bf16",
                    x,
                    x_layout,
                    cos,
                    cos_layout,
                    sin,
                    sin_layout,
                    *self,
                    shape,
                ),
                dtype => Err(Error::msg(format!(
                    "fused RoPE supports F32/F16/BF16 on CUDA, got {dtype:?}"
                ))),
            }
        }
    }

    fn validate_tensors(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(usize, usize, usize)> {
        if !x.device().same_device(cos.device())
            || !x.device().same_device(sin.device())
            || !x.device().is_cuda()
        {
            return Err(Error::msg(
                "fused RoPE requires tensors on the same CUDA device",
            ));
        }
        if x.dtype() != cos.dtype() || x.dtype() != sin.dtype() {
            return Err(Error::msg(format!(
                "fused RoPE requires matching dtypes, got {:?}, {:?}, {:?}",
                x.dtype(),
                cos.dtype(),
                sin.dtype()
            )));
        }
        if !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::msg(format!(
                "fused RoPE supports F32/F16/BF16 on CUDA, got {:?}",
                x.dtype()
            )));
        }
        let dims = x.dims();
        if dims.len() != 4 {
            return Err(Error::msg(format!(
                "fused RoPE expects [batch, seq, heads, head_dim], got {dims:?}"
            )));
        }
        let seq_len = dims[1];
        let heads = dims[2];
        let head_dim = dims[3];
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::msg(format!(
                "fused RoPE requires an even non-zero head_dim, got {head_dim}"
            )));
        }
        if cos.dims().len() != 2
            || sin.dims().len() != 2
            || cos.dims()[1] != head_dim / 2
            || sin.dims()[1] != head_dim / 2
            || cos.dims()[0] != sin.dims()[0]
            || seqlen_offset + seq_len > cos.dims()[0]
        {
            return Err(Error::msg(format!(
                "fused RoPE trig cache mismatch: x={:?}, cos={:?}, sin={:?}, offset={seqlen_offset}",
                x.dims(),
                cos.dims(),
                sin.dims()
            )));
        }
        Ok((seq_len, heads, head_dim))
    }

    fn validate_layout(layout: &Layout, name: &str) -> Result<()> {
        if !layout.is_contiguous() {
            return Err(Error::msg(format!(
                "{name} must be contiguous for fused CUDA RoPE"
            )));
        }
        Ok(())
    }

    fn launch<T>(
        fn_name: &str,
        x: &CudaStorage,
        x_layout: &Layout,
        cos: &CudaStorage,
        cos_layout: &Layout,
        sin: &CudaStorage,
        sin_layout: &Layout,
        params: FusedRope,
        shape: Shape,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
    {
        let dev = x.device.clone();
        let mut out = dev.alloc_zeros::<T>(params.total)?;
        let x_slice = x.as_cuda_slice::<T>()?.slice(x_layout.start_offset()..);
        let cos_slice = cos.as_cuda_slice::<T>()?.slice(cos_layout.start_offset()..);
        let sin_slice = sin.as_cuda_slice::<T>()?.slice(sin_layout.start_offset()..);
        let func = dev.get_or_load_custom_func(fn_name, MODULE, PTX)?;
        let cfg = LaunchConfig::for_num_elems(params.total as u32);
        let total = params.total as i32;
        let seq_len = params.seq_len as i32;
        let heads = params.heads as i32;
        let head_dim = params.head_dim as i32;
        let seqlen_offset = params.seqlen_offset as i32;
        let mut builder = func.builder();
        builder
            .arg(&mut out)
            .arg(&x_slice)
            .arg(&cos_slice)
            .arg(&sin_slice)
            .arg(&total)
            .arg(&seq_len)
            .arg(&heads)
            .arg(&head_dim)
            .arg(&seqlen_offset);
        unsafe { builder.launch(cfg).w()? };
        Ok((CudaStorage::wrap_cuda_slice(out, dev), shape))
    }
}
