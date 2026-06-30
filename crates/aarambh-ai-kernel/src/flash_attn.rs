use candle_core::{Error, Result, Tensor};

pub fn cuda_kernels_compiled() -> bool {
    cfg!(aarambh_cuda_kernels)
}

pub fn flash_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f64,
) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    {
        cuda::flash_attention_forward(q, k, v, causal, scale, false)
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    {
        let _ = (q, k, v, causal, scale);
        Err(Error::msg(
            "Flash Attention CUDA is unavailable because aarambh-ai-kernel was built without CUDA PTX kernels",
        ))
    }
}

pub fn flash_attention_forward_train(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    scale: f64,
) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
    {
        cuda::flash_attention_forward(q, k, v, causal, scale, true)
    }
    #[cfg(not(all(feature = "cuda", aarambh_cuda_kernels)))]
    {
        let _ = (q, k, v, causal, scale);
        Err(Error::msg(
            "Flash Attention CUDA is unavailable because aarambh-ai-kernel was built without CUDA PTX kernels",
        ))
    }
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CpuStorage, CustomOp3, DType, Error, Layout, Result, Shape, Tensor};
    use half::{bf16, f16};

    const FWD_PTX: &str = include_str!(env!("AARAMBH_CUDA_FLASH_ATTN_PTX"));
    const FWD_MODULE: &str = "aarambh_flash_attention";
    const BLOCK_SIZE: u32 = 256;

    #[derive(Clone, Copy, Debug)]
    struct FlashAttention {
        causal: bool,
        scale: f64,
        rows: usize,
        heads: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        with_bwd: bool,
    }

    pub fn flash_attention_forward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        scale: f64,
        with_bwd: bool,
    ) -> Result<Tensor> {
        let (rows, heads, q_len, kv_len, head_dim) = validate_tensors(q, k, v)?;
        let op = FlashAttention {
            causal,
            scale,
            rows,
            heads,
            q_len,
            kv_len,
            head_dim,
            with_bwd,
        };
        if with_bwd {
            q.apply_op3(k, v, op)
        } else {
            q.apply_op3_no_bwd(k, v, &op)
        }
    }

    impl CustomOp3 for FlashAttention {
        fn name(&self) -> &'static str {
            "aarambh-flash-attention"
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
            Err(Error::msg("Flash Attention custom op is CUDA-only"))
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
            validate_layout(q_layout, "attention q")?;
            validate_layout(k_layout, "attention k")?;
            validate_layout(v_layout, "attention v")?;
            let shape = q_layout.shape().clone();
            match q.dtype() {
                DType::F32 => launch::<f32>(
                    "aarambh_flash_attention_f32",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    *self,
                    shape,
                ),
                DType::F16 => launch::<f16>(
                    "aarambh_flash_attention_f16",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    *self,
                    shape,
                ),
                DType::BF16 => launch::<bf16>(
                    "aarambh_flash_attention_bf16",
                    q,
                    q_layout,
                    k,
                    k_layout,
                    v,
                    v_layout,
                    *self,
                    shape,
                ),
                dtype => Err(Error::msg(format!(
                    "Flash Attention supports F32/F16/BF16 on CUDA, got {dtype:?}"
                ))),
            }
        }

        fn bwd(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            out: &Tensor,
            grad_out: &Tensor,
        ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
            if !self.with_bwd {
                return Err(Error::msg(
                    "Flash Attention backward requested for inference-only op",
                ));
            }
            let (dq, dk, dv) =
                candle_flash_attention_backward(q, k, v, out, grad_out, self.causal, self.scale)?;
            Ok((Some(dq), Some(dk), Some(dv)))
        }
    }

    fn validate_tensors(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(usize, usize, usize, usize, usize)> {
        if !q.device().same_device(k.device())
            || !q.device().same_device(v.device())
            || !q.device().is_cuda()
        {
            return Err(Error::msg(
                "Flash Attention requires q/k/v on the same CUDA device",
            ));
        }
        if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
            return Err(Error::msg(format!(
                "Flash Attention requires matching dtypes, got {:?}, {:?}, {:?}",
                q.dtype(),
                k.dtype(),
                v.dtype()
            )));
        }
        if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::msg(format!(
                "Flash Attention supports F32/F16/BF16 on CUDA, got {:?}",
                q.dtype()
            )));
        }
        let dims = q.dims();
        if dims.len() != 4 || k.dims().len() != 4 || v.dims().len() != 4 {
            return Err(Error::msg(format!(
                "Flash Attention expects q/k/v rank 4, got {:?}, {:?}, {:?}",
                q.dims(),
                k.dims(),
                v.dims()
            )));
        }
        let [batch, heads, q_len, head_dim] = [dims[0], dims[1], dims[2], dims[3]];
        let [kb, kh, kv_len, kd] = [k.dims()[0], k.dims()[1], k.dims()[2], k.dims()[3]];
        let [vb, vh, vv_len, vd] = [v.dims()[0], v.dims()[1], v.dims()[2], v.dims()[3]];
        if batch != kb
            || batch != vb
            || heads != kh
            || heads != vh
            || kv_len != vv_len
            || head_dim != kd
            || head_dim != vd
            || head_dim == 0
            || head_dim > BLOCK_SIZE as usize
        {
            return Err(Error::msg(format!(
                "Flash Attention unsupported shapes q={:?}, k={:?}, v={:?}",
                q.dims(),
                k.dims(),
                v.dims()
            )));
        }
        Ok((batch * heads * q_len, heads, q_len, kv_len, head_dim))
    }

    fn validate_layout(layout: &Layout, name: &str) -> Result<()> {
        if !layout.is_contiguous() {
            return Err(Error::msg(format!(
                "{name} must be contiguous for Flash Attention"
            )));
        }
        Ok(())
    }

    fn launch<T>(
        fn_name: &str,
        q: &CudaStorage,
        q_layout: &Layout,
        k: &CudaStorage,
        k_layout: &Layout,
        v: &CudaStorage,
        v_layout: &Layout,
        params: FlashAttention,
        shape: Shape,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
    {
        let dev = q.device.clone();
        let mut out = dev.alloc_zeros::<T>(params.rows * params.head_dim)?;
        let mut lse = dev.alloc_zeros::<f32>(params.rows)?;
        let q_slice = q.as_cuda_slice::<T>()?.slice(q_layout.start_offset()..);
        let k_slice = k.as_cuda_slice::<T>()?.slice(k_layout.start_offset()..);
        let v_slice = v.as_cuda_slice::<T>()?.slice(v_layout.start_offset()..);
        let func = dev.get_or_load_custom_func(fn_name, FWD_MODULE, FWD_PTX)?;
        let cfg = LaunchConfig {
            grid_dim: (params.rows as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
        };
        let rows = params.rows as i32;
        let heads = params.heads as i32;
        let q_len = params.q_len as i32;
        let kv_len = params.kv_len as i32;
        let head_dim = params.head_dim as i32;
        let scale = params.scale as f32;
        let causal = i32::from(params.causal);
        let mut builder = func.builder();
        builder
            .arg(&mut out)
            .arg(&mut lse)
            .arg(&q_slice)
            .arg(&k_slice)
            .arg(&v_slice)
            .arg(&rows)
            .arg(&heads)
            .arg(&q_len)
            .arg(&kv_len)
            .arg(&head_dim)
            .arg(&scale)
            .arg(&causal);
        unsafe { builder.launch(cfg).w()? };
        Ok((CudaStorage::wrap_cuda_slice(out, dev), shape))
    }

    fn candle_flash_attention_backward(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &Tensor,
        grad_out: &Tensor,
        causal: bool,
        scale: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = if causal {
            scores.broadcast_add(&causal_mask(q, k)?)?
        } else {
            scores
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let dv = probs.transpose(2, 3)?.contiguous()?.matmul(grad_out)?;
        let dp = grad_out.matmul(&v.transpose(2, 3)?.contiguous()?)?;
        let delta = dp
            .broadcast_mul(&probs)?
            .sum_keepdim(candle_core::D::Minus1)?;
        let centered = (dp - delta)?;
        let ds = (probs.broadcast_mul(&centered)? * scale)?;
        let dq = ds.matmul(k)?;
        let dk = ds.transpose(2, 3)?.contiguous()?.matmul(q)?;
        let _ = out;
        Ok((dq, dk, dv))
    }

    fn causal_mask(q: &Tensor, k: &Tensor) -> Result<Tensor> {
        let q_len = q.dims()[2];
        let kv_len = k.dims()[2];
        let shift = kv_len.saturating_sub(q_len);
        let mask = (0..q_len)
            .flat_map(|q_idx| {
                (0..kv_len).map(move |kv_idx| {
                    if kv_idx <= shift + q_idx {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(mask, (1, 1, q_len, kv_len), q.device())?.to_dtype(q.dtype())
    }
}
