use candle_core::{Error, Result};

#[cfg(aarambh_cuda_stubs)]
unsafe extern "C" {
    fn aarambh_swiglu_fused_stub() -> i32;
}

pub fn cuda_stub_compiled() -> bool {
    cfg!(aarambh_cuda_stubs)
}

pub fn fused_swiglu_stub() -> Result<()> {
    Err(Error::msg(
        "Fused CUDA SwiGLU is prepared as a Phase 4 stub and is implemented in Phase 14",
    ))
}

#[cfg(aarambh_cuda_stubs)]
pub unsafe fn touch_cuda_stub_symbol() {
    // SAFETY: This no-op symbol is compiled from a Phase 4 CUDA stub file.
    unsafe {
        let _ = aarambh_swiglu_fused_stub();
    }
}
