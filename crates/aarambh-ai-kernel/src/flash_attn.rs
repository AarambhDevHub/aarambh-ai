use candle_core::{Error, Result};

#[cfg(aarambh_cuda_stubs)]
unsafe extern "C" {
    fn aarambh_flash_attention_stub() -> i32;
    fn aarambh_flash_attention_backward_stub() -> i32;
}

pub fn cuda_stub_compiled() -> bool {
    cfg!(aarambh_cuda_stubs)
}

pub fn flash_attention_forward_stub() -> Result<()> {
    Err(Error::msg(
        "Flash Attention CUDA execution is prepared as a Phase 4 stub and is implemented in Phase 14",
    ))
}

pub fn flash_attention_backward_stub() -> Result<()> {
    Err(Error::msg(
        "Flash Attention backward CUDA execution is prepared as a Phase 4 stub and is implemented in Phase 14",
    ))
}

#[cfg(aarambh_cuda_stubs)]
pub unsafe fn touch_cuda_stub_symbols() {
    // SAFETY: These no-op symbols are compiled from Phase 4 CUDA stub files.
    unsafe {
        let _ = aarambh_flash_attention_stub();
        let _ = aarambh_flash_attention_backward_stub();
    }
}
