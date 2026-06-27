use std::path::Path;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(aarambh_cuda_stubs)");

    let kernels = [
        "kernels/flash_attention.cu",
        "kernels/flash_attn_bwd.cu",
        "kernels/rms_norm_fused.cu",
        "kernels/rope_apply.cu",
        "kernels/swiglu_fused.cu",
    ];
    for kernel in kernels {
        println!("cargo:rerun-if-changed={kernel}");
    }

    if which::which("nvcc").is_err() {
        println!("cargo:warning=nvcc not found; CUDA kernel stubs are disabled");
        return;
    }

    let mut build = cc::Build::new();
    build.cuda(true);
    for kernel in kernels {
        if Path::new(kernel).exists() {
            build.file(kernel);
        }
    }
    build.compile("aarambh_cuda_stubs");
    println!("cargo:rustc-cfg=aarambh_cuda_stubs");
}
