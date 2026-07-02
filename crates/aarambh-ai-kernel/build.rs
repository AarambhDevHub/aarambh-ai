use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(aarambh_cuda_kernels)");

    let kernels = [
        (
            "AARAMBH_CUDA_FLASH_ATTN_PTX",
            "flash_attention",
            "kernels/flash_attention.cu",
        ),
        (
            "AARAMBH_CUDA_FLASH_ATTN_BWD_PTX",
            "flash_attn_bwd",
            "kernels/flash_attn_bwd.cu",
        ),
        (
            "AARAMBH_CUDA_RMS_NORM_PTX",
            "rms_norm_fused",
            "kernels/rms_norm_fused.cu",
        ),
        (
            "AARAMBH_CUDA_ROPE_PTX",
            "rope_apply",
            "kernels/rope_apply.cu",
        ),
        (
            "AARAMBH_CUDA_SWIGLU_PTX",
            "swiglu_fused",
            "kernels/swiglu_fused.cu",
        ),
    ];
    for (_, _, kernel) in kernels {
        println!("cargo:rerun-if-changed={kernel}");
    }

    let Ok(nvcc) = which::which("nvcc") else {
        println!("cargo:warning=nvcc not found; CUDA Phase 14 PTX kernels are disabled");
        return;
    };
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let gpu_arch = env::var("AARAMBH_CUDA_ARCH").unwrap_or_else(|_| "compute_75".to_string());

    let mut generated = Vec::with_capacity(kernels.len());
    for (env_name, module_name, kernel) in kernels {
        if Path::new(kernel).exists() {
            let ptx_path = out_dir.join(format!("{module_name}.ptx"));
            let output = Command::new(&nvcc)
                .arg("--ptx")
                .arg("-O3")
                .arg("--use_fast_math")
                .arg(format!("--gpu-architecture={gpu_arch}"))
                .arg("-o")
                .arg(&ptx_path)
                .arg(kernel)
                .output();

            match output {
                Ok(output) if output.status.success() => {
                    generated.push((env_name, ptx_path));
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    println!(
                        "cargo:warning=nvcc failed for {kernel}; CUDA Phase 14 PTX kernels are disabled: {stderr}"
                    );
                    return;
                }
                Err(err) => {
                    println!(
                        "cargo:warning=failed to run nvcc for {kernel}; CUDA Phase 14 PTX kernels are disabled: {err}"
                    );
                    return;
                }
            }
        }
    }

    for (env_name, ptx_path) in generated {
        println!("cargo:rustc-env={env_name}={}", ptx_path.display());
    }
    println!("cargo:rustc-cfg=aarambh_cuda_kernels");
}
