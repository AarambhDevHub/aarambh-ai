namespace {

__global__ void aarambh_rms_norm_noop_kernel(float *out, const float *x, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && out != nullptr && x != nullptr) {
        out[idx] = x[idx];
    }
}

} // namespace

extern "C" int aarambh_rms_norm_fused_stub() {
    // Phase 4 validates the fused RMSNorm CUDA translation unit only.
    return 0;
}
