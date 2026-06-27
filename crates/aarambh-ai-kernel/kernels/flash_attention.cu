namespace {

__global__ void aarambh_flash_attention_noop_kernel(float *out, const float *q, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && out != nullptr && q != nullptr) {
        out[idx] = q[idx];
    }
}

} // namespace

extern "C" int aarambh_flash_attention_stub() {
    // Phase 4 validates CUDA compilation/linking only. Runtime launch and
    // numerically-correct Flash Attention arrive in Phase 14.
    return 0;
}
