namespace {

__global__ void aarambh_rope_noop_kernel(float *q_out, const float *q_in, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && q_out != nullptr && q_in != nullptr) {
        q_out[idx] = q_in[idx];
    }
}

} // namespace

extern "C" int aarambh_rope_apply_stub() {
    // Phase 4 validates CUDA build plumbing for fused RoPE.
    return 0;
}
