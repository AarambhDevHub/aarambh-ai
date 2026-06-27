namespace {

__global__ void aarambh_swiglu_noop_kernel(float *out, const float *gate, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && out != nullptr && gate != nullptr) {
        out[idx] = gate[idx];
    }
}

} // namespace

extern "C" int aarambh_swiglu_fused_stub() {
    // Phase 4 validates CUDA build plumbing for fused SwiGLU.
    return 0;
}
