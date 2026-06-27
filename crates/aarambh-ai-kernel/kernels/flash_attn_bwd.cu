namespace {

__global__ void aarambh_flash_attention_backward_noop_kernel(
    float *dq,
    const float *dout,
    int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && dq != nullptr && dout != nullptr) {
        dq[idx] = dout[idx];
    }
}

} // namespace

extern "C" int aarambh_flash_attention_backward_stub() {
    // Phase 4 keeps backward CUDA code as a compile/link placeholder.
    return 0;
}
