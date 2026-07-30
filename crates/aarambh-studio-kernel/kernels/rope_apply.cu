#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T>
__device__ inline float load_value(const T *ptr, int idx);

template <>
__device__ inline float load_value<float>(const float *ptr, int idx) {
    return ptr[idx];
}

template <>
__device__ inline float load_value<half>(const half *ptr, int idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ inline float load_value<__nv_bfloat16>(const __nv_bfloat16 *ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ inline T store_value(float value);

template <>
__device__ inline float store_value<float>(float value) {
    return value;
}

template <>
__device__ inline half store_value<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 store_value<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
__device__ void rope_apply_kernel(
    T *out,
    const T *x,
    const T *cos,
    const T *sin,
    int total,
    int seq_len,
    int heads,
    int head_dim,
    int seqlen_offset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int half_dim = head_dim / 2;
    const int d = idx % head_dim;
    const int token = (idx / (heads * head_dim)) % seq_len;
    const int base = idx - d;
    const int pair = d < half_dim ? d : d - half_dim;
    const int x1_idx = base + pair;
    const int x2_idx = base + half_dim + pair;
    const int trig_idx = (seqlen_offset + token) * half_dim + pair;

    const float x1 = load_value(x, x1_idx);
    const float x2 = load_value(x, x2_idx);
    const float c = load_value(cos, trig_idx);
    const float s = load_value(sin, trig_idx);
    const float value = d < half_dim ? x1 * c - x2 * s : x1 * s + x2 * c;
    out[idx] = store_value<T>(value);
}

} // namespace

extern "C" __global__ void aarambh_rope_apply_f32(
    float *out,
    const float *x,
    const float *cos,
    const float *sin,
    int total,
    int seq_len,
    int heads,
    int head_dim,
    int seqlen_offset
) {
    rope_apply_kernel<float>(out, x, cos, sin, total, seq_len, heads, head_dim, seqlen_offset);
}

extern "C" __global__ void aarambh_rope_apply_f16(
    half *out,
    const half *x,
    const half *cos,
    const half *sin,
    int total,
    int seq_len,
    int heads,
    int head_dim,
    int seqlen_offset
) {
    rope_apply_kernel<half>(out, x, cos, sin, total, seq_len, heads, head_dim, seqlen_offset);
}

extern "C" __global__ void aarambh_rope_apply_bf16(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *cos,
    const __nv_bfloat16 *sin,
    int total,
    int seq_len,
    int heads,
    int head_dim,
    int seqlen_offset
) {
    rope_apply_kernel<__nv_bfloat16>(
        out, x, cos, sin, total, seq_len, heads, head_dim, seqlen_offset
    );
}
