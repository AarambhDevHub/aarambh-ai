#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

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
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const float gate_value = load_value(gate, idx);
    const float up_value = load_value(up, idx);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    out[idx] = store_value<T>(silu * up_value);
}

} // namespace

extern "C" __global__ void aarambh_swiglu_f32(
    float *out,
    const float *gate,
    const float *up,
    int total
) {
    swiglu_kernel<float>(out, gate, up, total);
}

extern "C" __global__ void aarambh_swiglu_f16(
    half *out,
    const half *gate,
    const half *up,
    int total
) {
    swiglu_kernel<half>(out, gate, up, total);
}

extern "C" __global__ void aarambh_swiglu_bf16(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    int total
) {
    swiglu_kernel<__nv_bfloat16>(out, gate, up, total);
}
