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

__device__ inline float block_reduce_sum(float value, float *shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__device__ void rms_norm_kernel(
    T *out,
    const T *x,
    const T *weight,
    int rows,
    int hidden,
    float eps
) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int base = row * hidden;
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = load_value(x, base + col);
        local_sum += value * value;
    }
    const float sum_sq = block_reduce_sum(local_sum, shared);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden) + eps);

    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = load_value(x, base + col) * inv_rms * load_value(weight, col);
        out[base + col] = store_value<T>(value);
    }
}

} // namespace

extern "C" __global__ void aarambh_rms_norm_f32(
    float *out,
    const float *x,
    const float *weight,
    int rows,
    int hidden,
    float eps
) {
    rms_norm_kernel<float>(out, x, weight, rows, hidden, eps);
}

extern "C" __global__ void aarambh_rms_norm_f16(
    half *out,
    const half *x,
    const half *weight,
    int rows,
    int hidden,
    float eps
) {
    rms_norm_kernel<half>(out, x, weight, rows, hidden, eps);
}

extern "C" __global__ void aarambh_rms_norm_bf16(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    int rows,
    int hidden,
    float eps
) {
    rms_norm_kernel<__nv_bfloat16>(out, x, weight, rows, hidden, eps);
}
