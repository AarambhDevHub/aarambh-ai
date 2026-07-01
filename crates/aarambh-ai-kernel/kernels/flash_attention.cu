#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kBlockSize = 256;

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

__device__ inline float block_reduce_max(float value, float *shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__device__ inline float attention_score(
    const T *q,
    const T *k,
    int q_base,
    int k_base,
    int key_idx,
    int head_dim,
    float scale
) {
    float acc = 0.0f;
    const int key_base = k_base + key_idx * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        acc += load_value(q, q_base + d) * load_value(k, key_base + d);
    }
    return acc * scale;
}

template <typename T>
__device__ void flash_attention_forward_kernel(
    T *out,
    float *lse,
    const T *q,
    const T *k,
    const T *v,
    int rows,
    int heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal
) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int q_idx = row % q_len;
    const int head = (row / q_len) % heads;
    const int batch = row / (heads * q_len);
    const int q_base = ((batch * heads + head) * q_len + q_idx) * head_dim;
    const int kv_base = ((batch * heads + head) * kv_len) * head_dim;

    int max_key = kv_len - 1;
    if (causal != 0) {
        max_key = min(max_key, kv_len - q_len + q_idx);
    }
    if (max_key < 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[q_base + d] = store_value<T>(0.0f);
        }
        if (threadIdx.x == 0 && lse != nullptr) {
            lse[row] = -INFINITY;
        }
        return;
    }

    float local_max = -FLT_MAX;
    for (int key_idx = threadIdx.x; key_idx <= max_key; key_idx += blockDim.x) {
        local_max = fmaxf(
            local_max,
            attention_score(q, k, q_base, kv_base, key_idx, head_dim, scale)
        );
    }
    const float row_max = block_reduce_max(local_max, shared);

    float local_sum = 0.0f;
    for (int key_idx = threadIdx.x; key_idx <= max_key; key_idx += blockDim.x) {
        const float score = attention_score(q, k, q_base, kv_base, key_idx, head_dim, scale);
        local_sum += expf(score - row_max);
    }
    const float denom = block_reduce_sum(local_sum, shared);
    const float inv_denom = 1.0f / denom;

    if (threadIdx.x == 0 && lse != nullptr) {
        lse[row] = row_max + logf(denom);
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx <= max_key; ++key_idx) {
            const float score = attention_score(q, k, q_base, kv_base, key_idx, head_dim, scale);
            const float prob = expf(score - row_max) * inv_denom;
            acc += prob * load_value(v, kv_base + key_idx * head_dim + d);
        }
        out[q_base + d] = store_value<T>(acc);
    }
}

} // namespace

extern "C" __global__ void aarambh_flash_attention_f32(
    float *out,
    float *lse,
    const float *q,
    const float *k,
    const float *v,
    int rows,
    int heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal
) {
    flash_attention_forward_kernel<float>(
        out, lse, q, k, v, rows, heads, q_len, kv_len, head_dim, scale, causal
    );
}

extern "C" __global__ void aarambh_flash_attention_f16(
    half *out,
    float *lse,
    const half *q,
    const half *k,
    const half *v,
    int rows,
    int heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal
) {
    flash_attention_forward_kernel<half>(
        out, lse, q, k, v, rows, heads, q_len, kv_len, head_dim, scale, causal
    );
}

extern "C" __global__ void aarambh_flash_attention_bf16(
    __nv_bfloat16 *out,
    float *lse,
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    int rows,
    int heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    int causal
) {
    flash_attention_forward_kernel<__nv_bfloat16>(
        out, lse, q, k, v, rows, heads, q_len, kv_len, head_dim, scale, causal
    );
}
