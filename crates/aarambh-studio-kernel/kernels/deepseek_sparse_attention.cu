#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
__device__ __forceinline__ float dsa_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float dsa_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float dsa_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T dsa_from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half dsa_from_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 dsa_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

extern "C" __global__ void aarambh_dsa_topk_f32(
    const float* scores,
    unsigned int* selected,
    int rows,
    int blocks,
    int top_k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int current = row % blocks;
    unsigned int* output = selected + row * top_k;
    for (int slot = 0; slot < top_k; ++slot) output[slot] = 0xffffffffu;
    const int ranked = min(top_k - 1, current);
    for (int block = 0; block < current; ++block) {
        const float value = scores[row * blocks + block];
        int insert_at = ranked;
        for (int slot = 0; slot < ranked; ++slot) {
            const unsigned int existing = output[slot];
            const float existing_value = existing == 0xffffffffu
                ? -CUDART_INF_F
                : scores[row * blocks + existing];
            if (value > existing_value || (value == existing_value && block < existing)) {
                insert_at = slot;
                break;
            }
        }
        if (insert_at < ranked) {
            for (int slot = ranked - 1; slot > insert_at; --slot) output[slot] = output[slot - 1];
            output[insert_at] = static_cast<unsigned int>(block);
        }
    }
    output[top_k - 1] = static_cast<unsigned int>(current);
}

template <typename T>
__device__ void dsa_sparse_forward(
    const T* q,
    const T* k,
    const T* v,
    const unsigned int* selected,
    T* output,
    int batch,
    int heads,
    int q_len,
    int kv_len,
    int head_dim,
    int selected_per_query,
    int block_size,
    float scale) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_count = batch * heads * q_len;
    if (element >= row_count * head_dim) return;
    const int dim = element % head_dim;
    const int row = element / head_dim;
    const int query = row % q_len;
    const int head = (row / q_len) % heads;
    const int batch_index = row / (heads * q_len);
    const int selection_row = batch_index * q_len + query;
    const int q_offset = row * head_dim;
    float running_max = -CUDART_INF_F;
    float running_sum = 0.0f;
    float accumulator = 0.0f;
    for (int slot = 0; slot < selected_per_query; ++slot) {
        const unsigned int selected_block = selected[selection_row * selected_per_query + slot];
        if (selected_block == 0xffffffffu) continue;
        const int start = selected_block * block_size;
        const int end = min(start + block_size, kv_len);
        for (int key_index = start; key_index < end; ++key_index) {
            if (key_index > kv_len - q_len + query) continue;
            float dot = 0.0f;
            const int key_offset = ((batch_index * heads + head) * kv_len + key_index) * head_dim;
            for (int index_dim = 0; index_dim < head_dim; ++index_dim) {
                dot += dsa_to_float(q[q_offset + index_dim])
                    * dsa_to_float(k[key_offset + index_dim]);
            }
            const float score = dot * scale;
            const float next_max = fmaxf(running_max, score);
            const float old_scale = __expf(running_max - next_max);
            const float new_scale = __expf(score - next_max);
            accumulator = accumulator * old_scale
                + new_scale * dsa_to_float(v[key_offset + dim]);
            running_sum = running_sum * old_scale + new_scale;
            running_max = next_max;
        }
    }
    output[element] = dsa_from_float<T>(running_sum > 0.0f ? accumulator / running_sum : 0.0f);
}

#define DEFINE_DSA_FORWARD(NAME, TYPE) \
extern "C" __global__ void NAME( \
    const TYPE* q, const TYPE* k, const TYPE* v, const unsigned int* selected, TYPE* output, \
    int batch, int heads, int q_len, int kv_len, int head_dim, int selected_per_query, \
    int block_size, float scale) { \
    dsa_sparse_forward(q, k, v, selected, output, batch, heads, q_len, kv_len, head_dim, \
        selected_per_query, block_size, scale); \
}

DEFINE_DSA_FORWARD(aarambh_dsa_sparse_f32, float)
DEFINE_DSA_FORWARD(aarambh_dsa_sparse_f16, __half)
DEFINE_DSA_FORWARD(aarambh_dsa_sparse_bf16, __nv_bfloat16)

extern "C" __global__ void aarambh_dsa_teacher_mass_f32(
    const float* attention_probabilities,
    float* block_mass,
    int rows,
    int kv_len,
    int block_size) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks = (kv_len + block_size - 1) / block_size;
    if (element >= rows * blocks) return;
    const int row = element / blocks;
    const int block = element % blocks;
    const int start = block * block_size;
    const int end = min(start + block_size, kv_len);
    float sum = 0.0f;
    for (int token = start; token < end; ++token) {
        sum += attention_probabilities[row * kv_len + token];
    }
    block_mass[element] = sum;
}
