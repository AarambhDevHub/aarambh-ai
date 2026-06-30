#include <float.h>
#include <math.h>

namespace {

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

__device__ inline float attention_score(
    const float *q,
    const float *k,
    int q_base,
    int k_base,
    int key_idx,
    int head_dim,
    float scale
) {
    float acc = 0.0f;
    const int key_base = k_base + key_idx * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        acc += q[q_base + d] * k[key_base + d];
    }
    return acc * scale;
}

} // namespace

extern "C" __global__ void aarambh_flash_attention_bwd_f32(
    float *dq,
    float *dk,
    float *dv,
    const float *dout,
    const float *out,
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
            dq[q_base + d] = 0.0f;
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

    float local_delta = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        local_delta += dout[q_base + d] * out[q_base + d];
    }
    const float delta = block_reduce_sum(local_delta, shared);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float dq_acc = 0.0f;
        const float dout_d = dout[q_base + d];
        const float q_d = q[q_base + d];

        for (int key_idx = 0; key_idx <= max_key; ++key_idx) {
            const int key_base = kv_base + key_idx * head_dim;
            const float score = attention_score(q, k, q_base, kv_base, key_idx, head_dim, scale);
            const float prob = expf(score - row_max) * inv_denom;

            float dp = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dp += dout[q_base + dd] * v[key_base + dd];
            }
            const float ds = prob * (dp - delta) * scale;
            dq_acc += ds * k[key_base + d];
            atomicAdd(&dk[key_base + d], ds * q_d);
            atomicAdd(&dv[key_base + d], prob * dout_d);
        }

        dq[q_base + d] = dq_acc;
    }
}
