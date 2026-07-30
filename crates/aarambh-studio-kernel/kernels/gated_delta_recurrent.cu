#include <cuda_runtime.h>

extern "C" __global__ void aarambh_gated_delta_recurrent_f32(
    float* output,
    const float* packed,
    const float* previous,
    int rows,
    int key_dim,
    int value_dim) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;

  const int packed_width = key_dim * 2 + value_dim + 2;
  const int state_width = key_dim * value_dim;
  const int output_width = state_width + value_dim;
  const float* input = packed + row * packed_width;
  const float* old_state = previous + row * state_width;
  float* next_state = output + row * output_width;
  float* mixed = next_state + state_width;
  const float* query = input;
  const float* key = input + key_dim;
  const float* value = input + key_dim * 2;
  const float alpha = input[key_dim * 2 + value_dim];
  const float beta = input[key_dim * 2 + value_dim + 1];

  for (int value_idx = static_cast<int>(threadIdx.x); value_idx < value_dim;
       value_idx += static_cast<int>(blockDim.x)) {
    float prediction = 0.0f;
    for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
      prediction = fmaf(key[key_idx],
                        old_state[key_idx * value_dim + value_idx] * alpha,
                        prediction);
    }
    const float error = value[value_idx] - prediction;
    float result = 0.0f;
    for (int key_idx = 0; key_idx < key_dim; ++key_idx) {
      const int state_idx = key_idx * value_dim + value_idx;
      const float updated = old_state[state_idx] * alpha +
                            beta * key[key_idx] * error;
      next_state[state_idx] = updated;
      result = fmaf(query[key_idx], updated, result);
    }
    mixed[value_idx] = result;
  }
}
