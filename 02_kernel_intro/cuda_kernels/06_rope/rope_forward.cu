#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(err) << std::endl;                      \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

constexpr float kBase = 10000.0f;

__global__ void rope_forward_kernel(const float* x, float* y, int seq_len,
                                    int num_heads, int head_dim, float base) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int total_rows = seq_len * num_heads;

  if (row >= total_rows) {
    return;
  }

  int token_idx = row / num_heads;
  int pair_dim = head_dim / 2;
  const float* src = x + static_cast<size_t>(row) * head_dim;
  float* dst = y + static_cast<size_t>(row) * head_dim;

  for (int pair_idx = tid; pair_idx < pair_dim; pair_idx += blockDim.x) {
    int even_col = 2 * pair_idx;
    int odd_col = even_col + 1;

    float x0 = src[even_col];
    float x1 = src[odd_col];

    float exponent = (2.0f * static_cast<float>(pair_idx)) /
                     static_cast<float>(head_dim);
    float theta = static_cast<float>(token_idx) / powf(base, exponent);
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    dst[even_col] = x0 * cos_theta - x1 * sin_theta;
    dst[odd_col] = x0 * sin_theta + x1 * cos_theta;
  }
}

void cpu_rope_forward(const std::vector<float>& x, std::vector<float>& y,
                      int seq_len, int num_heads, int head_dim, float base) {
  int pair_dim = head_dim / 2;
  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      size_t row = static_cast<size_t>(token_idx) * num_heads + head_idx;
      const float* src = x.data() + row * head_dim;
      float* dst = y.data() + row * head_dim;

      for (int pair_idx = 0; pair_idx < pair_dim; ++pair_idx) {
        int even_col = 2 * pair_idx;
        int odd_col = even_col + 1;

        float x0 = src[even_col];
        float x1 = src[odd_col];

        float exponent = (2.0f * static_cast<float>(pair_idx)) /
                         static_cast<float>(head_dim);
        float theta = static_cast<float>(token_idx) / std::pow(base, exponent);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);

        dst[even_col] = x0 * cos_theta - x1 * sin_theta;
        dst[odd_col] = x0 * sin_theta + x1 * cos_theta;
      }
    }
  }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float max_abs = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
  }
  return max_abs;
}

}  // namespace

int main() {
  constexpr int seq_len = 128;
  constexpr int num_heads = 8;
  constexpr int head_dim = 64;
  constexpr int threads_per_block = 128;

  static_assert(head_dim % 2 == 0, "RoPE requires even head_dim.");

  const size_t numel =
      static_cast<size_t>(seq_len) * num_heads * head_dim;
  const size_t bytes = numel * sizeof(float);
  const int total_rows = seq_len * num_heads;

  std::vector<float> h_x(numel);
  std::vector<float> h_y(numel, 0.0f);
  std::vector<float> h_ref(numel, 0.0f);

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        size_t offset =
            (static_cast<size_t>(token_idx) * num_heads + head_idx) * head_dim +
            dim_idx;
        float a = std::sin((token_idx + 1) * 0.11f);
        float b = std::cos((head_idx + 2) * 0.07f);
        float c = static_cast<float>(((dim_idx * 13 + token_idx * 5 +
                                       head_idx * 3) %
                                      29) -
                                     14) *
                  0.04f;
        h_x[offset] = 0.5f * a + 0.35f * b + c;
      }
    }
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  rope_forward_kernel<<<total_rows, threads_per_block>>>(d_x, d_y, seq_len,
                                                         num_heads, head_dim,
                                                         kBase);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  cpu_rope_forward(h_x, h_ref, seq_len, num_heads, head_dim, kBase);

  float max_abs = max_abs_diff(h_y, h_ref);
  bool ok = max_abs < 2e-5f;

  if (ok) {
    size_t sample_row = static_cast<size_t>(3) * num_heads + 1;
    size_t sample_base = sample_row * head_dim;
    std::cout << "rope_forward passed. seq_len=" << seq_len
              << ", num_heads=" << num_heads << ", head_dim=" << head_dim
              << ", threads_per_block=" << threads_per_block
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: y[0]=" << h_y[0]
              << ", y[1]=" << h_y[1]
              << ", y[sample_even]=" << h_y[sample_base]
              << ", y[sample_odd]=" << h_y[sample_base + 1] << std::endl;
  } else {
    std::cerr << "rope_forward failed. max_abs_diff=" << max_abs << std::endl;
    for (size_t i = 0; i < h_y.size(); ++i) {
      float abs_diff = std::fabs(h_y[i] - h_ref[i]);
      if (abs_diff == max_abs) {
        std::cerr << "first max diff at index " << i << ": got=" << h_y[i]
                  << ", ref=" << h_ref[i] << std::endl;
        break;
      }
    }
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return ok ? 0 : 1;
}
