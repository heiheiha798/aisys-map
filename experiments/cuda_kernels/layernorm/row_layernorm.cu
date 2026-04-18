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

constexpr float kEps = 1e-5f;

__global__ void row_layernorm_kernel(const float* x, float* y, int rows,
                                     int cols, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];
  float* s_sum = shared;
  float* s_sq_sum = shared + blockDim.x;

  float local_sum = 0.0f;
  float local_sq_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    float v = x[row * cols + col];
    local_sum += v;
    local_sq_sum += v * v;
  }

  s_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  float row_sum = s_sum[0];

  s_sq_sum[tid] = local_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sq_sum[tid] += s_sq_sum[tid + stride];
    }
    __syncthreads();
  }

  float row_sq_sum = s_sq_sum[0];
  float mean = row_sum / static_cast<float>(cols);
  float mean_sq = row_sq_sum / static_cast<float>(cols);
  float var = fmaxf(0.0f, mean_sq - mean * mean);
  float inv_std = rsqrtf(var + eps);

  for (int col = tid; col < cols; col += blockDim.x) {
    float v = x[row * cols + col];
    y[row * cols + col] = (v - mean) * inv_std;
  }
}

void cpu_row_layernorm(const std::vector<float>& x, std::vector<float>& y,
                       int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float v = x[static_cast<size_t>(r) * cols + c];
      sum += v;
      sq_sum += v * v;
    }

    float mean = sum / static_cast<float>(cols);
    float mean_sq = sq_sum / static_cast<float>(cols);
    float var = std::max(0.0f, mean_sq - mean * mean);
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int c = 0; c < cols; ++c) {
      float v = x[static_cast<size_t>(r) * cols + c];
      y[static_cast<size_t>(r) * cols + c] = (v - mean) * inv_std;
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
  constexpr int rows = 1024;
  constexpr int cols = 256;
  constexpr int threads_per_block = 256;

  const size_t numel = static_cast<size_t>(rows) * cols;
  const size_t bytes = numel * sizeof(float);
  const size_t shared_mem_bytes = 2 * threads_per_block * sizeof(float);

  std::vector<float> h_x(numel);
  std::vector<float> h_y(numel, 0.0f);
  std::vector<float> h_ref(numel, 0.0f);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float x = std::sin((r + 1) * 0.013f) + std::cos((c + 3) * 0.021f);
      float y = static_cast<float>(((r * 11 + c * 7) % 23) - 11) * 0.05f;
      h_x[static_cast<size_t>(r) * cols + c] = 0.7f * x + y;
    }
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  row_layernorm_kernel<<<rows, threads_per_block, shared_mem_bytes>>>(
      d_x, d_y, rows, cols, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
  cpu_row_layernorm(h_x, h_ref, rows, cols, kEps);

  float max_abs = max_abs_diff(h_y, h_ref);
  bool ok = max_abs < 2e-4f;

  if (ok) {
    std::cout << "row_layernorm passed. rows=" << rows << ", cols=" << cols
              << ", threads_per_block=" << threads_per_block
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: y[0]=" << h_y[0] << ", y[1]=" << h_y[1]
              << ", y[255]=" << h_y[255] << std::endl;
  } else {
    std::cerr << "row_layernorm failed. max_abs_diff=" << max_abs << std::endl;
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
