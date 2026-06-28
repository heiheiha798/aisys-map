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

__global__ void fused_residual_rmsnorm_kernel(const float* x,
                                              const float* residual,
                                              const float* gamma, float* y,
                                              int rows, int cols, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  extern __shared__ float s_sq_sum[];

  float local_sq_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    int idx = row * cols + col;
    float fused = x[idx] + residual[idx];
    local_sq_sum += fused * fused;
  }

  s_sq_sum[tid] = local_sq_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sq_sum[tid] += s_sq_sum[tid + stride];
    }
    __syncthreads();
  }

  float mean_sq = s_sq_sum[0] / static_cast<float>(cols);
  float inv_rms = rsqrtf(mean_sq + eps);

  for (int col = tid; col < cols; col += blockDim.x) {
    int idx = row * cols + col;
    float fused = x[idx] + residual[idx];
    y[idx] = gamma[col] * fused * inv_rms;
  }
}

void cpu_fused_residual_rmsnorm(const std::vector<float>& x,
                                const std::vector<float>& residual,
                                const std::vector<float>& gamma,
                                std::vector<float>& y, int rows, int cols,
                                float eps) {
  for (int r = 0; r < rows; ++r) {
    float sq_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r) * cols + c;
      float fused = x[idx] + residual[idx];
      sq_sum += fused * fused;
    }

    float mean_sq = sq_sum / static_cast<float>(cols);
    float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r) * cols + c;
      float fused = x[idx] + residual[idx];
      y[idx] = gamma[c] * fused * inv_rms;
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
  const size_t matrix_bytes = numel * sizeof(float);
  const size_t gamma_bytes = cols * sizeof(float);
  const size_t shared_mem_bytes = threads_per_block * sizeof(float);

  std::vector<float> h_x(numel);
  std::vector<float> h_residual(numel);
  std::vector<float> h_gamma(cols);
  std::vector<float> h_y(numel, 0.0f);
  std::vector<float> h_ref(numel, 0.0f);

  for (int c = 0; c < cols; ++c) {
    float periodic = std::sin((c + 1) * 0.015f);
    float offset = static_cast<float>((c % 17) - 8) * 0.01f;
    h_gamma[c] = 1.0f + 0.1f * periodic + offset;
  }

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r) * cols + c;
      float base = std::sin((r + 3) * 0.013f) + std::cos((c + 5) * 0.021f);
      float noise = static_cast<float>(((r * 11 + c * 7) % 31) - 15) * 0.03f;
      float residual_base =
          std::cos((r + 2) * 0.017f) - std::sin((c + 9) * 0.019f);
      float residual_noise =
          static_cast<float>(((r * 5 + c * 13) % 19) - 9) * 0.02f;
      h_x[idx] = 0.7f * base + noise;
      h_residual[idx] = 0.4f * residual_base + residual_noise;
    }
  }

  float* d_x = nullptr;
  float* d_residual = nullptr;
  float* d_gamma = nullptr;
  float* d_y = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, matrix_bytes));
  CUDA_CHECK(cudaMalloc(&d_residual, matrix_bytes));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma_bytes));
  CUDA_CHECK(cudaMalloc(&d_y, matrix_bytes));

  CUDA_CHECK(
      cudaMemcpy(d_x, h_x.data(), matrix_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), matrix_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), gamma_bytes,
                        cudaMemcpyHostToDevice));

  fused_residual_rmsnorm_kernel<<<rows, threads_per_block, shared_mem_bytes>>>(
      d_x, d_residual, d_gamma, d_y, rows, cols, kEps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_y.data(), d_y, matrix_bytes, cudaMemcpyDeviceToHost));
  cpu_fused_residual_rmsnorm(h_x, h_residual, h_gamma, h_ref, rows, cols,
                             kEps);

  float max_abs = max_abs_diff(h_y, h_ref);
  bool ok = max_abs < 2e-4f;

  if (ok) {
    std::cout << "fused_residual_rmsnorm passed. rows=" << rows
              << ", cols=" << cols
              << ", threads_per_block=" << threads_per_block
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: y[0]=" << h_y[0] << ", y[1]=" << h_y[1]
              << ", y[255]=" << h_y[255] << std::endl;
  } else {
    std::cerr << "fused_residual_rmsnorm failed. max_abs_diff=" << max_abs
              << std::endl;
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
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_y));
  return ok ? 0 : 1;
}
