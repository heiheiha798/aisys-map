#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
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

namespace gemm {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREADS = THREADS_X * THREADS_Y;
constexpr int TM = 4;
constexpr int TN = 4;

struct DiffStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
};

__global__ void tiled_gemm_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  float* c, int m, int n, int k) {
  __shared__ float a_tile[BLOCK_M][BLOCK_K];
  __shared__ float b_tile[BLOCK_K][BLOCK_N];

  const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_row = blockIdx.y * BLOCK_M;
  const int block_col = blockIdx.x * BLOCK_N;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int k0 = 0; k0 < k; k0 += BLOCK_K) {
    for (int idx = linear_tid; idx < BLOCK_M * BLOCK_K; idx += THREADS) {
      int tile_row = idx / BLOCK_K;
      int tile_col = idx % BLOCK_K;
      int global_row = block_row + tile_row;
      int global_col = k0 + tile_col;

      float val = 0.0f;
      if (global_row < m && global_col < k) {
        val = __bfloat162float(a[global_row * k + global_col]);
      }
      a_tile[tile_row][tile_col] = val;
    }

    for (int idx = linear_tid; idx < BLOCK_K * BLOCK_N; idx += THREADS) {
      int tile_row = idx / BLOCK_N;
      int tile_col = idx % BLOCK_N;
      int global_row = k0 + tile_row;
      int global_col = block_col + tile_col;

      float val = 0.0f;
      if (global_row < k && global_col < n) {
        val = __bfloat162float(b[global_row * n + global_col]);
      }
      b_tile[tile_row][tile_col] = val;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      float a_frag[TM];
      float b_frag[TN];

#pragma unroll
      for (int i = 0; i < TM; ++i) {
        a_frag[i] = a_tile[threadIdx.y * TM + i][kk];
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        b_frag[j] = b_tile[kk][threadIdx.x * TN + j];
      }

#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }

  const int row_base = block_row + threadIdx.y * TM;
  const int col_base = block_col + threadIdx.x * TN;

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int row = row_base + i;
    if (row >= m) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int col = col_base + j;
      if (col < n) {
        c[row * n + col] = acc[i][j];
      }
    }
  }
}

inline void fill_input(std::vector<float>& mat, int rows, int cols, int seed) {
  mat.resize(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float x = std::sin((r + 1) * 0.031f * (seed + 1)) +
                std::cos((c + 3) * 0.047f * (seed + 2));
      float y = static_cast<float>(((r * 17 + c * 13 + seed) % 19) - 9) * 0.08f;
      mat[static_cast<size_t>(r) * cols + c] = 0.6f * x + y;
    }
  }
}

inline std::vector<float> cpu_gemm(const std::vector<float>& a,
                                   const std::vector<float>& b, int m, int n,
                                   int k) {
  std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
      float a_val = a[static_cast<size_t>(i) * k + kk];
      for (int j = 0; j < n; ++j) {
        c[static_cast<size_t>(i) * n + j] +=
            a_val * b[static_cast<size_t>(kk) * n + j];
      }
    }
  }
  return c;
}

inline DiffStats compare_outputs(const std::vector<float>& got,
                                 const std::vector<float>& ref) {
  DiffStats stats;
  for (size_t i = 0; i < got.size(); ++i) {
    float abs_err = std::fabs(got[i] - ref[i]);
    float rel_err = abs_err / std::max(1e-6f, std::fabs(ref[i]));
    stats.max_abs = std::max(stats.max_abs, abs_err);
    stats.max_rel = std::max(stats.max_rel, rel_err);
  }
  return stats;
}

inline void convert_from_fp32(const std::vector<float>& src,
                              std::vector<__nv_bfloat16>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = __nv_bfloat16(src[i]);
  }
}

inline std::vector<float> run_once(const std::vector<__nv_bfloat16>& h_a,
                                   const std::vector<__nv_bfloat16>& h_b, int m,
                                   int n, int k) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return h_c;
}

inline float benchmark(const std::vector<__nv_bfloat16>& h_a,
                       const std::vector<__nv_bfloat16>& h_b, int m, int n,
                       int k, int warmup_iters = 5, int iters = 30) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  for (int i = 0; i < warmup_iters; ++i) {
    tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i) {
    tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return total_ms / static_cast<float>(iters);
}

inline double tflops_from_ms(int m, int n, int k, float ms) {
  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                 static_cast<double>(k);
  return flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

inline int run_experiment() {
  const bool profile_mode = (std::getenv("GEMM_PROFILE_ONCE") != nullptr);
  constexpr int check_m = 256;
  constexpr int check_n = 256;
  constexpr int check_k = 256;
  constexpr int bench_m = 1024;
  constexpr int bench_n = 1024;
  constexpr int bench_k = 1024;

  std::vector<float> a_fp32;
  std::vector<float> b_fp32;
  fill_input(a_fp32, check_m, check_k, 3);
  fill_input(b_fp32, check_k, check_n, 11);

  std::vector<__nv_bfloat16> a_bf16;
  std::vector<__nv_bfloat16> b_bf16;
  convert_from_fp32(a_fp32, a_bf16);
  convert_from_fp32(b_fp32, b_bf16);

  auto got = run_once(a_bf16, b_bf16, check_m, check_n, check_k);
  auto ref = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  DiffStats stats = compare_outputs(got, ref);
  bool ok = stats.max_abs < 0.2f;

  float avg_ms = 0.0f;
  std::vector<float> a_bench_fp32;
  std::vector<float> b_bench_fp32;
  fill_input(a_bench_fp32, bench_m, bench_k, 7);
  fill_input(b_bench_fp32, bench_k, bench_n, 19);

  std::vector<__nv_bfloat16> a_bench_bf16;
  std::vector<__nv_bfloat16> b_bench_bf16;
  convert_from_fp32(a_bench_fp32, a_bench_bf16);
  convert_from_fp32(b_bench_fp32, b_bench_bf16);

  if (profile_mode) {
    auto bench_out = run_once(a_bench_bf16, b_bench_bf16, bench_m, bench_n, bench_k);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    avg_ms = benchmark(a_bench_bf16, b_bench_bf16, bench_m, bench_n, bench_k);
  }

  std::cout << "bf16_cuda_core " << (ok ? "passed" : "failed")
            << ". max_abs_vs_fp32_ref=" << stats.max_abs
            << ", max_rel_vs_fp32_ref=" << stats.max_rel
            << ", avg_ms=" << avg_ms;
  if (!profile_mode) {
    std::cout << ", tflops=" << tflops_from_ms(bench_m, bench_n, bench_k, avg_ms);
  }
  std::cout << std::endl;

  return ok ? 0 : 1;
}

}  // namespace gemm

int main() { return gemm::run_experiment(); }
