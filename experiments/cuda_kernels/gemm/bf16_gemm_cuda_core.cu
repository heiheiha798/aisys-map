#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
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

template <typename T>
__host__ __device__ inline float storage_to_float(T x) {
  return static_cast<float>(x);
}

template <>
__host__ __device__ inline float storage_to_float<float>(float x) {
  return x;
}

template <>
__host__ __device__ inline float storage_to_float<int8_t>(int8_t x) {
  return static_cast<float>(x);
}

__host__ __device__ inline int unpack_int4(uint8_t packed, bool high_nibble) {
  int nibble = high_nibble ? ((packed >> 4) & 0xF) : (packed & 0xF);
  return (nibble >= 8) ? (nibble - 16) : nibble;
}

template <typename StorageT>
__global__ void tiled_gemm_kernel(const StorageT* a, const StorageT* b, float* c,
                                  int m, int n, int k, float scale_a,
                                  float scale_b) {
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
        val = storage_to_float(a[global_row * k + global_col]) * scale_a;
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
        val = storage_to_float(b[global_row * n + global_col]) * scale_b;
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

__global__ void tiled_gemm_int4_kernel(const uint8_t* a, const uint8_t* b,
                                       float* c, int m, int n, int k,
                                       float scale_a, float scale_b) {
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
        int linear_idx = global_row * k + global_col;
        uint8_t packed = a[linear_idx >> 1];
        val = static_cast<float>(unpack_int4(packed, linear_idx & 1)) * scale_a;
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
        int linear_idx = global_row * n + global_col;
        uint8_t packed = b[linear_idx >> 1];
        val = static_cast<float>(unpack_int4(packed, linear_idx & 1)) * scale_b;
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
  mat.resize(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float x = std::sin((r + 1) * 0.031f * (seed + 1)) +
                std::cos((c + 3) * 0.047f * (seed + 2));
      float y = static_cast<float>(((r * 17 + c * 13 + seed) % 19) - 9) * 0.08f;
      mat[r * cols + c] = 0.6f * x + y;
    }
  }
}

inline std::vector<float> cpu_gemm(const std::vector<float>& a,
                                   const std::vector<float>& b, int m, int n,
                                   int k) {
  std::vector<float> c(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
      float a_val = a[i * k + kk];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += a_val * b[kk * n + j];
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

template <typename StorageT>
inline void prepare_storage(const std::vector<float>& src,
                            std::vector<StorageT>& packed,
                            std::vector<float>& dequantized, float& scale) {
  scale = 1.0f;
  packed.resize(src.size());
  dequantized.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    packed[i] = StorageT(src[i]);
    dequantized[i] = storage_to_float(packed[i]);
  }
}

template <>
inline void prepare_storage<int8_t>(const std::vector<float>& src,
                                    std::vector<int8_t>& packed,
                                    std::vector<float>& dequantized,
                                    float& scale) {
  float max_abs = 0.0f;
  for (float v : src) {
    max_abs = std::max(max_abs, std::fabs(v));
  }
  scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
  packed.resize(src.size());
  dequantized.resize(src.size());

  for (size_t i = 0; i < src.size(); ++i) {
    int q = static_cast<int>(std::lrint(src[i] / scale));
    q = std::max(-127, std::min(127, q));
    packed[i] = static_cast<int8_t>(q);
    dequantized[i] = static_cast<float>(q) * scale;
  }
}

inline void prepare_int4_storage(const std::vector<float>& src,
                                 std::vector<uint8_t>& packed,
                                 std::vector<float>& dequantized,
                                 float& scale) {
  float max_abs = 0.0f;
  for (float v : src) {
    max_abs = std::max(max_abs, std::fabs(v));
  }
  scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;

  packed.assign((src.size() + 1) / 2, 0);
  dequantized.resize(src.size());

  for (size_t i = 0; i < src.size(); ++i) {
    int q = static_cast<int>(std::lrint(src[i] / scale));
    q = std::max(-8, std::min(7, q));
    dequantized[i] = static_cast<float>(q) * scale;
    uint8_t nibble = static_cast<uint8_t>(q & 0xF);
    if (i & 1U) {
      packed[i >> 1] |= static_cast<uint8_t>(nibble << 4);
    } else {
      packed[i >> 1] = nibble;
    }
  }
}

template <typename StorageT>
inline std::vector<float> run_kernel_once(const std::vector<StorageT>& h_a,
                                          const std::vector<StorageT>& h_b,
                                          int m, int n, int k, float scale_a,
                                          float scale_b) {
  StorageT* d_a = nullptr;
  StorageT* d_b = nullptr;
  float* d_c = nullptr;

  std::vector<float> h_c(m * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(StorageT)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(StorageT)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(StorageT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(StorageT),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return h_c;
}

inline std::vector<float> run_int4_kernel_once(const std::vector<uint8_t>& h_a,
                                               const std::vector<uint8_t>& h_b,
                                               int m, int n, int k,
                                               float scale_a, float scale_b) {
  uint8_t* d_a = nullptr;
  uint8_t* d_b = nullptr;
  float* d_c = nullptr;

  std::vector<float> h_c(m * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  tiled_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a,
                                          scale_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return h_c;
}

template <typename StorageT>
inline float benchmark_kernel(const std::vector<StorageT>& h_a,
                              const std::vector<StorageT>& h_b, int m, int n,
                              int k, float scale_a, float scale_b,
                              int warmup_iters = 5, int iters = 30) {
  StorageT* d_a = nullptr;
  StorageT* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(StorageT)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(StorageT)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(StorageT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(StorageT),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  for (int i = 0; i < warmup_iters; ++i) {
    tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i) {
    tiled_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
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

inline float benchmark_int4_kernel(const std::vector<uint8_t>& h_a,
                                   const std::vector<uint8_t>& h_b, int m,
                                   int n, int k, float scale_a, float scale_b,
                                   int warmup_iters = 5, int iters = 30) {
  uint8_t* d_a = nullptr;
  uint8_t* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice));

  dim3 block(THREADS_X, THREADS_Y);
  dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);

  for (int i = 0; i < warmup_iters; ++i) {
    tiled_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a,
                                            scale_b);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i) {
    tiled_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a,
                                            scale_b);
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
  double flops = 2.0 * static_cast<double>(m) * n * k;
  return flops / (static_cast<double>(ms) * 1.0e9);
}

template <typename StorageT>
inline int run_storage_experiment(const std::string& name) {
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

  std::vector<StorageT> a_storage;
  std::vector<StorageT> b_storage;
  std::vector<float> a_dequant;
  std::vector<float> b_dequant;
  float scale_a = 1.0f;
  float scale_b = 1.0f;

  prepare_storage(a_fp32, a_storage, a_dequant, scale_a);
  prepare_storage(b_fp32, b_storage, b_dequant, scale_b);

  auto ref_quant = cpu_gemm(a_dequant, b_dequant, check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got = run_kernel_once(a_storage, b_storage, check_m, check_n, check_k,
                             scale_a, scale_b);

  DiffStats stats_quant = compare_outputs(got, ref_quant);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);

  bool ok = stats_quant.max_abs < 0.2f && stats_quant.max_rel < 0.2f;

  float avg_ms = 0.0f;
  if (profile_mode) {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 7);
    fill_input(b_bench_fp32, bench_k, bench_n, 19);

    std::vector<StorageT> a_bench_storage;
    std::vector<StorageT> b_bench_storage;
    std::vector<float> dummy_a;
    std::vector<float> dummy_b;
    float bench_scale_a = 1.0f;
    float bench_scale_b = 1.0f;

    prepare_storage(a_bench_fp32, a_bench_storage, dummy_a, bench_scale_a);
    prepare_storage(b_bench_fp32, b_bench_storage, dummy_b, bench_scale_b);
    auto bench_out = run_kernel_once(a_bench_storage, b_bench_storage, bench_m,
                                     bench_n, bench_k, bench_scale_a,
                                     bench_scale_b);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 7);
    fill_input(b_bench_fp32, bench_k, bench_n, 19);

    std::vector<StorageT> a_bench_storage;
    std::vector<StorageT> b_bench_storage;
    std::vector<float> dummy_a;
    std::vector<float> dummy_b;
    float bench_scale_a = 1.0f;
    float bench_scale_b = 1.0f;

    prepare_storage(a_bench_fp32, a_bench_storage, dummy_a, bench_scale_a);
    prepare_storage(b_bench_fp32, b_bench_storage, dummy_b, bench_scale_b);
    avg_ms = benchmark_kernel(a_bench_storage, b_bench_storage, bench_m,
                              bench_n, bench_k, bench_scale_a, bench_scale_b);
  }

  std::cout << name << " "
            << (ok ? "passed" : "failed")
            << ". max_abs_vs_quant_ref=" << stats_quant.max_abs
            << ", max_rel_vs_quant_ref=" << stats_quant.max_rel
            << ", max_abs_vs_fp32_ref=" << stats_fp32.max_abs
            << ", avg_ms=" << avg_ms;
  if (!profile_mode) {
    std::cout << ", tflops="
              << tflops_from_ms(bench_m, bench_n, bench_k, avg_ms);
  }
  std::cout << std::endl;

  return ok ? 0 : 1;
}

inline int run_int4_experiment() {
  const bool profile_mode = (std::getenv("GEMM_PROFILE_ONCE") != nullptr);
  constexpr int check_m = 256;
  constexpr int check_n = 256;
  constexpr int check_k = 256;
  constexpr int bench_m = 1024;
  constexpr int bench_n = 1024;
  constexpr int bench_k = 1024;

  std::vector<float> a_fp32;
  std::vector<float> b_fp32;
  fill_input(a_fp32, check_m, check_k, 5);
  fill_input(b_fp32, check_k, check_n, 13);

  std::vector<uint8_t> a_packed;
  std::vector<uint8_t> b_packed;
  std::vector<float> a_dequant;
  std::vector<float> b_dequant;
  float scale_a = 1.0f;
  float scale_b = 1.0f;

  prepare_int4_storage(a_fp32, a_packed, a_dequant, scale_a);
  prepare_int4_storage(b_fp32, b_packed, b_dequant, scale_b);

  auto ref_quant = cpu_gemm(a_dequant, b_dequant, check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got =
      run_int4_kernel_once(a_packed, b_packed, check_m, check_n, check_k,
                           scale_a, scale_b);

  DiffStats stats_quant = compare_outputs(got, ref_quant);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);
  bool ok = stats_quant.max_abs < 1e-3f;

  float avg_ms = 0.0f;
  if (profile_mode) {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 17);
    fill_input(b_bench_fp32, bench_k, bench_n, 23);

    std::vector<uint8_t> a_bench_packed;
    std::vector<uint8_t> b_bench_packed;
    std::vector<float> dummy_a;
    std::vector<float> dummy_b;
    float bench_scale_a = 1.0f;
    float bench_scale_b = 1.0f;

    prepare_int4_storage(a_bench_fp32, a_bench_packed, dummy_a, bench_scale_a);
    prepare_int4_storage(b_bench_fp32, b_bench_packed, dummy_b, bench_scale_b);
    auto bench_out =
        run_int4_kernel_once(a_bench_packed, b_bench_packed, bench_m, bench_n,
                             bench_k, bench_scale_a, bench_scale_b);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 17);
    fill_input(b_bench_fp32, bench_k, bench_n, 23);

    std::vector<uint8_t> a_bench_packed;
    std::vector<uint8_t> b_bench_packed;
    std::vector<float> dummy_a;
    std::vector<float> dummy_b;
    float bench_scale_a = 1.0f;
    float bench_scale_b = 1.0f;

    prepare_int4_storage(a_bench_fp32, a_bench_packed, dummy_a, bench_scale_a);
    prepare_int4_storage(b_bench_fp32, b_bench_packed, dummy_b, bench_scale_b);

    avg_ms = benchmark_int4_kernel(a_bench_packed, b_bench_packed, bench_m,
                                   bench_n, bench_k, bench_scale_a,
                                   bench_scale_b);
  }

  std::cout << "int4 "
            << (ok ? "passed" : "failed")
            << ". max_abs_vs_quant_ref=" << stats_quant.max_abs
            << ", max_rel_vs_quant_ref=" << stats_quant.max_rel
            << ", max_abs_vs_fp32_ref=" << stats_fp32.max_abs
            << ", avg_ms=" << avg_ms;
  if (!profile_mode) {
    std::cout << ", tflops="
              << tflops_from_ms(bench_m, bench_n, bench_k, avg_ms);
  }
  std::cout << std::endl;

  return ok ? 0 : 1;
}

}  // namespace gemm

int main() { return gemm::run_storage_experiment<__nv_bfloat16>("bf16"); }
