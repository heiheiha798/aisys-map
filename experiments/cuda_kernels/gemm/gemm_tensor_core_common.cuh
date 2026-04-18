#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(err) << std::endl;                      \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace gemm_tc {

using namespace nvcuda;
namespace wmma_exp = nvcuda::wmma::experimental;

struct DiffStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
};

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

template <typename T>
inline void convert_from_fp32(const std::vector<float>& src, std::vector<T>& dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = T(src[i]);
  }
}

template <typename T>
inline std::vector<float> dequant_from_storage(const std::vector<T>& src) {
  std::vector<float> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = static_cast<float>(src[i]);
  }
  return out;
}

template <>
inline std::vector<float> dequant_from_storage<float>(const std::vector<float>& src) {
  return src;
}

inline void quantize_to_int8(const std::vector<float>& src, std::vector<int8_t>& dst,
                             std::vector<float>& dequantized, float& scale) {
  float max_abs = 0.0f;
  for (float v : src) {
    max_abs = std::max(max_abs, std::fabs(v));
  }
  scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
  dst.resize(src.size());
  dequantized.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    int q = static_cast<int>(std::lrint(src[i] / scale));
    q = std::max(-127, std::min(127, q));
    dst[i] = static_cast<int8_t>(q);
    dequantized[i] = static_cast<float>(q) * scale;
  }
}

inline void pack_int4_row_major(const std::vector<float>& src, int rows, int cols,
                                std::vector<int>& packed,
                                std::vector<float>& dequantized, float& scale) {
  float max_abs = 0.0f;
  for (float v : src) {
    max_abs = std::max(max_abs, std::fabs(v));
  }
  scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
  dequantized.resize(src.size());
  packed.assign((rows * cols + 7) / 8, 0);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      int idx = r * cols + c;
      int q = static_cast<int>(std::lrint(src[idx] / scale));
      q = std::max(-8, std::min(7, q));
      dequantized[idx] = static_cast<float>(q) * scale;

      int group = idx / 8;
      int offset = (idx % 8) * 4;
      packed[group] |= ((q & 0xF) << offset);
    }
  }
}

inline std::vector<int> transpose_int4_packed_col_major(const std::vector<int>& src,
                                                        int rows, int cols) {
  std::vector<int> out((rows * cols + 7) / 8, 0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      int src_idx = r * cols + c;
      int src_group = src_idx / 8;
      int src_offset = (src_idx % 8) * 4;
      int nibble = (src[src_group] >> src_offset) & 0xF;

      int dst_idx = c * rows + r;
      int dst_group = dst_idx / 8;
      int dst_offset = (dst_idx % 8) * 4;
      out[dst_group] |= (nibble << dst_offset);
    }
  }
  return out;
}

template <typename AType, typename BType>
__global__ void wmma_gemm_16x16x16_kernel(const AType* a, const BType* b, float* c,
                                          int m, int n, int k) {
  using FragA =
      wmma::fragment<wmma::matrix_a, 16, 16, 16, AType, wmma::row_major>;
  using FragB =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, BType, wmma::col_major>;
  using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  int lane_id = threadIdx.x % 32;
  (void)lane_id;

  int warps_per_block_m = blockDim.y;
  int warp_row = blockIdx.y * warps_per_block_m + threadIdx.y;
  int warp_col = blockIdx.x;

  int row = warp_row * 16;
  int col = warp_col * 16;

  if (row >= m || col >= n) {
    return;
  }

  FragC c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += 16) {
    FragA a_frag;
    FragB b_frag;
    const AType* a_ptr = a + row * k + k0;
    const BType* b_ptr = b + col * k + k0;
    wmma::load_matrix_sync(a_frag, a_ptr, k);
    wmma::load_matrix_sync(b_frag, b_ptr, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c + row * n + col, c_frag, n, wmma::mem_row_major);
}

__global__ void wmma_gemm_int8_kernel(const int8_t* a, const int8_t* b_col_major,
                                      float* c, int m, int n, int k,
                                      float scale_a, float scale_b) {
  using FragA =
      wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>;
  using FragB =
      wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major>;
  using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, int>;

  int row = (blockIdx.y * blockDim.y + threadIdx.y) * 16;
  int col = blockIdx.x * 16;
  if (row >= m || col >= n) {
    return;
  }

  FragC c_frag;
  wmma::fill_fragment(c_frag, 0);

  for (int k0 = 0; k0 < k; k0 += 16) {
    FragA a_frag;
    FragB b_frag;
    wmma::load_matrix_sync(a_frag, a + row * k + k0, k);
    wmma::load_matrix_sync(b_frag, b_col_major + col * k + k0, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ int c_tile[4][16 * 16];
  int warp_slot = threadIdx.y;
  wmma::store_matrix_sync(c_tile[warp_slot], c_frag, 16, wmma::mem_row_major);
  __syncthreads();

  if (threadIdx.x < 32) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
      int r = idx / 16;
      int cc = idx % 16;
      int global_r = row + r;
      int global_c = col + cc;
      if (global_r < m && global_c < n) {
        c[global_r * n + global_c] =
            static_cast<float>(c_tile[warp_slot][idx]) * scale_a * scale_b;
      }
    }
  }
}

__global__ void wmma_gemm_int4_kernel(const int* a_packed,
                                      const int* b_packed_col_major, int* c,
                                      int m, int n, int k) {
  using FragA = wmma::fragment<wmma::matrix_a, 8, 8, 32,
                               wmma_exp::precision::s4, wmma::row_major>;
  using FragB = wmma::fragment<wmma::matrix_b, 8, 8, 32,
                               wmma_exp::precision::s4, wmma::col_major>;
  using FragC = wmma::fragment<wmma::accumulator, 8, 8, 32, int>;

  int row = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
  int col = blockIdx.x * 8;
  if (row >= m || col >= n) {
    return;
  }

  FragC c_frag;
  wmma::fill_fragment(c_frag, 0);

  for (int k0 = 0; k0 < k; k0 += 32) {
    FragA a_frag;
    FragB b_frag;
    int a_offset = (row * k + k0) / 8;
    int b_offset = (col * k + k0) / 8;
    wmma::load_matrix_sync(a_frag, a_packed + a_offset, k);
    wmma::load_matrix_sync(b_frag, b_packed_col_major + b_offset, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ int c_tile[4][8 * 8];
  int warp_slot = threadIdx.y;
  wmma::store_matrix_sync(c_tile[warp_slot], c_frag, 8, wmma::mem_row_major);
  __syncthreads();

  if (threadIdx.x < 32) {
    for (int idx = threadIdx.x; idx < 8 * 8; idx += 32) {
      int r = idx / 8;
      int cc = idx % 8;
      int global_r = row + r;
      int global_c = col + cc;
      if (global_r < m && global_c < n) {
        c[global_r * n + global_c] = c_tile[warp_slot][idx];
      }
    }
  }
}

template <typename T>
inline std::vector<T> transpose_to_col_major_storage(const std::vector<T>& src,
                                                     int rows, int cols) {
  std::vector<T> out(src.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[c * rows + r] = src[r * cols + c];
    }
  }
  return out;
}

template <typename AType, typename BType>
inline std::vector<float> run_wmma_once(const std::vector<AType>& h_a,
                                        const std::vector<BType>& h_b_col_major,
                                        int m, int n, int k) {
  AType* d_a = nullptr;
  BType* d_b = nullptr;
  float* d_c = nullptr;
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(AType)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(BType)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(AType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(BType),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 15) / 16, (m + (block.y * 16) - 1) / (block.y * 16));

  wmma_gemm_16x16x16_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return h_c;
}

template <typename AType, typename BType>
inline float benchmark_wmma(const std::vector<AType>& h_a,
                            const std::vector<BType>& h_b_col_major, int m,
                            int n, int k, int warmup_iters = 5,
                            int iters = 30) {
  AType* d_a = nullptr;
  BType* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(AType)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(BType)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(AType),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(BType),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 15) / 16, (m + (block.y * 16) - 1) / (block.y * 16));

  for (int i = 0; i < warmup_iters; ++i) {
    wmma_gemm_16x16x16_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i) {
    wmma_gemm_16x16x16_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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

inline std::vector<float> run_wmma_int8_once(const std::vector<int8_t>& h_a,
                                             const std::vector<int8_t>& h_b_col_major,
                                             int m, int n, int k, float scale_a,
                                             float scale_b) {
  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  float* d_c = nullptr;
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 15) / 16, (m + (block.y * 16) - 1) / (block.y * 16));
  wmma_gemm_int8_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return h_c;
}

inline float benchmark_wmma_int8(const std::vector<int8_t>& h_a,
                                 const std::vector<int8_t>& h_b_col_major, int m,
                                 int n, int k, float scale_a, float scale_b,
                                 int warmup_iters = 5, int iters = 30) {
  int8_t* d_a = nullptr;
  int8_t* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 15) / 16, (m + (block.y * 16) - 1) / (block.y * 16));
  for (int i = 0; i < warmup_iters; ++i) {
    wmma_gemm_int8_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    wmma_gemm_int8_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k, scale_a, scale_b);
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

inline std::vector<float> run_wmma_int4_once(const std::vector<int>& h_a_packed,
                                             const std::vector<int>& h_b_packed_col_major,
                                             int m, int n, int k, float scale_a,
                                             float scale_b) {
  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c_int = nullptr;
  std::vector<int> h_c_int(static_cast<size_t>(m) * n, 0);
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a_packed.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_packed_col_major.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_c_int, h_c_int.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a_packed.data(), h_a_packed.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_packed_col_major.data(),
                        h_b_packed_col_major.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 7) / 8, (m + (block.y * 8) - 1) / (block.y * 8));
  wmma_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c_int, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c_int.data(), d_c_int, h_c_int.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < h_c.size(); ++i) {
    h_c[i] = static_cast<float>(h_c_int[i]) * scale_a * scale_b;
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c_int));
  return h_c;
}

inline float benchmark_wmma_int4(const std::vector<int>& h_a_packed,
                                 const std::vector<int>& h_b_packed_col_major, int m,
                                 int n, int k, int warmup_iters = 5,
                                 int iters = 30) {
  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a_packed.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_packed_col_major.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a_packed.data(), h_a_packed.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_packed_col_major.data(),
                        h_b_packed_col_major.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  dim3 block(32, 4);
  dim3 grid((n + 7) / 8, (m + (block.y * 8) - 1) / (block.y * 8));
  for (int i = 0; i < warmup_iters; ++i) {
    wmma_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    wmma_gemm_int4_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
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
inline int run_tensor_core_experiment(const std::string& name) {
  const bool profile_mode = (std::getenv("GEMM_PROFILE_ONCE") != nullptr);
  constexpr int check_m = 256;
  constexpr int check_n = 256;
  constexpr int check_k = 256;
  constexpr int bench_m = 1024;
  constexpr int bench_n = 1024;
  constexpr int bench_k = 1024;

  std::vector<float> a_fp32;
  std::vector<float> b_fp32;
  fill_input(a_fp32, check_m, check_k, 31);
  fill_input(b_fp32, check_k, check_n, 37);

  std::vector<StorageT> a_storage;
  std::vector<StorageT> b_storage;
  convert_from_fp32(a_fp32, a_storage);
  convert_from_fp32(b_fp32, b_storage);
  auto b_col_major = transpose_to_col_major_storage(b_storage, check_k, check_n);
  auto a_deq = dequant_from_storage(a_storage);
  auto b_deq = dequant_from_storage(b_storage);

  auto ref_quant = cpu_gemm(a_deq, b_deq, check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got = run_wmma_once(a_storage, b_col_major, check_m, check_n, check_k);

  DiffStats stats_quant = compare_outputs(got, ref_quant);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);
  bool ok = stats_quant.max_abs < 0.5f && stats_quant.max_rel < 0.5f;

  float avg_ms = 0.0f;
  if (profile_mode) {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 41);
    fill_input(b_bench_fp32, bench_k, bench_n, 43);

    std::vector<StorageT> a_bench_storage;
    std::vector<StorageT> b_bench_storage;
    convert_from_fp32(a_bench_fp32, a_bench_storage);
    convert_from_fp32(b_bench_fp32, b_bench_storage);
    auto b_bench_col_major =
        transpose_to_col_major_storage(b_bench_storage, bench_k, bench_n);
    auto bench_out =
        run_wmma_once(a_bench_storage, b_bench_col_major, bench_m, bench_n, bench_k);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    fill_input(a_bench_fp32, bench_m, bench_k, 41);
    fill_input(b_bench_fp32, bench_k, bench_n, 43);

    std::vector<StorageT> a_bench_storage;
    std::vector<StorageT> b_bench_storage;
    convert_from_fp32(a_bench_fp32, a_bench_storage);
    convert_from_fp32(b_bench_fp32, b_bench_storage);
    auto b_bench_col_major =
        transpose_to_col_major_storage(b_bench_storage, bench_k, bench_n);

    avg_ms = benchmark_wmma(a_bench_storage, b_bench_col_major, bench_m, bench_n,
                            bench_k);
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

inline int run_int8_tensor_core_experiment() {
  const bool profile_mode = (std::getenv("GEMM_PROFILE_ONCE") != nullptr);
  constexpr int check_m = 256, check_n = 256, check_k = 256;
  constexpr int bench_m = 1024, bench_n = 1024, bench_k = 1024;
  std::vector<float> a_fp32, b_fp32;
  fill_input(a_fp32, check_m, check_k, 47);
  fill_input(b_fp32, check_k, check_n, 53);

  std::vector<int8_t> a_q, b_q;
  std::vector<float> a_deq, b_deq;
  float scale_a = 1.0f, scale_b = 1.0f;
  quantize_to_int8(a_fp32, a_q, a_deq, scale_a);
  quantize_to_int8(b_fp32, b_q, b_deq, scale_b);
  auto b_col_major = transpose_to_col_major_storage(b_q, check_k, check_n);

  auto ref_quant = cpu_gemm(a_deq, b_deq, check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got = run_wmma_int8_once(a_q, b_col_major, check_m, check_n, check_k,
                                scale_a, scale_b);
  DiffStats stats_quant = compare_outputs(got, ref_quant);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);
  bool ok = stats_quant.max_abs < 0.5f && stats_quant.max_rel < 0.5f;

  float avg_ms = 0.0f;
  std::vector<float> a_bench_fp32, b_bench_fp32;
  fill_input(a_bench_fp32, bench_m, bench_k, 59);
  fill_input(b_bench_fp32, bench_k, bench_n, 61);
  std::vector<int8_t> a_bench_q, b_bench_q;
  std::vector<float> dummy_a, dummy_b;
  float bench_scale_a = 1.0f, bench_scale_b = 1.0f;
  quantize_to_int8(a_bench_fp32, a_bench_q, dummy_a, bench_scale_a);
  quantize_to_int8(b_bench_fp32, b_bench_q, dummy_b, bench_scale_b);
  auto b_bench_col_major =
      transpose_to_col_major_storage(b_bench_q, bench_k, bench_n);
  if (profile_mode) {
    auto bench_out = run_wmma_int8_once(a_bench_q, b_bench_col_major, bench_m,
                                        bench_n, bench_k, bench_scale_a,
                                        bench_scale_b);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    avg_ms = benchmark_wmma_int8(a_bench_q, b_bench_col_major, bench_m,
                                 bench_n, bench_k, bench_scale_a, bench_scale_b);
  }

  std::cout << "int8_tensor_core "
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

inline int run_int4_tensor_core_experiment() {
  const bool profile_mode = (std::getenv("GEMM_PROFILE_ONCE") != nullptr);
  constexpr int check_m = 256, check_n = 256, check_k = 256;
  constexpr int bench_m = 1024, bench_n = 1024, bench_k = 1024;
  std::vector<float> a_fp32, b_fp32;
  fill_input(a_fp32, check_m, check_k, 67);
  fill_input(b_fp32, check_k, check_n, 71);

  std::vector<int> a_packed, b_packed;
  std::vector<float> a_deq, b_deq;
  float scale_a = 1.0f, scale_b = 1.0f;
  pack_int4_row_major(a_fp32, check_m, check_k, a_packed, a_deq, scale_a);
  pack_int4_row_major(b_fp32, check_k, check_n, b_packed, b_deq, scale_b);
  auto b_col_major = transpose_int4_packed_col_major(b_packed, check_k, check_n);

  auto ref_quant = cpu_gemm(a_deq, b_deq, check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got = run_wmma_int4_once(a_packed, b_col_major, check_m, check_n, check_k,
                                scale_a, scale_b);
  DiffStats stats_quant = compare_outputs(got, ref_quant);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);
  bool ok = stats_quant.max_abs < 1.0f;

  float avg_ms = 0.0f;
  std::vector<float> a_bench_fp32, b_bench_fp32;
  fill_input(a_bench_fp32, bench_m, bench_k, 73);
  fill_input(b_bench_fp32, bench_k, bench_n, 79);
  std::vector<int> a_bench_packed, b_bench_packed;
  std::vector<float> dummy_a, dummy_b;
  float bench_scale_a = 1.0f, bench_scale_b = 1.0f;
  pack_int4_row_major(a_bench_fp32, bench_m, bench_k, a_bench_packed, dummy_a,
                      bench_scale_a);
  pack_int4_row_major(b_bench_fp32, bench_k, bench_n, b_bench_packed, dummy_b,
                      bench_scale_b);
  auto b_bench_col_major =
      transpose_int4_packed_col_major(b_bench_packed, bench_k, bench_n);
  if (profile_mode) {
    auto bench_out =
        run_wmma_int4_once(a_bench_packed, b_bench_col_major, bench_m, bench_n,
                           bench_k, bench_scale_a, bench_scale_b);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    avg_ms = benchmark_wmma_int4(a_bench_packed, b_bench_col_major, bench_m,
                                 bench_n, bench_k);
  }

  std::cout << "int4_tensor_core "
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

}  // namespace gemm_tc
