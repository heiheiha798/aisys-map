#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_BLOCK_WARPS_M = 8;
constexpr int WMMA_BLOCK_WARPS = WMMA_BLOCK_WARPS_M;
constexpr int WMMA_B_SKEW = 8;
constexpr int WMMA_B_LD = WMMA_K + WMMA_B_SKEW;

struct DiffStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
};

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

inline std::vector<float> dequant_from_storage(
    const std::vector<__nv_bfloat16>& src) {
  std::vector<float> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = __bfloat162float(src[i]);
  }
  return out;
}

inline std::vector<__nv_bfloat16> transpose_to_col_major_storage(
    const std::vector<__nv_bfloat16>& src, int rows, int cols) {
  std::vector<__nv_bfloat16> out(src.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c) * rows + r] =
          src[static_cast<size_t>(r) * cols + c];
    }
  }
  return out;
}

__global__ void wmma_gemm_16x16x16_kernel(const __nv_bfloat16* a,
                                          const __nv_bfloat16* b_col_major,
                                          float* c, int m, int n, int k) {
  using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               __nv_bfloat16, wmma::row_major>;
  using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               __nv_bfloat16, wmma::col_major>;
  using FragC =
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

  int warp_slot = threadIdx.y;
  int row = (blockIdx.y * WMMA_BLOCK_WARPS_M + warp_slot) * WMMA_M;
  int col = blockIdx.x * WMMA_N;
  bool warp_active = (row < m && col < n);

  __shared__ __nv_bfloat16 b_tile[WMMA_N * WMMA_B_LD];
  const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

  FragC c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += WMMA_K) {
    FragA a_frag;
    FragB b_frag;

    for (int idx = linear_tid; idx < WMMA_N * WMMA_K;
         idx += blockDim.x * blockDim.y) {
      int tile_col = idx / WMMA_K;
      int tile_row = idx % WMMA_K;
      int global_col = col + tile_col;
      int global_row = k0 + tile_row;
      if (global_col < n && global_row < k) {
        b_tile[tile_col * WMMA_B_LD + tile_row] =
            b_col_major[global_col * k + global_row];
      } else {
        b_tile[tile_col * WMMA_B_LD + tile_row] = __nv_bfloat16(0.0f);
      }
    }
    __syncthreads();

    if (warp_active) {
      wmma::load_matrix_sync(a_frag, a + row * k + k0, k);
      wmma::load_matrix_sync(b_frag, b_tile, WMMA_B_LD);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }

  if (warp_active) {
    wmma::store_matrix_sync(c + row * n + col, c_frag, n, wmma::mem_row_major);
  }
}

inline std::vector<float> run_once(const std::vector<__nv_bfloat16>& h_a,
                                   const std::vector<__nv_bfloat16>& h_b_col_major,
                                   int m, int n, int k) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  dim3 block(32, WMMA_BLOCK_WARPS);
  dim3 grid((n + WMMA_N - 1) / WMMA_N,
            (m + WMMA_BLOCK_WARPS_M * WMMA_M - 1) /
                (WMMA_BLOCK_WARPS_M * WMMA_M));

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

inline float benchmark(const std::vector<__nv_bfloat16>& h_a,
                       const std::vector<__nv_bfloat16>& h_b_col_major, int m,
                       int n, int k, int warmup_iters = 5, int iters = 30) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b_col_major.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b_col_major.data(),
                        h_b_col_major.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  dim3 block(32, WMMA_BLOCK_WARPS);
  dim3 grid((n + WMMA_N - 1) / WMMA_N,
            (m + WMMA_BLOCK_WARPS_M * WMMA_M - 1) /
                (WMMA_BLOCK_WARPS_M * WMMA_M));

  for (int i = 0; i < warmup_iters; ++i) {
    wmma_gemm_16x16x16_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
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

inline double tflops_from_ms(int m, int n, int k, float ms) {
  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                 static_cast<double>(k);
  return flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

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

  std::vector<__nv_bfloat16> a_bf16;
  std::vector<__nv_bfloat16> b_bf16;
  convert_from_fp32(a_fp32, a_bf16);
  convert_from_fp32(b_fp32, b_bf16);

  auto b_col_major = transpose_to_col_major_storage(b_bf16, check_k, check_n);
  auto got = run_once(a_bf16, b_col_major, check_m, check_n, check_k);
  auto ref_bf16 =
      cpu_gemm(dequant_from_storage(a_bf16), dequant_from_storage(b_bf16),
               check_m, check_n, check_k);
  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);

  DiffStats stats_bf16 = compare_outputs(got, ref_bf16);
  DiffStats stats_fp32 = compare_outputs(got, ref_fp32);
  bool ok = stats_bf16.max_abs < 0.5f && stats_bf16.max_rel < 0.5f;

  float avg_ms = 0.0f;
  std::vector<float> a_bench_fp32;
  std::vector<float> b_bench_fp32;
  fill_input(a_bench_fp32, bench_m, bench_k, 41);
  fill_input(b_bench_fp32, bench_k, bench_n, 43);

  std::vector<__nv_bfloat16> a_bench_bf16;
  std::vector<__nv_bfloat16> b_bench_bf16;
  convert_from_fp32(a_bench_fp32, a_bench_bf16);
  convert_from_fp32(b_bench_fp32, b_bench_bf16);
  auto b_bench_col_major =
      transpose_to_col_major_storage(b_bench_bf16, bench_k, bench_n);

  if (profile_mode) {
    auto bench_out =
        run_once(a_bench_bf16, b_bench_col_major, bench_m, bench_n, bench_k);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    avg_ms = benchmark(a_bench_bf16, b_bench_col_major, bench_m, bench_n, bench_k);
  }

  std::cout << name << " " << (ok ? "passed" : "failed")
            << ". max_abs_vs_bf16_ref=" << stats_bf16.max_abs
            << ", max_rel_vs_bf16_ref=" << stats_bf16.max_rel
            << ", max_abs_vs_fp32_ref=" << stats_fp32.max_abs
            << ", avg_ms=" << avg_ms;
  if (!profile_mode) {
    std::cout << ", tflops=" << tflops_from_ms(bench_m, bench_n, bench_k, avg_ms);
  }
  std::cout << std::endl;

  return ok ? 0 : 1;
}

}  // namespace gemm_tc

int main() { return gemm_tc::run_tensor_core_experiment("bf16_tensor_core"); }
