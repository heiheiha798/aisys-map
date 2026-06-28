#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(err) << std::endl;                      \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t status = (call);                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__         \
                << " - status=" << static_cast<int>(status) << std::endl;    \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace gemm_cublas {

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

inline double tflops_from_ms(int m, int n, int k, float ms) {
  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                 static_cast<double>(k);
  return flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

inline void launch_bf16_gemm(cublasHandle_t handle, const __nv_bfloat16* d_a,
                             const __nv_bfloat16* d_b, float* d_c, int m, int n,
                             int k) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS assumes column-major storage. Treat row-major:
  //   A[m, k] as column-major A^T[k, m]
  //   B[k, n] as column-major B^T[n, k]
  // and compute C^T[n, m] = B^T[n, k] * A^T[k, m].
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                            d_b, CUDA_R_16BF, n, d_a, CUDA_R_16BF, k, &beta,
                            d_c, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline std::vector<float> run_once(const std::vector<__nv_bfloat16>& h_a,
                                   const std::vector<__nv_bfloat16>& h_b, int m,
                                   int n, int k) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;
  std::vector<float> h_c(static_cast<size_t>(m) * n, 0.0f);

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  launch_bf16_gemm(handle, d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUBLAS_CHECK(cublasDestroy(handle));

  return h_c;
}

inline float benchmark(const std::vector<__nv_bfloat16>& h_a,
                       const std::vector<__nv_bfloat16>& h_b, int m, int n,
                       int k, int warmup_iters = 5, int iters = 30) {
  __nv_bfloat16* d_a = nullptr;
  __nv_bfloat16* d_b = nullptr;
  float* d_c = nullptr;

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup_iters; ++i) {
    launch_bf16_gemm(handle, d_a, d_b, d_c, m, n, k);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < iters; ++i) {
    launch_bf16_gemm(handle, d_a, d_b, d_c, m, n, k);
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
  CUBLAS_CHECK(cublasDestroy(handle));

  return total_ms / static_cast<float>(iters);
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
  fill_input(a_fp32, check_m, check_k, 31);
  fill_input(b_fp32, check_k, check_n, 37);

  std::vector<__nv_bfloat16> a_bf16;
  std::vector<__nv_bfloat16> b_bf16;
  convert_from_fp32(a_fp32, a_bf16);
  convert_from_fp32(b_fp32, b_bf16);

  auto ref_fp32 = cpu_gemm(a_fp32, b_fp32, check_m, check_n, check_k);
  auto got = run_once(a_bf16, b_bf16, check_m, check_n, check_k);
  DiffStats stats = compare_outputs(got, ref_fp32);
  bool ok = stats.max_abs < 0.5f;

  float avg_ms = 0.0f;
  if (profile_mode) {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    std::vector<__nv_bfloat16> a_bench_bf16;
    std::vector<__nv_bfloat16> b_bench_bf16;
    fill_input(a_bench_fp32, bench_m, bench_k, 41);
    fill_input(b_bench_fp32, bench_k, bench_n, 43);
    convert_from_fp32(a_bench_fp32, a_bench_bf16);
    convert_from_fp32(b_bench_fp32, b_bench_bf16);
    auto bench_out = run_once(a_bench_bf16, b_bench_bf16, bench_m, bench_n, bench_k);
    avg_ms = bench_out[0] * 0.0f;
  } else {
    std::vector<float> a_bench_fp32;
    std::vector<float> b_bench_fp32;
    std::vector<__nv_bfloat16> a_bench_bf16;
    std::vector<__nv_bfloat16> b_bench_bf16;
    fill_input(a_bench_fp32, bench_m, bench_k, 41);
    fill_input(b_bench_fp32, bench_k, bench_n, 43);
    convert_from_fp32(a_bench_fp32, a_bench_bf16);
    convert_from_fp32(b_bench_fp32, b_bench_bf16);
    avg_ms = benchmark(a_bench_bf16, b_bench_bf16, bench_m, bench_n, bench_k);
  }

  std::cout << "bf16_cublas " << (ok ? "passed" : "failed")
            << ". max_abs_vs_fp32_ref=" << stats.max_abs
            << ", max_rel_vs_fp32_ref=" << stats.max_rel
            << ", avg_ms=" << avg_ms;
  if (!profile_mode) {
    std::cout << ", tflops=" << tflops_from_ms(bench_m, bench_n, bench_k, avg_ms);
  }
  std::cout << std::endl;

  return ok ? 0 : 1;
}

}  // namespace gemm_cublas

int main() { return gemm_cublas::run_experiment(); }
