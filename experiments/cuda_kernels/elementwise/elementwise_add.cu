#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                << cudaGetErrorString(err) << std::endl;                      \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void elementwise_add_kernel(const float* a, const float* b, float* c,
                                       int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  constexpr int n = 1 << 20;
  constexpr int bytes = n * sizeof(float);
  constexpr int threads_per_block = 256;
  const int blocks = (n + threads_per_block - 1) / threads_per_block;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(2 * i);
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  elementwise_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = h_a[i] + h_b[i];
    if (std::fabs(h_c[i] - expected) > 1e-5f) {
      std::cerr << "Mismatch at index " << i << ": got " << h_c[i]
                << ", expected " << expected << std::endl;
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout << "elementwise_add passed. "
              << "blocks=" << blocks
              << ", threads_per_block=" << threads_per_block << std::endl;
    std::cout << "sample output: c[0]=" << h_c[0]
              << ", c[1]=" << h_c[1]
              << ", c[1024]=" << h_c[1024] << std::endl;
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return ok ? 0 : 1;
}
