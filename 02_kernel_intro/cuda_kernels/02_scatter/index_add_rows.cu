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

__global__ void index_add_rows_kernel(const float* src, const int* ids, float* dst,
                                      int src_rows, int dst_rows, int dim) {
  int src_row = blockIdx.x;
  int tid = threadIdx.x;

  if (src_row >= src_rows) {
    return;
  }

  int dst_row = ids[src_row];
  if (dst_row < 0 || dst_row >= dst_rows) {
    return;
  }

  const float* src_ptr = src + static_cast<size_t>(src_row) * dim;
  float* dst_ptr = dst + static_cast<size_t>(dst_row) * dim;

  for (int col = tid; col < dim; col += blockDim.x) {
    atomicAdd(dst_ptr + col, src_ptr[col]);
  }
}

void cpu_index_add_rows(const std::vector<float>& src, const std::vector<int>& ids,
                        std::vector<float>& dst, int src_rows, int dst_rows,
                        int dim) {
  for (int r = 0; r < src_rows; ++r) {
    int dst_row = ids[r];
    if (dst_row < 0 || dst_row >= dst_rows) {
      continue;
    }
    const float* src_ptr = src.data() + static_cast<size_t>(r) * dim;
    float* dst_ptr = dst.data() + static_cast<size_t>(dst_row) * dim;
    for (int c = 0; c < dim; ++c) {
      dst_ptr[c] += src_ptr[c];
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
  constexpr int src_rows = 4096;
  constexpr int dst_rows = 512;
  constexpr int dim = 256;
  constexpr int threads_per_block = 256;

  const size_t src_numel = static_cast<size_t>(src_rows) * dim;
  const size_t dst_numel = static_cast<size_t>(dst_rows) * dim;
  const size_t src_bytes = src_numel * sizeof(float);
  const size_t dst_bytes = dst_numel * sizeof(float);
  const size_t ids_bytes = static_cast<size_t>(src_rows) * sizeof(int);

  std::vector<float> h_src(src_numel);
  std::vector<int> h_ids(src_rows);
  std::vector<float> h_dst(dst_numel, 0.0f);
  std::vector<float> h_ref(dst_numel, 0.0f);

  for (int r = 0; r < src_rows; ++r) {
    for (int c = 0; c < dim; ++c) {
      float x = std::sin((r + 1) * 0.0031f) - std::cos((c + 9) * 0.015f);
      float y = static_cast<float>(((r * 13 + c * 7) % 37) - 18) * 0.02f;
      h_src[static_cast<size_t>(r) * dim + c] = 0.5f * x + y;
    }
  }

  for (int i = 0; i < src_rows; ++i) {
    h_ids[i] = (i * 17 + (i / 5) * 29) % dst_rows;
  }

  float* d_src = nullptr;
  float* d_dst = nullptr;
  int* d_ids = nullptr;

  CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, dst_bytes));
  CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));

  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dst, 0, dst_bytes));

  index_add_rows_kernel<<<src_rows, threads_per_block>>>(d_src, d_ids, d_dst,
                                                         src_rows, dst_rows, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost));
  cpu_index_add_rows(h_src, h_ids, h_ref, src_rows, dst_rows, dim);

  float max_abs = max_abs_diff(h_dst, h_ref);
  bool ok = max_abs < 1e-5f;

  if (ok) {
    std::cout << "index_add_rows passed. src_rows=" << src_rows
              << ", dst_rows=" << dst_rows << ", dim=" << dim
              << ", threads_per_block=" << threads_per_block
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: dst[0]=" << h_dst[0]
              << ", dst[1]=" << h_dst[1]
              << ", dst[last]=" << h_dst.back() << std::endl;
  } else {
    std::cerr << "index_add_rows failed. max_abs_diff=" << max_abs << std::endl;
  }

  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  CUDA_CHECK(cudaFree(d_ids));
  return ok ? 0 : 1;
}
