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

__global__ void row_gather_kernel(const float* table, const int* ids, float* out,
                                  int batch, int dim) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= batch) {
    return;
  }

  int token_id = ids[row];
  const float* src = table + static_cast<size_t>(token_id) * dim;
  float* dst = out + static_cast<size_t>(row) * dim;

  for (int col = tid; col < dim; col += blockDim.x) {
    dst[col] = src[col];
  }
}

void cpu_row_gather(const std::vector<float>& table, const std::vector<int>& ids,
                    std::vector<float>& out, int batch, int dim) {
  for (int r = 0; r < batch; ++r) {
    int token_id = ids[r];
    const float* src = table.data() + static_cast<size_t>(token_id) * dim;
    float* dst = out.data() + static_cast<size_t>(r) * dim;
    for (int c = 0; c < dim; ++c) {
      dst[c] = src[c];
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
  constexpr int vocab = 8192;
  constexpr int dim = 256;
  constexpr int batch = 4096;
  constexpr int threads_per_block = 256;

  const size_t table_numel = static_cast<size_t>(vocab) * dim;
  const size_t out_numel = static_cast<size_t>(batch) * dim;
  const size_t table_bytes = table_numel * sizeof(float);
  const size_t out_bytes = out_numel * sizeof(float);
  const size_t ids_bytes = static_cast<size_t>(batch) * sizeof(int);

  std::vector<float> h_table(table_numel);
  std::vector<int> h_ids(batch);
  std::vector<float> h_out(out_numel, 0.0f);
  std::vector<float> h_ref(out_numel, 0.0f);
  const char* id_mode_env = std::getenv("GATHER_ID_MODE");
  const bool repeated_mode =
      (id_mode_env != nullptr && std::string(id_mode_env) == "repeated");

  for (int r = 0; r < vocab; ++r) {
    for (int c = 0; c < dim; ++c) {
      float x = std::sin((r + 1) * 0.0013f) + std::cos((c + 5) * 0.017f);
      float y = static_cast<float>(((r * 7 + c * 11) % 31) - 15) * 0.03f;
      h_table[static_cast<size_t>(r) * dim + c] = 0.6f * x + y;
    }
  }

  for (int i = 0; i < batch; ++i) {
    if (repeated_mode) {
      h_ids[i] = (i / 64) % 32;
    } else {
      h_ids[i] = (i * 37 + (i / 7) * 17) % vocab;
    }
  }

  float* d_table = nullptr;
  float* d_out = nullptr;
  int* d_ids = nullptr;

  CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_ids, ids_bytes));

  CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), table_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_ids, h_ids.data(), ids_bytes, cudaMemcpyHostToDevice));

  row_gather_kernel<<<batch, threads_per_block>>>(d_table, d_ids, d_out, batch,
                                                  dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  cpu_row_gather(h_table, h_ids, h_ref, batch, dim);

  float max_abs = max_abs_diff(h_out, h_ref);
  bool ok = max_abs < 1e-6f;

  if (ok) {
    std::cout << "row_gather passed. vocab=" << vocab << ", batch=" << batch
              << ", dim=" << dim
              << ", threads_per_block=" << threads_per_block
              << ", id_mode=" << (repeated_mode ? "repeated" : "random")
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: out[0]=" << h_out[0]
              << ", out[1]=" << h_out[1]
              << ", out[last]=" << h_out.back() << std::endl;
  } else {
    std::cerr << "row_gather failed. max_abs_diff=" << max_abs << std::endl;
  }

  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_ids));
  return ok ? 0 : 1;
}
