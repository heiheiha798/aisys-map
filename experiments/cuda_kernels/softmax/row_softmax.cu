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

__global__ void row_softmax_kernel(const float* x, float* y, int rows,
                                   int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];
  float* s_max = shared;
  float* s_sum = shared;

  float local_max = -INFINITY;
  for (int col = tid; col < cols; col += blockDim.x) {
    float val = x[row * cols + col];
    if (val > local_max) {
      local_max = val;
    }
  }

  s_max[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
    }
    __syncthreads();
  }

  float row_max = s_max[0];

  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    float exp_val = expf(x[row * cols + col] - row_max);
    y[row * cols + col] = exp_val;
    local_sum += exp_val;
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

  for (int col = tid; col < cols; col += blockDim.x) {
    y[row * cols + col] /= row_sum;
  }
}

void cpu_row_softmax(const std::vector<float>& x, std::vector<float>& y, int rows,
                     int cols) {
  for (int r = 0; r < rows; ++r) {
    float row_max = -INFINITY;
    for (int c = 0; c < cols; ++c) {
      row_max = std::max(row_max, x[r * cols + c]);
    }

    float row_sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float exp_val = std::exp(x[r * cols + c] - row_max);
      y[r * cols + c] = exp_val;
      row_sum += exp_val;
    }

    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] /= row_sum;
    }
  }
}

int main() {
  constexpr int rows = 128;
  constexpr int cols = 256;
  constexpr int threads_per_block = 256;
  constexpr int bytes = rows * cols * sizeof(float);
  constexpr int shared_mem_bytes = threads_per_block * sizeof(float);

  std::vector<float> h_x(rows * cols);
  std::vector<float> h_y(rows * cols, 0.0f);
  std::vector<float> h_ref(rows * cols, 0.0f);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      h_x[r * cols + c] = static_cast<float>((r + c) % 17 - 8);
    }
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

  row_softmax_kernel<<<rows, threads_per_block, shared_mem_bytes>>>(d_x, d_y, rows,
                                                                    cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  cpu_row_softmax(h_x, h_ref, rows, cols);

  bool ok = true;
  for (int i = 0; i < rows * cols; ++i) {
    if (std::fabs(h_y[i] - h_ref[i]) > 1e-4f) {
      std::cerr << "Mismatch at index " << i << ": got " << h_y[i]
                << ", expected " << h_ref[i] << std::endl;
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout << "row_softmax passed. rows=" << rows << ", cols=" << cols
              << ", threads_per_block=" << threads_per_block << std::endl;
    std::cout << "sample output: y[0]=" << h_y[0]
              << ", y[1]=" << h_y[1]
              << ", y[255]=" << h_y[255] << std::endl;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  return ok ? 0 : 1;
}
