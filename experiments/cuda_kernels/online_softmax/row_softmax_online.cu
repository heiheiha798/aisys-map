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

struct OnlineSoftmaxState {
  float m;
  float l;
};

__device__ __forceinline__ OnlineSoftmaxState merge_states(OnlineSoftmaxState a,
                                                           OnlineSoftmaxState b) {
  if (a.l == 0.0f) {
    return b;
  }
  if (b.l == 0.0f) {
    return a;
  }

  OnlineSoftmaxState out;
  out.m = fmaxf(a.m, b.m);
  out.l = a.l * expf(a.m - out.m) + b.l * expf(b.m - out.m);
  return out;
}

__device__ __forceinline__ OnlineSoftmaxState warp_reduce_state(
    OnlineSoftmaxState state) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    OnlineSoftmaxState other;
    other.m = __shfl_down_sync(0xffffffff, state.m, offset);
    other.l = __shfl_down_sync(0xffffffff, state.l, offset);
    state = merge_states(state, other);
  }
  return state;
}

__global__ void row_softmax_online_kernel(const float* x, float* y, int rows,
                                          int cols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x >> 5;

  if (row >= rows) {
    return;
  }

  __shared__ float warp_max[8];
  __shared__ float warp_sum[8];

  OnlineSoftmaxState state;
  state.m = -INFINITY;
  state.l = 0.0f;

  for (int col = tid; col < cols; col += blockDim.x) {
    float v = x[row * cols + col];
    OnlineSoftmaxState cur;
    cur.m = v;
    cur.l = 1.0f;
    state = merge_states(state, cur);
  }

  state = warp_reduce_state(state);

  if (lane == 0) {
    warp_max[warp_id] = state.m;
    warp_sum[warp_id] = state.l;
  }

  __syncthreads();

  OnlineSoftmaxState block_state;
  if (warp_id == 0) {
    if (lane < num_warps) {
      block_state.m = warp_max[lane];
      block_state.l = warp_sum[lane];
    } else {
      block_state.m = -INFINITY;
      block_state.l = 0.0f;
    }

    block_state = warp_reduce_state(block_state);

    if (lane == 0) {
      warp_max[0] = block_state.m;
      warp_sum[0] = block_state.l;
    }
  }

  __syncthreads();

  float row_max = warp_max[0];
  float row_sum = warp_sum[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    float exp_val = expf(x[row * cols + col] - row_max);
    y[row * cols + col] = exp_val / row_sum;
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
  constexpr int rows = 4096;
  constexpr int cols = 256;
  constexpr int threads_per_block = 256;
  constexpr int bytes = rows * cols * sizeof(float);

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

  row_softmax_online_kernel<<<rows, threads_per_block>>>(d_x, d_y, rows, cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

  cpu_row_softmax(h_x, h_ref, rows, cols);

  bool ok = true;
  for (int i = 0; i < rows * cols; ++i) {
    if (std::isnan(h_y[i]) || std::fabs(h_y[i] - h_ref[i]) > 1e-4f) {
      std::cerr << "Mismatch at index " << i << ": got " << h_y[i]
                << ", expected " << h_ref[i] << std::endl;
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout << "row_softmax_online passed. rows=" << rows
              << ", cols=" << cols
              << ", threads_per_block=" << threads_per_block << std::endl;
    std::cout << "sample output: y[0]=" << h_y[0]
              << ", y[1]=" << h_y[1]
              << ", y[255]=" << h_y[255] << std::endl;
  }

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  return ok ? 0 : 1;
}
