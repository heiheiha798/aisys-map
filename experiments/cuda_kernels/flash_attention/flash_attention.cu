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

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kRowTile = kWarpsPerBlock;
constexpr int kColTile = 64;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void flash_attention_kernel(const float* q, const float* k,
                                       const float* v, float* out,
                                       int seq_len, int head_dim,
                                       float scale) {
  int warp_id = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int query_row = blockIdx.x * kRowTile + warp_id;

  extern __shared__ float shared[];
  float* k_tile = shared;
  float* v_tile = k_tile + kColTile * head_dim;
  float* score_tile = v_tile + kColTile * head_dim;

  float q_lane = 0.0f;
  if (query_row < seq_len && lane < head_dim) {
    q_lane = q[static_cast<size_t>(query_row) * head_dim + lane];
  }

  float row_out = 0.0f;
  float row_m = -INFINITY;
  float row_l = 0.0f;

  for (int tile_start = 0; tile_start < seq_len; tile_start += kColTile) {
    int tile_cols = min(kColTile, seq_len - tile_start);

    for (int idx = threadIdx.x; idx < tile_cols * head_dim; idx += blockDim.x) {
      int local_col = idx / head_dim;
      int d = idx % head_dim;
      int global_col = tile_start + local_col;
      k_tile[idx] = k[static_cast<size_t>(global_col) * head_dim + d];
      v_tile[idx] = v[static_cast<size_t>(global_col) * head_dim + d];
    }
    __syncthreads();

    if (query_row < seq_len && lane < head_dim) {
      for (int local_col = 0; local_col < tile_cols; ++local_col) {
        float dot = q_lane * k_tile[local_col * head_dim + lane];
        dot = warp_sum(dot);
        if (lane == 0) {
          score_tile[warp_id * kColTile + local_col] = dot * scale;
        }
      }

      float old_scale = 0.0f;
      if (lane == 0) {
        float tile_m = -INFINITY;
        for (int local_col = 0; local_col < tile_cols; ++local_col) {
          tile_m = fmaxf(tile_m, score_tile[warp_id * kColTile + local_col]);
        }

        float new_m = fmaxf(row_m, tile_m);
        old_scale = (row_l == 0.0f) ? 0.0f : expf(row_m - new_m);
        float tile_l = 0.0f;
        for (int local_col = 0; local_col < tile_cols; ++local_col) {
          float weight = expf(score_tile[warp_id * kColTile + local_col] - new_m);
          score_tile[warp_id * kColTile + local_col] = weight;
          tile_l += weight;
        }
        row_l = row_l * old_scale + tile_l;
        row_m = new_m;
      }

      old_scale = __shfl_sync(0xffffffff, old_scale, 0);
      row_out *= old_scale;
      for (int local_col = 0; local_col < tile_cols; ++local_col) {
        float weight = (lane == 0) ? score_tile[warp_id * kColTile + local_col] : 0.0f;
        weight = __shfl_sync(0xffffffff, weight, 0);
        float v_val = v_tile[local_col * head_dim + lane];
        row_out += weight * v_val;
      }
    }
    __syncthreads();
  }

  if (query_row < seq_len && lane < head_dim) {
    float final_l = __shfl_sync(0xffffffff, row_l, 0);
    out[static_cast<size_t>(query_row) * head_dim + lane] = row_out / final_l;
  }
}

void cpu_flash_attention(const std::vector<float>& q,
                         const std::vector<float>& k,
                         const std::vector<float>& v,
                         std::vector<float>& out, int seq_len,
                         int head_dim, float scale) {
  std::vector<float> scores(seq_len, 0.0f);
  for (int row = 0; row < seq_len; ++row) {
    const float* q_row = q.data() + static_cast<size_t>(row) * head_dim;
    float row_max = -INFINITY;
    for (int col = 0; col < seq_len; ++col) {
      const float* k_row = k.data() + static_cast<size_t>(col) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q_row[d] * k_row[d];
      }
      scores[col] = dot * scale;
      row_max = std::max(row_max, scores[col]);
    }

    float row_sum = 0.0f;
    for (int col = 0; col < seq_len; ++col) {
      scores[col] = std::exp(scores[col] - row_max);
      row_sum += scores[col];
    }

    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int col = 0; col < seq_len; ++col) {
        acc += (scores[col] / row_sum) *
               v[static_cast<size_t>(col) * head_dim + d];
      }
      out[static_cast<size_t>(row) * head_dim + d] = acc;
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
  constexpr int seq_len = 512;
  constexpr int head_dim = 32;
  constexpr int threads_per_block = kWarpsPerBlock * kWarpSize;

  const size_t numel = static_cast<size_t>(seq_len) * head_dim;
  const size_t bytes = numel * sizeof(float);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const size_t shared_mem_bytes =
      static_cast<size_t>((2 * kColTile * head_dim) + (kRowTile * kColTile)) *
      sizeof(float);

  std::vector<float> h_q(numel);
  std::vector<float> h_k(numel);
  std::vector<float> h_v(numel);
  std::vector<float> h_out(numel, 0.0f);
  std::vector<float> h_ref(numel, 0.0f);

  for (int row = 0; row < seq_len; ++row) {
    for (int d = 0; d < head_dim; ++d) {
      size_t idx = static_cast<size_t>(row) * head_dim + d;
      h_q[idx] = 0.08f * std::sin((row + 2) * (d + 1) * 0.09f) +
                 0.02f * static_cast<float>((row + d) % 5 - 2);
      h_k[idx] = 0.07f * std::cos((row + 4) * (d + 2) * 0.06f) +
                 0.03f * static_cast<float>((row * 3 + d) % 7 - 3);
      h_v[idx] = 0.05f * std::sin((row + 6) * (d + 1) * 0.11f) +
                 0.02f * static_cast<float>((row * 5 + d * 2) % 9 - 4);
    }
  }

  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_q, bytes));
  CUDA_CHECK(cudaMalloc(&d_k, bytes));
  CUDA_CHECK(cudaMalloc(&d_v, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

  int grid = (seq_len + kRowTile - 1) / kRowTile;
  flash_attention_kernel<<<grid, threads_per_block, shared_mem_bytes>>>(
      d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  cpu_flash_attention(h_q, h_k, h_v, h_ref, seq_len, head_dim, scale);

  float max_abs = max_abs_diff(h_out, h_ref);
  bool ok = max_abs < 2e-4f;

  if (ok) {
    std::cout << "flash_attention passed. seq_len=" << seq_len
              << ", head_dim=" << head_dim
              << ", threads_per_block=" << threads_per_block
              << ", row_tile=" << kRowTile
              << ", col_tile=" << kColTile
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: out[0]=" << h_out[0]
              << ", out[1]=" << h_out[1]
              << ", out[last]=" << h_out.back() << std::endl;
  } else {
    std::cerr << "flash_attention failed. max_abs_diff=" << max_abs
              << std::endl;
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
