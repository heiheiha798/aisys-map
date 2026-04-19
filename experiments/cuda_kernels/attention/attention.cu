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

__global__ void attention_kernel(const float* q, const float* k, const float* v,
                                 float* out, int seq_len, int head_dim,
                                 float scale) {
  int query_row = blockIdx.x;
  int tid = threadIdx.x;

  if (query_row >= seq_len) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + seq_len;

  for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
    float dot = 0.0f;
    const float* q_row = q + static_cast<size_t>(query_row) * head_dim;
    const float* k_row = k + static_cast<size_t>(key_row) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += q_row[d] * k_row[d];
    }
    scores[key_row] = dot * scale;
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
    local_max = fmaxf(local_max, scores[key_row]);
  }
  reduce[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }

  float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
    float exp_score = expf(scores[key_row] - row_max);
    scores[key_row] = exp_score;
    local_sum += exp_score;
  }
  reduce[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  float row_sum = reduce[0];
  for (int key_row = tid; key_row < seq_len; key_row += blockDim.x) {
    scores[key_row] /= row_sum;
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int key_row = 0; key_row < seq_len; ++key_row) {
      acc += scores[key_row] * v[static_cast<size_t>(key_row) * head_dim + d];
    }
    out[static_cast<size_t>(query_row) * head_dim + d] = acc;
  }
}

void cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                   const std::vector<float>& v, std::vector<float>& out,
                   int seq_len, int head_dim, float scale) {
  std::vector<float> scores(seq_len, 0.0f);

  for (int query_row = 0; query_row < seq_len; ++query_row) {
    float row_max = -INFINITY;
    const float* q_row = q.data() + static_cast<size_t>(query_row) * head_dim;

    for (int key_row = 0; key_row < seq_len; ++key_row) {
      const float* k_row = k.data() + static_cast<size_t>(key_row) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q_row[d] * k_row[d];
      }
      scores[key_row] = dot * scale;
      row_max = std::max(row_max, scores[key_row]);
    }

    float row_sum = 0.0f;
    for (int key_row = 0; key_row < seq_len; ++key_row) {
      scores[key_row] = std::exp(scores[key_row] - row_max);
      row_sum += scores[key_row];
    }

    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int key_row = 0; key_row < seq_len; ++key_row) {
        float weight = scores[key_row] / row_sum;
        acc += weight * v[static_cast<size_t>(key_row) * head_dim + d];
      }
      out[static_cast<size_t>(query_row) * head_dim + d] = acc;
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
  constexpr int seq_len = 64;
  constexpr int head_dim = 32;
  constexpr int threads_per_block = 128;

  const size_t numel = static_cast<size_t>(seq_len) * head_dim;
  const size_t bytes = numel * sizeof(float);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const size_t shared_mem_bytes =
      (static_cast<size_t>(seq_len) + threads_per_block) * sizeof(float);

  std::vector<float> h_q(numel);
  std::vector<float> h_k(numel);
  std::vector<float> h_v(numel);
  std::vector<float> h_out(numel, 0.0f);
  std::vector<float> h_ref(numel, 0.0f);

  for (int row = 0; row < seq_len; ++row) {
    for (int d = 0; d < head_dim; ++d) {
      size_t idx = static_cast<size_t>(row) * head_dim + d;
      h_q[idx] = 0.07f * std::sin((row + 1) * (d + 1) * 0.13f) +
                 0.03f * static_cast<float>((row + d) % 5 - 2);
      h_k[idx] = 0.05f * std::cos((row + 3) * (d + 1) * 0.11f) +
                 0.04f * static_cast<float>((row * 3 + d) % 7 - 3);
      h_v[idx] = 0.06f * std::sin((row + 5) * (d + 2) * 0.09f) +
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

  attention_kernel<<<seq_len, threads_per_block, shared_mem_bytes>>>(
      d_q, d_k, d_v, d_out, seq_len, head_dim, scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  cpu_attention(h_q, h_k, h_v, h_ref, seq_len, head_dim, scale);

  float max_abs = max_abs_diff(h_out, h_ref);
  bool ok = max_abs < 1e-4f;

  if (ok) {
    std::cout << "attention passed. seq_len=" << seq_len
              << ", head_dim=" << head_dim
              << ", threads_per_block=" << threads_per_block
              << ", max_abs_diff=" << max_abs << std::endl;
    std::cout << "sample output: out[0]=" << h_out[0]
              << ", out[1]=" << h_out[1]
              << ", out[last]=" << h_out.back() << std::endl;
  } else {
    std::cerr << "attention failed. max_abs_diff=" << max_abs << std::endl;
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
