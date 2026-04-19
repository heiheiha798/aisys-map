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

struct KvOp {
  int head;
  int slot;
  int token_id;
};

__global__ void kv_cache_append_update_kernel(const float* k_src,
                                              const float* v_src,
                                              const KvOp* ops,
                                              float* k_cache,
                                              float* v_cache,
                                              int num_ops,
                                              int num_heads,
                                              int max_seq_len,
                                              int head_dim) {
  int op_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (op_idx >= num_ops) {
    return;
  }

  KvOp op = ops[op_idx];
  if (op.head < 0 || op.head >= num_heads || op.slot < 0 ||
      op.slot >= max_seq_len) {
    return;
  }

  const size_t src_offset = static_cast<size_t>(op.token_id) * num_heads * head_dim +
                            static_cast<size_t>(op.head) * head_dim;
  const size_t cache_offset =
      static_cast<size_t>(op.head) * max_seq_len * head_dim +
      static_cast<size_t>(op.slot) * head_dim;

  const float* k_src_ptr = k_src + src_offset;
  const float* v_src_ptr = v_src + src_offset;
  float* k_cache_ptr = k_cache + cache_offset;
  float* v_cache_ptr = v_cache + cache_offset;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    k_cache_ptr[d] = k_src_ptr[d];
    v_cache_ptr[d] = v_src_ptr[d];
  }
}

void cpu_kv_cache_append_update(const std::vector<float>& k_src,
                                const std::vector<float>& v_src,
                                const std::vector<KvOp>& ops,
                                std::vector<float>& k_cache,
                                std::vector<float>& v_cache,
                                int num_heads,
                                int max_seq_len,
                                int head_dim) {
  for (const KvOp& op : ops) {
    if (op.head < 0 || op.head >= num_heads || op.slot < 0 ||
        op.slot >= max_seq_len) {
      continue;
    }

    const size_t src_offset =
        static_cast<size_t>(op.token_id) * num_heads * head_dim +
        static_cast<size_t>(op.head) * head_dim;
    const size_t cache_offset =
        static_cast<size_t>(op.head) * max_seq_len * head_dim +
        static_cast<size_t>(op.slot) * head_dim;

    const float* k_src_ptr = k_src.data() + src_offset;
    const float* v_src_ptr = v_src.data() + src_offset;
    float* k_cache_ptr = k_cache.data() + cache_offset;
    float* v_cache_ptr = v_cache.data() + cache_offset;

    for (int d = 0; d < head_dim; ++d) {
      k_cache_ptr[d] = k_src_ptr[d];
      v_cache_ptr[d] = v_src_ptr[d];
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

void fill_source(std::vector<float>& k_src, std::vector<float>& v_src, int tokens,
                 int num_heads, int head_dim) {
  for (int token = 0; token < tokens; ++token) {
    for (int head = 0; head < num_heads; ++head) {
      for (int d = 0; d < head_dim; ++d) {
        size_t idx = static_cast<size_t>(token) * num_heads * head_dim +
                     static_cast<size_t>(head) * head_dim + d;
        float base = 0.01f * static_cast<float>(token) +
                     0.1f * static_cast<float>(head) +
                     0.001f * static_cast<float>(d);
        k_src[idx] = std::sin(base * 3.0f) + 0.25f * std::cos(base * 5.0f);
        v_src[idx] = std::cos(base * 2.0f) - 0.35f * std::sin(base * 7.0f);
      }
    }
  }
}

void print_sample(const std::vector<float>& k_cache, const std::vector<float>& v_cache,
                  int head, int slot, int max_seq_len, int head_dim) {
  size_t base = static_cast<size_t>(head) * max_seq_len * head_dim +
                static_cast<size_t>(slot) * head_dim;
  std::cout << "sample cache slice: "
            << "K[h=" << head << ", s=" << slot << ", d=0]=" << k_cache[base]
            << ", K[..., d=1]=" << k_cache[base + 1]
            << ", V[..., d=0]=" << v_cache[base]
            << ", V[..., d=1]=" << v_cache[base + 1] << std::endl;
}

}  // namespace

int main() {
  constexpr int num_tokens = 8;
  constexpr int num_heads = 2;
  constexpr int max_seq_len = 6;
  constexpr int head_dim = 16;
  constexpr int threads_per_block = 128;

  const size_t src_numel =
      static_cast<size_t>(num_tokens) * num_heads * head_dim;
  const size_t cache_numel =
      static_cast<size_t>(num_heads) * max_seq_len * head_dim;
  const size_t src_bytes = src_numel * sizeof(float);
  const size_t cache_bytes = cache_numel * sizeof(float);

  std::vector<KvOp> h_append_ops = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 1}, {1, 1, 1}, {0, 2, 2}, {1, 2, 2},
  };
  std::vector<KvOp> h_update_ops = {
      {0, 1, 5},
      {1, 1, 5},
  };

  std::vector<float> h_k_src(src_numel);
  std::vector<float> h_v_src(src_numel);
  std::vector<float> h_k_cache(cache_numel, -7.0f);
  std::vector<float> h_v_cache(cache_numel, -11.0f);
  std::vector<float> h_k_ref = h_k_cache;
  std::vector<float> h_v_ref = h_v_cache;

  fill_source(h_k_src, h_v_src, num_tokens, num_heads, head_dim);
  cpu_kv_cache_append_update(h_k_src, h_v_src, h_append_ops, h_k_ref, h_v_ref,
                             num_heads, max_seq_len, head_dim);
  cpu_kv_cache_append_update(h_k_src, h_v_src, h_update_ops, h_k_ref, h_v_ref,
                             num_heads, max_seq_len, head_dim);

  float* d_k_src = nullptr;
  float* d_v_src = nullptr;
  float* d_k_cache = nullptr;
  float* d_v_cache = nullptr;
  KvOp* d_append_ops = nullptr;
  KvOp* d_update_ops = nullptr;

  CUDA_CHECK(cudaMalloc(&d_k_src, src_bytes));
  CUDA_CHECK(cudaMalloc(&d_v_src, src_bytes));
  CUDA_CHECK(cudaMalloc(&d_k_cache, cache_bytes));
  CUDA_CHECK(cudaMalloc(&d_v_cache, cache_bytes));
  CUDA_CHECK(cudaMalloc(&d_append_ops, h_append_ops.size() * sizeof(KvOp)));
  CUDA_CHECK(cudaMalloc(&d_update_ops, h_update_ops.size() * sizeof(KvOp)));

  CUDA_CHECK(cudaMemcpy(d_k_src, h_k_src.data(), src_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_src, h_v_src.data(), src_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_append_ops, h_append_ops.data(),
                        h_append_ops.size() * sizeof(KvOp),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_update_ops, h_update_ops.data(),
                        h_update_ops.size() * sizeof(KvOp),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), cache_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), cache_bytes, cudaMemcpyHostToDevice));

  kv_cache_append_update_kernel<<<static_cast<int>(h_append_ops.size()),
                                  threads_per_block>>>(
      d_k_src, d_v_src, d_append_ops, d_k_cache, d_v_cache,
      static_cast<int>(h_append_ops.size()), num_heads, max_seq_len, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  kv_cache_append_update_kernel<<<static_cast<int>(h_update_ops.size()),
                                  threads_per_block>>>(
      d_k_src, d_v_src, d_update_ops, d_k_cache, d_v_cache,
      static_cast<int>(h_update_ops.size()), num_heads, max_seq_len, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(h_k_cache.data(), d_k_cache, cache_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_v_cache.data(), d_v_cache, cache_bytes, cudaMemcpyDeviceToHost));

  float k_max_abs = max_abs_diff(h_k_cache, h_k_ref);
  float v_max_abs = max_abs_diff(h_v_cache, h_v_ref);
  bool ok = k_max_abs < 1e-6f && v_max_abs < 1e-6f;

  if (ok) {
    std::cout << "kv_cache_append_update passed. num_tokens=" << num_tokens
              << ", num_heads=" << num_heads
              << ", max_seq_len=" << max_seq_len
              << ", head_dim=" << head_dim
              << ", append_ops=" << h_append_ops.size()
              << ", update_ops=" << h_update_ops.size()
              << ", k_max_abs_diff=" << k_max_abs
              << ", v_max_abs_diff=" << v_max_abs << std::endl;
    std::cout << "append example: token 2 -> slot 2" << std::endl;
    std::cout << "update example: token 5 overwrites slot 1" << std::endl;
    print_sample(h_k_cache, h_v_cache, 0, 1, max_seq_len, head_dim);
  } else {
    std::cerr << "kv_cache_append_update failed. k_max_abs_diff=" << k_max_abs
              << ", v_max_abs_diff=" << v_max_abs << std::endl;
  }

  CUDA_CHECK(cudaFree(d_k_src));
  CUDA_CHECK(cudaFree(d_v_src));
  CUDA_CHECK(cudaFree(d_k_cache));
  CUDA_CHECK(cudaFree(d_v_cache));
  CUDA_CHECK(cudaFree(d_append_ops));
  CUDA_CHECK(cudaFree(d_update_ops));
  return ok ? 0 : 1;
}
