import math
import sys

try:
  import torch
  import triton
  import triton.language as tl
except ImportError as exc:
  print(
      "Missing dependency: this script requires PyTorch + Triton in a CUDA-enabled "
      "Python environment.",
      file=sys.stderr,
  )
  raise SystemExit(1) from exc


@triton.jit
def kv_cache_append_update_kernel(k_src_ptr, v_src_ptr, op_heads_ptr,
                                  op_slots_ptr, op_token_ids_ptr, k_cache_ptr,
                                  v_cache_ptr, num_ops, num_tokens, num_heads,
                                  max_seq_len, head_dim,
                                  BLOCK_D: tl.constexpr):
  op_idx = tl.program_id(0)
  d_block = tl.program_id(1)

  valid_op = op_idx < num_ops

  head = tl.load(op_heads_ptr + op_idx, mask=valid_op, other=0).to(tl.int64)
  slot = tl.load(op_slots_ptr + op_idx, mask=valid_op, other=0).to(tl.int64)
  token_id = tl.load(op_token_ids_ptr + op_idx, mask=valid_op, other=0).to(
      tl.int64)

  d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

  valid_write = (valid_op & (head >= 0) & (head < num_heads) & (slot >= 0) &
                 (slot < max_seq_len) & (token_id >= 0) &
                 (token_id < num_tokens) & (d_offsets < head_dim))

  src_base = (token_id * num_heads + head) * head_dim
  cache_base = (head * max_seq_len + slot) * head_dim

  k_vals = tl.load(k_src_ptr + src_base + d_offsets, mask=valid_write, other=0.0)
  v_vals = tl.load(v_src_ptr + src_base + d_offsets, mask=valid_write, other=0.0)

  tl.store(k_cache_ptr + cache_base + d_offsets, k_vals, mask=valid_write)
  tl.store(v_cache_ptr + cache_base + d_offsets, v_vals, mask=valid_write)


def fill_source(num_tokens: int, num_heads: int, head_dim: int) -> tuple[torch.Tensor,
                                                                          torch.Tensor]:
  k_src = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float32)
  v_src = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float32)

  for token in range(num_tokens):
    for head in range(num_heads):
      for dim in range(head_dim):
        base = 0.01 * token + 0.1 * head + 0.001 * dim
        k_src[token, head, dim] = math.sin(base * 3.0) + 0.25 * math.cos(base * 5.0)
        v_src[token, head, dim] = math.cos(base * 2.0) - 0.35 * math.sin(base * 7.0)

  return k_src, v_src


def cpu_kv_cache_append_update(k_src: torch.Tensor, v_src: torch.Tensor,
                               op_heads: torch.Tensor, op_slots: torch.Tensor,
                               op_token_ids: torch.Tensor, k_cache: torch.Tensor,
                               v_cache: torch.Tensor) -> None:
  num_ops = op_heads.numel()
  num_tokens, num_heads, _ = k_src.shape
  _, max_seq_len, _ = k_cache.shape

  for op_idx in range(num_ops):
    head = int(op_heads[op_idx].item())
    slot = int(op_slots[op_idx].item())
    token_id = int(op_token_ids[op_idx].item())

    if head < 0 or head >= num_heads:
      continue
    if slot < 0 or slot >= max_seq_len:
      continue
    if token_id < 0 or token_id >= num_tokens:
      continue

    k_cache[head, slot, :] = k_src[token_id, head, :]
    v_cache[head, slot, :] = v_src[token_id, head, :]


def launch_kv_cache_append_update(k_src: torch.Tensor, v_src: torch.Tensor,
                                  op_heads: torch.Tensor, op_slots: torch.Tensor,
                                  op_token_ids: torch.Tensor,
                                  k_cache: torch.Tensor,
                                  v_cache: torch.Tensor,
                                  block_d: int = 16) -> None:
  assert k_src.is_cuda and v_src.is_cuda
  assert op_heads.is_cuda and op_slots.is_cuda and op_token_ids.is_cuda
  assert k_cache.is_cuda and v_cache.is_cuda

  num_tokens, num_heads, head_dim = k_src.shape
  _, max_seq_len, _ = k_cache.shape
  num_ops = op_heads.numel()

  grid = (num_ops, triton.cdiv(head_dim, block_d))
  kv_cache_append_update_kernel[grid](
      k_src,
      v_src,
      op_heads,
      op_slots,
      op_token_ids,
      k_cache,
      v_cache,
      num_ops,
      num_tokens,
      num_heads,
      max_seq_len,
      head_dim,
      BLOCK_D=block_d,
  )


def main() -> int:
  if not torch.cuda.is_available():
    print("CUDA is not available in the current PyTorch environment.", file=sys.stderr)
    return 1

  device = torch.device("cuda")

  num_tokens = 8
  num_heads = 2
  max_seq_len = 6
  head_dim = 16

  append_heads = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32)
  append_slots = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
  append_token_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)

  update_heads = torch.tensor([0, 1], dtype=torch.int32)
  update_slots = torch.tensor([1, 1], dtype=torch.int32)
  update_token_ids = torch.tensor([5, 5], dtype=torch.int32)

  k_src_cpu, v_src_cpu = fill_source(num_tokens, num_heads, head_dim)
  k_cache_cpu = torch.full((num_heads, max_seq_len, head_dim), -7.0, dtype=torch.float32)
  v_cache_cpu = torch.full((num_heads, max_seq_len, head_dim), -11.0, dtype=torch.float32)
  k_ref = k_cache_cpu.clone()
  v_ref = v_cache_cpu.clone()

  cpu_kv_cache_append_update(k_src_cpu, v_src_cpu, append_heads, append_slots,
                             append_token_ids, k_ref, v_ref)
  cpu_kv_cache_append_update(k_src_cpu, v_src_cpu, update_heads, update_slots,
                             update_token_ids, k_ref, v_ref)

  k_src_gpu = k_src_cpu.to(device)
  v_src_gpu = v_src_cpu.to(device)
  k_cache_gpu = k_cache_cpu.to(device)
  v_cache_gpu = v_cache_cpu.to(device)

  append_heads_gpu = append_heads.to(device)
  append_slots_gpu = append_slots.to(device)
  append_token_ids_gpu = append_token_ids.to(device)
  update_heads_gpu = update_heads.to(device)
  update_slots_gpu = update_slots.to(device)
  update_token_ids_gpu = update_token_ids.to(device)

  launch_kv_cache_append_update(k_src_gpu, v_src_gpu, append_heads_gpu,
                                append_slots_gpu, append_token_ids_gpu,
                                k_cache_gpu, v_cache_gpu)
  torch.cuda.synchronize()

  launch_kv_cache_append_update(k_src_gpu, v_src_gpu, update_heads_gpu,
                                update_slots_gpu, update_token_ids_gpu,
                                k_cache_gpu, v_cache_gpu)
  torch.cuda.synchronize()

  k_cache_out = k_cache_gpu.cpu()
  v_cache_out = v_cache_gpu.cpu()

  k_max_abs = (k_cache_out - k_ref).abs().max().item()
  v_max_abs = (v_cache_out - v_ref).abs().max().item()
  ok = k_max_abs < 1e-6 and v_max_abs < 1e-6

  if ok:
    print(
        "kv_cache_append_update_triton passed. "
        f"num_tokens={num_tokens}, num_heads={num_heads}, "
        f"max_seq_len={max_seq_len}, head_dim={head_dim}, "
        f"append_ops={append_heads.numel()}, update_ops={update_heads.numel()}, "
        f"k_max_abs_diff={k_max_abs}, v_max_abs_diff={v_max_abs}"
    )
    print("append example: token 2 -> slot 2")
    print("update example: token 5 overwrites slot 1")
    print(
        "sample cache slice: "
        f"K[h=0, s=1, d=0]={k_cache_out[0, 1, 0].item()}, "
        f"K[..., d=1]={k_cache_out[0, 1, 1].item()}, "
        f"V[..., d=0]={v_cache_out[0, 1, 0].item()}, "
        f"V[..., d=1]={v_cache_out[0, 1, 1].item()}"
    )
    return 0

  print(
      "kv_cache_append_update_triton failed. "
      f"k_max_abs_diff={k_max_abs}, v_max_abs_diff={v_max_abs}",
      file=sys.stderr,
  )
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
