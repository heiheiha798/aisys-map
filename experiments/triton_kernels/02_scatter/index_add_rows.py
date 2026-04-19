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
def index_add_rows_kernel(src_ptr, ids_ptr, dst_ptr, src_rows, dst_rows, dim,
                          BLOCK_D: tl.constexpr):
  src_row = tl.program_id(0)
  d_block = tl.program_id(1)

  valid_src_row = src_row < src_rows
  dst_row = tl.load(ids_ptr + src_row, mask=valid_src_row, other=0).to(tl.int64)

  d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
  valid_write = (valid_src_row & (dst_row >= 0) & (dst_row < dst_rows) &
                 (d_offsets < dim))

  src_offsets = src_row * dim + d_offsets
  dst_offsets = dst_row * dim + d_offsets

  src_vals = tl.load(src_ptr + src_offsets, mask=valid_write, other=0.0)
  tl.atomic_add(dst_ptr + dst_offsets, src_vals, mask=valid_write)


def cpu_index_add_rows(src: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor) -> None:
  src_rows, dim = src.shape
  dst_rows, _ = dst.shape

  for src_row in range(src_rows):
    dst_row = int(ids[src_row].item())
    if dst_row < 0 or dst_row >= dst_rows:
      continue
    dst[dst_row, :] += src[src_row, :]


def launch_index_add_rows(src: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor,
                          block_d: int = 128) -> None:
  assert src.is_cuda and ids.is_cuda and dst.is_cuda
  src_rows, dim = src.shape
  dst_rows, _ = dst.shape

  grid = (src_rows, triton.cdiv(dim, block_d))
  index_add_rows_kernel[grid](
      src,
      ids,
      dst,
      src_rows,
      dst_rows,
      dim,
      BLOCK_D=block_d,
  )


def main() -> int:
  if not torch.cuda.is_available():
    print("CUDA is not available in the current PyTorch environment.", file=sys.stderr)
    return 1

  device = torch.device("cuda")

  src_rows = 4096
  dst_rows = 512
  dim = 256

  src_cpu = torch.empty((src_rows, dim), dtype=torch.float32)
  ids_cpu = torch.empty((src_rows,), dtype=torch.int32)
  dst_cpu = torch.zeros((dst_rows, dim), dtype=torch.float32)
  ref_cpu = torch.zeros((dst_rows, dim), dtype=torch.float32)

  for row in range(src_rows):
    ids_cpu[row] = (row * 17 + (row // 5) * 29) % dst_rows
    for col in range(dim):
      # Use exactly representable integer-valued floats so that atomic-add order
      # does not change the mathematically correct result in this teaching example.
      src_cpu[row, col] = float(((row * 13 + col * 7) % 37) - 18)

  cpu_index_add_rows(src_cpu, ids_cpu, ref_cpu)

  src_gpu = src_cpu.to(device)
  ids_gpu = ids_cpu.to(device)
  dst_gpu = dst_cpu.to(device)

  launch_index_add_rows(src_gpu, ids_gpu, dst_gpu)
  torch.cuda.synchronize()

  dst_out = dst_gpu.cpu()
  max_abs = (dst_out - ref_cpu).abs().max().item()
  ok = max_abs == 0.0

  if ok:
    print(
        "index_add_rows_triton passed. "
        f"src_rows={src_rows}, dst_rows={dst_rows}, dim={dim}, "
        f"max_abs_diff={max_abs}"
    )
    print(
        "sample output: "
        f"dst[0, 0]={dst_out[0, 0].item()}, "
        f"dst[0, 1]={dst_out[0, 1].item()}, "
        f"dst[last]={dst_out[-1, -1].item()}"
    )
    return 0

  print(f"index_add_rows_triton failed. max_abs_diff={max_abs}", file=sys.stderr)
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
