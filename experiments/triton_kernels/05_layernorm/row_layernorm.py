import math

import torch
import triton
import triton.language as tl


EPS = 1e-5


@triton.jit
def row_layernorm_kernel(x_ptr, y_ptr, rows, cols, stride, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    row_ptr = x_ptr + row * stride + offsets
    x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

    row_sum = tl.sum(x, axis=0)
    row_sq_sum = tl.sum(x * x, axis=0)

    mean = row_sum / cols
    mean_sq = row_sq_sum / cols
    var = tl.maximum(mean_sq - mean * mean, 0.0)
    inv_std = tl.rsqrt(var + eps)

    y = (x - mean) * inv_std
    tl.store(y_ptr + row * stride + offsets, y, mask=mask)


def cpu_row_layernorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    mean_sq = (x * x).mean(dim=1, keepdim=True)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    return (x - mean) * torch.rsqrt(var + eps)


def make_input(rows: int, cols: int, device: str) -> torch.Tensor:
    r = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    c = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0)
    x = torch.sin((r + 1.0) * 0.013) + torch.cos((c + 3.0) * 0.021)
    y = (((r * 11.0 + c * 7.0) % 23.0) - 11.0) * 0.05
    return 0.7 * x + y


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton example.")

    rows = 1024
    cols = 256
    block_size = triton.next_power_of_2(cols)

    x = make_input(rows, cols, "cuda")
    y = torch.empty_like(x)

    grid = (rows,)
    row_layernorm_kernel[grid](x, y, rows, cols, x.stride(0), EPS, BLOCK_SIZE=block_size)

    ref = cpu_row_layernorm(x, EPS)
    max_abs_diff = torch.max(torch.abs(y - ref)).item()
    ok = max_abs_diff < 2e-4

    if ok:
        print(
            f"row_layernorm passed. rows={rows}, cols={cols}, "
            f"block_size={block_size}, max_abs_diff={max_abs_diff}"
        )
        print(f"sample output: y[0]={y.flatten()[0].item()}, y[1]={y.flatten()[1].item()}, y[255]={y.flatten()[255].item()}")
    else:
        idx = torch.argmax(torch.abs(y - ref)).item()
        got = y.flatten()[idx].item()
        expected = ref.flatten()[idx].item()
        raise AssertionError(
            f"row_layernorm failed. max_abs_diff={max_abs_diff}, idx={idx}, got={got}, ref={expected}"
        )


if __name__ == "__main__":
    run()
