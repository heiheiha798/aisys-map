import torch
import triton
import triton.language as tl


@triton.jit
def row_mean_sq_kernel(x_ptr, residual_ptr, mean_sq_ptr, rows, cols,
                       x_stride_row, residual_stride_row,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    if row >= rows:
        return

    x_row_ptr = x_ptr + row * x_stride_row
    residual_row_ptr = residual_ptr + row * residual_stride_row

    cols_idx = tl.arange(0, BLOCK_SIZE)
    mask = cols_idx < cols
    x = tl.load(x_row_ptr + cols_idx, mask=mask, other=0.0)
    residual = tl.load(residual_row_ptr + cols_idx, mask=mask, other=0.0)
    fused = x + residual
    acc = fused * fused
    mean_sq = tl.sum(acc, axis=0) / cols
    tl.store(mean_sq_ptr + row, mean_sq)


@triton.jit
def fused_residual_rmsnorm_kernel(x_ptr, residual_ptr, gamma_ptr, mean_sq_ptr,
                                  y_ptr, rows, cols, eps, x_stride_row,
                                  residual_stride_row, y_stride_row,
                                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = pid // tl.cdiv(cols, BLOCK_SIZE)
    block_col = pid % tl.cdiv(cols, BLOCK_SIZE)

    if row >= rows:
        return

    cols_idx = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols_idx < cols

    x = tl.load(x_ptr + row * x_stride_row + cols_idx, mask=mask, other=0.0)
    residual = tl.load(
        residual_ptr + row * residual_stride_row + cols_idx,
        mask=mask,
        other=0.0,
    )
    gamma = tl.load(gamma_ptr + cols_idx, mask=mask, other=0.0)
    mean_sq = tl.load(mean_sq_ptr + row)
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = gamma * (x + residual) * inv_rms
    tl.store(y_ptr + row * y_stride_row + cols_idx, y, mask=mask)


def fused_residual_rmsnorm_triton(x: torch.Tensor, residual: torch.Tensor,
                                  gamma: torch.Tensor, eps: float = 1e-5,
                                  block_size: int = 256) -> torch.Tensor:
    if x.ndim != 2 or residual.ndim != 2:
        raise ValueError(
            f"expected x and residual to be 2D, got x={tuple(x.shape)}, "
            f"residual={tuple(residual.shape)}"
        )
    if x.shape != residual.shape:
        raise ValueError(
            f"expected x and residual to share shape, got {tuple(x.shape)} "
            f"vs {tuple(residual.shape)}"
        )
    if gamma.ndim != 1 or gamma.shape[0] != x.shape[1]:
        raise ValueError(
            f"expected gamma shape ({x.shape[1]},), got {tuple(gamma.shape)}"
        )

    tensors = {"x": x, "residual": residual, "gamma": gamma}
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"expected {name} to live on CUDA device")
        if tensor.dtype != torch.float32:
            raise ValueError(f"expected {name} to be float32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"expected {name} to be contiguous")

    rows, cols = x.shape
    if cols > block_size:
        raise ValueError(
            f"expected cols <= block_size for this teaching kernel, got "
            f"cols={cols}, block_size={block_size}"
        )
    mean_sq = torch.empty((rows,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    row_mean_sq_kernel[(rows,)](
        x,
        residual,
        mean_sq,
        rows,
        cols,
        x.stride(0),
        residual.stride(0),
        BLOCK_SIZE=block_size,
    )

    grid = (rows * triton.cdiv(cols, block_size),)
    fused_residual_rmsnorm_kernel[grid](
        x,
        residual,
        gamma,
        mean_sq,
        y,
        rows,
        cols,
        eps,
        x.stride(0),
        residual.stride(0),
        y.stride(0),
        BLOCK_SIZE=block_size,
    )
    return y


def fused_residual_rmsnorm_reference(x: torch.Tensor, residual: torch.Tensor,
                                     gamma: torch.Tensor,
                                     eps: float = 1e-5) -> torch.Tensor:
    fused = x + residual
    mean_sq = fused.pow(2).mean(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    return gamma.unsqueeze(0) * fused * inv_rms


def build_inputs(rows: int, cols: int, device: torch.device):
    row_ids = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)
    col_ids = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(0)

    gamma_periodic = torch.sin((col_ids + 1.0) * 0.015)
    gamma_offset = ((torch.remainder(col_ids, 17.0)) - 8.0) * 0.01
    gamma = (1.0 + 0.1 * gamma_periodic + gamma_offset).reshape(cols).contiguous()

    base = torch.sin((row_ids + 3.0) * 0.013) + torch.cos((col_ids + 5.0) * 0.021)
    noise = (((row_ids * 11.0 + col_ids * 7.0) % 31.0) - 15.0) * 0.03
    residual_base = torch.cos((row_ids + 2.0) * 0.017) - torch.sin((col_ids + 9.0) * 0.019)
    residual_noise = (((row_ids * 5.0 + col_ids * 13.0) % 19.0) - 9.0) * 0.02

    x = (0.7 * base + noise).contiguous()
    residual = (0.4 * residual_base + residual_noise).contiguous()
    return x, residual, gamma


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Please run this script on a CUDA machine.")

    torch.manual_seed(0)

    rows = 1024
    cols = 256
    block_size = 256
    eps = 1e-5
    device = torch.device("cuda")

    x, residual, gamma = build_inputs(rows, cols, device)
    y = fused_residual_rmsnorm_triton(
        x,
        residual,
        gamma,
        eps=eps,
        block_size=block_size,
    )
    ref = fused_residual_rmsnorm_reference(x, residual, gamma, eps=eps)

    max_abs_diff = (y - ref).abs().max().item()
    ok = max_abs_diff < 2e-4

    if ok:
        print(
            "fused_residual_rmsnorm passed. "
            f"rows={rows}, cols={cols}, block_size={block_size}, "
            f"max_abs_diff={max_abs_diff}"
        )
        print(
            "sample output: "
            f"y[0]={y[0, 0].item()}, "
            f"y[1]={y[0, 1].item()}, "
            f"y[255]={y[0, cols - 1].item()}"
        )
    else:
        diff = (y - ref).abs()
        flat_idx = int(diff.argmax().item())
        row = flat_idx // cols
        col = flat_idx % cols
        print(f"fused_residual_rmsnorm failed. max_abs_diff={max_abs_diff}")
        print(
            f"first max diff at row={row}, col={col}: "
            f"got={y[row, col].item()}, ref={ref[row, col].item()}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
